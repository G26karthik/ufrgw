"""Accepted v3 post-rules, applied to any decoded test submission (id,label). Label-free: only test features + the sub's own decode.

Rules (notes/v3_final.md; evidence in notes/v3_cells.md, notes/v3_zprop.md):
  QRN (v3_cells, subsumes qrm): QR & MERCHANT & qr_vpa_mismatch null & amount % 100 != 0 & no other null-mm merchant-QR row at
      that receiver in the split -> qr_swap. 0 train rows (asserted).
  R2 (v3_zprop takeover): off-zone row decoded sim/ATO; zone rows (sim/ATO zone) sent by the same sender within +-1 h, one zone
      type, != decode -> that zone type.
  R3 (v3_zprop chain hop): off-zone row decoded mule/vel/lay; zone rows (mule/vel/lay zone) whose receiver is this row's sender
      (R3a) or whose sender is this row's receiver (R3b) within +-1 h, one zone type, != decode -> that zone type.
  Zone devices in BAND (test mule|velocity edge where DP and victim cores disagree) are never zprop sources.
Precedence: zprop hits are computed on the input decode; QRN is applied last and wins (rows are disjoint in practice: QRN rows
are merchant-QR rows decoded legit/qr_swap, zprop rows are decoded in {mule,vel,lay,sim,ATO}; asserted on overlap).
Zones: test edges from ts_judge_post.zones (dp_edges) on the input decode unless --edges is given; the zone override makes every
zone-device row carry its zone type, so the DP recovers the judge's edges (checked in scratch_v3_final/chk_final.py).
usage (cwd=work): python v3_rules.py subs/ts_judge_h6_qrm.csv v3_final [--edges e0,e1,...]
"""
import argparse
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all
from feats.ts_event import zone_ids, test_block
import ts_judge_post as J

c = pl.col
BAND = [(473749, 473783)]
G1, G2 = [L["mule_account"], L["velocity_bot"], L["layering_chain"]], [L["sim_swap"], L["account_takeover"]]
# (name, group, candidate key, zone-row key, window s)
ZRULES = [("R2", G2, "sender_id", "sender_id", 3600), ("R3a", G1, "sender_id", "receiver_id", 3600),
          ("R3b", G1, "receiver_id", "sender_id", 3600)]


def qrn_ids(tx):
    m = tx.filter((c("txn_type") == "QR") & (c("receiver_type") == "MERCHANT") & c("qr_vpa_mismatch").is_null())
    m = m.with_columns(pl.len().over("receiver_id", "is_test").alias("n_rcv"))
    hit = m.filter((c("amount") % 100 != 0) & (c("n_rcv") == 1))
    assert hit.filter(~c("is_test")).height == 0, "QRN must have 0 train rows"
    return hit.filter(c("is_test"))["txn_id"]


def test_zones(D, q, tx, edges=None):
    """D: test rows (txn_id, dn) aligned with decode q -> zone label id per row (0 = none), edges."""
    if edges is None:
        _, edges = J.zones(D, q, *test_block(tx))
    return zone_ids(D["dn"].to_numpy(), np.ones(D.height, bool), edges), edges


def zprop(D, q, z):
    """D: test rows (txn_id, ts, sender_id, receiver_id, dn) aligned with decode q and zones z -> DataFrame txn_id, zz, rule."""
    D = D.with_columns(pl.Series("q", q), pl.Series("z", z))
    src = D.with_columns(pl.when(pl.any_horizontal([c("dn").is_between(lo, hi - 1) for lo, hi in BAND]))
                         .then(0).otherwise(c("z")).alias("z"))
    F = D.filter(c("z") == 0)
    out = []
    for nm, g, a, b, W in ZRULES:
        zr = src.filter(c("z").is_in(g)).select(c("ts").alias("zts"), c("z").alias("zz"), c(b).alias("_k"))
        m = F.filter(c("q").is_in(g)).join(zr, left_on=a, right_on="_k").filter((c("ts") - c("zts")).abs() <= W) \
            .group_by("txn_id").agg(c("zz").unique().alias("zs"), c("q").first())
        out.append(m.filter(c("zs").list.len() == 1).with_columns(c("zs").list.first().alias("zz"), pl.lit(nm).alias("rule"))
                   .filter(c("zz") != c("q")).select("txn_id", "zz", "rule"))
    h = pl.concat(out).group_by("txn_id").agg(c("zz").unique(), c("rule").str.join("+"))
    return h.filter(c("zz").list.len() == 1).with_columns(c("zz").list.first())  # conflicting types -> left alone


def apply_rules(sub, tx, edges=None):
    """sub: id,label (sample order) -> (new sub, per-row change log DataFrame id/from/to/rule)."""
    D = sub.select(c("id").alias("txn_id")).join(
        tx.select("txn_id", "ts", "sender_id", "receiver_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")),
        on="txn_id", how="left", maintain_order="left")
    assert D["dn"].null_count() == 0
    q = np.array([L[s] for s in sub["label"].to_list()])
    z, edges = test_zones(D, q, tx, edges)
    print("test edges", list(edges))
    h = zprop(D, q, z)
    lab = sub["label"].to_numpy().copy()
    pos = {t: i for i, t in enumerate(sub["id"].to_list())}
    log = []
    for t, zz, r in h.iter_rows():
        log.append((t, lab[pos[t]], LABELS[zz], r)); lab[pos[t]] = LABELS[zz]
    zp = set(h["txn_id"].to_list())
    for t in qrn_ids(tx).to_list():
        assert t not in zp, f"QRN/zprop overlap {t}"
        if lab[pos[t]] != "qr_swap":
            log.append((t, lab[pos[t]], "qr_swap", "QRN")); lab[pos[t]] = "qr_swap"
    log = pl.DataFrame(log, schema=["id", "from", "to", "rule"], orient="row")
    return sub.with_columns(pl.Series("label", lab)), log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--edges", default="", help="comma-separated test zone edges (default: DP on the input decode)")
    a = ap.parse_args()
    sub = pl.read_csv(ROOT / a.inp)
    new, log = apply_rules(sub, load_all(), [int(x) for x in a.edges.split(",")] if a.edges else None)
    print("rows changed:", log.height)
    print(log.group_by("rule", "from", "to").len().sort("rule").rows())
    path = ROOT / "subs" / f"{a.out}.csv"
    assert not path.exists(), f"{path} exists; never overwrite"
    assert new.height == sub.height and new["id"].equals(sub["id"]) and new["label"].is_in(LABELS).all()
    new.write_csv(path); log.write_csv(ROOT / "notes" / f"{a.out}_changes.csv")  # not in subs/: keep candidate globs clean
    print("wrote", path)
