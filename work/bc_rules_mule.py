"""bc_mule: resolve the disputed test mule|velocity band by device-id occupancy, then propagate zone types through fan-ins.

Evidence (notes/bc_mule.md, work/scratch_bc_mule/a4_occ.py):
  Fresh-phone zones have class-specific occupancy (used ids / ids): train mule .53, velocity .35; test mule .68, velocity .33.
  DV0473782 being mule needs the whole old band [473749, 473783) to be mule zone. The band has 11 used ids of 34. Train
  34-id windows (overlapping): mule zone 0/625 with <=11 used (min 12; <=12 in 1%), velocity zone 41%. So LR >= ~50 for
  velocity, conservative since the test mule zone is denser than train. The changepoint itself is noisy at test scale
  (block bootstrap: |err| <= 8 in only ~25%), so it is NOT used to place [473749, 473757) -> BAND guard there.
  The victim-core "mule up to 473782" comes from the host's own decode (velocity fan-ins decode mule 15/16).
Rules (zone rows outside BAND are sources; candidate = off-zone P2P row decoded mule/velocity/layering):
  R1  (v3_zprop R1, unchanged): a zone row pays the candidate's receiver within +-48 h.
  R1f (forward): the candidate's sender received a zone row in the previous 48 h (mule forwards of a hosted fan-in).
  One zone type among the links, != decode -> that type. Train label purity / OOF gate: scratch_bc_mule/chk_rules.py.
usage (cwd=work): python bc_rules_mule.py   -> subs/bc_mule.csv from subs/sc_final.csv
"""
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all
from v3_rules import test_zones

c = pl.col
BAND = [(473749, 473757)]  # occupancy-ambiguous ids of the old band [473749, 473783)
G1 = [L["mule_account"], L["velocity_bot"], L["layering_chain"]]
W = 48 * 3600


def hits(D, q, z, band=BAND, band_as=None):
    """D: rows (txn_id, ts, sender_id, receiver_id, receiver_type, dn) aligned with decode q and zone ids z -> txn_id, zz, rule.
    band_as: None = BAND devices are no source; else they are sources of that type (variant bc_mule_b: mule)."""
    D = D.with_columns(pl.Series("q", q), pl.Series("z", z))
    inb = pl.any_horizontal([c("dn").is_between(lo, hi - 1) for lo, hi in band]) if band else pl.lit(False)
    sz = c("z") if band_as is None else pl.when(inb).then(band_as).otherwise(c("z"))
    src = D.with_columns(sz.alias("sz")).filter(c("sz").is_in(G1) & (~inb if band_as is None else pl.lit(True))).select(
        c("ts").alias("zts"), c("sz").alias("zz"), "receiver_id")
    F = D.filter((c("z") == 0) & c("q").is_in(G1) & (c("receiver_type") == "P2P"))
    r1 = F.join(src, on="receiver_id").filter((c("ts") - c("zts")).abs() <= W).with_columns(pl.lit("R1").alias("rule"))
    r1f = F.join(src.rename({"receiver_id": "sender_id"}), on="sender_id").filter(
        (c("ts") > c("zts")) & (c("ts") - c("zts") <= W)).with_columns(pl.lit("R1f").alias("rule"))
    h = pl.concat([r1.select("txn_id", "q", "zz", "rule"), r1f.select("txn_id", "q", "zz", "rule")]).group_by("txn_id").agg(
        c("q").first(), c("zz").unique(), c("rule").unique().sort().str.join("+"))
    return h.filter(c("zz").list.len() == 1).with_columns(c("zz").list.first()).filter(c("zz") != c("q")).select("txn_id", "zz", "rule")


def apply(sub_df, tx, edges=None, band_as=None):
    """sub_df: id,label (sample order) -> new sub_df (zone edges: DP on the sub's own decode, as v3_rules)."""
    D = sub_df.select(c("id").alias("txn_id")).join(
        tx.select("txn_id", "ts", "sender_id", "receiver_id", "receiver_type", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")),
        on="txn_id", how="left", maintain_order="left")
    assert D["dn"].null_count() == 0
    q = np.array([L[s] for s in sub_df["label"].to_list()])
    z, edges = test_zones(D, q, tx, edges)
    h = hits(D, q, z, band_as=band_as)
    pos = {t: i for i, t in enumerate(sub_df["id"].to_list())}
    lab = sub_df["label"].to_numpy().copy()
    for t, zz, r in h.iter_rows():
        lab[pos[t]] = LABELS[zz]
    apply.log = h.with_columns(c("zz").map_elements(lambda k: LABELS[k], return_dtype=pl.String).alias("to"))
    apply.edges = edges
    return sub_df.with_columns(pl.Series("label", lab))


if __name__ == "__main__":
    import sys
    b = "--b" in sys.argv  # variant: guarded ids [473749, 473757) are mule sources
    out = ROOT / "subs" / ("bc_mule_b.csv" if b else "bc_mule.csv")
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / "subs" / "sc_final.csv")
    new = apply(base, load_all(), band_as=L["mule_account"] if b else None)
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("test edges", apply.edges)
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows())
    print(apply.log.sort("rule").rows())
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    new.write_csv(out)
    print("wrote", out)
