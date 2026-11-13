"""Training-target relabelling for TRAIN rows (labels only, never features). See notes/relabel.md.

relabel(tx, variant) -> pl.Series "y" aligned with tx: relabelled train labels, -1 = drop the row from the fit set,
null on test rows. `variant` is a string of letters applied in this order (e.g. "AC", "ABCD"):
  A  clean_labels.clean: event-majority type + legit rows inside mule/velocity/collusion events
  B  flow mode C: flow chain (chain_size>=3) mule/legit -> layering_chain; legit in a mule fan-in burst -> mule_account
  C  takeover archetype: sim/ATO labels -> archetype type (arch_sim -> sim_swap, arch_ato -> ATO)   <- the one that helps
  D  ring: legit/qr_swap/refund_abuse rows in round-amount small-merchant bursts -> merchant_collusion
  E  drop fraud labels with no structural support for their class (`support` below)
  Z  device-id zone truth (feats/ts_event.py, train edges): every row on a zone device -> the zone's type (after C, so it
     also undoes C on sim-zone rows that carry the ATO archetype). notes/idea_zclean.md
  P  with Z: rows linked to a mule/layering/phishing/fake_cashback zone row within 6 h (receiver == zone sender, or sender
     == zone P2P receiver) -> the zone's type (85% label agreement, legit share ~13% = noise rate)
  S  with Z: sim/ATO rows inside the sim/ATO zones keep C's archetype type
Rules read the full-density module caches (common.run_feature), which never read labels. Rules that use labels
(A, and the label conditions here) see only the rows of `tx`, so pass the fit-set rows to avoid using val labels.
"""
import polars as pl
from common import LABEL2ID as L, run_feature
from clean_labels import clean

ROUND = [1000., 2000., 2500., 5000., 10000.]


def _feats(tx: pl.DataFrame) -> pl.DataFrame:
    cols = {"flow": ["chain_size", "rin_n12h", "rin_np_frac12h", "rin_r10_frac12h", "pair_n12h", "sin_n24h"],
            "takeover": ["sess_n", "arch_sim", "arch_ato"],
            "ring": ["rcv_n", "seg_n", "seg_r500", "s4_np_int_eve", "s10_np_int_eve"],
            "burst": ["dev_c300", "snd_c300"],
            "scam": ["bait_before_n", "pair_tiny_n", "rev_pair"]}
    f = tx.select("txn_id", "amount", "txn_type", "qr_vpa_mismatch")
    for m, cs in cols.items():
        f = f.join(run_feature(m).select("txn_id", *[pl.col(f"{m}__{c}").fill_nan(None).alias(c) for c in cs]),
                   on="txn_id", how="left", maintain_order="left")
    return f


def support(f: pl.DataFrame) -> pl.Expr:
    """Per-class structural support of a row (True = the row shows its class's event signature)."""
    tt, a = pl.col("txn_type"), pl.col("amount")
    rules = {
        "mule_account": (pl.col("rin_n12h") >= 3) | (pl.col("sin_n24h") >= 3),
        "velocity_bot": (pl.col("dev_c300") >= 3) | (pl.col("snd_c300") >= 3),
        "layering_chain": pl.col("chain_size") >= 2,
        "sim_swap": pl.col("sess_n") >= 2,
        "account_takeover": pl.col("sess_n") >= 2,
        "phishing_collect": (tt == "COLLECT") | (a <= 10) | (pl.col("bait_before_n") > 0),
        "qr_swap": tt == "QR",
        "refund_abuse": tt == "REFUND",
        "fake_cashback": (a <= 10) | (pl.col("bait_before_n") > 0) | (pl.col("pair_tiny_n") > 0) | (pl.col("rev_pair") > 0),
        "loan_app_extortion": (pl.col("s4_np_int_eve") >= 2) | (pl.col("s10_np_int_eve") >= 2),
        "merchant_collusion": (pl.col("seg_n") >= 3) & (pl.col("rcv_n") < 500),
    }
    e = pl.lit(True)
    for k, r in rules.items():
        e = pl.when(pl.col("y") == L[k]).then(r.fill_null(False)).otherwise(e)
    return e


def _zone(tx: pl.DataFrame, propagate: bool) -> pl.Series:
    """Zone label id per row (0 = none), train edges only (test rows are nulled by the caller); label-free."""
    import numpy as np
    from feats.ts_event import zone_ids
    d = tx["device_id"].str.slice(2).cast(pl.Int64).to_numpy()
    z = zone_ids(d, np.zeros(len(d), bool), [2 ** 62] * 12)
    if propagate:
        t = tx.select("txn_id", "ts", "sender_id", "receiver_id", "receiver_type").with_columns(pl.Series("z", z))
        k = t.filter(pl.col("z").is_in([L["mule_account"], L["layering_chain"], L["phishing_collect"], L["fake_cashback"]]))
        links = [(k.select(pl.col("sender_id").alias("key"), "z", pl.col("ts").alias("zt")), "receiver_id"),
                 (k.filter(pl.col("receiver_type") == "P2P").select(pl.col("receiver_id").alias("key"), "z", pl.col("ts").alias("zt")), "sender_id")]
        hit = pl.concat([t.filter(pl.col("z") == 0).select("txn_id", "ts", lk).join(zk, left_on=lk, right_on="key")
                         .filter((pl.col("ts") - pl.col("zt")).abs() <= 6 * 3600).select("txn_id", "z") for zk, lk in links])
        hit = hit.unique("txn_id", keep="first", maintain_order=True)
        zp = t.select("txn_id").join(hit, on="txn_id", how="left", maintain_order="left")["z"].fill_null(0).to_numpy()
        z = np.where(z > 0, z, zp)
    return pl.Series("z", z, pl.Int8)


def relabel(tx: pl.DataFrame, variant: str) -> pl.Series:
    y = tx["y"] if "A" not in variant else clean(tx)
    f = _feats(tx).with_columns(y.cast(pl.Int8).alias("y"))
    Y, c = pl.col("y"), lambda k: pl.lit(L[k], pl.Int8)
    if "B" in variant:
        chain = pl.col("chain_size") >= 3
        burst = (pl.col("rin_n12h") >= 6) & (pl.col("rin_np_frac12h") >= 0.7) & (pl.col("rin_r10_frac12h") >= 0.9) & (pl.col("pair_n12h") == 1)
        f = f.with_columns(pl.when(chain.fill_null(False) & Y.is_in([0, L["mule_account"]])).then(c("layering_chain"))
                           .when(burst.fill_null(False) & (Y == 0)).then(c("mule_account")).otherwise(Y).alias("y"))
    if "C" in variant:
        st = Y.is_in([L["sim_swap"], L["account_takeover"]])
        # type only: also sending 1-row-session sim/ATO labels to legit measured -0.006 pooled (notes/relabel.md)
        f = f.with_columns(pl.when(st & (pl.col("arch_sim") == 1)).then(c("sim_swap"))
                           .when(st & (pl.col("arch_ato") == 1)).then(c("account_takeover")).otherwise(Y).alias("y"))
    if "D" in variant:
        cb = (pl.col("amount").is_in(ROUND) & (pl.col("rcv_n") < 500) & (pl.col("seg_r500") >= 0.6) & (pl.col("seg_n") >= 3)
              & (pl.col("txn_type") != "REFUND") & (pl.col("qr_vpa_mismatch").fill_null(0) != 1)).fill_null(False)
        f = f.with_columns(pl.when(cb & Y.is_in([0, L["qr_swap"], L["refund_abuse"]])).then(c("merchant_collusion"))
                           .otherwise(Y).alias("y"))
    if "Z" in variant:
        sa = [L["sim_swap"], L["account_takeover"]]
        keep_c = pl.lit("S" in variant) & Y.is_in(sa) & pl.col("z").is_in(sa)  # S: keep C's sim/ATO type inside sim/ATO zones
        f = f.with_columns(_zone(tx, "P" in variant).alias("z")).with_columns(
            pl.when((pl.col("z") > 0) & ~keep_c).then(pl.col("z")).otherwise(Y).alias("y")).drop("z")
    if "E" in variant:
        f = f.with_columns(pl.when((Y > 0) & ~support(f)).then(pl.lit(-1, pl.Int8)).otherwise(Y).alias("y"))
    return f["y"].set(tx["is_test"], None)


if __name__ == "__main__":
    import sys
    from common import load_all, LABELS
    tx = load_all().filter(~pl.col("is_test"))
    for v in sys.argv[1:] or ["A", "B", "C", "D", "E"]:
        y2 = relabel(tx, v)
        assert y2.len() == tx.height and y2.null_count() == 0
        ch = pl.DataFrame({"y": tx["y"], "y2": y2}).filter(pl.col("y") != pl.col("y2"))
        print(f"variant {v}: changed {ch.height} rows")
        print(ch.group_by("y", "y2").len().sort("len", descending=True).head(15)
              .with_columns(pl.col("y").map_elements(lambda i: LABELS[i], return_dtype=pl.String),
                            pl.col("y2").map_elements(lambda i: LABELS[i] if i >= 0 else "DROP", return_dtype=pl.String)))
