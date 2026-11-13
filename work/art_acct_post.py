"""Account-link velocity propagation on top of ts_judge_h6 (notes/art_acct.md).

Rule (chosen on Aug from scratch_art_acct/a14.py grid, scored on Sep): a row NOT on a zone device, decoded fraud, whose P2P
receiver account also received velocity-zone rows (pool DV0420000-422999 or the velocity zone) within +-24 h, and no other
zone type in that window -> velocity_bot. Label-free at prediction time: test zones are ts_judge_h6's DP zones (zt).
usage (cwd=work): python art_acct_post.py [TAG=h6] [OUT=art_acct]   -> subs/OUT.csv (built on subs/ts_judge_TAG.csv)
"""
import sys
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all
from feats.ts_event import zone_ids
from train import write_sub
from ts_judge_post import get, gate

c = pl.col
W = 86400
VEL = L["velocity_bot"]


def vel_link(d, ids):
    """bool per txn in ids: rule applies (before the decoded-fraud condition)."""
    tx = load_all().select("txn_id", "ts", "is_test", pl.when(c("receiver_id").str.starts_with("AC")).then(c("receiver_id")).alias("rn"),
                           c("device_id").str.slice(2).cast(pl.Int64).alias("dn"))
    z = zone_ids(tx["dn"].to_numpy(), np.zeros(tx.height, bool), [10**9])  # train rows: TR_EDGES (+ pool)
    tx = tx.with_columns(pl.Series("z", z)).join(pl.DataFrame({"txn_id": d["tid"], "zt": d["zt"]}), on="txn_id", how="left") \
           .with_columns(pl.when(c("is_test")).then(c("zt")).otherwise(c("z")).alias("z"))
    Z = tx.filter((c("z") > 0) & c("rn").is_not_null()).select("rn", c("ts").alias("zts"), c("z").alias("zz"))
    g = (tx.filter((c("z") == 0) & c("rn").is_not_null()).select("txn_id", "rn", "ts").join(Z, on="rn")
         .filter((c("ts") - c("zts")).abs() <= W).group_by("txn_id").agg((c("zz") == VEL).all().alias("hit")))
    return pl.DataFrame({"txn_id": ids}).join(g, on="txn_id", how="left", maintain_order="left")["hit"].fill_null(False).to_numpy()


def main(tag, out):
    d = get(tag)
    mo, mt = vel_link(d, d["oid"]), vel_link(d, d["tid"])
    ap = lambda q, m: np.where(m & (q > 0), VEL, q)
    gate(d, ap(d["qo"], mo), ap(d["qa"], mo), "art_acct")
    qt = ap(d["qt"], mt)
    ref = pl.read_csv(ROOT / "subs" / f"ts_judge_{tag}.csv")["label"].to_numpy()
    lab = np.array(LABELS)[qt]
    ch = lab != ref
    print(f"test rows changed vs subs/ts_judge_{tag}.csv: {int(ch.sum())}")
    print(pl.DataFrame({"from": ref[ch], "to": lab[ch]}).group_by("from", "to").len().sort("len", descending=True))
    write_sub(np.eye(len(LABELS), dtype=np.float32)[qt], np.zeros(len(LABELS)), out)


if __name__ == "__main__":
    a = sys.argv[1:] + [None, None]
    main(a[0] or "h6", a[1] or "art_acct")
