"""Refund post-rule SEGR (notes/sc_refund.md): round-amount small-merchant segments with 4 <= n <= 8 rows are refund events.

Segment = feats/pairs.py definition (merchant receiver, non-REFUND, amount in {1000,2000,2500,5000,10000}, qr_vpa_mismatch != 1,
24 h gap per merchant), built per split (train / test). Label-free: uses only test columns and the sub's own decode.
Rule: a row in a test segment with 4 <= n <= 8 decoded merchant_collusion -> refund_abuse.
Train evidence (Jul-Sep segment labels): 0 collusion rows in any segment with 4 <= n <= 8 (Jul+Aug 45 refund segs, Sep 20);
smallest collusion segment with n >= 4 is n=10. Row precision for refund in 4-8 segments: Jul+Aug .82, Sep .76. Naive-Bayes LR refund:collusion for an n=8, 18 h segment >= ~20 per factor (size, duration); sender
concentration is higher in refund segments than in collusion ones, so it does not favour collusion.
OOF: the rule fires on 0 Aug/Sep rows (h6 never splits a small segment refund/collusion), so it cannot be scored on validation.
usage (cwd=work): python sc_rules_refund.py subs/v3_final.csv sc_refund
"""
import argparse
import polars as pl
from common import ROOT, LABELS, load_all

c = pl.col
G = 24 * 3600
RND = [1000.0, 2000.0, 2500.0, 5000.0, 10000.0]


def seg_ids(tx):
    m = (tx.filter(c("is_test") & (c("receiver_type") == "MERCHANT") & (c("txn_type") != "REFUND") & c("amount").is_in(RND)
                   & (c("qr_vpa_mismatch").fill_null(0) != 1)).select("txn_id", "ts", "receiver_id").sort("receiver_id", "ts")
         .with_columns(((c("ts").diff().over("receiver_id").fill_null(G + 1) > G).cum_sum()).alias("seg")))
    return m.with_columns(pl.len().over("seg").alias("n")).filter(c("n").is_between(4, 8))["txn_id"]


def apply(sub_df, tx):
    """sub_df: id,label -> new sub_df (same order)."""
    hit = c("id").is_in(seg_ids(tx).implode()) & (c("label") == "merchant_collusion")
    return sub_df.with_columns(pl.when(hit).then(pl.lit("refund_abuse")).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    a = ap.parse_args()
    sub = pl.read_csv(ROOT / a.inp)
    new = apply(sub, load_all())
    ch = new["label"] != sub["label"]
    print("rows changed:", int(ch.sum()), pl.DataFrame({"from": sub["label"].filter(ch), "to": new["label"].filter(ch)}).group_by("from", "to").len().rows())
    assert new.height == sub.height and new["id"].equals(sub["id"]) and new["label"].is_in(LABELS).all()
    if a.inp.endswith("v3_final.csv"):
        assert int(ch.sum()) == 3, "expected 3 rows on v3_final"
    path = ROOT / "subs" / f"{a.out}.csv"
    assert not path.exists(), f"{path} exists; never overwrite"
    new.write_csv(path)
    print("wrote", path)
