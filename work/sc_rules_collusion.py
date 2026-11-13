"""Collusion post-rule (notes/sc_collusion.md). Label-free: test features + the sub's own decode.

RS (refund-sized segment): rows at a merchant with amount in the round set {1000,2000,2500,5000,10000}, txn_type != REFUND,
qr_vpa_mismatch != 1 are cut into per-merchant segments with a 24 h gap (whole data, train+test, so events crossing the split
stay whole). A row decoded merchant_collusion inside a segment of 4-8 such rows -> refund_abuse.
Train evidence (label-only): such segments hold 0 collusion labels in Jul 0/121, Aug 0/142, Sep 0/108 rows (refund .83/.81/.76,
the rest legit noise and a little qr); collusion appears only from n=10 up. Judge OOF decodes 0 rows into the cell on Aug/Sep.
usage (cwd=work): python sc_rules_collusion.py subs/v3_final.csv sc_collusion
"""
import sys
import polars as pl
from common import ROOT, LABELS, load_all

c = pl.col
RND = [1000.0, 2000.0, 2500.0, 5000.0, 10000.0]
GAP = 24 * 3600


def segments(tx):
    """txn_id, is_test, n (segment size) for round-set merchant rows."""
    return (tx.filter((c("receiver_type") == "MERCHANT") & (c("txn_type") != "REFUND") & c("amount").is_in(RND)
                      & (c("qr_vpa_mismatch").fill_null(0) != 1))
            .sort("receiver_id", "ts").with_columns(((c("ts").diff().over("receiver_id").fill_null(GAP + 1) > GAP).cum_sum()).alias("seg"))
            .with_columns(pl.len().over("seg").alias("n")).select("txn_id", "is_test", "n", "y"))


def apply(sub_df, tx):
    cell = segments(tx).filter(c("is_test") & c("n").is_between(4, 8))["txn_id"]
    hit = c("id").is_in(cell.implode()) & (c("label") == "merchant_collusion")
    return sub_df.with_columns(pl.when(hit).then(pl.lit("refund_abuse")).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    inp, out = sys.argv[1], sys.argv[2]
    tx = load_all()
    s = segments(tx).filter(~c("is_test") & c("n").is_between(4, 8))
    assert s.height > 300 and (s["y"] == LABELS.index("merchant_collusion")).sum() == 0, "train cell must hold 0 collusion labels"
    sub = pl.read_csv(ROOT / inp)
    new = apply(sub, tx)
    ch = sub.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows())
    path = ROOT / "subs" / f"{out}.csv"
    assert not path.exists(), f"{path} exists; never overwrite"
    assert new.height == sub.height and new["id"].equals(sub["id"]) and new["label"].is_in(LABELS).all()
    new.write_csv(path)
    print("wrote", path)
