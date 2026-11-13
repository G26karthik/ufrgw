"""Takeover session-consistency post-rule (notes/sc_takeover.md). Label-free: test features + the sub's own decode.

SC1: sender session = rows of one sender with gaps <= 2 h (test split only). drain row = P2P, new_payee == 1, integer amount,
     amount / balance_before > .4. A drain row off the test zone block, decoded a fraud class other than sim/ATO, whose session
     holds >= 2 OTHER drain rows all decoded one type t in {sim_swap, account_takeover} -> t.
Evidence (train labels, Jul/Aug/Sep): a drain row in a session with >= 2 other drain rows labelled t carries t .86-.87,
legit .09-.12, a non-sim/ATO fraud label only .02-.03, spread over 8+ types (noise, no concentration). Decode-conditioned OOF
hits (h6 judge): Aug 1 (labelled legit), Sep 0, so NOT validated out of time. Legit-decoded rows are left alone (OOF: 25/26 legit).
usage (cwd=work): python sc_rules_takeover.py subs/v3_final.csv sc_takeover
"""
import sys
import polars as pl
from common import ROOT, LABELS, load_all
from feats.ts_event import test_block

c = pl.col
SA = ["sim_swap", "account_takeover"]


def apply(sub_df, tx):
    lo, hi = test_block(tx)
    D = sub_df.select(c("id").alias("txn_id"), "label").join(
        tx.filter(c("is_test")).select("txn_id", "ts", "sender_id", "receiver_type", "new_payee", "amount", "balance_before",
                                       c("device_id").str.slice(2).cast(pl.Int64).alias("dn")),
        on="txn_id", how="left", maintain_order="left").with_row_index("i")
    assert D["ts"].null_count() == 0
    D = D.sort("sender_id", "ts").with_columns(
        (c("ts").diff().over("sender_id").fill_null(10**9) > 7200).cum_sum().over("sender_id").alias("sx"),
        ((c("receiver_type") == "P2P") & (c("new_payee") == 1) & (c("amount") % 1 == 0)
         & (c("amount") / c("balance_before") > .4)).alias("drain"))
    dr = D.filter(c("drain"))
    # per drain row: decodes of the OTHER drain rows of its session
    o = dr.select("i", "sender_id", "sx").join(dr.select("sender_id", "sx", c("i").alias("j"), c("label").alias("ol")), on=["sender_id", "sx"]) \
        .filter(c("i") != c("j")).group_by("i").agg(pl.len().alias("n"), c("ol").unique().alias("ols"))
    hit = dr.join(o, on="i").filter(c("n") >= 2, c("ols").list.len() == 1, ~c("dn").is_between(lo, hi - 1),
                                    ~c("label").is_in(["legit", *SA])) \
        .with_columns(c("ols").list.first().alias("to")).filter(c("to").is_in(SA))
    lab = sub_df["label"].to_list()
    log = []
    for i, t, frm, to in hit.select("i", "txn_id", "label", "to").iter_rows():
        log.append((t, frm, to, "SC1")); lab[i] = to
    return sub_df.with_columns(pl.Series("label", lab)), pl.DataFrame(log, schema=["id", "from", "to", "rule"], orient="row")


if __name__ == "__main__":
    inp, out = sys.argv[1], sys.argv[2]
    sub = pl.read_csv(ROOT / inp)
    new, log = apply(sub, load_all())
    print("rows changed:", log.height, log.rows())
    assert log.height <= 5, "SC1 expected ~1 test row on v3_final; inspect before shipping more"
    path = ROOT / "subs" / f"{out}.csv"
    assert not path.exists(), f"{path} exists; never overwrite"
    assert new.height == sub.height and new["id"].equals(sub["id"]) and new["label"].is_in(LABELS).all()
    new.write_csv(path); log.write_csv(ROOT / "notes" / f"{out}_changes.csv")
    print("wrote", path)
