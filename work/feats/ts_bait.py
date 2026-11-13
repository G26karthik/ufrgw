"""ts_bait: event-level phishing vs fake_cashback type for bait pairs (notes/typesplit_bait.md).

Bait event = pair (f, v) with a tiny integer P2P f->v (Rs 1-10); rows = all f->v and v->f rows of the pair.
  nb = 0 (unanswered bait)                         -> phishing   (train: 99% ph)
  nb >= 1 and (nf >= 2 or dt > 180 min or the payback is not new_payee) -> fake_cashback (train: ~100% fc)
  nf = 1, nb = 1, dt <= 180                        -> ambiguous  (ph 60 / fc 40, no label-free split found)
ts_bait__rule: 0 not a bait event (or nf > 2 / nb > 1), 1 phishing, 2 fake_cashback, 3 ambiguous.
"""
import polars as pl

c = pl.col


def events(tx: pl.DataFrame) -> pl.DataFrame:
    """txn_id -> rule (rows of bait pairs only)."""
    tiny = tx.filter((c("receiver_type") == "P2P") & (c("amount") <= 10) & (c("amount") == c("amount").round(0)))
    pairs = tiny.select(c("sender_id").alias("f"), c("receiver_id").alias("v")).unique()
    cols = ["txn_id", "ts", "new_payee"]
    fw = tx.select(*cols, "sender_id", "receiver_id").join(pairs, left_on=["sender_id", "receiver_id"], right_on=["f", "v"]) \
        .select(*cols, c("sender_id").alias("f"), c("receiver_id").alias("v"), pl.lit(True).alias("fwd"))
    bk = tx.select(*cols, "sender_id", "receiver_id").join(pairs, left_on=["receiver_id", "sender_id"], right_on=["f", "v"]) \
        .select(*cols, c("receiver_id").alias("f"), c("sender_id").alias("v"), pl.lit(False).alias("fwd"))
    R = pl.concat([fw, bk])
    E = R.group_by("f", "v").agg(
        c("fwd").sum().alias("nf"), (~c("fwd")).sum().alias("nb"),
        ((c("ts").filter(~c("fwd")).min() - c("ts").filter(c("fwd")).max()) / 60).alias("dt"),
        c("new_payee").filter(~c("fwd")).min().alias("v_np"))
    E = E.with_columns(
        pl.when((c("nf") > 2) | (c("nb") > 1)).then(0)
        .when(c("nb") == 0).then(1)
        .when((c("nf") >= 2) | (c("dt") > 180) | (c("v_np") == 0)).then(2)
        .otherwise(3).cast(pl.Int8).alias("rule"))
    return R.join(E.select("f", "v", "rule"), on=["f", "v"]).group_by("txn_id").agg(c("rule").max())


def build(tx: pl.DataFrame) -> pl.DataFrame:
    r = tx.select("txn_id").join(events(tx), on="txn_id", how="left", maintain_order="left").with_columns(c("rule").fill_null(0))
    return r.rename({"rule": "ts_bait__rule"})
