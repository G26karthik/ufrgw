"""pairs: features for the top core-row type confusions (see notes/confusion.md).

Only one confusion is separable: small-merchant round-amount bursts filed as qr_swap / refund_abuse / merchant_collusion.
They share row signatures (amount in {1000,2000,2500,5000,10000}, QR or PAY, no mismatch) and differ by burst size:
qr ~3 rows / 2 senders / <6 h, refund ~5 rows / 3 senders / ~10 h, collusion ~17 rows / 4 senders / ~45 h, colluders repeat.
ring's segments (all rows, 12 h gap) blur this with background rows and split bursts; here segments are built on the
round-amount rows only, per merchant, with a 24 h gap. NaN on every other row. No labels read.
Build it at FULL density (common.run_feature("pairs"), see scratch_confusion/mk_matrix_fd.py), not density.run_feature_matched:
density.halves links fraud rows only by sender / sender|merchant / device, so a multi-sender merchant burst is split across
the two halves (81% of refund rows, 92% of collusion rows get a smaller segment) while Oct bursts stay whole. These segments
are ~all fraud rows, so the full-density build already matches Oct. dm build: +.0003 pooled; full: +.0059 (notes/confusion.md).
"""
import polars as pl

P = "pairs__"
G = 24 * 3600
RND = [1000.0, 2000.0, 2500.0, 5000.0, 10000.0]


def build(tx: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    m = (tx.filter((c("receiver_type") == "MERCHANT") & (c("txn_type") != "REFUND") & c("amount").is_in(RND)
                   & (c("qr_vpa_mismatch").fill_null(0) != 1))
         .select("txn_id", "ts", "sender_id", "receiver_id", "txn_type").sort("receiver_id", "ts")
         .with_columns(((c("ts").diff().over("receiver_id").fill_null(G + 1) > G).cum_sum()).alias("seg")))
    m = m.with_columns(
        pl.len().over("seg").alias("n"), c("sender_id").n_unique().over("seg").alias("nsnd"),
        ((c("ts").max() - c("ts").min()).over("seg") / 3600).alias("dur_h"), pl.len().over("seg", "sender_id").alias("pair_n"),
        (c("txn_type") == "QR").mean().over("seg").alias("qr_share"),
        ((c("ts").max() - c("ts").min()).over("seg", "sender_id") / 3600).alias("pair_dur_h"))
    m = m.with_columns(c("pair_n").max().over("seg").alias("pair_max"), (c("n") / c("nsnd")).alias("rows_per_snd"))
    cols = ["n", "nsnd", "dur_h", "pair_n", "qr_share", "pair_dur_h", "pair_max", "rows_per_snd"]
    f = m.select("txn_id", *[c(x).cast(pl.Float32).alias(P + "rseg_" + x) for x in cols])
    return tx.select("txn_id").join(f, on="txn_id", how="left", maintain_order="left").fill_null(float("nan"))  # NaN like the matrix
