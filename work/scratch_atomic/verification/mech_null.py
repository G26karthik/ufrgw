"""Null mechanism (noise floor of the harness): one column of uniform noise from a hash of txn_id. Carries no information, so
the recovered / damaged / macro-F1 differences it produces are what 'one more column, same seeds' does by chance."""
import polars as pl


def build(tx):
    return tx.select("txn_id", (pl.col("txn_id").hash(seed=11) % 1_000_003 / 1_000_003).cast(pl.Float32).alias("va_null__u"))
