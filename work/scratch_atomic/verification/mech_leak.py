"""Self-test of the permutation check: a deliberately label-dependent column (fraud-label share of the sender). va_feat.py must abort."""
import polars as pl


def build(tx):
    return tx.select("txn_id", (pl.col("y") > 0).mean().over("sender_id").cast(pl.Float32).alias("va_leak__te"))
