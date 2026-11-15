"""Audit columns for agent A's `sd_first` trace (first row of a sender on a device that is not the sender's first row), causal
variants only (rank by time inside the frame, so a row never depends on later rows)."""
import polars as pl

c = pl.col


def build(tx):
    num = lambda k: c(k).str.slice(2).cast(pl.Int64)
    return tx.sort("ts", "txn_id").select(
        "txn_id",
        ((c("ts").rank("ordinal").over("sender_id", "device_id") == 1) & (c("ts").rank("ordinal").over("sender_id") > 1)).cast(pl.Float32).alias("va_sd__first"),
        (num("device_id") == num("sender_id")).cast(pl.Float32).alias("va_sd__home"),
        (c("ts").rank("ordinal").over("sender_id", "device_id") - 1).cast(pl.Float32).alias("va_sd__k_before"),
        (c("ts").rank("ordinal").over("sender_id") - 1).cast(pl.Float32).alias("va_sd__snd_before"))
