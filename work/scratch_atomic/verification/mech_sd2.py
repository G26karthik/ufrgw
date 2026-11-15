"""Audit columns, alternative readings of agent A's `sd_first` trace: does the device of a row equal the device of the sender's
previous / next row, is it the sender's first row, and is a later row of the sender on the same device present (non-causal,
audit only). All label-free."""
import polars as pl

c = pl.col


def build(tx):
    t = tx.sort("ts", "txn_id")
    o = lambda e: e.over("sender_id")
    return t.select(
        "txn_id",
        (c("device_id") == o(c("device_id").shift(1))).cast(pl.Float32).alias("va_sd2__same_prev"),   # null on the sender's first row
        (c("device_id") == o(c("device_id").shift(-1))).cast(pl.Float32).alias("va_sd2__same_next"),  # null on the sender's last row
        (o(c("ts").rank("ordinal")) == 1).cast(pl.Float32).alias("va_sd2__snd_first"),
        (c("ts").rank("ordinal").over("sender_id", "device_id") == pl.len().over("sender_id", "device_id")).cast(pl.Float32).alias("va_sd2__sd_last"),
        pl.len().over("sender_id", "device_id").cast(pl.Float32).alias("va_sd2__sd_n"),
        c("device_id").n_unique().over("sender_id").cast(pl.Float32).alias("va_sd2__snd_ndev"))
