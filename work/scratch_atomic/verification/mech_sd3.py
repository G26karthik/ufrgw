"""Audit columns: device number and sender number, to split agent A's `sd_first` trace by the device classes of
notes/fresh_errors.md (own: device number == sender number; base_other: < 420000; pool 420000-422999; fraud zones 423000-428454;
legit fresh-phone region >= 428455, notes/art_acct.md:25)."""
import polars as pl

c = pl.col


def build(tx):
    return tx.select("txn_id", c("device_id").str.slice(2).cast(pl.Int64).cast(pl.Float64).alias("va_sd3__dn"),
                     c("sender_id").str.slice(2).cast(pl.Int64).cast(pl.Float64).alias("va_sd3__sn"))
