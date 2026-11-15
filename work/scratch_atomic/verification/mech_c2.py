"""Audit columns for agent A's C2 cells (label-free, row-level original fields only)."""
import polars as pl

c = pl.col


def build(tx):
    nonint = c("amount") != c("amount").round(0)
    return tx.select(
        "txn_id",
        ((c("txn_type") == "QR") & (c("receiver_type") == "MERCHANT") & (c("qr_vpa_mismatch") == 1) & nonint).fill_null(False).cast(pl.Float32).alias("va_c2__qr"),
        ((c("txn_type") == "QR") & (c("receiver_type") == "MERCHANT") & (c("qr_vpa_mismatch") == 1)).fill_null(False).cast(pl.Float32).alias("va_c2__qr_any_amt"),
        ((c("txn_type") == "COLLECT") & (c("receiver_type") == "P2P") & (c("new_payee") == 1) & ~nonint).fill_null(False).cast(pl.Float32).alias("va_c2__collect"),
        ((c("app_version").cast(pl.Float64) < 4) & (c("ip_risk_score") >= 0.441)).fill_null(False).cast(pl.Float32).alias("va_c2__J"))
