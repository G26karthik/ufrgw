"""Does the dropped-field trace of agent A's C1 (high ip_risk_score, old app_version on fraud-path rows) exist in October?
Label-free test: stamp rows S = non-QR rows with qr_vpa_mismatch == 0 (fraud-path marker, notes/fresh_columns.md section 2) against
all other rows, per month, test rows separately. No label and no model output is used.
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_ipoct.py"""
import polars as pl
from common import load_all

c = pl.col
tx = load_all().with_columns(((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False).alias("S"),
                             (c("ip_risk_score").rank() / pl.len()).over("month", "is_test").alias("ip_rank"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
print(tx.group_by("month", "is_test", "S").agg(pl.len().alias("rows"), c("ip_rank").mean().round(3).alias("mean ip rank in month"),
      (c("ip_rank") >= .95).mean().round(3).alias("share ip rank >= .95"), c("ip_risk_score").median().round(3).alias("median ip"),
      (c("app_version").cast(pl.Float64) < 4).mean().round(3).alias("share app < 4")).sort("is_test", "month", "S"))
