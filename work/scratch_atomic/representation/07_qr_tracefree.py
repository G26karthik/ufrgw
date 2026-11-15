"""Agent A: what drives the trace-free qr_swap AUC .56 of 04_mix.py part 2?  cwd=work, PYTHONPATH=.
python scratch_atomic/representation/07_qr_tracefree.py > scratch_atomic/representation/07_out.txt
Cohort = home device, no stamp, app_version >= 4, ip < July q95, non-integer amount (same filter as 04 part 2).
Per month: cell (txn_type, receiver_type, qr_vpa_mismatch) counts for qr_swap-labelled rows against legit-labelled rows,
and the final decoder's label on them (benchmark N, months 8 and 9)."""
import polars as pl
from common import load_all

c = pl.col; OUT = "scratch_atomic/representation/"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
Z = tx.filter((num("device_id") == num("sender_id")).fill_null(False) & ~((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
              & (c("app_version") >= 4) & (c("ip_risk_score") < q95) & (c("amount") != c("amount").round(0)))
Z = Z.join(pl.read_parquet(f"{OUT}held_N.parquet").select("txn_id", "q"), on="txn_id", how="left").with_columns(
    pl.when(c("qr_vpa_mismatch").is_null()).then(pl.lit("null")).otherwise(c("qr_vpa_mismatch").cast(pl.Int8).cast(pl.String)).alias("mm"))
for m in (7, 8, 9):
    M = Z.filter(c("month") == m); nq, nl = M.filter(c("y") == 7).height, M.filter(c("y") == 0).height
    g = M.group_by("txn_type", "receiver_type", "mm").agg(
        (c("y") == 7).sum().alias("qr_label"), (c("y") == 0).sum().alias("legit_label"), ((c("y") > 0) & (c("y") != 7)).sum().alias("other_fraud_label"),
        ((c("y") == 7) & (c("q") == 0)).sum().alias("qr_label_dec_legit"), ((c("y") == 0) & (c("q") > 0)).sum().alias("legit_label_dec_fraud"),
        ((c("y") == 0) & (c("q") == 0)).sum().alias("legit_label_dec_legit")).with_columns(
        (c("qr_label") / nq).round(4).alias("qr_share"), (c("legit_label") / nl).round(4).alias("legit_share"),
        (c("qr_label") / (c("qr_label") + c("legit_label") + c("other_fraud_label"))).round(4).alias("qr_rate_in_cell")).sort("qr_label", descending=True)
    print(f"\n== month {m}: trace-free cohort, qr_swap labels {nq}, legit labels {nl} (decoder columns are null in July)"); print(g.head(8))
