"""Agent A: where are the fraud-path rows that the final decoder calls legit?  cwd=work, PYTHONPATH=.
python scratch_atomic/representation/06_lost_cells.py > scratch_atomic/representation/06_out.txt   (needs held_N.parquet, held_D.parquet)
Joint trace J = app_version < 4 AND ip_risk_score >= July all-row q95 (two fields the vector drops; independent draws on the
legit path, so the legit rate is about .15 * .044 = .0066; fraud-path rate about .28: fresh_columns.md s2).
For rows the decoder calls legit (ALL labels, so the count is label-free), per cell of original fields:
  n, J count, expected J count under the legit path (cell n * legit J rate of the month's home rows), excess, and the
  supplied-label mix of the J rows (only to read what the excess is).  Excess / .28 estimates fraud-path rows lost in the cell.
J itself cannot be an October feature (app_version is 5.x for every October row, ip drifts: drift.md); it is a train-month
truth proxy that does not come from the model or the labels."""
import sys
import numpy as np
import polars as pl
from common import load_all, LABELS

c = pl.col; OUT = "scratch_atomic/representation/"
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
tx = tx.with_columns(
    (num("device_id") == num("sender_id")).fill_null(False).alias("home_dev"),
    ((c("app_version") < 4) & (c("ip_risk_score") >= q95)).alias("J"),
    (c("amount") == c("amount").round(0)).alias("int_amt"),
    pl.when(c("qr_vpa_mismatch").is_null()).then(pl.lit("null")).otherwise(c("qr_vpa_mismatch").cast(pl.Int8).cast(pl.String)).alias("mm"),
    (c("new_payee") == 1).alias("np"))
CELL = ["txn_type", "receiver_type", "mm", "int_amt", "np"]
for B in sys.argv[1:] or ["N", "D"]:
    H = tx.join(pl.read_parquet(f"{OUT}held_{B}.parquet").select("txn_id", "q"), on="txn_id")
    for m in (8, 9):
        M = H.filter(c("month") == m)
        rJ = M.filter((c("y") == 0) & (c("q") == 0))["J"].mean()   # legit-path J rate (legit label, decoded legit)
        rF = M.filter((c("y") > 0) & (c("q") > 0))["J"].mean()     # J rate on rows that are fraud by label AND decode
        D0 = M.filter(c("q") == 0)
        tot = D0["J"].sum() - rJ * D0.height
        print(f"\n== benchmark {B} month {m}: decoded-legit rows {D0.height}; J rate legit-labelled {rJ:.5f}, fraud label+decode {rF:.3f}; "
              f"J rows among decoded legit {int(D0['J'].sum())}, expected {rJ * D0.height:.0f}, excess {tot:.0f} -> about {tot / (rF - rJ):.0f} fraud-path rows decoded legit")
        g = (D0.group_by(CELL).agg(pl.len().alias("n"), c("J").sum().alias("J_n"), (c("J") & (c("y") > 0)).sum().alias("J_fraud_label"),
                                   c("y").filter(c("J") & (c("y") > 0)).mode().first().alias("J_top_cls"), c("home_dev").filter(c("J")).mean().round(2).alias("J_home"))
             .with_columns((c("n") * rJ).round(1).alias("J_exp")).with_columns((c("J_n") - c("J_exp")).round(1).alias("excess"),
                                                                             ((c("J_n") - c("J_exp")) / (c("J_exp") + 1e-9).sqrt()).round(1).alias("z"))
             .sort("excess", descending=True))
        print(g.head(12))
        print("by supplied class of the decoded-legit rows (reading aid): class, n, J_n, expected, excess")
        for k in range(0, 12):
            d = D0.filter(c("y") == k); print(f"   {LABELS[k]:20s} {d.height:7d} {int(d['J'].sum()):5d} {rJ * d.height:7.1f} {d['J'].sum() - rJ * d.height:7.1f}")
