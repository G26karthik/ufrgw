"""Agent A: which cells hold the same-class sender pairs of the trace-free cohort (test 3 of 09_placement.py)?
cwd=work, PYTHONPATH=.  python scratch_atomic/representation/12_sender_pairs.py > scratch_atomic/representation/12_out.txt
A pair = two trace-free fraud rows of one sender inside one month. Same-class pairs are tabulated by class and by the
(txn_type, receiver_type) of both rows; 'other' = the two rows sit in different cells."""
import polars as pl
from common import load_all, LABELS

c = pl.col; pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
A = tx.filter((num("device_id") == num("sender_id")).fill_null(False) & ~((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
              & (c("app_version") >= 4) & (c("ip_risk_score") < q95) & (c("amount") != c("amount").round(0)) & (c("y") > 0))
A = A.select("month", "sender_id", "receiver_id", "ts", "y", pl.concat_str("txn_type", pl.lit("-"), "receiver_type").alias("cell"))
P = A.join(A, on=["month", "sender_id"], suffix="_b").filter(c("ts") < c("ts_b"))
print("all sender pairs per month:", P.group_by("month").len().sort("month").rows())
S = P.filter(c("y") == c("y_b")).with_columns(pl.when(c("cell") == c("cell_b")).then(c("cell")).otherwise(pl.lit("other")).alias("cell2"),
                                             (c("receiver_id") == c("receiver_id_b")).alias("same_receiver"))
print("same-class sender pairs per month:", S.group_by("month").len().sort("month").rows())
g = S.group_by("month", "y", "cell2", "same_receiver").len().sort("month", "len", descending=[False, True])
print(g.with_columns(c("y").map_elements(lambda k: LABELS[k], return_dtype=pl.String).alias("class")))
# expected same-class pairs under class shuffle = pairs x sum_k p_k^2, p_k = class share among the month's cohort fraud rows
for m in (7, 8, 9):
    sh = A.filter(c("month") == m).group_by("y").len().with_columns((c("len") / c("len").sum()).alias("p"))
    n = P.filter(c("month") == m).height
    print(f"month {m}: pairs {n}, same-class observed {S.filter(c('month') == m).height}, expected {n * float((sh['p'] ** 2).sum()):.1f}; "
          f"without REFUND-MERCHANT and QR-P2P same-cell pairs: observed {S.filter((c('month') == m) & ~c('cell2').is_in(['REFUND-MERCHANT', 'QR-P2P'])).height}")
