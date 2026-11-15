"""Agent A: model-free check of the P2P-receiver recurrence predicate inside the trace-free cohort, QR rows excluded
(QR-to-P2P is the known qr_swap signature). cwd=work, PYTHONPATH=.
python scratch_atomic/representation/11_recur_tracefree.py > scratch_atomic/representation/11_out.txt
Predicate (label-informed): receiver_type == P2P and the receiver held ANY fraud label in the source month(s).
Target rows: trace-free home-device P2P rows with txn_type != QR. Binomial z against the cohort rate of the target month."""
import polars as pl
from common import load_all

c = pl.col
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
tx = tx.with_columns((c("y") > 0).alias("fraud"),
                     ((num("device_id") == num("sender_id")).fill_null(False) & ~((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
                      & (c("app_version") >= 4) & (c("ip_risk_score") < q95) & (c("amount") != c("amount").round(0))).alias("tf"))
for src, tgt in (([7], 8), ([8], 9), ([7, 8], 9)):
    S = tx.filter(c("month").is_in(src) & c("fraud") & (c("receiver_type") == "P2P"))["receiver_id"].unique().to_list()
    T = tx.filter((c("month") == tgt) & c("tf") & (c("receiver_type") == "P2P") & (c("txn_type") != "QR"))
    P = T.filter(c("receiver_id").is_in(S)); r0 = T["fraud"].mean(); n = P.height; k = int(P["fraud"].sum())
    z = (k - n * r0) / (n * r0 * (1 - r0)) ** .5
    print(f"source {src} -> month {tgt}: trace-free non-QR P2P rows {T.height}, fraud-label rate {r0:.4f}; predicate rows {n}, fraud-labelled {k} "
          f"(rate {k / n:.4f}, expected {n * r0:.1f}, lift {k / n / r0:.2f}, z {z:+.1f}); by txn_type {P.group_by('txn_type').agg(pl.len(), c('fraud').sum()).rows()}")
