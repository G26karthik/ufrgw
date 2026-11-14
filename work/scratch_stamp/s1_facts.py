"""A1 step 1: what qr_vpa_mismatch looks like, how many held-month non-QR rows carry the stamp, label mix."""
import polars as pl
from common import load_all, LABELS

c = pl.col; pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
tx = load_all()
print(tx.select("qr_vpa_mismatch").dtypes, tx.height)
t = tx.with_columns((c("txn_type") == "QR").alias("qr"))
print("value counts of qr_vpa_mismatch by QR / receiver_type / is_test:")
print(t.group_by("is_test", "qr", "receiver_type", "qr_vpa_mismatch").len().sort("is_test", "qr", "receiver_type", "qr_vpa_mismatch"))
tr = t.filter(~c("is_test"))
st = (~c("qr") & (c("qr_vpa_mismatch") == 0)).fill_null(False).alias("stamp_nonqr")  # fill_null: mm == 0 is null on null mm
sq = (c("qr") & (c("receiver_type") == "MERCHANT") & c("qr_vpa_mismatch").is_null()).alias("stamp_qr")
tr = tr.with_columns(st, sq)
print("train rows by month: non-QR rows, stamped non-QR, fraud-labelled share of stamped, QR-stamped")
print(tr.group_by("month").agg((~c("qr")).sum().alias("nonqr"), c("stamp_nonqr").sum(), (c("stamp_nonqr") & (c("y") > 0)).sum().alias("stamp_fraud"),
                               (c("stamp_nonqr") & (c("y") == 0)).sum().alias("stamp_legit"), c("stamp_qr").sum(),
                               (c("y") > 0).sum().alias("fraud_rows"), ((c("y") > 0) & ~c("qr")).sum().alias("fraud_nonqr")).sort("month"))
print("stamped non-QR rows by month and label:")
print(tr.filter(c("stamp_nonqr") & c("month").is_in([8, 9])).group_by("month", "y").len().sort("month", "y")
        .with_columns(c("y").map_elements(lambda k: LABELS[k], return_dtype=pl.String).alias("label")))
print("fraud-labelled non-QR rows by class: stamped share (Aug, Sep)")
print(tr.filter((c("y") > 0) & ~c("qr") & c("month").is_in([8, 9])).group_by("month", "y").agg(pl.len().alias("n"), c("stamp_nonqr").mean().round(3).alias("stamped"))
        .sort("month", "y").with_columns(c("y").map_elements(lambda k: LABELS[k], return_dtype=pl.String).alias("label")))
print("stamped non-QR rows by txn_type / receiver_type (train):")
print(tr.filter(c("stamp_nonqr")).group_by("txn_type", "receiver_type").len().sort("len", descending=True))
