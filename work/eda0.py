import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
D = "work/data/"
tr = pl.read_parquet(D + "train.parquet"); te = pl.read_parquet(D + "test.parquet")
print(tr["label"].value_counts().sort("count", descending=True))
tr = tr.with_columns(pl.col("timestamp").dt.month().alias("m"))
print(tr.group_by("m", "label").len().pivot(on="m", index="label", values="len").sort("label"))
both = pl.concat([tr.drop("label", "m").with_columns(pl.lit("tr").alias("src")), te.with_columns(pl.lit("te").alias("src"))])
both = both.with_columns(pl.col("timestamp").dt.month().alias("m"))
for c in ["app", "app_version", "txn_type", "receiver_type"]:
    print(both.group_by("m", c).len().pivot(on="m", index=c, values="len").sort(c))
print(both.group_by("m").agg(pl.col(c).mean().alias(c) for c in ["amount", "sim_age_hours", "device_age_days", "new_payee", "pin_retries", "login_to_txn_sec", "ip_risk_score", "qr_vpa_mismatch"]).sort("m"))
num = ["amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries", "login_to_txn_sec", "balance_before", "ip_risk_score", "qr_vpa_mismatch"]
print(tr.group_by("label").agg([pl.col(c).median().alias(c) for c in num] + [pl.len()]).sort("label"))
print(tr.group_by("label", "txn_type").len().pivot(on="txn_type", index="label", values="len").sort("label"))
print(tr.group_by("label", "receiver_type").len().pivot(on="receiver_type", index="label", values="len").sort("label"))
print("ip_risk by label x month"); print(tr.group_by("label", "m").agg(pl.col("ip_risk_score").mean()).pivot(on="m", index="label", values="ip_risk_score").sort("label"))
print("devices/sender:", both.group_by("sender_id").agg(pl.col("device_id").n_unique()).describe())
print("n senders", both["sender_id"].n_unique(), "n devices", both["device_id"].n_unique(), "n receivers", both["receiver_id"].n_unique())
