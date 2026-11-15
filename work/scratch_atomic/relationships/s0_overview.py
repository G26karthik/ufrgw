import numpy as np, polars as pl
from common import load_all, LABELS
from clean_labels import fraud_events
tx = load_all()
print(tx.columns); print(tx.height, tx.group_by("month","is_test").len().sort("month"))
rows = pl.read_parquet("scratch_gap/N_sep_rows.parquet"); print(rows.columns, rows.height)
d = np.load("scratch_gap/dec_N_wt.npz"); print({k: d[k].shape for k in d.files})
rows = rows.with_columns(pl.Series("q", d["C_q"]), pl.Series("yy", d["y"]))
print("y_true==dec y", (rows["y_true"].to_numpy() == d["y"]).mean() if "y_true" in rows.columns else None)
t, F = fraud_events(tx, 48 * 3600)
m = rows.filter((pl.col("yy") > 0)).join(F.select("txn_id", "ev"), on="txn_id", how="left")
print(m.group_by("yy").agg(pl.len().alias("n"), (pl.col("q") == 0).sum().alias("miss"), pl.col("ev").n_unique().alias("ev"),
      pl.col("ev").filter(pl.col("q") == 0).n_unique().alias("ev_miss")).sort("yy").with_columns(pl.col("yy").map_elements(lambda i: LABELS[i], return_dtype=pl.String).alias("lab")))
# event-level: fraction of rows missed per event
e = m.group_by("ev").agg(pl.len().alias("n"), (pl.col("q") == 0).sum().alias("miss"), pl.col("yy").mode().first().alias("y"))
print(e.with_columns((pl.col("miss") == pl.col("n")).alias("all_missed"), (pl.col("n") == 1).alias("single")).group_by("y", "all_missed", "single").len().sort("y", "all_missed", "single"))
c = pl.read_parquet("scratch_gap/N_f1.parquet", n_rows=5); print([x for x in c.columns])
