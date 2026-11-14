"""D_f2: the production density-matched matrix with every train row as fit row (relabel-C target) and the real October
test rows as held rows. Same columns as scratch_gap/D_f0.parquet / D_f1.parquet (d_data.py). Label use: train labels only.
usage (cwd=work, PYTHONPATH=.): python scratch_mix/mix_data.py -> scratch_gap/D_f2.parquet"""
import os
import polars as pl
import train as T

c = pl.col; out = "scratch_gap/D_f2.parquet"; assert not os.path.exists(out)
M = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet").join(T.fit_target("C"), on="txn_id", how="left", maintain_order="left")
cols = pl.read_parquet_schema("scratch_gap/D_f1.parquet").names()
D = M.select(cols)
assert D.filter(c("is_test"))["y_fit"].null_count() == int(D["is_test"].sum()) and D.filter(~c("is_test"))["y_fit"].null_count() == 0
D.write_parquet(out); print("written", out, D.shape, "test rows", int(D["is_test"].sum()), "fit rows", int((~D["is_test"]).sum()), "y_fit min", D["y_fit"].min())
