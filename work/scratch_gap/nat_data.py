"""Save the benchmark-N feature tables (same worlds as bench_nat.py) for models other than LightGBM.
scratch_gap/N_f0.parquet: fit Jul, held Aug.  N_f1.parquet: fit Jul+Aug, held Sep (+ Oct-dated train rows, hidden).
Columns: txn_id, month, is_test (held), y_fit (relabel C on fit rows from the world's own features; null on held rows),
then the 190 features. Held labels are not stored. usage (cwd=work, PYTHONPATH=.): python scratch_gap/nat_data.py"""
import sys
import polars as pl
import relabel as R
import train as T
from common import load_all

sys.path.insert(0, "scratch_gap")
from bench_nat import world, build  # noqa: E402

c = pl.col
tx = load_all()
for fi, (fm, vm) in enumerate(T.FOLDS):
    w = world(tx, fm, vm); X, F, feats = build(w)
    fit = w.filter(~c("is_test")); R.run_feature = lambda m: F[m]
    yf = fit.select("txn_id", R.relabel(fit, "C").alias("y_fit"))
    D = w.select("txn_id", "month", "is_test").join(yf, on="txn_id", how="left", maintain_order="left").join(X, on="txn_id", how="left", maintain_order="left")
    assert D.filter(c("is_test"))["y_fit"].null_count() == D.filter(c("is_test")).height and len(feats) == 190
    D.write_parquet(f"scratch_gap/N_f{fi}.parquet"); print("written fold", fi, D.shape, flush=True)
