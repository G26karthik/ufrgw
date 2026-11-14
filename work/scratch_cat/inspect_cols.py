"""Step 0: how the nine CAT_COLS are stored in the benchmark tables (dtype, distinct codes, nulls/NaN, vocabulary size)."""
import numpy as np
import polars as pl
from quick_eval import CAT_COLS, cat_dtypes

E = cat_dtypes()
for f in ("N_f0", "N_f1", "D_f0", "D_f1"):
    D = pl.read_parquet(f"scratch_gap/{f}.parquet", columns=["is_test", "month", "y_fit"] + CAT_COLS)
    print(f, "rows", D.height, "fit", int((~D["is_test"]).sum()), "held months", D.filter(pl.col("is_test"))["month"].unique().to_list())
    for k in CAT_COLS:
        x = D[k]; v = x.to_numpy(); u = np.unique(v[~np.isnan(v)])
        cats = E[k].categories.to_list()
        print(f"  {k:18s} {str(x.dtype):8s} nulls {x.null_count()} nan {int(np.isnan(v).sum())} distinct {len(u)} min {u.min():.0f} max {u.max():.0f} "
              f"integer-valued {bool((u == np.round(u)).all())} vocab {len(cats)} NA code {cats.index('NA')} rows at NA code {int((v == cats.index('NA')).sum())}"
              f" | held-only codes {sorted(set(np.unique(v[D['is_test'].to_numpy()])) - set(np.unique(v[~D['is_test'].to_numpy()])))}")
