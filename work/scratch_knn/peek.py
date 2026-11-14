"""Peek at benchmark N frames: cardinalities, NaN shares of row-level columns, RAM/CPU load."""
import polars as pl, numpy as np, psutil
from quick_eval import CAT_COLS
D = pl.read_parquet("scratch_gap/N_f0.parquet")
row = D.columns[4:31]
for k in row:
    s = D[k]; print(f"{k:20s} nuniq {s.n_unique():8d} nan {s.is_nan().mean():.3f} min {s.fill_nan(None).min():.3g} max {s.fill_nan(None).max():.3g}", "CAT" if k in CAT_COLS else "")
print("cpu load %", psutil.cpu_percent(interval=2), "free RAM GB", round(psutil.virtual_memory().available / 2**30, 1))
