"""Read-only: dtypes of the nine CAT_COLS in the cached production matrices (train.build_matrix --matrix path)."""
import glob
import polars as pl
from quick_eval import CAT_COLS

for f in sorted(glob.glob("feats_cache/matrix_*.parquet")):
    s = pl.scan_parquet(f).collect_schema()
    print(f, "| columns", len(s), "| dtypes overall", sorted({str(v) for k, v in s.items() if k not in ("txn_id", "month", "y", "is_test")}),
          "| nine:", {k: str(s[k]) for k in CAT_COLS if k in s})
