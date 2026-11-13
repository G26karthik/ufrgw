"""Build the final feature matrix from scratch: base row features + 6 modules -> one float32 parquet.

Same columns, order and dtypes as feats_cache/matrix_m5dm2_pairsfd.parquet (the matrix behind subs/h6_ppq.csv):
  txn_id, month, y, is_test | quick_eval.base_features | flow, takeover (density.halves)
  | burst, scam, ring (density.halves + 24 h merchant-burst linking, cache __dm2) | pairs (full density, see feats/pairs.py)
Module caches go to common.CACHE (env UPI_CACHE overrides feats_cache/); --force rebuilds them.

--core2: the c2s6 matrix (= feats_cache/matrix_core2swap.parquet): the halved copies of feats/core2.py's 28 columns are
dropped and the full-density core2 build is appended (still 190 features; train with --modules ...,pairs,core2).

usage (cwd=work): python make_matrix.py OUT.parquet [--force] [--core2]
  check:  $env:UPI_CACHE="feats_cache_repro"; python make_matrix.py feats_cache_repro/matrix.parquet --force
"""
import argparse
import polars as pl
from common import load_all, run_feature
from density import run_feature_matched
from quick_eval import base_features
from feats.core2 import COLS as CORE2

ML = 24 * 3600  # merchant-burst link gap of the __dm2 builds (scratch_halves/b1.py)
MODULES = [("flow", 0), ("burst", ML), ("takeover", 0), ("scam", ML), ("ring", ML)]  # column order of the matrix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--core2", action="store_true", help="swap in the full-density core2 columns (c2s6 matrix)")
    a = ap.parse_args()
    tx = load_all()
    X = base_features(tx)
    for name, ml in MODULES:
        X = X.join(run_feature_matched(name, tx, force=a.force, mlink=ml), on="txn_id", how="left", maintain_order="left")
        print(name, X.shape, flush=True)
    X = X.join(run_feature("pairs", tx, force=a.force), on="txn_id", how="left", maintain_order="left")
    feats = [c for c in X.columns if c != "txn_id"]
    D = tx.select("txn_id", "month", "y", "is_test").join(X, on="txn_id", how="left", maintain_order="left")
    D = D.with_columns([pl.col(c).cast(pl.Float32) for c in feats])
    assert D.height == tx.height and len(feats) == 190, (D.height, len(feats))
    if a.core2:  # same steps as scratch_core2/mk_matrix.py (matrix_core2swap)
        f = run_feature("core2", tx, force=a.force)
        f = f.with_columns([pl.col(c).cast(pl.Float32) for c in f.columns if c != "txn_id"])
        dm = [f"{m.__name__.split('.')[-1]}__{c}" for m, cols in CORE2.items() for c in cols]
        assert set(dm) <= set(D.columns)
        D = D.join(f, on="txn_id", how="left", maintain_order="left").drop(dm)
        assert D.height == tx.height and D.width == 194, D.shape
    D.write_parquet(a.out)
    print("written", a.out, D.shape)


if __name__ == "__main__":
    main()
