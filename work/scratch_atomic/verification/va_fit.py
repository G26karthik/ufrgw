"""Matched-baseline fit for agent C: benchmark table + optional new label-free columns, same recipe as scratch_wt/wt.py arm wC
(same rows, keep mask, seeds 0,1, T.LGB, 150 rounds, learning rate .05, weights legit 5 / fraud 1, num_threads=4).
NAME = C  : control, no new columns; must equal the stored reference bit for bit (va_score.py check).
NAME = xyz: joins the new columns written by va_feat.py (txn_id + columns) to the benchmark table by txn_id; every benchmark row
            must be present. N: feat/<xyz>_N_f<fold>.parquet (built on the fold's world). D: feat/<xyz>_ALL.parquet (built on all rows) or, with env DFEAT=DM, feat/<xyz>_DM.parquet (density-matched build).
env SEED0 (default 0): first seed; SEED0=2 gives a second seed pair, files get the suffix b2.
Output: scratch_atomic/verification/P<BM>_va<NAME>_<aug|sep>.npy (never overwritten).
usage (cwd=work, PYTHONPATH=., env BM=N|D, FOLDS=0,1): python scratch_atomic/verification/va_fit.py NAME"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T

c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2
SEED0 = int(os.environ.get("SEED0", 0)); SUF = "" if SEED0 == 0 else "b" + str(SEED0)
DFEAT = os.environ.get("DFEAT", "ALL")  # benchmark D: ALL = full-density build, DM = density-matched build (va_feat.py)
TAG = os.environ.get("TAG", "")  # appended to the mechanism name in the output file (e.g. dm)
BM = os.environ.get("BM", "N"); FOLDS = [int(k) for k in os.environ.get("FOLDS", "0,1").split(",")]
META = ("txn_id", "month", "is_test", "y_fit"); HERE = "scratch_atomic/verification"


def run(name):
    for fi in FOLDS:
        fm, vm = T.FOLDS[fi]
        out = f"{HERE}/P{BM}_va{name}{TAG}{SUF}_{'aug' if vm == 8 else 'sep'}.npy"
        assert not os.path.exists(out), out  # never overwrite
        D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]
        if name != "C":
            F = pl.read_parquet(f"{HERE}/feat/{name}_N_f{fi}.parquet" if BM == "N" else f"{HERE}/feat/{name}_{DFEAT}.parquet"); new = [k for k in F.columns if k != "txn_id"]
            assert F["txn_id"].n_unique() == F.height and not set(new) & set(D.columns), "duplicate ids or column clash"
            n0 = D.height; D = D.join(F.with_columns(pl.lit(True).alias("_in")), on="txn_id", how="left", maintain_order="left")
            assert D.height == n0 and D["_in"].null_count() == 0, "benchmark rows without feature rows"
            D = D.drop("_in").with_columns([c(k).cast(pl.Float32) for k in new]); feats = feats + new
            print(f"{BM} {name} fold{fi}: added {new}", flush=True)
        params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=4, verbose=-1)
        fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
        P = np.zeros((V.height, T.K), np.float32); t = time.time(); gain = np.zeros(len(feats))
        for s in range(SEED0, SEED0 + SEEDS):
            keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
            w = np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32)
            ds = lgb.Dataset(fit.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=w, feature_name=feats, categorical_feature="auto")
            m = lgb.train({**params, "seed": s}, ds, ROUNDS)
            P += m.predict(V.select(feats).to_numpy()).astype(np.float32) / SEEDS
            gain += m.feature_importance("gain") / SEEDS
        np.save(out, P)
        if name != "C":
            r = np.argsort(-gain); rank = {feats[j]: i + 1 for i, j in enumerate(r)}
            print(f"{BM} {name} fold{fi} gain share / rank of new columns (of {len(feats)}): "
                  + ", ".join(f"{k} {gain[feats.index(k)] / gain.sum():.4f} / {rank[k]}" for k in new), flush=True)
        print(f"{BM} va{name} fold{fi} done ({time.time() - t:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]: run(a)
