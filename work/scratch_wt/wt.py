"""Training influence of legit versus fraud classes. Same harness as scratch_gap/numrep.py::run arm A_C (same keep mask,
seeds, features, T.LGB params, 150 rounds, lr .05, target y_fit), only the sample weights differ; num_threads=4.
arms : wC  baseline weights (legit 5, fraud 1) = harness control, must reproduce the reference
       wL1 every sampled row weight 1 (no sampling correction)
       wSQ baseline weights x n_c^-0.5, n_c = class counts of ALL fit rows of the fold (y_fit >= 0, before legit downsampling)
Every arm is rescaled so that sum(weights) == sum(baseline weights) on the same sampled rows (asserted after the float32 cast).
usage (cwd=work, PYTHONPATH=., env BM=N|D, SEED0=0|2): python scratch_wt/wt.py ARM [ARM ...]"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T

c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2
SEED0, BM = int(os.environ.get("SEED0", 0)), os.environ.get("BM", "N")
SUF = "" if SEED0 == 0 else "b" + str(SEED0)
META = ("txn_id", "month", "is_test", "y_fit")


def weights(arm, yk, n):
    w0 = np.where(yk == 0, 1 / LF, 1.0)
    w = {"wC": w0, "wL1": np.ones(len(yk)), "wSQ": w0 * n[yk] ** -0.5}[arm]
    w = (w * (w0.sum() / w.sum())).astype(np.float32)
    assert abs(float(w.sum(dtype=np.float64)) / w0.sum() - 1) < 1e-5, arm  # total weight unchanged
    return w0, w


def run(arm):
    for fi, (fm, vm) in enumerate(T.FOLDS):
        out = f"scratch_gap/P{BM}_{arm}{SUF}_{'aug' if vm == 8 else 'sep'}.npy"
        assert not os.path.exists(out), out  # never overwrite
        D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]
        params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=4, verbose=-1)
        fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
        n = np.bincount(y[y >= 0], minlength=T.K).astype(np.float64)
        P = np.zeros((V.height, T.K), np.float32); t = time.time()
        for s in range(SEED0, SEED0 + SEEDS):
            keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
            w0, w = weights(arm, y[keep], n)
            pc = [float(w[y[keep] == k][0]) for k in range(T.K)]
            print(f"{BM} {arm} fold{fi} seed{s}: rows {int(keep.sum())} (legit {int((y[keep] == 0).sum())}) | total weight baseline {w0.sum():.1f} arm {w.sum(dtype=np.float64):.1f}"
                  f" | legit share of weight baseline {w0[y[keep] == 0].sum() / w0.sum():.4f} arm {w[y[keep] == 0].sum(dtype=np.float64) / w.sum(dtype=np.float64):.4f}"
                  f" | per-row weight by class {[round(v, 3) for v in pc]} | fit-row counts {n.astype(int).tolist()}", flush=True)
            ds = lgb.Dataset(fit.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=w, feature_name=feats, categorical_feature="auto")
            m = lgb.train({**params, "seed": s}, ds, ROUNDS)
            P += m.predict(V.select(feats).to_numpy()).astype(np.float32) / SEEDS
        np.save(out, P)
        print(f"{BM} {arm}{SUF} fold{fi} done ({time.time() - t:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]: run(a)
