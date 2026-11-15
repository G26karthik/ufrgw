"""October fit of the wSQ arm: wt.py recipe (same keep mask, weights, params, 150 rounds) on fold 2 of the D tables
(all train rows fit, real test rows held). Output rows are in D_f2 test-row order, as scratch_mix/mix_lgb.py.
usage (cwd=work, PYTHONPATH=., env BM=D): python scratch_wt/wt_oct.py SEED [SEED ...]  -> scratch_gap/PD_wSQo<first seed>_oct.npy"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T
from scratch_wt.wt import weights, META, ROUNDS, LF

c = pl.col; seeds = [int(s) for s in sys.argv[1:]]
out = f"scratch_gap/PD_wSQo{seeds[0]}_oct.npy"; assert not os.path.exists(out), out  # never overwrite
D = pl.read_parquet("scratch_gap/D_f2.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
fit = D.filter(~c("is_test")); Xh = D.filter(c("is_test")).select(feats).to_numpy(); y = fit["y_fit"].to_numpy().astype(np.int32)
n = np.bincount(y[y >= 0], minlength=T.K).astype(np.float64); P = np.zeros((len(Xh), T.K)); t = time.time()
params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=4, verbose=-1)
for s in seeds:
    keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
    w0, w = weights("wSQ", y[keep], n)
    ds = lgb.Dataset(fit.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=w, feature_name=feats, categorical_feature="auto")
    P += lgb.train({**params, "seed": s}, ds, ROUNDS).predict(Xh) / len(seeds)
    print(f"oct wSQ seed{s}: rows {int(keep.sum())} | total weight baseline {w0.sum():.1f} arm {w.sum(dtype=np.float64):.1f} ({time.time() - t:.0f}s)", flush=True)
assert abs(P.sum(1) - 1).max() < 1e-4
np.save(out, P.astype(np.float32)); print("saved", out, P.shape, flush=True)
