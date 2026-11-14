"""Neighbour columns of scratch_knn/knn_feats.py (same space, banks, cross-fit; its functions are imported) on the D tables.
fold 0/1 as there; fold 2 = every train row is a fit row, the real October test rows are the held rows.
usage (cwd=work, PYTHONPATH=.): python scratch_mix/mix_knn.py FOLD -> scratch_mix/KFD_f<FOLD>.parquet"""
import sys, time
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
import train as T
from common import load_all
from clean_labels import fraud_events
from scratch_knn.knn_feats import space, kd, KS, NF, LEGIT_FRAC, SEEDS, LF, NAMES

c = pl.col; fi = int(sys.argv[1]); fm = T.FOLDS[fi][0] if fi < 2 else [7, 8, 9, 10]; t0 = time.time(); rng = np.random.default_rng(100 + fi)
D = pl.read_parquet(f"scratch_gap/D_f{fi}.parquet"); n = D.height
fit = ~D["is_test"].to_numpy(); held = D.select(c("is_test") & (c("month") == 8 + fi)).to_series().to_numpy() if fi < 2 else ~fit
y = np.full(n, -1); y[fit] = D.filter(~c("is_test"))["y_fit"].to_numpy(); yf = y[fit]
need = np.zeros(n, bool); need[fit] = (yf != 0) | np.any([np.random.default_rng(s).random(len(yf)) < LF for s in range(SEEDS)], axis=0)
_, FE = fraud_events(load_all().filter(~c("is_test") & c("month").is_in(fm)), g=48 * 3600)
ev = D.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
assert not np.isfinite(ev[~fit]).any()
fold = np.where(np.isnan(ev), rng.integers(0, NF, n), rng.integers(0, NF, int(np.nanmax(ev)) + 1)[np.nan_to_num(ev).astype(np.int64)]); fold[~fit] = -1
print(f"fold{fi}: rows {n}, fit {int(fit.sum())}, held {int(held.sum())}, fit rows queried {int(need.sum())}", flush=True)
with threadpool_limits(4):
    X = space(D, fit); nL = int(round(LEGIT_FRAC * (y == 0).sum())); print(f"space {X.shape}, legit bank {nL} ({time.time() - t0:.0f}s)", flush=True)
    F = np.full((n, T.K, len(KS)), np.nan, np.float32)
    for j in list(range(NF)) + [-1]:
        q = held if j < 0 else need & (fold == j)
        for k in range(T.K):
            bank = np.flatnonzero((y == k) & (fold != j))
            if k == 0: bank = rng.choice(bank, nL, replace=False)
            assert not q[bank].any(); F[q, k] = kd(X[bank], X[q])
        print(f"  {'held' if j < 0 else 'fold ' + str(j)}: queries {int(q.sum())} ({time.time() - t0:.0f}s)", flush=True)
lf = need & (y == 0)
print("median 5th-neighbour distance, fit legit rows vs held rows:", " ".join(f"c{k} {np.median(F[lf, k, 1]):.3f}/{np.median(F[held, k, 1]):.3f}" for k in range(T.K)))
e = 1e-3; G = np.hstack([F.reshape(n, -1), F[:, 1:].min(1) / (F[:, 0] + e), F[:, 1:, 1] / (F[:, :1, 1] + e)])
assert G.shape[1] == len(NAMES) and not np.isnan(G[need | held]).any()
pl.DataFrame({"txn_id": D["txn_id"], **{k: G[:, i] for i, k in enumerate(NAMES)}}).write_parquet(f"scratch_mix/KFD_f{fi}.parquet")
print(f"fold{fi} written ({time.time() - t0:.0f}s)", flush=True)
