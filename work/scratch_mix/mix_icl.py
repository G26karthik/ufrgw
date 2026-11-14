"""TabICLv2 on the D tables (production density-matched matrix); fit_ctx / predict copied from scratch_icl/icl.py
(that file runs at import). Context: NLEG legit + up to CAP per fraud class of the fold's fit rows, seed 0, ensemble 1,
prior-corrected. fold 0 -> aug, 1 -> sep, 2 (all train rows -> real test rows) -> oct (row order = D_f2 test rows).
usage (cwd=work, PYTHONPATH=.): python scratch_mix/mix_icl.py FOLD -> scratch_gap/PD_xICL_<kind>.npy"""
import os, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
import numpy as np
import polars as pl
import torch
from tabicl import TabICLClassifier

torch.set_num_threads(4)
fi = int(sys.argv[1]); CAP, NLEG, NEST, CHUNK, SEED = 2500, 20000, 1, 20000, 0
META = ("txn_id", "month", "is_test", "y_fit"); c = pl.col; kind = ("aug", "sep", "oct")[fi]
out = f"scratch_gap/PD_xICL_{kind}.npy"; assert not os.path.exists(out), out
D = pl.read_parquet(f"scratch_gap/D_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
fit = D.filter(~c("is_test")); held = D.filter(c("is_test") & (c("month") == 8 + fi)) if fi < 2 else D.filter(c("is_test"))
if fi < 2: assert held["txn_id"].equals(pl.read_parquet(f"scratch_gap/D_{kind}_rows.parquet", columns=["txn_id"])["txn_id"])
X, y, Xh = fit.select(feats).to_numpy(), fit["y_fit"].to_numpy().astype(np.int64), held.select(feats).to_numpy(); assert y.min() >= 0
rng = np.random.default_rng(SEED)
idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], min((y == k).sum(), CAP if k else NLEG), replace=False) for k in range(12)]))
prior, cprior = np.bincount(y, minlength=12) / len(y), np.bincount(y[idx], minlength=12) / len(idx)
print(f"fold{fi} context {len(idx)} rows, class counts {np.bincount(y[idx], minlength=12).tolist()}", flush=True)
m = TabICLClassifier(n_estimators=NEST, batch_size=1, device="cuda", random_state=SEED).fit(X[idx], y[idx])
assert (m.classes_ == np.arange(12)).all(); del X, D
res, t = [], time.time()
for i in range(0, len(Xh), CHUNK):
    res.append(m.predict_proba(Xh[i:i + CHUNK]))
    if i % (CHUNK * 5) == 0: print(f"  {i + CHUNK}/{len(Xh)} rows {time.time() - t:.0f}s", flush=True)
P = np.vstack(res) * (prior / cprior); P = (P / P.sum(1, keepdims=True)).astype(np.float32)
assert P.shape == (len(Xh), 12) and np.isfinite(P).all()
np.save(out, P); print(f"saved {out} in {time.time() - t:.0f}s; mean predicted share {np.round(P.mean(0), 4).tolist()}", flush=True)
if fi == 2: held.select("txn_id").write_parquet("scratch_mix/oct_ids_icl.parquet")
