"""TabICLv2 (tabicl 2.2.0, checkpoint tabicl-classifier-v2-20260212) on all 12 classes of benchmark N, 190 features as floats.
Context = stratified sample of the fold's fit rows: every fraud class capped at CAP rows, plus NLEG legit rows (seed SEED).
12 classes > the model's 10 -> the package's many-class path (mixed-radix embedding + hierarchical ICL); it refuses kv_cache,
so every predict call re-encodes the context: predict in chunks several times the context.
Prior correction: p *= (fit-month class share / context class share), renormalised (the context over-represents fraud).
usage (cwd=work, PYTHONPATH=.):
  python scratch_icl/icl.py sub  CAP NLEG NEST CHUNK   -> timing + quality on an August subset (all fraud + 30k legit, legit reweighted)
  python scratch_icl/icl.py full CAP NLEG NEST CHUNK NAME -> scratch_gap/PN_i<NAME>_{aug,sep}.npy and PN_i<NAME>b35_{aug,sep}.npy (0.65 C + 0.35 model)"""
import os, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "6"
import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, log_loss
from tabicl import TabICLClassifier

torch.set_num_threads(6)
MODE, CAP, NLEG, NEST, CHUNK = sys.argv[1], *(int(a) for a in sys.argv[2:6]); SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 0
META = ("txn_id", "month", "is_test", "y_fit"); c = pl.col; G = "scratch_gap"


def load(fold):
    D = pl.read_parquet(f"{G}/N_f{fold}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
    fit, held = D.filter(~c("is_test")), D.filter(c("is_test") & (c("month") == 8 + fold))  # N_f1 also holds 147 month-10 rows: excluded
    rows = pl.read_parquet(f"{G}/N_{'aug' if fold == 0 else 'sep'}_rows.parquet", columns=["txn_id", "y_true"])
    assert (held["txn_id"] == rows["txn_id"]).all()
    return fit.select(feats).to_numpy(), fit["y_fit"].to_numpy().astype(np.int64), held.select(feats).to_numpy(), rows["y_true"].to_numpy()


def fit_ctx(X, y):
    rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], min((y == k).sum(), CAP if k else NLEG), replace=False) for k in range(12)]))
    prior, cprior = np.bincount(y, minlength=12) / len(y), np.bincount(y[idx], minlength=12) / len(idx)
    print(f"context {len(idx)} rows, class counts {np.bincount(y[idx], minlength=12).tolist()}", flush=True)
    m = TabICLClassifier(n_estimators=NEST, batch_size=1, device="cuda", random_state=SEED).fit(X[idx], y[idx])
    assert (m.classes_ == np.arange(12)).all()
    return m, prior / cprior


def predict(m, w, X):
    out, t = [], time.time(); torch.cuda.reset_peak_memory_stats()
    for i in range(0, len(X), CHUNK):
        out.append(m.predict_proba(X[i:i + CHUNK]))
        if i % (CHUNK * 10) == 0:
            print(f"  {i + CHUNK}/{len(X)} rows {time.time() - t:.0f}s VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
                  f"(device used {(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30:.1f} GB)", flush=True)
    P = np.vstack(out) * w; P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    assert P.shape == (len(X), 12) and np.isfinite(P).all()
    print(f"  predicted {len(X)} rows in {time.time() - t:.0f}s ({len(X) / (time.time() - t):.0f} rows/s)", flush=True)
    return P


if MODE == "sub":
    Xf, yf, Xh, yt = load(0); rng = np.random.default_rng(7)
    sub = np.sort(np.concatenate([np.where(yt != 0)[0], rng.choice(np.where(yt == 0)[0], 30_000, replace=False)]))
    wt = np.where(yt[sub] == 0, (yt == 0).sum() / 30_000, 1.0); ys = yt[sub]
    m, w = fit_ctx(Xf, yf)
    Pq = predict(m, w, Xh[sub[:10_000]])  # two query sizes: separates the fixed context cost from the per-row cost
    P = predict(m, w, Xh[sub])
    d = np.abs(Pq[:CHUNK] - P[:len(Pq[:CHUNK])]); print(f"chunking check on the first {len(d)} rows: max abs diff {d.max():.4f}, mean {d.mean():.6f}, share of rows with a diff > 0.01 {(d.max(1) > 0.01).mean():.4f}")
    C = np.load(f"{G}/PN_C_aug.npy")[sub]
    print("mean predicted share (weighted) vs fit prior:", np.round(np.average(P, 0, wt), 4).tolist(), np.round(np.bincount(yf, minlength=12) / len(yf), 4).tolist())
    for nm, Q in (("C", C), ("ICL", P), ("0.65C+0.35ICL", 0.65 * C + 0.35 * P)):
        ap = [average_precision_score(ys == k, Q[:, k], sample_weight=wt) for k in range(1, 12)]
        print(f"{nm:14s} weighted log-loss {log_loss(ys, Q, sample_weight=wt, labels=range(12)):.5f} | mean fraud-class AP {np.mean(ap):.4f} | AP {np.round(ap, 3).tolist()}", flush=True)
else:
    NAME = sys.argv[6]
    for fold, mo in ((0, "aug"), (1, "sep")):
        out = f"{G}/PN_i{NAME}_{mo}.npy"; assert not os.path.exists(out), out  # new names only
        Xf, yf, Xh, yt = load(fold); m, w = fit_ctx(Xf, yf); del Xf
        P = predict(m, w, Xh); np.save(out, P)
        np.save(f"{G}/PN_i{NAME}b35_{mo}.npy", (0.65 * np.load(f"{G}/PN_C_{mo}.npy") + 0.35 * P).astype(np.float32))
        print(f"saved {out}; mean predicted share {np.round(P.mean(0), 4).tolist()}", flush=True)
        del m, Xh; torch.cuda.empty_cache()
