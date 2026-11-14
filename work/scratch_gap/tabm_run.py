"""12-class TabM (official package `tabm` 0.0.3, https://github.com/yandex-research/tabm) on benchmark N, 190 features.
Same fit rows and target as the LightGBM recipe: relabel-C target, legit rows downsampled to 20% with weight 5.
Preprocessing, fitted on the fit rows only (no held label is read):
  numeric (181): constant columns dropped; quantile transform to a normal (noise 1e-3 std for ties, as in the TabM repo);
    missing -> 0 after the transform plus a 0/1 indicator column for every column with > 0.1% missing;
  categorical (9, quick_eval.CAT_COLS): integer codes of the label-free vocabularies, one-hot inside TabM.
Configs: plain = TabM, 3 blocks x 512, dropout 0.1, k=32.  ple = same backbone with piecewise-linear embeddings
  (48 quantile bins, d=16), the package default "tabm-mini" for embedded inputs.
fold0 (fit Jul): early stopping on August (development) weighted log-loss, patience 5, max 40 epochs.
fold1 (fit Jul+Aug): the epoch count chosen in fold0; September labels are never loaded here.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/tabm_run.py plain|ple SEED [N|D]
  -> scratch_gap/PN_TABM<cfg><seed>_aug.npy, _sep.npy (mean of the k softmax outputs), tabm_<cfg><seed>.json"""
import json, sys, time
import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn
from sklearn.preprocessing import QuantileTransformer
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from quick_eval import CAT_COLS, cat_dtypes

CFG, SEED = sys.argv[1], int(sys.argv[2]); TAG = f"TABM{CFG}{SEED}"
BM = sys.argv[3] if len(sys.argv) > 3 else "N"  # benchmark: N (label-hidden worlds) or D (production matrix, scratch_gap/d_data.py)
K, LF, BS, LR, WD, MAXE, PAT = 12, 0.2, 1024, 2e-3, 3e-4, 40, 5
META = ("txn_id", "month", "is_test", "y_fit"); dev = torch.device("cuda"); c = pl.col
CH = 1024 if CFG == "ple" else 16384  # prediction chunk: the embedded input is copied k times (16384 rows overflowed the 8 GB card)


class Prep:
    def __init__(self, X, rng):
        sd = np.nanstd(X, 0); self.keep = sd > 0; X = X[:, self.keep]; sd = sd[self.keep]
        self.ind = np.isnan(X).mean(0) > 0.001
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10 ** 9, random_state=0)
        self.qt.fit(X + rng.normal(0, 1e-3, X.shape).astype(np.float32) * sd)

    def __call__(self, X):
        X = X[:, self.keep]; out = []
        for i in range(0, len(X), 200_000):
            x = X[i:i + 200_000]; nan = np.isnan(x)
            t = np.nan_to_num(self.qt.transform(x), nan=0.0).astype(np.float32)
            out.append(np.hstack([t, nan[:, self.ind].astype(np.float32)]))
        return np.vstack(out)


def predict(model, xn, xc):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(xn), CH):
            lg = model(torch.as_tensor(xn[i:i + CH], device=dev), torch.as_tensor(xc[i:i + CH], device=dev))
            out.append(lg.float().softmax(-1).mean(1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def run(fold, epochs=None, devset=None):
    D = pl.read_parquet(f"scratch_gap/{BM}_f{fold}.parquet"); feats = [k for k in D.columns if k not in META]
    num = [k for k in feats if k not in CAT_COLS]; cards = [len(cat_dtypes()[k].categories) for k in CAT_COLS]
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    fit = D.filter(~c("is_test")); y = fit["y_fit"].to_numpy().astype(np.int64)
    keep = ((y != 0) | (rng.random(len(y)) < LF)) & (y >= 0)
    fit = fit.filter(pl.Series(keep)); y = y[keep]; w = np.where(y == 0, 1 / LF, 1.0).astype(np.float32)
    Xn = fit.select(num).to_numpy().astype(np.float32); P = Prep(Xn, rng); Xn = P(Xn)
    Xc = fit.select(CAT_COLS).to_numpy().astype(np.int64)
    held = D.filter(c("is_test") & (c("month") == (8 if fold == 0 else 9)))
    Hn, Hc = P(held.select(num).to_numpy().astype(np.float32)), held.select(CAT_COLS).to_numpy().astype(np.int64)
    del D
    emb = None
    if CFG == "ple":
        emb = PiecewiseLinearEmbeddings(compute_bins(torch.as_tensor(Xn), n_bins=48), d_embedding=16, activation=False, version="B")
    model = TabM.make(n_num_features=Xn.shape[1], cat_cardinalities=cards, d_out=K, num_embeddings=emb, n_blocks=3, d_block=512, dropout=0.1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); scaler = torch.amp.GradScaler("cuda")
    tn, tc, ty, tw = (torch.as_tensor(a, device=dev) for a in (Xn, Xc, y, w)); n = len(y); k = model.backbone.k if hasattr(model.backbone, "k") else 32
    print(f"{TAG} fold{fold}: fit rows {n}, numeric {Xn.shape[1]} (indicators {int(P.ind.sum())}, dropped {int((~P.keep).sum())}), cat {cards}, k {k}", flush=True)
    best, best_ep, best_state, bad = 1e9, 0, None, 0
    for ep in range(1, (epochs or MAXE) + 1):
        model.train(); t = time.time(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(tn[b], tc[b])  # (batch, k, classes): every member is trained on its own loss
            kk = lg.shape[1]
            loss = (Fn.cross_entropy(lg.float().flatten(0, 1), ty[b].repeat_interleave(kk), reduction="none") * tw[b].repeat_interleave(kk)).sum() / (tw[b].sum() * kk)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
        msg = f"  epoch {ep} train {tot / n:.4f} ({time.time() - t:.0f}s)"
        if devset is not None:
            idx, yd, wd = devset; p = predict(model, Hn[idx], Hc[idx])
            ll = float((-np.log(p[np.arange(len(yd)), yd] + 1e-12) * wd).sum() / wd.sum()); msg += f" dev logloss {ll:.5f}"
            if ll < best - 1e-5:
                best, best_ep, bad = ll, ep, 0; best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg, flush=True)
        if devset is not None and bad >= PAT:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    np.save(f"scratch_gap/P{BM}_{TAG}_{'aug' if fold == 0 else 'sep'}.npy", predict(model, Hn, Hc))
    return best_ep, best


if __name__ == "__main__":
    rows = pl.read_parquet(f"scratch_gap/{BM}_aug_rows.parquet")  # August = development: labels used for stopping only
    ya = rows["y_true"].to_numpy().astype(np.int64); r = np.random.default_rng(0)
    idx = np.flatnonzero((ya != 0) | (r.random(len(ya)) < LF))
    ep, ll = run(0, devset=(idx, ya[idx], np.where(ya[idx] == 0, 1 / LF, 1.0)))
    print(f"{TAG}: fold0 best epoch {ep}, dev logloss {ll:.5f}", flush=True)
    run(1, epochs=ep)
    json.dump(dict(tag=TAG, cfg=CFG, seed=SEED, epochs=ep, dev_logloss=ll, lr=LR, wd=WD, batch=BS), open(f"scratch_gap/tabm_{BM}_{CFG}{SEED}.json", "w"))
    print(f"{TAG}: done", flush=True)
