"""Final trainer: base features + feature modules -> GBDT multiclass, temporal CV, bias-decoded submission.

usage (cwd=work):
  python train.py --modules a,b --mode val  --threads 4 --seeds 1 --tag NAME
  python train.py --modules a,b --mode full --threads 4 --seeds 3 --tag NAME   # runs val first if oof/NAME_meta.json missing
best (h6; matrix from make_matrix.py, full recipe in ../run_all.ps1):
  python train.py --modules flow,burst,takeover,scam,ring,pairs --matrix feats_cache/matrix_m5dm2_pairsfd.parquet --relabel C \
      --rounds 150 --mode val|full --threads 10 --seeds 6 --tag h6
Outputs: oof/<tag>.npy (+ _ids.parquet, _meta.json), preds/<tag>_test.npy (sample_submission order),
         subs/<tag>.csv. See notes/trainer.md.
"""
import argparse, json, time
import numpy as np
import polars as pl
from common import ROOT, load_all, LABELS
from density import run_feature_matched
from quick_eval import base_features

K = len(LABELS)
FOLDS = [([7], 8), ([7, 8], 9)]  # mimic train -> next month
SAMPLE = ROOT.parent / "upi-fraud-rings" / "sample_submission.csv"
for d in ("oof", "preds", "subs"):
    (ROOT / d).mkdir(exist_ok=True)


# ---------------------------------------------------------------- metrics / decoding
def macro_f1(y, pred):
    tp = np.bincount(y[pred == y], minlength=K)
    den = np.bincount(y, minlength=K) + np.bincount(pred, minlength=K)
    return 2 * tp / np.maximum(den, 1)


def tune_bias(P, y, grid=np.arange(-4, 6.001, 0.1), passes=4, smooth=5):
    """Coordinate ascent on per-class log-prob bias (legit fixed at 0). Score curve is smoothed over the grid
    before argmax so the chosen bias sits on a plateau, not a noisy spike."""
    L = np.log(P + 1e-12).astype(np.float32); b = np.zeros(K, np.float32)
    ny = np.bincount(y, minlength=K)
    for _ in range(passes):
        changed = False
        for k in range(1, K):
            o = L + b; o[:, k] = -np.inf
            oth = o.argmax(1); m = o[np.arange(len(y)), oth]; d = m - L[:, k]  # class k wins iff g > d
            sc = []
            for g in grid:
                pred = np.where(g > d, k, oth)
                sc.append((2 * np.bincount(y[pred == y], minlength=K) / np.maximum(ny + np.bincount(pred, minlength=K), 1)).mean())
            sc = np.convolve(np.pad(sc, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
            g = float(grid[sc.argmax()])
            if abs(g - b[k]) > 1e-6: b[k] = g; changed = True
        if not changed: break
    return b


def decode(P, b):
    return (np.log(P + 1e-12) + b).argmax(1)


def report(name, y, P, b=None):
    fa = macro_f1(y, P.argmax(1)); line = f"{name}: macro argmax={fa.mean():.4f}"
    if b is not None:
        fb = macro_f1(y, decode(P, b)); line += f" biased={fb.mean():.4f}"
    print(line)
    for i, l in enumerate(LABELS):
        print(f"  {l:20s} argmax={fa[i]:.3f}" + (f" biased={fb[i]:.3f}" if b is not None else ""))
    return fa.mean(), (fb.mean() if b is not None else None)


# ---------------------------------------------------------------- models (pluggable)
# notes/tuning.md: strong L2 + smaller trees is the one LGB knob that matters under label noise (+.009 pooled vs l2=1)
LGB = dict(num_leaves=31, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0)


def lgb_fit(X, y, w, Xv, yv, wv, a, seed, rounds):
    import lightgbm as lgb
    p = dict(objective="multiclass", num_class=K, learning_rate=a.lr, **LGB, num_threads=a.threads, verbose=-1, seed=seed)
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    if Xv is None:
        m = lgb.train(p, dtr, rounds)
        return m, rounds
    dva = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
    m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(a.patience, verbose=False)])
    return m, m.best_iteration


def lgb_pred(m, X):
    return m.predict(X, num_iteration=m.best_iteration or None).astype(np.float32)


def xgb_fit(X, y, w, Xv, yv, wv, a, seed, rounds):
    import xgboost as xgb
    p = dict(objective="multi:softprob", num_class=K, eta=a.lr, max_depth=8, min_child_weight=5, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", nthread=a.threads, seed=seed,
             eval_metric="mlogloss", device=a.device)
    dtr = xgb.DMatrix(X, y, weight=w)
    if Xv is None:
        return xgb.train(p, dtr, rounds), rounds
    m = xgb.train(p, dtr, rounds, evals=[(xgb.DMatrix(Xv, yv, weight=wv), "va")],
                  early_stopping_rounds=a.patience, verbose_eval=False)
    return m, m.best_iteration + 1


def xgb_pred(m, X):
    import xgboost as xgb
    it = getattr(m, "best_iteration", None)
    return m.predict(xgb.DMatrix(X), iteration_range=(0, it + 1) if it is not None else (0, 0)).astype(np.float32)


MODELS = {"lgb": (lgb_fit, lgb_pred), "xgb": (xgb_fit, xgb_pred)}


def predict_chunked(pred, m, D, feats, chunk=200_000):
    return np.concatenate([pred(m, D[i:i + chunk].select(feats).to_numpy()) for i in range(0, D.height, chunk)])


# ---------------------------------------------------------------- data
def build_matrix(modules, drop="", matrix=None):
    if matrix:  # cached output of an earlier build (txn_id, month, y, is_test + float32 feats, load_all row order)
        D = pl.read_parquet(matrix)
        drop = [c for c in drop.split(",") if c]
        assert set(drop) <= set(D.columns), f"--drop: unknown columns {sorted(set(drop) - set(D.columns))}"
        D = D.drop(drop)
        return D, [c for c in D.columns if c not in ("txn_id", "month", "y", "is_test")]
    tx = load_all()
    X = base_features(tx)
    for mname in modules:
        f = run_feature_matched(mname, tx)
        bad = [c for c in f.columns if c != "txn_id" and not c.startswith(f"{mname}__")]
        assert not bad, f"{mname}: unprefixed columns {bad}"
        X = X.join(f, on="txn_id", how="left")
    drop = [c for c in drop.split(",") if c]
    assert set(drop) <= set(X.columns) - {"txn_id"}, f"--drop: unknown columns {sorted(set(drop) - set(X.columns))}"
    X = X.drop(drop)
    feats = [c for c in X.columns if c != "txn_id"]
    D = tx.select("txn_id", "month", "y", "is_test").join(X, on="txn_id", how="left", maintain_order="left")
    del tx, X
    D = D.with_columns([pl.col(c).cast(pl.Float32) for c in feats])  # RAM: float32 everywhere
    return D, feats


def fit_target(variant):
    """(txn_id, y_fit): relabel.relabel on all rows (train labels relabelled, -1 = drop, null on test)."""
    from relabel import relabel
    tx = load_all()
    return tx.select("txn_id", relabel(tx, variant).alias("y_fit"))


def class_w(y, a):
    if a.cw_pow == 0: return np.ones(len(y), np.float32)
    n = np.bincount(y, minlength=K).astype(float)
    cw = (n[1:].max() / np.maximum(n, 1)) ** a.cw_pow; cw[0] = 1.0  # fraud classes scaled to the largest fraud class
    return cw[y].astype(np.float32)


def sample_fit(F, a, seed, rows=None, col="y_fit"):
    """Legit downsample (weight 1/frac) on the F rows picked by expr `rows` (default all), target `col` (falls back to y;
    -1 = drop row); returns (X, y, w). Only the sampled rows are copied (RAM)."""
    sel = np.ones(F.height, bool) if rows is None else F.select(rows).to_series().to_numpy()
    y = F[col if col in F.columns else "y"].to_numpy()[sel].astype(np.int32)
    keep = ((y != 0) | (np.random.default_rng(seed).random(len(y)) < a.legit_frac)) & (y >= 0)
    full = np.zeros(F.height, bool); full[np.flatnonzero(sel)[keep]] = True
    X = F.select(a.feats).filter(pl.Series(full)).to_numpy(); y = y[keep]
    w = np.where(y == 0, 1.0 / a.legit_frac, 1.0).astype(np.float32) * class_w(y, a)
    return X, y, w


def fit_one(F, a, seed, V=None, rounds=None, rows=None):
    fit, _ = MODELS[a.model]
    X, y, w = sample_fit(F, a, seed, rows)
    if V is None or getattr(a, "rounds", 0):
        return fit(X, y, w, None, None, None, a, seed, rounds or a.rounds)
    Xv, yv, wv = sample_fit(V, a, seed + 1000, col="y")  # early stopping on legit-downsampled raw-label val
    return fit(X, y, w, Xv, yv, wv, a, seed, a.max_rounds)


# ---------------------------------------------------------------- modes
def run_val(D, a):
    _, pred = MODELS[a.model]
    P_all, ids, ys, folds, best = [], [], [], [], []
    for fi, (fm, vm) in enumerate(FOLDS):
        F = pl.col("month").is_in(fm) & ~pl.col("is_test")
        V = D.filter((pl.col("month") == vm) & ~pl.col("is_test"))
        P = np.zeros((V.height, K), np.float32); its = []
        for s in range(a.seeds):
            t = time.time()
            m, it = fit_one(D, a, s, V, rows=F)
            P += predict_chunked(pred, m, V, a.feats) / a.seeds; its.append(it)
            print(f"  fold{fi} seed{s}: best_iter={it} ({time.time()-t:.0f}s)", flush=True)
        best.append(float(np.mean(its)))
        report(f"fold{fi} fit{fm}->val{vm}", V["y"].to_numpy(), P)
        P_all.append(P); ids.append(V["txn_id"]); ys.append(V["y"].to_numpy()); folds.append(np.full(V.height, fi))
        del V
    P = np.concatenate(P_all); y = np.concatenate(ys).astype(np.int64); fold = np.concatenate(folds)
    b = tune_bias(P, y)
    out = {"tag": a.tag, "modules": a.modules, "model": a.model, "best_iters": best, "bias": b.tolist(), "lgb": LGB,
           "args": {k: v for k, v in vars(a).items() if k != "feats"}, "n_feats": len(a.feats)}
    out["pooled_argmax"], out["pooled_biased"] = report("POOLED OOF (bias tuned on pooled)", y, P, b)
    for fi in range(len(FOLDS)):
        mk = fold == fi
        out[f"fold{fi}_argmax"], out[f"fold{fi}_biased"] = report(f"fold{fi} w/ pooled bias", y[mk], P[mk], b)
    np.save(ROOT / "oof" / f"{a.tag}.npy", P)
    pl.DataFrame({"txn_id": pl.concat(ids), "y": y, "fold": fold}).write_parquet(ROOT / "oof" / f"{a.tag}_ids.parquet")
    (ROOT / "oof" / f"{a.tag}_meta.json").write_text(json.dumps(out, indent=1))
    return out


def write_sub(P_test, b, tag):
    """P_test rows are in sample_submission order."""
    ids = pl.read_csv(SAMPLE)["id"]
    lab = np.array(LABELS)[decode(P_test, np.asarray(b))]
    sub = pl.DataFrame({"id": ids, "label": lab})
    assert sub.height == 260582 and sub["id"].n_unique() == sub.height
    assert sub["label"].is_in(LABELS).all()
    path = ROOT / "subs" / f"{tag}.csv"; sub.write_csv(path)
    chk = pl.read_csv(path)
    assert chk.columns == ["id", "label"] and chk["id"].equals(ids)
    print(f"submission {path}  rows={chk.height}  label counts:")
    print(chk["label"].value_counts().sort("count", descending=True))
    return path


def run_full(D, a):
    meta_p = ROOT / "oof" / f"{a.tag}_meta.json"
    if not meta_p.exists():
        print("no val record for this tag -> running val first", flush=True); run_val(D, a)
    meta = json.loads(meta_p.read_text())
    keys = [k for k in ("modules", "model", "lr", "legit_frac", "cw_pow",
                        "drop", "max_rounds", "patience", "rounds", "relabel") if k in meta["args"]]
    assert all(meta["args"][k] == getattr(a, k) for k in keys) and meta["n_feats"] == len(a.feats) and \
        (a.model != "lgb" or meta.get("lgb") == LGB), "val meta from a different config; rerun --mode val"
    rounds = int(np.mean(meta["best_iters"]) * 1.1)
    _, pred = MODELS[a.model]
    T = pl.read_csv(SAMPLE).select(pl.col("id").alias("txn_id")).join(D.filter(pl.col("is_test")), on="txn_id",
                                                                        how="left", maintain_order="left")
    assert T.height == 260582 and T["month"].null_count() == 0, "test ids missing from data"
    P = np.zeros((T.height, K), np.float32)
    for s in range(a.seeds):
        t = time.time(); m, _ = fit_one(D, a, s, rounds=rounds, rows=~pl.col("is_test"))
        P += predict_chunked(pred, m, T, a.feats) / a.seeds
        print(f"  full seed{s}: rounds={rounds} ({time.time()-t:.0f}s)", flush=True)
    np.save(ROOT / "preds" / f"{a.tag}_test.npy", P)
    return write_sub(P, meta["bias"], a.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="")
    ap.add_argument("--mode", choices=["val", "full"], default="val")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", choices=list(MODELS), default="lgb")
    ap.add_argument("--device", default="cpu", help="xgb only: cpu|cuda")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--legit-frac", type=float, default=0.2)
    ap.add_argument("--cw-pow", type=float, default=0.0, help="class weight (n_maxfraud/n_c)^pow, 0=off")
    ap.add_argument("--drop", default="", help="comma list of feature columns to drop")
    ap.add_argument("--rounds", type=int, default=0, help="fixed rounds, no early stopping (0 = early stop on val logloss)")
    ap.add_argument("--relabel", default="", help="fit target = relabel.relabel(tx, VARIANT) on train rows, e.g. C")
    ap.add_argument("--matrix", default="", help="read features from this cached matrix parquet instead of building")
    a = ap.parse_args(); t0 = time.time()
    a.modules = [m for m in a.modules.split(",") if m]
    if a.relabel:  # child process: relabel's load_all + module caches would otherwise stay in this process's heap (~3 GB)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(1) as ex: yf = ex.submit(fit_target, a.relabel).result()
    D, a.feats = build_matrix(a.modules, a.drop, a.matrix)
    D = D.join(yf, on="txn_id", how="left", maintain_order="left") if a.relabel else D.with_columns(pl.col("y").alias("y_fit"))
    print(f"tag={a.tag} modules={a.modules} n_feats={len(a.feats)} rows={D.height}", flush=True)
    run_val(D, a) if a.mode == "val" else run_full(D, a)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
