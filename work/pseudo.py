"""Transductive adaptation: pseudo-labelling / self-training and adversarial domain weights (notes/pseudo.md).

sim: for each train.FOLDS fold (fit Jul -> target Aug, fit Jul+Aug -> target Sep) the target month is treated as
  unlabeled. Pseudo labels = source probs (tun_best OOF, or another variant's cached probs for round 2) decoded with
  the bias tuned on the OTHER fold (no target labels used). Refit = tun_best config on fit rows (relabel C) + pseudo rows,
  score on the true target labels. Probs cached as scratch_pseudo/P_<variant>_s<seed>.npy (rows = oof/tun_best_ids order).
full: fit all train rows + pseudo-labelled test rows (source preds/tun_best_test.npy, tun_best meta bias)
  -> preds/pseudo_test.npy, subs/pseudo.csv (decoded with the tun_best meta bias).

usage (cwd=work): python pseudo.py prep | sim VAR[,VAR] --seeds 0,1,2 | score VAR[,VAR] | full VAR
"""
import argparse, json, sys, time
from types import SimpleNamespace
import numpy as np
import polars as pl
from common import ROOT
from train import FOLDS, K, MODELS, sample_fit, tune_bias, decode, macro_f1, predict_chunked, write_sub, SAMPLE

S = ROOT / "scratch_pseudo"; S.mkdir(exist_ok=True)
MATRIX = ROOT / "feats_cache" / "matrix_m5dm.parquet"
# thr: min P of the decoded fraud class; q: keep the top-q share of each decoded fraud class (per-class threshold);
# legit: also the top-x share of rows by P[legit] (P[legit] tops out at .99, so a share, not a threshold); ev: event-majority labels (clean_labels.clean on the decoded target rows),
# rows clean changed are kept whatever their confidence; w: pseudo row weight; dw: adversarial domain weights on fit rows;
# src: pseudo source = mean of that variant's cached seeds (round 2).
VARIANTS = {
    "base": {}, "frall": dict(thr=0), "fr50": dict(thr=.5), "fr80": dict(thr=.8), "frq50": dict(q=.5),
    "fr50w05": dict(thr=.5, w=.5), "fr50leg": dict(thr=.5, legit=.5), "ev": dict(ev=1, thr=0), "ev50": dict(ev=1, thr=.5),
    "dw": dict(dw=1),
}


def prep():
    """lab.parquet: event keys + relabel-C fit target + clean_labels proxy, all rows in load_all order (needs ~3 GB)."""
    from common import load_all
    from clean_labels import clean
    from relabel import relabel
    tx = load_all()
    tx.select("txn_id", "ts", "sender_id", "device_id", "receiver_type", "receiver_id",
              relabel(tx, "C").alias("y_fit"), clean(tx).alias("yc")).write_parquet(S / "lab.parquet")


def load(test):
    q = pl.scan_parquet(MATRIX)
    if not test: q = q.filter(~pl.col("is_test"))
    D = q.join(pl.scan_parquet(S / "lab.parquet").select("txn_id", "y_fit"), on="txn_id", how="left", maintain_order="left").collect()
    return D, [c for c in D.columns if c not in ("txn_id", "month", "y", "is_test", "y_fit")]


def args(feats):
    return SimpleNamespace(model="lgb", lr=0.05, threads=3, legit_frac=0.2, cw_pow=0.0, feats=feats)


def pseudo_labels(P, b, cfg, slim):
    """Pseudo label per target row (-1 = not used)."""
    d = decode(P, b); conf = P[np.arange(len(d)), d]
    keep = d > 0
    if "q" in cfg:
        for k in range(1, K):
            m = d == k
            if m.any(): keep[m] &= conf[m] >= np.quantile(conf[m], 1 - cfg["q"])
    else:
        keep &= conf >= cfg.get("thr", 0)
    if cfg.get("ev"):
        from clean_labels import clean
        ye = clean(slim.with_columns(pl.Series("y", d, dtype=pl.Int8))).to_numpy().astype(np.int64)
        keep = (ye > 0) & (keep | (ye != d)); d = ye
    lab = np.where(keep, d, -1)
    if "legit" in cfg: lab[(d == 0) & (P[:, 0] >= np.quantile(P[:, 0], 1 - cfg["legit"]))] = 0
    return lab


def domain_weights(D, a, fit_rows, tgt_rows, seed=0):
    """p(target)/p(fit) from an LGB fit-vs-target classifier (uniform 200k rows per side, AUC on a held-out half)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    fi, ti = (np.flatnonzero(D.select(r).to_series().to_numpy()) for r in (fit_rows, tgt_rows))
    fs, ts = rng.choice(fi, 200_000, replace=False), rng.choice(ti, 200_000, replace=False)
    idx = np.concatenate([fs, ts]); z = np.r_[np.zeros(len(fs)), np.ones(len(ts))]
    X = D[idx].select(a.feats).to_numpy(); h = rng.random(len(z)) < .5
    p = dict(objective="binary", learning_rate=.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=.5, lambda_l2=10,
             num_threads=a.threads, verbose=-1, seed=seed)
    m = lgb.train(p, lgb.Dataset(X[h], z[h]), 200)
    print(f"  adversarial AUC (held-out) = {roc_auc_score(z[~h], m.predict(X[~h])):.4f}", flush=True)
    imp = sorted(zip(m.feature_importance("gain"), a.feats), reverse=True)[:8]
    print("  top features:", [f for _, f in imp], flush=True)
    m = lgb.train(p, lgb.Dataset(X, z), 200)
    pr = np.clip(np.concatenate([m.predict(D[fi[i:i + 200_000]].select(a.feats).to_numpy()) for i in range(0, len(fi), 200_000)]), .02, .98)
    w = np.ones(D.height, np.float32); r = pr / (1 - pr); w[fi] = r / r.mean()
    print(f"  weight quantiles .01/.5/.99/max: {np.round(np.quantile(w[fi], [.01, .5, .99, 1]), 3).tolist()}", flush=True)
    return w


def fit_predict(D, a, seed, fit_rows, tgt_rows, lab, cfg, Vt, rounds, dw=None):
    fit, pred = MODELS["lgb"]
    if dw is None:
        X, y, w = sample_fit(D, a, seed, fit_rows)
    else:  # carry the domain weight through sample_fit's row pick as an extra column
        X, y, w = sample_fit(D.with_columns(pl.Series("dw", dw)), args(a.feats + ["dw"]), seed, fit_rows)
        w = w * X[:, -1]; X = X[:, :-1]
    if lab is not None and (lab >= 0).any():
        yp = np.full(D.height, -1, np.int32); yp[np.flatnonzero(D.select(tgt_rows).to_series().to_numpy())] = lab
        Xp, ypp, wp = sample_fit(D.with_columns(pl.Series("y_ps", yp)), a, seed + 500, tgt_rows, col="y_ps")
        X, y, w = np.concatenate([X, Xp]), np.concatenate([y, ypp]), np.concatenate([w, wp * cfg.get("w", 1.0)])
        print(f"    pseudo rows in fit: {len(ypp)} (fraud {int((ypp > 0).sum())})", flush=True)
    m, _ = fit(X, y, w, None, None, None, a, seed, rounds); del X
    return predict_chunked(pred, m, Vt, a.feats)


def source(name):
    if name == "tun_best": return np.load(ROOT / "oof" / "tun_best.npy")
    fs = sorted(S.glob(f"P_{name}_s*.npy")); assert fs, f"no cached probs for {name}"
    return np.mean([np.load(f) for f in fs], 0)


def sim(variants, seeds):
    D, feats = load(test=False); a = args(feats)
    ids = pl.read_parquet(ROOT / "oof" / "tun_best_ids.parquet"); y = ids["y"].to_numpy().astype(np.int64); fold = ids["fold"].to_numpy()
    lab_all = pl.read_parquet(S / "lab.parquet")
    for v in variants:
        cfg = VARIANTS[v] if v in VARIANTS else dict(VARIANTS[v.split("_r2")[0]], src=v.split("_r2")[0])  # NAME_r2 = round 2
        Psrc = source(cfg.get("src", "tun_best"))
        for s in seeds:
            out = S / f"P_{v}_s{s}.npy"
            if out.exists(): continue
            t = time.time(); Ps = []
            for fi, (fm, vm) in enumerate(FOLDS):
                fit_rows = pl.col("month").is_in(fm); tgt_rows = pl.col("month") == vm
                Vt = D.filter(tgt_rows); assert Vt["txn_id"].equals(ids.filter(pl.col("fold") == fi)["txn_id"])
                lab = None
                if cfg and not cfg.get("dw"):
                    o = fold != fi; b = tune_bias(Psrc[o], y[o])  # bias from the other fold: no target labels
                    slim = Vt.select("txn_id").join(lab_all, on="txn_id", how="left", maintain_order="left")
                    lab = pseudo_labels(Psrc[fold == fi], b, cfg, slim)
                    yt = y[fold == fi]; u = lab >= 0
                    print(f"  fold{fi}: pseudo {int(u.sum())} rows, fraud {int((lab > 0).sum())}, label acc vs true y "
                          f"{(lab[u] == yt[u]).mean():.3f} (fraud rows {(lab[lab > 0] == yt[lab > 0]).mean():.3f})", flush=True)
                dw = None
                if cfg.get("dw"):
                    f = S / f"dw_fold{fi}.npy"
                    if not f.exists(): np.save(f, domain_weights(D, a, fit_rows, tgt_rows))
                    dw = np.load(f)
                Ps.append(fit_predict(D, a, s, fit_rows, tgt_rows, lab, cfg, Vt, 150, dw)); del Vt
            np.save(out, np.concatenate(Ps)); print(f"{v} seed{s} done ({time.time() - t:.0f}s)", flush=True)


def metrics(P, y, yc, sep):
    b = tune_bias(P, y); pr = decode(P, b); bo = tune_bias(P[~sep], y[~sep]); po = decode(P[sep], bo)
    f = lambda t, p: macro_f1(t, p).mean()
    return [f(y, pr), f(y[sep], pr[sep]), f(yc, pr), f(yc[sep], pr[sep]), f(y[sep], po), f(yc[sep], po)]


def score(variants):
    ids = pl.read_parquet(ROOT / "oof" / "tun_best_ids.parquet"); y = ids["y"].to_numpy().astype(np.int64); sep = ids["fold"].to_numpy() == 1
    yc = ids.join(pl.read_parquet(S / "lab.parquet").select("txn_id", "yc"), on="txn_id", how="left", maintain_order="left")["yc"].to_numpy().astype(np.int64)
    print("variant            pooled   sep  cPool  cSep | oosSep oosCSep", flush=True)
    rows = [("tun_best(3s)", np.load(ROOT / "oof" / "tun_best.npy"))]
    for v in variants:
        fs = sorted(S.glob(f"P_{v}_s*.npy"))
        rows += [(f"{v} {f.stem[-2:]}", np.load(f)) for f in fs] + ([(f"{v} mean{len(fs)}", np.mean([np.load(f) for f in fs], 0))] if len(fs) > 1 else [])
    res = {}
    for name, P in rows:
        res[name] = m = metrics(P, y, yc, sep)
        print(f"{name:18s} " + " ".join(f"{x:.4f}" for x in m[:4]) + " | " + " ".join(f"{x:.4f}" for x in m[4:]), flush=True)
    (S / "score.json").write_text(json.dumps({**json.loads((S / "score.json").read_text()), **res} if (S / "score.json").exists() else res, indent=1))


def full(v, seeds):
    cfg = VARIANTS[v]; meta = json.loads((ROOT / "oof" / "tun_best_meta.json").read_text()); b = np.asarray(meta["bias"], np.float32)
    D, feats = load(test=True); a = args(feats)
    tid = pl.read_csv(SAMPLE).select(pl.col("id").alias("txn_id"))
    T = tid.join(D.filter(pl.col("is_test")), on="txn_id", how="left", maintain_order="left")
    assert T.height == 260582 and T["month"].null_count() == 0
    # pseudo labels in D's test-row order (the order sample_fit walks the target rows)
    Dt = D.filter(pl.col("is_test")).select("txn_id")
    src = Dt.join(tid.with_columns(pl.Series("r", np.arange(tid.height))), on="txn_id", how="left", maintain_order="left")["r"].to_numpy()
    Psrc = np.load(ROOT / "preds" / "tun_best_test.npy")[src]
    slim = Dt.join(pl.read_parquet(S / "lab.parquet"), on="txn_id", how="left", maintain_order="left")
    lab = pseudo_labels(Psrc, b, cfg, slim); print(f"Oct pseudo rows {int((lab >= 0).sum())}, fraud {int((lab > 0).sum())}", flush=True)
    P = np.mean([fit_predict(D, a, s, ~pl.col("is_test"), pl.col("is_test"), lab, cfg, T, int(150 * 1.1)) for s in seeds], 0)
    np.save(ROOT / "preds" / "pseudo_test.npy", P)
    write_sub(P, b, "pseudo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prep", "sim", "score", "full"]); ap.add_argument("variants", nargs="?", default="")
    ap.add_argument("--seeds", default="0"); x = ap.parse_args(); vs = [v for v in x.variants.split(",") if v]
    seeds = [int(s) for s in x.seeds.split(",")]
    {"prep": lambda: prep(), "sim": lambda: sim(vs, seeds), "score": lambda: score(vs), "full": lambda: full(vs[0], seeds)}[x.cmd]()
