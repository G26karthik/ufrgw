"""Density-view augmentation: train on the union of K halvings of train (density.halves seeds), weight 1/K per copy.

Each view = matrix_m5dm2_pairsfd with flow/takeover (__dm) and burst/scam/ring (__dm2, mlink 24 h) rebuilt under
another halving seed; base features and full-density `pairs` are view-independent. View 7 is the original matrix.
Validation rows and test rows always take the seed-7 view (comparable with h6); <tag>tta averages predictions over views.
Fit rows, legit downsample (sample_fit) and target (relabel) are identical to train.py, so `--views 7` reproduces h3/h6.

usage (cwd=work):
  python augment.py --views 7,8,9 --seeds 3 [--seed-off 0] [--mdl-scale] --mode val|full --threads 4 --tag NAME
  then: python oos.py NAME; python postproc.py NAME --qr-null   (same for NAMEtta)
Caches (never overwrites existing ones): feats_cache/<module>__dm[2]s<seed>.parquet, matrix_m5dm2_pairsfd_s<seed>.parquet.
"""
import argparse, json, time
import importlib
import numpy as np
import polars as pl
from common import ROOT, CACHE, load_all
from density import halves
from make_matrix import MODULES
import train as T

BASE = ROOT / "feats_cache" / "matrix_m5dm2_pairsfd.parquet"
META = ("txn_id", "month", "y", "is_test")


def module_view(name, ml, seed, tx):
    """density.run_feature_matched with a halving seed (no mtime cache logic; density.py untouched)."""
    out = CACHE / f"{name}__dm{'2' if ml else ''}s{seed}.parquet"
    if out.exists():
        return pl.read_parquet(out)
    h = halves(tx, seed=seed, mlink=ml)
    mod = importlib.import_module(f"feats.{name}")
    parts = []
    for k in (0, 1):
        sub = tx.filter((h == k).fill_null(False) | tx["is_test"])
        own = sub.filter(~pl.col("is_test") | pl.lit(k == 0)).select("txn_id")
        parts.append(mod.build(sub).join(own, on="txn_id", how="semi"))
    f = pl.concat(parts)
    assert f.height == tx.height and f["txn_id"].n_unique() == tx.height, name
    f.write_parquet(out)
    return f


def view_path(seed):
    if seed == 7:
        return BASE
    out = CACHE / f"matrix_m5dm2_pairsfd_s{seed}.parquet"
    if out.exists():
        return out
    tx = load_all(); D = pl.read_parquet(BASE); cols = D.columns
    for name, ml in MODULES:
        f = module_view(name, ml, seed, tx)
        fc = [c for c in f.columns if c != "txn_id"]
        D = D.drop(fc).join(f.with_columns([pl.col(c).cast(pl.Float32) for c in fc]), on="txn_id", how="left", maintain_order="left")
        print(f"view {seed}: {name}", flush=True)
    D = D.select(cols)
    assert D.height == tx.height and D["txn_id"].equals(tx["txn_id"])
    D.write_parquet(out)
    return out


def load_view(seed, yfit):
    D = pl.read_parquet(view_path(seed))
    assert D["txn_id"].equals(yfit["txn_id"]), "view row order differs from load_all"
    return D.with_columns(yfit["y_fit"])


def sample_views(a, seed, rows):
    """sample_fit on every view (same rows: same y_fit/order/rng), each copy weighted 1/K."""
    Xs, ys, ws = [], [], []
    for v in a.views:
        D = load_view(v, a.yfit)
        X, y, w = T.sample_fit(D, a, seed, rows); del D
        Xs.append(X); ys.append(y); ws.append(w / len(a.views))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def subset_views(a, expr):
    return {v: load_view(v, a.yfit).filter(expr).select(a.feats) for v in a.views}


def save_oof(tag, P, ids, y, fold, a, extra):
    b = T.tune_bias(P, y)
    out = {"tag": tag, "bias": b.tolist(), "lgb": T.LGB, "best_iters": [a.rounds] * len(T.FOLDS), "n_feats": len(a.feats),
           "args": {k: v for k, v in vars(a).items() if k not in ("feats", "yfit")}, **extra}
    out["pooled_argmax"], out["pooled_biased"] = T.report(f"{tag} POOLED", y, P, b)
    for fi in range(len(T.FOLDS)):
        mk = fold == fi
        out[f"fold{fi}_argmax"], out[f"fold{fi}_biased"] = T.macro_f1(y[mk], P[mk].argmax(1)).mean(), T.macro_f1(y[mk], T.decode(P[mk], b)).mean()
        print(f"  fold{fi} argmax={out[f'fold{fi}_argmax']:.4f} biased={out[f'fold{fi}_biased']:.4f}")
    np.save(ROOT / "oof" / f"{tag}.npy", P)
    pl.DataFrame({"txn_id": ids, "y": y, "fold": fold}).write_parquet(ROOT / "oof" / f"{tag}_ids.parquet")
    (ROOT / "oof" / f"{tag}_meta.json").write_text(json.dumps(out, indent=1))


def run_val(a):
    fit, pred = T.MODELS[a.model]
    P_all, Pt_all, ids, ys, folds = [], [], [], [], []
    for fi, (fm, vm) in enumerate(T.FOLDS):
        vexpr = (pl.col("month") == vm) & ~pl.col("is_test")
        V = subset_views(a, vexpr)
        Vm = pl.read_parquet(BASE, columns=list(META)).filter(vexpr)
        P = np.zeros((Vm.height, T.K), np.float32); Pt = np.zeros_like(P)
        for s in range(a.seed_off, a.seed_off + a.seeds):
            t = time.time()
            X, y, w = sample_views(a, s, pl.col("month").is_in(fm) & ~pl.col("is_test"))
            m, _ = fit(X, y, w, None, None, None, a, s, a.rounds); del X
            pv = {v: T.predict_chunked(pred, m, V[v], a.feats) for v in a.views}
            P += pv[7] / a.seeds; Pt += sum(pv.values()) / len(pv) / a.seeds
            print(f"  fold{fi} seed{s}: n_fit={len(y)} ({time.time()-t:.0f}s)", flush=True)
        P_all.append(P); Pt_all.append(Pt); ids.append(Vm["txn_id"]); ys.append(Vm["y"].to_numpy()); folds.append(np.full(Vm.height, fi))
        del V
    y = np.concatenate(ys).astype(np.int64); fold = np.concatenate(folds); ids = pl.concat(ids)
    save_oof(a.tag, np.concatenate(P_all), ids, y, fold, a, {})
    if len(a.views) > 1:
        save_oof(a.tag + "tta", np.concatenate(Pt_all), ids, y, fold, a, {})


def run_full(a):
    fit, pred = T.MODELS[a.model]
    meta = json.loads((ROOT / "oof" / f"{a.tag}_meta.json").read_text())
    keys = ("views", "rounds", "relabel", "legit_frac", "mdl_scale", "lr")
    assert all(meta["args"][k] == getattr(a, k) for k in keys), "val meta from a different config"
    rounds = int(a.rounds * 1.1)
    tid = pl.read_csv(T.SAMPLE).select(pl.col("id").alias("txn_id"))
    Tv = {}
    for v in a.views:  # test rows in sample_submission order
        D = load_view(v, a.yfit).filter(pl.col("is_test"))
        Tv[v] = tid.join(D, on="txn_id", how="left", maintain_order="left").select(a.feats); del D
        assert Tv[v].height == 260582
    P = np.zeros((260582, T.K), np.float32); Pt = np.zeros_like(P)
    for s in range(a.seed_off, a.seed_off + a.seeds):
        t = time.time()
        X, y, w = sample_views(a, s, ~pl.col("is_test"))
        m, _ = fit(X, y, w, None, None, None, a, s, rounds); del X
        pv = {v: T.predict_chunked(pred, m, Tv[v], a.feats) for v in a.views}
        P += pv[7] / a.seeds; Pt += sum(pv.values()) / len(pv) / a.seeds
        print(f"  full seed{s}: rounds={rounds} ({time.time()-t:.0f}s)", flush=True)
    np.save(ROOT / "preds" / f"{a.tag}_test.npy", P)
    T.write_sub(P, meta["bias"], a.tag)
    if len(a.views) > 1:
        np.save(ROOT / "preds" / f"{a.tag}tta_test.npy", Pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="7,8,9")
    ap.add_argument("--mode", choices=["val", "full", "build"], default="val")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--relabel", default="C")
    ap.add_argument("--mdl-scale", action="store_true", help="min_data_in_leaf x K (copies count as rows in LightGBM)")
    a = ap.parse_args(); t0 = time.time()
    a.views = [int(v) for v in a.views.split(",")]
    assert a.views[0] == 7, "view 7 (the h6 matrix) must be first: validation/test use it"
    if a.mode == "build":
        for v in a.views: print(view_path(v))
        return
    a.model, a.lr, a.legit_frac, a.cw_pow = "lgb", 0.05, 0.2, 0.0  # train.py defaults behind h6
    if a.mdl_scale:
        T.LGB["min_data_in_leaf"] *= len(a.views)
    from concurrent.futures import ProcessPoolExecutor  # child process: load_all heap (~3 GB) is freed (as train.py)
    with ProcessPoolExecutor(1) as ex: a.yfit = ex.submit(T.fit_target, a.relabel).result()
    a.feats = [c for c in pl.read_parquet_schema(BASE) if c not in META]
    assert len(a.feats) == 190
    print(f"tag={a.tag} views={a.views} seeds={a.seed_off}..{a.seed_off + a.seeds - 1} LGB={T.LGB}", flush=True)
    run_val(a) if a.mode == "val" else run_full(a)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
