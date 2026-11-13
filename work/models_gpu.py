"""GPU model diversity (XGBoost cuda / CatBoost GPU) on the cached m5dm matrix, train.py protocol and output formats.

usage (cwd=work):
  python models_gpu.py --model xgb --tag gpu_xgb --mode val  --rounds 400 [--ckpt 100,200,300] [--lr .05 --depth 8]
  python models_gpu.py --model xgb --tag gpu_xgb --mode full --rounds 400 --seeds 3     # same flags as val
Same folds, legit downsample (train.sample_fit rng) and bias tuning as train.py; fixed rounds, no early stopping
(noisy val logloss stops at ~30 iters, notes/gpu.md). Rows are read from feats_cache/matrix_m5dm.parquet by index.
val writes oof/<tag>.npy (+ _ids.parquet, _meta.json); --ckpt also saves scratch_gpu/<tag>_r<k>.npy per checkpoint.
full writes preds/<tag>_test.npy (sample_submission order) and subs/<tag>.csv.
"""
import argparse, json, time
import numpy as np
import polars as pl
from common import ROOT
from train import FOLDS, K, SAMPLE, tune_bias, decode, macro_f1, report, write_sub

MATRIX = ROOT / "feats_cache" / "matrix_m5dm.parquet"


def xgb_fit(X, y, w, a, seed):
    import xgboost as xgb
    p = dict(objective="multi:softprob", num_class=K, eta=a.lr, max_depth=a.depth, min_child_weight=a.mcw,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=a.l2, tree_method="hist", device="cuda", seed=seed,
             max_bin=256, nthread=3)
    m = xgb.train(p, xgb.DMatrix(X, y, weight=w), a.rounds)
    return lambda Z, k: m.predict(xgb.DMatrix(Z), iteration_range=(0, k)).astype(np.float32)


def cat_fit(X, y, w, a, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function="MultiClass", iterations=a.rounds, learning_rate=a.lr, depth=a.depth,
                           l2_leaf_reg=a.l2, task_type="GPU", devices="0", random_seed=seed, border_count=128,
                           thread_count=3, verbose=0)
    m.fit(X, y, sample_weight=w)
    return lambda Z, k: m.predict_proba(Z, ntree_end=k).astype(np.float32)


FIT = {"xgb": xgb_fit, "cat": cat_fit}


def load_rows(idx, feats, chunk=300_000):
    """Feature rows at matrix row indices idx (sorted), float32."""
    q = pl.scan_parquet(MATRIX).with_row_index("i")
    return np.concatenate([q.filter(pl.col("i").is_in(pl.Series(idx[j:j + chunk]).cast(pl.UInt32).implode())).select(feats)
                           .collect().to_numpy().astype(np.float32) for j in range(0, len(idx), chunk)])


def fit_set(M, rows, a, seed):
    """train.sample_fit on matrix rows `rows`: same rng draw over the same row order -> same legit keep mask."""
    y = M["y"].to_numpy()[rows].astype(np.int32)
    keep = (y != 0) | (np.random.default_rng(seed).random(len(y)) < a.legit_frac)
    y = y[keep]
    return load_rows(rows[keep], a.feats), y, np.where(y == 0, 1.0 / a.legit_frac, 1.0).astype(np.float32)


def run_val(M, a):
    ck = sorted({int(c) for c in a.ckpt.split(",") if c} | {a.rounds})
    P = {k: [] for k in ck}; ids, ys, folds = [], [], []
    for fi, (fm, vm) in enumerate(FOLDS):
        tr = np.flatnonzero((M["month"].is_in(fm) & ~M["is_test"]).to_numpy())
        va = np.flatnonzero(((M["month"] == vm) & ~M["is_test"]).to_numpy())
        Xv = load_rows(va, a.feats); acc = {k: 0 for k in ck}
        for s in range(a.seeds):
            t = time.time(); pr = FIT[a.model](*fit_set(M, tr, a, s), a, s)
            for k in ck: acc[k] = acc[k] + pr(Xv, k) / a.seeds
            print(f"  fold{fi} seed{s} ({time.time()-t:.0f}s)", flush=True)
        for k in ck: P[k].append(acc[k])
        ids.append(M["txn_id"].gather(va)); ys.append(M["y"].to_numpy()[va]); folds.append(np.full(len(va), fi))
        del Xv
    y = np.concatenate(ys).astype(np.int64); fold = np.concatenate(folds)
    for k in ck:  # tune_bias costs ~2 min, so checkpoints get argmax only (scratch_gpu/ev.py for biased)
        Pk = np.concatenate(P[k])
        if k != a.rounds:
            (ROOT / "scratch_gpu").mkdir(exist_ok=True); np.save(ROOT / "scratch_gpu" / f"{a.tag}_r{k}.npy", Pk)
            print(f"ckpt {k}: pooled argmax={macro_f1(y, Pk.argmax(1)).mean():.4f} Sep argmax="
                  f"{macro_f1(y[fold == 1], Pk[fold == 1].argmax(1)).mean():.4f}", flush=True)
    b = tune_bias(Pk, y)
    out = {"tag": a.tag, "model": a.model, "bias": b.tolist(), "n_feats": len(a.feats),
           "args": {k: v for k, v in vars(a).items() if k != "feats"}}
    out["pooled_argmax"], out["pooled_biased"] = report("POOLED OOF (bias tuned on pooled)", y, Pk, b)
    for fi in range(len(FOLDS)):
        out[f"fold{fi}_argmax"], out[f"fold{fi}_biased"] = report(f"fold{fi} w/ pooled bias", y[fold == fi], Pk[fold == fi], b)
    np.save(ROOT / "oof" / f"{a.tag}.npy", Pk)
    pl.DataFrame({"txn_id": pl.concat(ids), "y": y, "fold": fold}).write_parquet(ROOT / "oof" / f"{a.tag}_ids.parquet")
    (ROOT / "oof" / f"{a.tag}_meta.json").write_text(json.dumps(out, indent=1))


def run_full(M, a):
    meta = json.loads((ROOT / "oof" / f"{a.tag}_meta.json").read_text())
    same = ("model", "rounds", "lr", "depth", "l2", "mcw", "legit_frac")
    assert all(meta["args"][k] == getattr(a, k) for k in same) and meta["n_feats"] == len(a.feats), \
        "val meta from a different config; rerun --mode val"
    tr = np.flatnonzero((~M["is_test"]).to_numpy())
    te = M.with_row_index("i").filter(pl.col("is_test")).select("i", "txn_id")
    te = pl.read_csv(SAMPLE).select(pl.col("id").alias("txn_id")).join(te, on="txn_id", how="left", maintain_order="left")
    assert te.height == 260582 and te["i"].null_count() == 0, "test ids missing from matrix"
    order = te["i"].to_numpy(); srt = np.argsort(order)
    Xt = np.empty((len(order), len(a.feats)), np.float32); Xt[srt] = load_rows(order[srt], a.feats)
    P = np.zeros((len(order), K), np.float32)
    for s in range(a.seeds):
        t = time.time(); P += FIT[a.model](*fit_set(M, tr, a, s), a, s)(Xt, a.rounds) / a.seeds
        print(f"  full seed{s} ({time.time()-t:.0f}s)", flush=True)
    np.save(ROOT / "preds" / f"{a.tag}_test.npy", P)
    write_sub(P, meta["bias"], a.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(FIT), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["val", "full"], default="val")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--mcw", type=float, default=5.0, help="xgb min_child_weight")
    ap.add_argument("--legit-frac", type=float, default=0.2)
    a = ap.parse_args(); t0 = time.time()
    M = pl.read_parquet(MATRIX, columns=["txn_id", "month", "y", "is_test"])
    a.feats = [c for c in pl.scan_parquet(MATRIX).collect_schema().names() if c not in M.columns]
    print(f"tag={a.tag} model={a.model} n_feats={len(a.feats)}", flush=True)
    run_val(M, a) if a.mode == "val" else run_full(M, a)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
