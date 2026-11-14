"""New prediction sources on benchmark N, same fit rows / target / weights as the LightGBM recipe (bench_nat.py):
relabel-C target, rows with y_fit < 0 dropped, legit rows downsampled to 20% with weight 5, 190 features.
  xgb: XGBoost hist, train.xgb_fit parameters, eta .05, 200 rounds, 2 seeds           -> PN_sXGB
  cat: CatBoost MultiClass (CPU), depth 8, lr .1, l2 3, 1000 iterations, 1 seed        -> PN_sCAT
  sub: LightGBM recipe (train.LGB, 150 rounds, lr .05, 2 seeds) on 113 features: burst__ and ring__ modules removed -> PN_sSUB
  ts : soft two-stage LightGBM (train.LGB, 150 rounds, lr .05, 2 seeds):                -> PN_sTS
       binary fraud vs legit on the same weighted fit rows, 11-class type model on the fraud-labelled fit rows,
       P(legit) = 1 - P(fraud), P(k) = P(fraud) * P(k | fraud). No hard gate.
usage (cwd=work, PYTHONPATH=.): python scratch_stack/n14/gbdt_n.py xgb|cat|ts|sub|lgb [gpu]  -> scratch_gap/PN_s<XGB|CAT|TS|SUB>_<aug|sep>.npy
replication: env BM=D (benchmark D matrix -> PD_s...), env SEED0=2 (seeds 2,3 -> ..._s<M>b2)"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T
from nb import fold_data, K, G, BM

M = sys.argv[1]; TH, LF = 5, 0.2; SEEDS = 1 if M == "cat" else 2
S0 = int(os.environ.get("SEED0", 0)); OUT = f"P{BM}_s{M.upper()}" + (f"b{S0}" if S0 else "")  # SEED0 / BM: replication runs


def fit_pred(Xf, y, w, Xh, seed):
    assert len(np.unique(y)) == K
    if M == "xgb":
        import xgboost as xgb
        p = dict(objective="multi:softprob", num_class=K, eta=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", nthread=TH, seed=seed)
        return xgb.train(p, xgb.DMatrix(Xf, y, weight=w), 200).predict(xgb.DMatrix(Xh))
    if M == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(loss_function="MultiClass", iterations=1000, learning_rate=0.1, depth=8, l2_leaf_reg=3,
                               random_seed=seed, thread_count=TH, verbose=250, allow_writing_files=False, **(dict(task_type="GPU") if sys.argv[2:] == ["gpu"] else {}))
        m.fit(Xf, y, sample_weight=w); assert [int(k) for k in m.classes_] == list(range(K))
        return m.predict_proba(Xh)
    lp = dict(learning_rate=0.05, **T.LGB, num_threads=TH, verbose=-1, seed=seed); fr = y > 0
    if M in ("sub", "lgb"):  # lgb = the plain 12-class recipe (control for seed averaging)
        return lgb.train(dict(objective="multiclass", num_class=K, **lp), lgb.Dataset(Xf, y, weight=w), 150).predict(Xh)
    b = lgb.train(dict(objective="binary", **lp), lgb.Dataset(Xf, (y > 0).astype(np.int32), weight=w), 150)
    m = lgb.train(dict(objective="multiclass", num_class=K - 1, **lp), lgb.Dataset(Xf[fr], y[fr] - 1), 150)  # class j = label j + 1
    pf = b.predict(Xh)[:, None]
    return np.hstack([1 - pf, pf * m.predict(Xh)])


for fi in (0, 1):
    t = time.time(); feats, fit, y, Xh, kind = fold_data(fi); P = np.zeros((len(Xh), K), np.float64)
    if M == "sub":  # feature variant: no burst__ / ring__ columns
        ok = [not k.startswith(("burst__", "ring__")) for k in feats]; Xh = Xh[:, ok]; feats = [k for k, o in zip(feats, ok) if o]; assert len(feats) == 113
    for s in range(S0, S0 + SEEDS):
        keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)  # same draw as bench_nat.py
        P += fit_pred(fit.filter(pl.Series(keep)).select(feats).to_numpy(), y[keep], np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32), Xh, s) / SEEDS
        print(f"{M} fold{fi} seed{s}: fit rows {int(keep.sum())} ({time.time() - t:.0f}s)", flush=True)
    assert abs(P.sum(1) - 1).max() < 1e-4
    np.save(f"{G}/{OUT}_{kind}.npy", P.astype(np.float32))
    if kind == "aug":  # September labels are not read here
        ya = pl.read_parquet(f"{G}/{BM}_aug_rows.parquet")["y_true"].to_numpy().astype(np.int64)
        C = np.load(f"{G}/P{BM}_C_aug.npy")
        print(f"{M} Aug: macro argmax {T.macro_f1(ya, P.argmax(1)).mean():.4f} (LightGBM C {T.macro_f1(ya, C.argmax(1)).mean():.4f}), "
              f"log-loss {-np.log(P[np.arange(len(ya)), ya] + 1e-12).mean():.5f} (C {-np.log(C[np.arange(len(ya)), ya] + 1e-12).mean():.5f})", flush=True)
print(M, "done", flush=True)
