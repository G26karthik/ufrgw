"""LightGBM sources of the combined submission on the D tables (production density-matched matrix).
  sts  soft two-stage (scratch_stack/n14/gbdt_n.py recipe), seeds 0-3        -> PD_xSTS_<kind>.npy
  bsel 1023 bins on the row-level numeric columns (numrep.py A_BSEL), seeds 0,1 -> PD_xBSEL_<kind>.npy
  ka   190 + 50 neighbour columns (scratch_mix/KFD_f<fold>.parquet), seeds 0,1  -> PD_xKA_<kind>.npy
Same fit rows, target and weights as the LightGBM recipe: y_fit, legit rows downsampled to 20% with weight 5.
fold 0 -> aug, fold 1 -> sep, fold 2 (all train rows fit, real test rows held) -> oct (row order = D_f2 test rows).
usage (cwd=work, PYTHONPATH=.): python scratch_mix/mix_lgb.py sts|bsel|ka FOLD[,FOLD]"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T

c = pl.col; SRC, FOLDS = sys.argv[1], [int(f) for f in sys.argv[2].split(",")]; TH, LF, K = 4, 0.2, T.K
META = ("txn_id", "month", "is_test", "y_fit"); SEEDS = range(4) if SRC == "sts" else range(2)
ROWNUM = ["amount", "log_amount", "cents", "sim_age_hours", "device_age_days", "login_to_txn_sec", "balance_before", "amt_share_bal", "minute_of_day"]
for fi in FOLDS:
    kind = ("aug", "sep", "oct")[fi]; out = f"scratch_gap/PD_x{SRC.upper()}_{kind}.npy"; assert not os.path.exists(out), out
    D = pl.read_parquet(f"scratch_gap/D_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
    if SRC == "ka":
        KF = pl.read_parquet(f"scratch_mix/KFD_f{fi}.parquet"); assert KF["txn_id"].equals(D["txn_id"])
        D = pl.concat([D, KF.drop("txn_id")], how="horizontal"); feats = feats + KF.columns[1:]
    fit = D.filter(~c("is_test")); held = D.filter(c("is_test") & (c("month") == 8 + fi)) if fi < 2 else D.filter(c("is_test"))
    if fi < 2: assert held["txn_id"].equals(pl.read_parquet(f"scratch_gap/D_{kind}_rows.parquet", columns=["txn_id"])["txn_id"])
    y = fit["y_fit"].to_numpy().astype(np.int32); Xh = held.select(feats).to_numpy(); P = np.zeros((len(Xh), K)); t = time.time()
    lp = dict(learning_rate=0.05, **T.LGB, num_threads=TH, verbose=-1)
    for s in SEEDS:
        keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
        Xf = fit.select(feats).filter(pl.Series(keep)).to_numpy(); yk = y[keep]; w = np.where(yk == 0, 1 / LF, 1.0).astype(np.float32)
        if SRC == "sts":
            fr = yk > 0
            b = lgb.train(dict(objective="binary", seed=s, **lp), lgb.Dataset(Xf, (yk > 0).astype(np.int32), weight=w), 150)
            m = lgb.train(dict(objective="multiclass", num_class=K - 1, seed=s, **lp), lgb.Dataset(Xf[fr], yk[fr] - 1), 150)
            pf = b.predict(Xh)[:, None]; p = np.hstack([1 - pf, pf * m.predict(Xh)])
        else:
            if SRC == "ka": assert not np.isnan(Xf[:, -50:]).any() and not np.isnan(Xh[:, -50:]).any()
            extra = dict(max_bin_by_feature=[1023 if k in ROWNUM else 255 for k in feats]) if SRC == "bsel" else {}
            p = lgb.train(dict(objective="multiclass", num_class=K, seed=s, **lp, **extra), lgb.Dataset(Xf, yk, weight=w, feature_name=feats), 150).predict(Xh)
        P += p / len(SEEDS); print(f"{SRC} fold{fi} seed{s}: fit rows {int(keep.sum())} ({time.time() - t:.0f}s)", flush=True)
    assert abs(P.sum(1) - 1).max() < 1e-4
    np.save(out, P.astype(np.float32)); print("saved", out, P.shape, flush=True)
    if fi == 2: held.select("txn_id").write_parquet(f"scratch_mix/oct_ids_{SRC}.parquet")
