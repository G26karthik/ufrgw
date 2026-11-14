"""Experiment B trainer: baseline recipe of scratch_gap/numrep.py::run on 190 columns + the neighbour columns of knn_feats.py.
REP FOLD : zero new columns (harness check against PN_C_<aug|sep>.npy) -> scratch_knn/REP_<aug|sep>.npy
kA FOLD  : 190 + 50 columns -> scratch_gap/PN_kA_<aug|sep>.npy (refuses to overwrite) and the blend PN_kBL = 0.65 C + 0.35 kA
usage (cwd=work, PYTHONPATH=.): python scratch_knn/knn_train.py REP|kA FOLD"""
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T

c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2; META = ("txn_id", "month", "is_test", "y_fit")
arm, fi = sys.argv[1], int(sys.argv[2]); fm, vm = T.FOLDS[fi]; tag = "aug" if vm == 8 else "sep"
out = f"scratch_knn/REP_{tag}.npy" if arm == "REP" else f"scratch_gap/PN_kA_{tag}.npy"
assert not os.path.exists(out) and not os.path.exists(f"scratch_gap/PN_kBL_{tag}.npy"), "refusing to overwrite"
D = pl.read_parquet(f"scratch_gap/N_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; new = []
if arm == "kA":
    KF = pl.read_parquet(f"scratch_knn/KF_f{fi}.parquet"); assert KF["txn_id"].equals(D["txn_id"]); new = KF.columns[1:]
    D = pl.concat([D, KF.drop("txn_id")], how="horizontal"); feats = feats + new
fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
assert V["txn_id"].equals(pl.read_parquet(f"scratch_gap/N_{tag}_rows.parquet")["txn_id"]), "held rows are not in prediction order"
params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=10, verbose=-1)
P = np.zeros((V.height, T.K), np.float32); t = time.time(); Xv = V.select(feats).to_numpy()
for s in range(SEEDS):
    keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
    Xf = fit.select(feats).filter(pl.Series(keep)).to_numpy()
    if new: assert not np.isnan(Xf[:, -len(new):]).any() and not np.isnan(Xv[:, -len(new):]).any()  # every kept row was queried
    m = lgb.train({**params, "seed": s}, lgb.Dataset(Xf, y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32), feature_name=feats), ROUNDS)
    P += m.predict(Xv).astype(np.float32) / SEEDS
    if new:
        g = m.feature_importance("gain"); o = np.argsort(-g)
        print(f"seed {s}: gain share of the {len(new)} new columns {g[-len(new):].sum() / g.sum():.4f}; new columns in the top 30: {[feats[i] for i in o[:30] if feats[i] in new]}", flush=True)
np.save(out, P); C = np.load(f"scratch_gap/PN_C_{tag}.npy")
print(f"{arm} fold{fi} saved {out} ({time.time() - t:.0f}s); vs C: max abs probability difference {np.abs(P - C).max():.5f}, argmax differs on {int((P.argmax(1) != C.argmax(1)).sum())} of {len(P)} rows", flush=True)
if arm == "kA": np.save(f"scratch_gap/PN_kBL_{tag}.npy", (0.65 * C + 0.35 * P).astype(np.float32))
