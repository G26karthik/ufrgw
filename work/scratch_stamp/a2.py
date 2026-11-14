"""A2 variant m50: controlled stamp masking on FIT rows (declared in PROGRESS.md before fitting).
Each stamped non-QR fit row (txn_type != QR and qr_vpa_mismatch == 0), whatever its label, gets qr_vpa_mismatch = NaN with
probability SHARE (mask seed 7, independent of the model seeds). a1.py showed that a full rebuild changes only this column.
Held rows are NOT masked. Recipe = numrep.run arm A_C otherwise (seeds 0,1, 150 rounds, 4 threads), labels unchanged.
out: scratch_gap/P<BM>_m50_<aug|sep>.npy and P<BM>_m50b35_<aug|sep>.npy = .65 * P<BM>_C + .35 * m50.
usage (cwd=work, PYTHONPATH=.): [BM=D] python scratch_stamp/a2.py"""
import os, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T
from common import load_all

c = pl.col; SEEDS, ROUNDS, LF, SHARE, MSEED = 2, 150, 0.2, 0.5, 7; BM = os.environ.get("BM", "N")
META = ("txn_id", "month", "is_test", "y_fit")
tt = load_all().select("txn_id", "txn_type")
for fi, (fm, vm) in enumerate(T.FOLDS):
    t = time.time(); tag = "aug" if vm == 8 else "sep"
    for nm in ("m50", "m50b35"): assert not os.path.exists(f"scratch_gap/P{BM}_{nm}_{tag}.npy"), "output name already used"
    D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]
    fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
    ty = fit.select("txn_id").join(tt, on="txn_id", how="left", maintain_order="left")["txn_type"]; assert ty.null_count() == 0
    X = fit.select(feats).to_numpy(); j = feats.index("qr_vpa_mismatch")
    st = (ty != "QR").to_numpy() & (X[:, j] == 0); mk = st & (np.random.default_rng(MSEED).random(len(st)) < SHARE)
    assert np.isnan(X[(ty != "QR").to_numpy() & ~st & (X[:, j] != 1), j]).all()  # ordinary non-QR rows are NaN
    X[mk, j] = np.nan
    print(f"{BM} fold{fi}: fit rows {len(y)}, stamped non-QR fit rows {int(st.sum())} (target fraud {int((st & (y > 0)).sum())}, legit {int((st & (y == 0)).sum())}, dropped {int((st & (y < 0)).sum())}), masked {int(mk.sum())}", flush=True)
    P = np.zeros((V.height, T.K), np.float32)
    params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=4, verbose=-1)
    for s in range(SEEDS):
        keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
        ds = lgb.Dataset(X[keep], y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32), feature_name=feats, categorical_feature="auto")
        m = lgb.train({**params, "seed": s}, ds, ROUNDS)
        P += m.predict(V.select(feats).to_numpy()).astype(np.float32) / SEEDS
        g = dict(zip(feats, m.feature_importance("gain"))); print(f"  seed {s}: gain share of qr_vpa_mismatch {g['qr_vpa_mismatch'] / sum(g.values()):.4f} ({time.time()-t:.0f}s)", flush=True)
    np.save(f"scratch_gap/P{BM}_m50_{tag}.npy", P)
    np.save(f"scratch_gap/P{BM}_m50b35_{tag}.npy", (0.65 * np.load(f"scratch_gap/P{BM}_C_{tag}.npy") + 0.35 * P).astype(np.float32))
    print(f"  written P{BM}_m50_{tag}.npy, P{BM}_m50b35_{tag}.npy", flush=True)
