"""A1: counterfactual stamp masking on the held month of benchmark N.
Held-month stamped non-QR rows (txn_type != QR and qr_vpa_mismatch == 0) get qr_vpa_mismatch = null (the ordinary non-QR
value) in the raw world; the whole world is rebuilt with bench_nat.build and compared column by column with N_f<fold>.
Not masked: fit rows, the Oct-dated hidden rows of fold 1, non-QR rows with mismatch == 1, every QR row.
Reference recipe C (numrep.run arm A_C, 4 threads) is fitted once per fold and seed on the unmasked fit rows; the held
month is predicted from the original and from the masked features with the same boosters.
out (scratch_stamp/): P_orig_<aug|sep>.npy, P_mask_<aug|sep>.npy, mask_<aug|sep>.npy (bool, held-row order), model_f<fold>_s<seed>.txt
usage (cwd=work, PYTHONPATH=.): python scratch_stamp/a1.py"""
import sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T
from common import load_all

sys.path.insert(0, "scratch_gap")
from bench_nat import world, build  # noqa: E402

c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2; META = ("txn_id", "month", "is_test", "y_fit")
tx = load_all()
for fi, (fm, vm) in enumerate(T.FOLDS):
    t = time.time(); tag = "aug" if vm == 8 else "sep"
    D = pl.read_parquet(f"scratch_gap/N_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]
    w = world(tx, fm, vm); assert w["txn_id"].equals(D["txn_id"])
    msk = (c("is_test") & (c("month") == vm) & (c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
    wm = w.with_columns(pl.when(msk).then(None).otherwise(c("qr_vpa_mismatch")).alias("qr_vpa_mismatch"))
    mrow = w.select(msk).to_series().to_numpy()
    Xm, _, f2 = build(wm); assert f2 == feats and Xm["txn_id"].equals(D["txn_id"])
    diff = {k: int((~((a := D[k].to_numpy()) == (b := Xm[k].to_numpy())) & ~(np.isnan(a) & np.isnan(b))).sum()) for k in feats}
    changed = {k: n for k, n in diff.items() if n}
    a, b = D["qr_vpa_mismatch"].to_numpy(), Xm["qr_vpa_mismatch"].to_numpy()
    drow = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    print(f"fold{fi}: masked rows {int(mrow.sum())}; columns that differ after the rebuild: {changed}; "
          f"qr_vpa_mismatch differs exactly on the masked rows: {bool((drow == mrow).all())}; masked value now NaN: {bool(np.isnan(b[mrow]).all())}; "
          f"all other 189 columns identical on all rows: {set(changed) <= {'qr_vpa_mismatch'}} ({time.time()-t:.0f}s)", flush=True)
    held = (c("is_test") & (c("month") == vm))
    fit = D.filter(~c("is_test")); V = D.filter(held); hrow = D.select(held).to_series().to_numpy()
    Vm = Xm.filter(pl.Series(hrow)); assert Vm["txn_id"].equals(V["txn_id"])
    np.save(f"scratch_stamp/mask_{tag}.npy", mrow[hrow])
    y = fit["y_fit"].to_numpy().astype(np.int32)
    params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=4, verbose=-1)
    P = np.zeros((V.height, T.K), np.float32); Pm = np.zeros_like(P)
    for s in range(SEEDS):
        keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
        ds = lgb.Dataset(fit.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32),
                         feature_name=feats, categorical_feature="auto")
        m = lgb.train({**params, "seed": s}, ds, ROUNDS); m.save_model(f"scratch_stamp/model_f{fi}_s{s}.txt")
        P += m.predict(V.select(feats).to_numpy()).astype(np.float32) / SEEDS
        Pm += m.predict(Vm.select(feats).to_numpy()).astype(np.float32) / SEEDS
        g = dict(zip(feats, m.feature_importance("gain"))); tot = sum(g.values())
        print(f"  seed {s}: gain share of qr_vpa_mismatch {g['qr_vpa_mismatch']/tot:.4f} (rank {sorted(g.values(), reverse=True).index(g['qr_vpa_mismatch'])+1} of {len(feats)}),"
              f" scam__rcv_qr_mm1_frac_15d {g['scam__rcv_qr_mm1_frac_15d']/tot:.4f}, scam__rcv_qr_mmnull_15d {g['scam__rcv_qr_mmnull_15d']/tot:.4f}, scam__snd_qr_mmnull_15d {g['scam__snd_qr_mmnull_15d']/tot:.4f} ({time.time()-t:.0f}s)", flush=True)
    np.save(f"scratch_stamp/P_orig_{tag}.npy", P); np.save(f"scratch_stamp/P_mask_{tag}.npy", Pm)
    R = np.load(f"scratch_gap/PN_C_{tag}.npy")
    print(f"  self-check vs PN_C_{tag}.npy: max abs diff {np.abs(P - R).max():.2e}, argmax differs on {int((P.argmax(1) != R.argmax(1)).sum())} rows of {len(P)}", flush=True)
    print(f"  rows not masked with any change in P: {int((np.abs(P - Pm).max(1) > 0)[~mrow[hrow]].sum())}; masked rows with a change: {int((np.abs(P - Pm).max(1) > 0)[mrow[hrow]].sum())} of {int(mrow[hrow].sum())}", flush=True)
