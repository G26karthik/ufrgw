"""Density stress test D (label-informed): the production matrix feats_cache/matrix_m5dm2_pairsfd.parquet, whose
neighbourhood features are built on density.halves (uses the labels of every train month, including the scored one, to
keep fraud events whole). Same four fit targets as bench_nat.py via train.fit_target (relabel on all train rows).
usage (cwd=work, PYTHONPATH=.): python scratch_gap/bench_den.py -> scratch_gap/PD_<variant>_<aug|sep>.npy, D_<aug|sep>_rows.parquet"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np, polars as pl, lightgbm as lgb
import train as T
from common import load_all, run_feature
from density import run_feature_matched
c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2; VARIANTS = {"RAW": "", "C": "C", "AC": "AC", "CE": "CE"}


def main():
    M = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet"); feats = [k for k in M.columns if k not in ("txn_id", "month", "y", "is_test")]
    tx = load_all()
    D = tx.select("txn_id", "month", "txn_type", "receiver_type", "qr_vpa_mismatch", c("y").alias("y_true"), c("device_id").str.slice(2).cast(pl.Int64).alias("dn")) \
          .join(run_feature_matched("takeover", tx).select("txn_id", "takeover__arch_sim", "takeover__arch_ato"), on="txn_id", how="left", maintain_order="left") \
          .join(run_feature("ts_bait", tx), on="txn_id", how="left", maintain_order="left")
    del tx
    for v, code in VARIANTS.items():
        if code:
            with ProcessPoolExecutor(1) as ex: yf = ex.submit(T.fit_target, code).result()
            F = M.join(yf, on="txn_id", how="left", maintain_order="left")
        else:
            F = M.with_columns(c("y").alias("y_fit"))
        for fm, vm in T.FOLDS:
            V = M.filter((c("month") == vm) & ~c("is_test")); P = np.zeros((V.height, T.K), np.float32)
            sel = F.select(c("month").is_in(fm) & ~c("is_test")).to_series().to_numpy(); y = F["y_fit"].to_numpy()[sel].astype(np.int32)
            for s in range(SEEDS):
                keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
                full = np.zeros(F.height, bool); full[np.flatnonzero(sel)[keep]] = True
                m = lgb.train(dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=10, verbose=-1, seed=s),
                              lgb.Dataset(F.select(feats).filter(pl.Series(full)).to_numpy(), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32)), ROUNDS)
                P += T.predict_chunked(T.lgb_pred, m, V, feats) / SEEDS
            k = "aug" if vm == 8 else "sep"; np.save(f"scratch_gap/PD_{v}_{k}.npy", P)
            V.select("txn_id").join(D, on="txn_id", how="left", maintain_order="left").write_parquet(f"scratch_gap/D_{k}_rows.parquet")
            print(v, vm, "macro argmax", round(float(T.macro_f1(V["y"].to_numpy().astype(np.int64), P.argmax(1)).mean()), 4), flush=True)


if __name__ == "__main__":
    main()
