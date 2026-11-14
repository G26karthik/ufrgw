"""Controlled arms for the bait-family generator. Same fit rows, legit downsample, target (relabel C) and LightGBM recipe
as train.py (h6). fold0: fit Jul -> predict Aug (development); fold1: fit Jul+Aug -> predict Sep (evaluation).
Generated events enter a fold only if their source month is a fit month. cwd=work, PYTHONPATH=.
usage: python scratch_gen/arms.py ARM [ARM ...]   -> scratch_gen/P_<arm>_<aug|sep>.npy (seed-averaged probabilities)
arms: A (h6 recipe) | B05, B1 (source-event rows weighted x1.5, x2) | BR (min_data_in_leaf 50, lambda_l2 2.5)
      | C05, C1 (plus 0.5x / 1x generated events)"""
import sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, polars as pl, lightgbm as lgb
import train as T
SEEDS, ROUNDS, LF, THREADS = 2, 150, 0.2, 8
META = ("txn_id", "month", "y", "is_test", "y_fit")
c = pl.col

def main():
    with ProcessPoolExecutor(1) as ex: yf = ex.submit(T.fit_target, "C").result()
    D = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet").join(yf, on="txn_id", how="left", maintain_order="left")
    feats = [k for k in D.columns if k not in META]
    G = pl.read_parquet("scratch_gen/gen_v2.parquet").with_columns(c("y").alias("y_fit"))
    src = pl.read_parquet("scratch_gen/src_rows.parquet")
    is_src = D["txn_id"].is_in(src["txn_id"].implode()).to_numpy()
    half_ev = np.random.default_rng(123).random(int(G["gen_ev"].max()) + 1) < 0.5  # fixed 0.5x subset of generated events
    ids = {vm: D.filter((c("month") == vm) & ~c("is_test")).select("txn_id", "y") for vm in (8, 9)}
    for vm, d in ids.items(): d.write_parquet(f"scratch_gen/ids_{'aug' if vm == 8 else 'sep'}.parquet")
    for arm in sys.argv[1:]:
        params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=THREADS, verbose=-1)
        if arm == "BR": params.update(min_data_in_leaf=50, lambda_l2=2.5)
        g = G if arm == "C1" else G.filter(pl.Series(half_ev[G["gen_ev"].to_numpy()])) if arm == "C05" else G.head(0)
        F = pl.concat([D.select(list(META) + feats), g.select(list(META) + feats)])  # generated rows last: legit draws unchanged
        extra = np.ones(F.height, np.float32)
        if arm in ("B05", "B1"): extra[:D.height][is_src] = 1.5 if arm == "B05" else 2.0
        for fm, vm in T.FOLDS:
            V = D.filter((c("month") == vm) & ~c("is_test")); P = np.zeros((V.height, T.K), np.float32)
            sel = F.select(c("month").is_in(fm) & ~c("is_test")).to_series().to_numpy()
            y = F["y_fit"].to_numpy()[sel].astype(np.int32)
            for s in range(SEEDS):
                t = time.time()
                keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
                full = np.zeros(F.height, bool); full[np.flatnonzero(sel)[keep]] = True
                X = F.select(feats).filter(pl.Series(full)).to_numpy(); yy = y[keep]
                w = np.where(yy == 0, 1.0 / LF, 1.0).astype(np.float32) * extra[full]
                m = lgb.train({**params, "seed": s}, lgb.Dataset(X, yy, weight=w), ROUNDS)
                P += T.predict_chunked(T.lgb_pred, m, V, feats) / SEEDS
                print(f"{arm} fit{fm}->val{vm} seed{s}: rows {len(yy)} (generated {int(full[D.height:].sum())}, src weight {extra[full].max():.1f}) {time.time()-t:.0f}s", flush=True)
            np.save(f"scratch_gen/P_{arm}_{'aug' if vm == 8 else 'sep'}.npy", P)
            f = T.macro_f1(V["y"].to_numpy().astype(np.int64), P.argmax(1))
            print(f"{arm} val{vm}: macro argmax {f.mean():.4f} phishing {f[6]:.3f} fake_cashback {f[9]:.3f}", flush=True)

if __name__ == "__main__":
    main()
