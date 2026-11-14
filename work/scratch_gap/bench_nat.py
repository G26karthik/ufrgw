"""Natural temporal benchmark N: held-out labels hidden, no later observations, no label-based density split.
World(fit months, held month): rows of those months only (real test rows and later months absent); held rows get
is_test=True and null labels. Features: quick_eval.base_features + flow, burst, takeover, scam, ring, pairs built once on
the world at full density (density.halves is NOT used, so no label reaches a feature), + ts_bait for the decoder.
fold0: fit Jul, held Aug (calibration).  fold1: fit Jul+Aug, held Sep (+ the 147 Oct-dated train rows, hidden, unscored).
Fit targets are computed on the fit rows only, from the world's own features (relabel.run_feature is pointed at them):
  RAW supplied labels | C takeover archetype type (current recipe) | AC event-majority cleaning + C | CE C + drop fraud
  labels without structural support (model fit on structured rows only).
Check: features rebuilt with the held labels permuted instead of hidden must be identical.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/bench_nat.py   -> scratch_gap/N_<fold>_*.parquet, PN_<variant>_<aug|sep>.npy"""
import importlib, time
import numpy as np, polars as pl, lightgbm as lgb
import train as T, relabel as R
from common import load_all
from quick_eval import base_features
c = pl.col; MODS = ["flow", "burst", "takeover", "scam", "ring"]; VARIANTS = {"RAW": "", "C": "C", "AC": "AC", "CE": "CE"}
SEEDS, ROUNDS, LF = 2, 150, 0.2

def world(tx, fm, vm, permute=False):
    w = tx.filter(~c("is_test") & (c("month").is_in(fm + [vm]) | pl.lit(vm == 9) & (c("month") == 10)))
    held = ~w["month"].is_in(fm)
    if permute:  # held labels shuffled among the held rows instead of hidden
        idx = np.flatnonzero(held.to_numpy()); sh = np.random.default_rng(1).permutation(idx)
        y, lab = w["y"].to_numpy().copy(), w["label"].to_list(); y[idx] = y[sh]
        for i, j in zip(idx, sh): lab[i] = w["label"][int(j)]
        return w.with_columns(pl.Series("y", y, pl.Int8), pl.Series("label", lab), held.alias("is_test"))
    return w.with_columns(pl.when(held).then(None).otherwise(c("y")).alias("y"), pl.when(held).then(None).otherwise(c("label")).alias("label"), held.alias("is_test"))

def build(w):
    F = {m: importlib.import_module(f"feats.{m}").build(w) for m in MODS + ["pairs", "ts_bait"]}
    X = base_features(w)
    for m in MODS + ["pairs"]: X = X.join(F[m], on="txn_id", how="left", maintain_order="left")
    feats = [k for k in X.columns if k != "txn_id"]
    return X.with_columns([c(k).cast(pl.Float32) for k in feats]), F, feats

def main():
    tx = load_all(); truth = tx.select("txn_id", c("y").alias("y_true"))
    for fi, (fm, vm) in enumerate(T.FOLDS):
        t = time.time(); w = world(tx, fm, vm); X, F, feats = build(w)
        assert len(feats) == 190 and w.filter(c("is_test"))["y"].null_count() == w.filter(c("is_test")).height
        if fi == 1:
            X2, _, _ = build(world(tx, fm, vm, permute=True))
            assert X.equals(X2), "features changed when held-out labels were permuted"
            print("held-label permutation check: features identical (190 columns,", X.height, "rows)", flush=True)
        D = w.select("txn_id", "month", "y", "is_test", "txn_type", "receiver_type", "qr_vpa_mismatch", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")) \
             .join(F["takeover"].select("txn_id", "takeover__arch_sim", "takeover__arch_ato"), on="txn_id", how="left", maintain_order="left") \
             .join(F["ts_bait"], on="txn_id", how="left", maintain_order="left").join(truth, on="txn_id", how="left", maintain_order="left")
        fit = w.filter(~c("is_test")); R.run_feature = lambda m: F[m]
        yf = {v: R.relabel(fit, code) if code else fit["y"] for v, code in VARIANTS.items()}
        Xf = X.join(fit.select("txn_id"), on="txn_id", how="semi", maintain_order="left"); assert Xf["txn_id"].equals(fit["txn_id"])
        V = X.join(w.filter(c("is_test") & (c("month") == vm)).select("txn_id"), on="txn_id", how="semi", maintain_order="left")
        D.join(V.select("txn_id"), on="txn_id", how="semi", maintain_order="left").write_parquet(f"scratch_gap/N_{'aug' if vm == 8 else 'sep'}_rows.parquet")
        print(f"fold{fi} world rows {w.height} fit {fit.height} held scored {V.height} ({time.time()-t:.0f}s)", flush=True)
        for v, ys in yf.items():
            y = ys.to_numpy().astype(np.int32); P = np.zeros((V.height, T.K), np.float32)
            print(f"  {v}: fit target differs from supplied label on {int((y != fit['y'].to_numpy()).sum())} rows, dropped {int((y < 0).sum())}", flush=True)
            for s in range(SEEDS):
                keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
                m = lgb.train(dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=10, verbose=-1, seed=s),
                              lgb.Dataset(Xf.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32)), ROUNDS)
                P += T.predict_chunked(T.lgb_pred, m, V, feats) / SEEDS
            np.save(f"scratch_gap/PN_{v}_{'aug' if vm == 8 else 'sep'}.npy", P)
            print(f"  {v} held {vm}: macro argmax {T.macro_f1(D.join(V.select('txn_id'), on='txn_id', how='semi', maintain_order='left')['y_true'].to_numpy().astype(np.int64), P.argmax(1)).mean():.4f} ({time.time()-t:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
