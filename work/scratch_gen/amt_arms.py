"""Mechanism 2 comparison on an Oct-like Sep (legit amounts x1.17, scratch_gen/amt_world.py). cwd=work, PYTHONPATH=.
A = h6 recipe fit on the real Jul / Jul+Aug rows.  M = same recipe fit on the amount-scaled Jul / Jul+Aug rows.
PASS RULE (fixed before reading the result): M passes if on Oct-like Sep, with supplied labels,
  M (bias tuned on Oct-like Aug) - A (bias tuned on real Aug) >= .002, M - A (bias retuned on Oct-like Aug) >= .001,
  and the event-bootstrap 90% interval of the first difference excludes 0."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np, polars as pl, lightgbm as lgb
import train as T
from clean_labels import fraud_events
from common import load_all
c = pl.col; K = T.K; SEEDS, ROUNDS, LF = 2, 150, 0.2
META = ("txn_id", "month", "y", "is_test", "y_fit")

def f1w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w; ok = p == y
    return 2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)

def main():
    with ProcessPoolExecutor(1) as ex: yf = ex.submit(T.fit_target, "C").result()
    W = {"real": pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet").join(yf, on="txn_id", how="left", maintain_order="left"),
         "oct": pl.read_parquet("scratch_gen/matrix_amt.parquet").join(yf, on="txn_id", how="left", maintain_order="left")}
    feats = [k for k in W["real"].columns if k not in META]
    P = {}
    for arm, fitw in (("A", "real"), ("M", "oct")):
        F = W[fitw]
        for fm, vm in T.FOLDS:
            sel = F.select(c("month").is_in(fm) & ~c("is_test")).to_series().to_numpy(); y = F["y_fit"].to_numpy()[sel].astype(np.int32)
            V = {k: W[k].filter((c("month") == vm) & ~c("is_test")) for k in W}
            for k in W: P[arm, k, vm] = np.zeros((V[k].height, K), np.float32)
            for s in range(SEEDS):
                keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
                full = np.zeros(F.height, bool); full[np.flatnonzero(sel)[keep]] = True
                m = lgb.train(dict(objective="multiclass", num_class=K, learning_rate=0.05, **T.LGB, num_threads=8, verbose=-1, seed=s),
                              lgb.Dataset(F.select(feats).filter(pl.Series(full)).to_numpy(), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32)), ROUNDS)
                for k in W: P[arm, k, vm] += T.predict_chunked(T.lgb_pred, m, V[k], feats) / SEEDS
                print(arm, fm, vm, "seed", s, flush=True)
            for k in W: np.save(f"scratch_gen/Pamt_{arm}_{k}_{vm}.npy", P[arm, k, vm])
    ya = W["real"].filter((c("month") == 8) & ~c("is_test"))["y"].to_numpy().astype(np.int64)
    S9 = W["real"].filter((c("month") == 9) & ~c("is_test")).select("txn_id", "y"); y = S9["y"].to_numpy().astype(np.int64)
    b = {(arm, k): T.tune_bias(P[arm, k, 8], ya) for arm in "AM" for k in W}
    dec = {"A real-fit, real Sep (reference)": T.decode(P["A", "real", 9], b["A", "real"]),
           "A real-fit, Oct-like Sep, bias from real Aug": T.decode(P["A", "oct", 9], b["A", "real"]),
           "A real-fit, Oct-like Sep, bias from Oct-like Aug": T.decode(P["A", "oct", 9], b["A", "oct"]),
           "M scaled-fit, Oct-like Sep, bias from Oct-like Aug": T.decode(P["M", "oct", 9], b["M", "oct"]),
           "M scaled-fit, real Sep, bias from Oct-like Aug": T.decode(P["M", "real", 9], b["M", "oct"])}
    for k, p in dec.items():
        f = f1w(y, p); print(f"{k:52s} macro {f.mean():.4f} legit_fp {int(((y == 0) & (p > 0)).sum()):5d} | " + " ".join(f"{T.LABELS[i][:5]}={f[i]:.3f}" for i in range(1, K)))
    tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
    ev = S9.join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
    grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026); n = len(y)
    boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]
    pub = [rng.random(n) < 0.3 for _ in range(200)]
    ks = list(dec); base = dec[ks[1]]
    for k in ks[2:4]:
        db = np.array([f1w(y, dec[k], w).mean() - f1w(y, base, w).mean() for w in boot])
        dq = np.array([f1w(y[~m], dec[k][~m]).mean() - f1w(y[~m], base[~m]).mean() for m in pub])
        dp = np.array([f1w(y[m], dec[k][m]).mean() - f1w(y[m], base[m]).mean() for m in pub])
        print(f"{k} minus '{ks[1]}': full {f1w(y, dec[k]).mean() - f1w(y, base).mean():+.4f} | event bootstrap 5%..95% {np.quantile(db, .05):+.4f}..{np.quantile(db, .95):+.4f}"
              f" | 30/70 splits: public {dp.mean():+.4f} sd {dp.std():.4f}, private {dq.mean():+.4f} sd {dq.std():.4f}, private gain in {(dq > 0).mean():.0%}")

if __name__ == "__main__":
    main()
