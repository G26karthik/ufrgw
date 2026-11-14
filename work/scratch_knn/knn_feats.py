"""Experiment B: class-conditional neighbour features on benchmark N (design declared in PROGRESS.md before the run).
Per world (fold0: fit Jul, held Aug; fold1: fit Jul+Aug, held Sep), using y_fit of fit rows only:
  space : CONT columns -> QuantileTransformer(normal) fitted on fit rows, clipped to +-3, missing -> 0; hour as sin/cos;
          BIN columns raw; OH columns one-hot / sqrt(2) (a category mismatch costs 1 unit, null is its own level).
  banks : fraud class = all fit rows of the class; legit = random sample of LEGIT_FRAC of the fit legit rows.
  cross-fit: fit rows are split in NF folds (fraud events whole, events from fit-month labels only, other rows random);
          a fit row in fold j queries banks built from the other folds (legit bank resampled to the same absolute size),
          so it never meets itself or its own event. Held rows query the full fit bank.
  output: distance to the 1st, 5th, 15th neighbour in each of the 12 class banks + ratios -> scratch_knn/KF_f<fold>.parquet
Fit legit rows that neither training seed keeps (64%) are not queried (NaN, never read by the trainer).
usage (cwd=work, PYTHONPATH=.): python scratch_knn/knn_feats.py time | python scratch_knn/knn_feats.py FOLD"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "8"  # the machine is shared
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
import train as T
from common import load_all
from clean_labels import fraud_events

c = pl.col; KS = (1, 5, 15); NF = 10; LEGIT_FRAC = 0.2; SEEDS, LF = 2, 0.2
CONT = ["log_amount", "cents", "sim_age_hours", "device_age_days", "login_to_txn_sec", "balance_before", "amt_share_bal",
        "burst__dev_c300", "burst__snd_c300", "burst__snd_ref7d", "flow__rin_n12h", "flow__sin_n24h", "flow__rout_n6h",
        "takeover__dev_frac", "takeover__n_other_dev", "scam__rcv_in_15d", "ring__rcv_n"]
BIN = ["round100", "round1000", "new_payee", "pin_retries", "away_from_home", "is_p2p"]
OH = ["txn_type", "receiver_type", "app", "qr_vpa_mismatch", "mer_same_city"]
NAMES = [f"kn_c{k}_k{j}" for k in range(T.K) for j in KS] + [f"kn_rmin_k{j}" for j in KS] + [f"kn_r5_c{k}" for k in range(1, T.K)]


def space(D, fit):
    num = lambda cols: D.select([c(k).fill_nan(None) for k in cols]).to_numpy().astype(np.float64)  # null and NaN -> NaN
    x = num(CONT); print("missing share of CONT columns:", {k: round(float(v), 3) for k, v in zip(CONT, np.isnan(x).mean(0)) if v > 0}, flush=True)
    z = np.clip(QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=0).fit(x[fit]).transform(x), -3, 3)
    h = D["hour"].to_numpy() * (2 * np.pi / 24); b = num(BIN); assert not np.isnan(b).any()
    oh = [(v[:, None] == np.unique(v)[None]) / np.sqrt(2) for v in (np.nan_to_num(num([k])[:, 0], nan=-1) for k in OH)]
    return np.ascontiguousarray(np.hstack([np.nan_to_num(z), np.sin(h)[:, None], np.cos(h)[:, None], b, *oh]), dtype=np.float32)


def kd(bank, Q):
    d, _ = NearestNeighbors(n_neighbors=max(KS), algorithm="brute").fit(bank).kneighbors(Q)
    return d[:, [k - 1 for k in KS]].astype(np.float32)


def main(fi):
    fm, vm = T.FOLDS[fi]; t0 = time.time(); rng = np.random.default_rng(100 + fi)
    D = pl.read_parquet(f"scratch_gap/N_f{fi}.parquet"); n = D.height
    fit = ~D["is_test"].to_numpy(); held = D.select(c("is_test") & (c("month") == vm)).to_series().to_numpy()
    y = np.full(n, -1); y[fit] = D.filter(~c("is_test"))["y_fit"].to_numpy(); yf = y[fit]
    need = np.zeros(n, bool); need[fit] = (yf != 0) | np.any([np.random.default_rng(s).random(len(yf)) < LF for s in range(SEEDS)], axis=0)  # rows a training seed keeps
    _, FE = fraud_events(load_all().filter(~c("is_test") & c("month").is_in(fm)), g=48 * 3600)  # events from fit-month labels only
    ev = D.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
    assert not np.isfinite(ev[~fit]).any()
    fold = np.where(np.isnan(ev), rng.integers(0, NF, n), rng.integers(0, NF, int(np.nanmax(ev)) + 1)[np.nan_to_num(ev).astype(np.int64)]); fold[~fit] = -1
    print(f"fold{fi}: rows {n}, fit {int(fit.sum())}, held scored {int(held.sum())}, fit rows queried {int(need.sum())}, fraud fit rows without an event id {int(((y > 0) & np.isnan(ev)).sum())}", flush=True)
    X = space(D, fit); nL = int(round(LEGIT_FRAC * (y == 0).sum())); print(f"space {X.shape}, legit bank size {nL} ({time.time() - t0:.0f}s)", flush=True)
    F = np.full((n, T.K, len(KS)), np.nan, np.float32); sizes = np.zeros((NF + 1, T.K), int)
    for j in list(range(NF)) + [-1]:  # -1 = held rows against the full fit bank
        q = held if j < 0 else need & (fold == j)
        for k in range(T.K):
            bank = np.flatnonzero((y == k) & (fold != j))
            if k == 0: bank = rng.choice(bank, nL, replace=False)
            assert not q[bank].any(); sizes[j, k] = len(bank)
            F[q, k] = kd(X[bank], X[q])
        print(f"  {'held' if j < 0 else 'fold ' + str(j)}: queries {int(q.sum())} ({time.time() - t0:.0f}s)", flush=True)
    print("bank sizes per class (legit, then fraud classes in T.LABELS order):\n  cross-fit min", sizes[:NF].min(0).tolist(), "\n  cross-fit max", sizes[:NF].max(0).tolist(), "\n  held (full)  ", sizes[-1].tolist())
    lf = need & (y == 0)
    print("density check, median distance to the 5th neighbour, fit legit rows (cross-fit) vs held rows (full bank):")
    print("  " + " ".join(f"c{k} {np.median(F[lf, k, 1]):.3f}/{np.median(F[held, k, 1]):.3f}" for k in range(T.K)))
    print("own-class check, fit rows of class k: share with distance 0 to the 1st own-class neighbour:", " ".join(f"c{k} {(F[need & (y == k), k, 0] == 0).mean():.3f}" for k in range(T.K)))
    e = 1e-3; G = np.hstack([F.reshape(n, -1), F[:, 1:].min(1) / (F[:, 0] + e), F[:, 1:, 1] / (F[:, :1, 1] + e)])
    assert G.shape[1] == len(NAMES) and not np.isnan(G[need | held]).any()
    pl.DataFrame({"txn_id": D["txn_id"], **{k: G[:, i] for i, k in enumerate(NAMES)}}).write_parquet(f"scratch_knn/KF_f{fi}.parquet")
    print(f"fold{fi} written, {len(NAMES)} columns ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "time":
        r = np.random.default_rng(0); B, Q = r.normal(size=(100_000, 50)).astype(np.float32), r.normal(size=(20_000, 50)).astype(np.float32)
        t = time.time(); kd(B, Q); print(f"20k queries x 100k bank x 50 dims: {time.time() - t:.1f}s")
    else: main(int(sys.argv[1]))
