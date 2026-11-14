"""Shared helpers for the 14 Nov combination pass on benchmark N (cwd=work, PYTHONPATH=.).
fold_data: fit/held matrices of scratch_gap/N_f<fold>.parquet, held rows checked against N_<aug|sep>_rows.parquet.
aug_ctx / aug_score: the one August selection score used for every candidate. The per-class bias is tuned on one
event-grouped half of August and the other half is scored, then swapped; the mean of the two halves is the score.
Archetype forcing, device zones and the bait rule read no label, so they run on all August rows (as bench_eval.py
does on September)."""
import os
import numpy as np
import polars as pl
import train as T
from common import LABEL2ID as L

c = pl.col; K = T.K; G = "scratch_gap"; HERE = os.path.dirname(os.path.abspath(__file__)); META = ("txn_id", "month", "is_test", "y_fit")
BM = os.environ.get("BM", "N")  # benchmark for fold_data: N (default) or D (replication runs only)


def fold_data(fi):
    """-> feats, fit frame (all fit rows), y_fit, held feature matrix (rows = N_<aug|sep>_rows order)."""
    D = pl.read_parquet(f"{G}/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
    kind, vm = ("aug", 8) if fi == 0 else ("sep", 9)
    fit = D.filter(~c("is_test")); held = D.filter(c("is_test") & (c("month") == vm))
    assert held["txn_id"].equals(pl.read_parquet(f"{G}/{BM}_{kind}_rows.parquet", columns=["txn_id"])["txn_id"]), "held rows not aligned"
    return feats, fit, fit["y_fit"].to_numpy().astype(np.int32), held.select(feats).to_numpy(), kind


def aug_ctx():
    """August rows, labels, event groups (fraud rows: clean_labels.fraud_events, 48 h; every legit row its own group)."""
    Da = pl.read_parquet(f"{G}/N_aug_rows.parquet"); ya = Da["y_true"].to_numpy().astype(np.int64); f = f"{HERE}/aug_groups.npy"
    if not os.path.exists(f):
        from common import load_all
        from clean_labels import fraud_events
        tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
        ev = Da.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
        assert not np.isnan(ev[ya > 0]).any()
        np.save(f, np.where(np.isnan(ev), -1 - np.arange(len(ev)), ev).astype(np.int64))  # legit rows: unique negative ids
    grp = np.load(f); assert len(grp) == len(ya)
    return Da, ya, grp


def split(grp, n, seed=0):
    """Event-grouped assignment of rows to n parts."""
    u, inv = np.unique(grp, return_inverse=True)
    return np.random.default_rng(seed).integers(0, n, len(u))[inv]


def force(P, D):  # same as bench_eval.force
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


def aug_score(Pa, Da, ya, half):
    """-> (after bias + forcing, after the full decoder, [full-decoder score of each half])."""
    from feats.ts_event import TR_EDGES
    from ts_judge_post import zones, override
    Pf = force(Pa, Da); rule = Da["ts_bait__rule"].fill_null(0).to_numpy(); a, b = [], []
    for h in (0, 1):
        p1 = T.decode(Pf, T.tune_bias(Pf[half == h], ya[half == h]))
        z, _ = zones(Da, p1, TR_EDGES[0], TR_EDGES[-1]); q = override(p1, z, rule); m = half != h
        a.append(T.macro_f1(ya[m], p1[m]).mean()); b.append(T.macro_f1(ya[m], q[m]).mean())
    return float(np.mean(a)), float(np.mean(b)), [round(float(x), 4) for x in b]


def load(name, kind):
    P = np.load(f"{G}/PN_{name}_{kind}.npy"); assert P.shape[1] == K and abs(P.sum(1) - 1).max() < 1e-3, name
    return P
