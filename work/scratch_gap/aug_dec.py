"""August selection score AFTER the decoder, for benchmark-D sources (a model's gain can vanish under the overrides).
August rows are split into two event-grouped halves (fraud events whole, legit rows at random). The per-class bias is tuned
on one half; the other half is scored after archetype forcing, device zones (edges from the August decode) and the bait
rule; then the halves swap. Reported: mean of the two halves, before and after the decoder, and each half's value.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/aug_dec.py D VAR1,VAR2,..."""
import sys
import numpy as np
import polars as pl
import train as T
from common import load_all, LABEL2ID as L
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES
from ts_judge_post import zones, override

c = pl.col; B, VARS = sys.argv[1], sys.argv[2].split(",")
Da = pl.read_parquet(f"scratch_gap/{B}_aug_rows.parquet"); ya = Da["y_true"].to_numpy().astype(np.int64); n = len(ya)
tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
ev = Da.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
rng = np.random.default_rng(2026); evh = rng.integers(0, 2, int(np.nanmax(ev)) + 1)
half = np.where(np.isnan(ev), rng.integers(0, 2, n), evh[np.nan_to_num(ev).astype(np.int64)])
rule = Da["ts_bait__rule"].fill_null(0).to_numpy()


def force(P):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = Da.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


print(f"benchmark {B}, August rows {n}, halves {int((half == 0).sum())} / {int((half == 1).sum())}")
for v in VARS:
    P = np.load(f"scratch_gap/P{B}_{v}_aug.npy"); F = force(P); pre, post = [], []
    for a, b in ((0, 1), (1, 0)):
        A_, B_ = half == a, half == b
        pre.append(T.macro_f1(ya[B_], T.decode(P[B_], T.tune_bias(P[A_], ya[A_]))).mean())
        p1 = T.decode(F, T.tune_bias(F[A_], ya[A_])); z, _ = zones(Da, p1, TR_EDGES[0], TR_EDGES[-1]); q = override(p1, z, rule)
        post.append(T.macro_f1(ya[B_], q[B_]).mean())
    print(f"{v:10s} August before decoder {np.mean(pre):.4f} ({pre[0]:.4f}, {pre[1]:.4f}) | after decoder {np.mean(post):.4f} ({post[0]:.4f}, {post[1]:.4f})", flush=True)
