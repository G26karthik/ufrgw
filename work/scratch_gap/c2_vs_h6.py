"""h6 vs c2s6 on September (benchmark D) with identical downstream rules: common decoder, then the four generic later
layers (rules_sep.py union). Paired event bootstrap of the difference, changed rows, corrections and new errors.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/c2_vs_h6.py   (after rules_sep.py D _ens h6 and rules_sep.py D _ens c2s6)"""
import numpy as np
import polars as pl
import train as T
from common import load_all
from clean_labels import fraud_events

c = pl.col; K = T.K
z = np.load("scratch_gap/dec_D_ens.npz"); y = z["y"]
rows = pl.read_parquet("scratch_gap/D_sep_rows.parquet").select("txn_id")
tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
ev = rows.join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026); n = len(y)
boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]


def f1w(p, w=None):
    w = np.ones(n) if w is None else w; ok = p == y
    return 2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)


for stage, a, b in (("before decoder", z["h6_pre"], z["c2s6_pre"]), ("after decoder", z["h6_q"], z["c2s6_q"]),
                    ("after decoder + generic later layers", np.load("scratch_gap/rules_union_D_h6.npy"), np.load("scratch_gap/rules_union_D_c2s6.npy"))):
    d = [f1w(b, w).mean() - f1w(a, w).mean() for w in boot]; ch = a != b; fa, fb = f1w(a), f1w(b)
    print(f"{stage:38s} h6 {fa.mean():.4f} c2s6 {fb.mean():.4f} ({fb.mean() - fa.mean():+.4f}, boot {np.quantile(d, .05):+.4f}..{np.quantile(d, .95):+.4f}) | rows differing {int(ch.sum())}:"
          f" c2s6 right / h6 wrong {int((ch & (b == y)).sum())}, h6 right / c2s6 wrong {int((ch & (a == y)).sum())}, both wrong {int((ch & (a != y) & (b != y)).sum())}")
    print("      per-class F1 change: " + " ".join(f"{T.LABELS[k][:5]} {fb[k] - fa[k]:+.3f}" for k in range(1, K)))
