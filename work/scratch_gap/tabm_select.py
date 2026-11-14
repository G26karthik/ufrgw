"""Development-only selection for TabM (benchmark N): config, then blend weight with LightGBM, both by August macro-F1
with a per-class bias tuned on August. September labels are not read here.
Writes PN_TABM_<aug|sep>.npy (chosen config, seeds averaged) and PN_BLEND_<aug|sep>.npy (w * TabM + (1 - w) * LightGBM).
usage (cwd=work, PYTHONPATH=.): python scratch_gap/tabm_select.py SEED[,SEED]"""
import sys
import numpy as np
import polars as pl
import train as T

seeds = sys.argv[1].split(",")
ya = pl.read_parquet("scratch_gap/N_aug_rows.parquet")["y_true"].to_numpy().astype(np.int64)
wa = np.where(ya == 0, 1.0, 1.0)


def load(tag, k):
    return np.mean([np.load(f"scratch_gap/PN_TABM{tag}{s}_{k}.npy") for s in seeds], 0)


def dev(P):
    return float(T.macro_f1(ya, T.decode(P, T.tune_bias(P, ya))).mean()), float(-np.log(P[np.arange(len(ya)), ya] + 1e-12).mean())


L = {k: np.load(f"scratch_gap/PN_C_{k}.npy") for k in ("aug", "sep")}
res = {"LightGBM": dev(L["aug"])}
for cfg in ("plain", "ple"):
    res[cfg] = dev(load(cfg, "aug"))
for k, (f, ll) in res.items():
    print(f"August {k:9s} macro-F1 (August bias) {f:.4f}  log-loss {ll:.5f}")
best = max(("plain", "ple"), key=lambda k: res[k][0]); print("config chosen on August:", best)
Tm = {k: load(best, k) for k in ("aug", "sep")}
ws = {}
for w in (0.2, 0.35, 0.5, 0.65, 0.8):
    ws[w] = dev(w * Tm["aug"] + (1 - w) * L["aug"]); print(f"August blend w={w}: macro-F1 {ws[w][0]:.4f}  log-loss {ws[w][1]:.5f}")
w = max(ws, key=lambda k: ws[k][0]); print("blend weight chosen on August:", w)
for k in ("aug", "sep"):
    np.save(f"scratch_gap/PN_TABM_{k}.npy", Tm[k]); np.save(f"scratch_gap/PN_BLEND_{k}.npy", (w * Tm[k] + (1 - w) * L[k]).astype(np.float32))
