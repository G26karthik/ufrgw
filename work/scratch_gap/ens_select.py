"""Ensemble of saved historical models + TabM on benchmark D. Everything is chosen on August; September is not read.
Pool (class probabilities aligned by txn_id, scratch_gap/d_data.py): h6 (production LightGBM), c2s6 (core2 features),
ls_halves (other density split), aug3btta (density-view augmentation + TTA), xgbC (XGBoost), gpu_cat (CatBoost, raw labels),
TABM (tabm 0.0.3 ple, seeds 0+1, trained on the benchmark-D matrix).
Predefined combinations (arithmetic mean of probabilities):
  E_LGB = h6 + c2s6 + ls_halves + aug3btta;  E_GBDT = E_LGB + xgbC + gpu_cat;  E_ALL = E_GBDT + TABM;
  E_H6TABM = 0.65 h6 + 0.35 TABM (weight frozen from the benchmark-N selection);
  E_GREEDY = forward selection with replacement on August log-loss, at most 10 picks, started from the best single model.
The combination with the best August macro-F1 (August-tuned bias) is saved as COMB.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/ens_select.py -> scratch_gap/PD_<name>_<aug|sep>.npy for every combination"""
import numpy as np
import polars as pl
import train as T

POOL = ["h6", "c2s6", "ls_halves", "aug3btta", "xgbC", "gpu_cat", "TABM"]
ya = pl.read_parquet("scratch_gap/D_aug_rows.parquet")["y_true"].to_numpy().astype(np.int64)
for k in ("aug", "sep"):
    np.save(f"scratch_gap/PD_TABM_{k}.npy", np.mean([np.load(f"scratch_gap/PD_TABMple{s}_{k}.npy") for s in (0, 1)], 0))
A = {m: np.load(f"scratch_gap/PD_{m}_aug.npy").astype(np.float64) for m in POOL}
ll = lambda P: float(-np.log(P[np.arange(len(ya)), ya] + 1e-12).mean())
mf = lambda P: float(T.macro_f1(ya, T.decode(P, T.tune_bias(P, ya))).mean())
print("August, single models:")
for m in POOL:
    print(f"  {m:10s} macro-F1 {mf(A[m]):.4f}  log-loss {ll(A[m]):.5f}", flush=True)
C = np.corrcoef([(A[m].argmax(1) != A["h6"].argmax(1)).astype(float) + 0 for m in POOL[1:]])
print("  rows where argmax differs from h6:", {m: int((A[m].argmax(1) != A["h6"].argmax(1)).sum()) for m in POOL[1:]})
W = {"E_LGB": {m: 1 for m in POOL[:4]}, "E_GBDT": {m: 1 for m in POOL[:6]}, "E_ALL": {m: 1 for m in POOL}, "E_H6TABM": {"h6": 0.65, "TABM": 0.35}}
picks = [min(POOL, key=lambda m: ll(A[m]))]; cur = A[picks[0]].copy(); best = ll(cur)
for _ in range(9):
    cand = {m: ll((cur * len(picks) + A[m]) / (len(picks) + 1)) for m in POOL}; m = min(cand, key=cand.get)
    if cand[m] >= best - 1e-6: break
    picks.append(m); cur = (cur * (len(picks) - 1) + A[m]) / len(picks); best = cand[m]
W["E_GREEDY"] = {m: picks.count(m) for m in set(picks)}; print("greedy picks (August log-loss):", picks)
res = {}
for name, w in W.items():
    s = sum(w.values()); P = sum(A[m] * v for m, v in w.items()) / s; res[name] = mf(P)
    print(f"  {name:9s} weights {({m: round(v / s, 3) for m, v in w.items()})} macro-F1 {res[name]:.4f}  log-loss {ll(P):.5f}", flush=True)
    for k in ("aug", "sep"):
        np.save(f"scratch_gap/PD_{name}_{k}.npy", (sum(np.load(f"scratch_gap/PD_{m}_{k}.npy") * v for m, v in w.items()) / s).astype(np.float32))
comb = max(res, key=res.get); print("combination chosen on August:", comb)
for k in ("aug", "sep"):
    np.save(f"scratch_gap/PD_COMB_{k}.npy", np.load(f"scratch_gap/PD_{comb}_{k}.npy"))
