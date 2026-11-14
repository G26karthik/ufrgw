"""Experiment B diagnostics.
aug : August only. Among rows the baseline C calls legit (bias tuned on August), does a neighbour column separate
      fraud-labelled rows (missed fraud) from legit rows? AUC per column family, and of the kA / C fraud probability.
sep : after the freeze. From scratch_gap/dec_N_knn.npz: fraud rows C calls legit that a variant types correctly (recovered),
      or calls some fraud class (flagged), against rows C had right that the variant gets wrong; before and after the decoder.
usage (cwd=work, PYTHONPATH=.): python scratch_knn/recov.py aug|sep"""
import sys
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import train as T

if sys.argv[1] == "aug":
    R = pl.read_parquet("scratch_gap/N_aug_rows.parquet"); y = R["y_true"].to_numpy().astype(np.int64)
    KF = R.select("txn_id").join(pl.read_parquet("scratch_knn/KF_f0.parquet"), on="txn_id", how="left", maintain_order="left")
    C = np.load("scratch_gap/PN_C_aug.npy"); pc = T.decode(C, T.tune_bias(C, y)); m = pc == 0; f = (y > 0)[m]
    print(f"August: rows {len(y)}, fraud-labelled {int((y > 0).sum())}, C calls legit {int(m.sum())}, of which fraud-labelled (missed) {int(f.sum())}")
    auc = lambda v: roc_auc_score(f, v[m])
    print("AUC missed fraud vs legit among C-legit rows (higher = more fraud-like):")
    print("  -kn_rmin_k1/k5/k15 (nearest fraud bank / legit bank):", " ".join(f"{auc(-KF[f'kn_rmin_k{k}'].to_numpy()):.3f}" for k in (1, 5, 15)))
    print("  kn_c0_k1/k5/k15 (distance to the legit bank):", " ".join(f"{auc(KF[f'kn_c0_k{k}'].to_numpy()):.3f}" for k in (1, 5, 15)))
    print("  -kn_r5_c<class>, that class's missed rows vs legit:", " ".join(f"{T.LABELS[k][:5]} {roc_auc_score((y == k)[m][(y[m] == 0) | (y[m] == k)], -KF[f'kn_r5_c{k}'].to_numpy()[m][(y[m] == 0) | (y[m] == k)]):.3f}" for k in range(1, T.K)))
    print(f"  1 - P(legit): C {auc(1 - C[:, 0]):.3f}", end="")
    try: print(f" | kA {auc(1 - np.load('scratch_gap/PN_kA_aug.npy')[:, 0]):.3f}")
    except FileNotFoundError: print(" | kA not built yet")
else:
    z = np.load("scratch_gap/dec_N_knn.npz"); y = z["y"]; V = [k[:-2] for k in z.files if k.endswith("_q")]
    for st, nm in (("pre", "before decoder"), ("q", "after decoder")):
        a = z[f"{V[0]}_{st}"]; miss = (y > 0) & (a == 0)
        print(f"--- {nm}: fraud-labelled rows {int((y > 0).sum())}, {V[0]} calls legit {int(miss.sum())}")
        for v in V[1:]:
            b = z[f"{v}_{st}"]; rec = miss & (b == y); lost = (y > 0) & (a == y) & (b != y); fp = (y == 0) & (a == 0) & (b > 0); fpfix = (y == 0) & (a > 0) & (b == 0)
            print(f"{v:5s} missed fraud now typed right {int(rec.sum())} | missed fraud now flagged with a wrong type {int((miss & (b > 0) & (b != y)).sum())}"
                  f" | fraud rows C had right, now wrong {int(lost.sum())} (now legit {int((lost & (b == 0)).sum())}) | new legit rows called fraud {int(fp.sum())} | legit rows no longer called fraud {int(fpfix.sum())}")
            print("      recovered by class: " + " ".join(f"{T.LABELS[k][:5]} {int((rec & (y == k)).sum())}" for k in range(1, T.K)))
            print("      lost by class:      " + " ".join(f"{T.LABELS[k][:5]} {int((lost & (y == k)).sum())}" for k in range(1, T.K)))
