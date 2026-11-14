"""Does TabM (or the blend) recover different errors from LightGBM? September, benchmark N, supplied labels.
Reads scratch_gap/dec_N_tabm.npz (bench_eval.py N C,TABM,BLEND _tabm). `pre` = before the decoder, `q` = after it.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/tabm_diff.py"""
import numpy as np
import train as T

z = np.load("scratch_gap/dec_N_tabm.npz"); y = z["y"]
for stage in ("pre", "q"):
    a = z[f"C_{stage}"]
    print(f"--- {'before decoder' if stage == 'pre' else 'after decoder'}: LightGBM macro {T.macro_f1(y, a).mean():.4f}, right fraud rows {int(((a == y) & (y > 0)).sum())}, predicted fraud {int((a > 0).sum())}")
    for v in ("TABM", "BLEND"):
        b = z[f"{v}_{stage}"]; fr = y > 0
        only_b, only_a = fr & (b == y) & (a != y), fr & (a == y) & (b != y)
        either = np.where(fr & (b == y), b, a)  # oracle: take the other model wherever it is right on a fraud row
        print(f"{v:5s} macro {T.macro_f1(y, b).mean():.4f} | rows decoded differently {int((a != b).sum())} | fraud rows right only in {v} {int(only_b.sum())} "
              f"(of them LightGBM said legit {int((only_b & (a == 0)).sum())}, other class {int((only_b & (a > 0)).sum())}) | right only in LightGBM {int(only_a.sum())}"
              f" | legit-labelled rows called fraud: LightGBM {int(((y == 0) & (a > 0)).sum())}, {v} {int(((y == 0) & (b > 0)).sum())}, only {v} {int(((y == 0) & (b > 0) & (a == 0)).sum())}"
              f" | oracle union {T.macro_f1(y, either).mean():.4f}")
        fa, fb = T.macro_f1(y, a), T.macro_f1(y, b)
        print("      per-class F1 change: " + " ".join(f"{T.LABELS[k][:5]} {fb[k] - fa[k]:+.3f}" for k in range(1, T.K)))
