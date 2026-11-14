"""Rows changed / corrected / newly wrong / missed fraud recovered for variants against a reference, from a bench_eval decode file.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/dec_diff.py dec_N_icl.npz C iICLs0,iICLs0b35"""
import sys
import numpy as np
import train as T

z = np.load("scratch_gap/" + sys.argv[1]); y = z["y"]; R = sys.argv[2]
for v in sys.argv[3].split(","):
    for st in ("pre", "q"):
        a, b = z[f"{R}_{st}"], z[f"{v}_{st}"]; ch = a != b; d = T.macro_f1(y, b) - T.macro_f1(y, a)
        top = sorted(range(1, T.K), key=lambda k: -abs(d[k]))[:3]
        print(f"{v} {st} changed {int(ch.sum())} corrected {int((ch & (b == y)).sum())} newly wrong {int((ch & (a == y)).sum())}"
              f" missed fraud recovered {int(((a == 0) & (y > 0) & (b == y)).sum())} new legit called fraud {int(((a == 0) & (y == 0) & (b > 0)).sum())} "
              + " ".join(f"{T.LABELS[k][:6]} {d[k]:+.3f}" for k in top))
