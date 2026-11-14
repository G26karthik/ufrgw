"""Corrected rows, new errors and per-class F1 change of every variant against the first one (C), September, benchmark N.
Reads scratch_gap/dec_N<SUFFIX>.npz written by bench_eval.py. `pre` = before the decoder, `q` = after the full decoder.
usage (cwd=work, PYTHONPATH=.): python scratch_stack/n14/diff.py SUFFIX"""
import sys
import numpy as np
import train as T

z = np.load(f"scratch_gap/dec_{sys.argv[1] if sys.argv[1][:2] in ('N_', 'D_') else 'N' + sys.argv[1]}.npz"); y = z["y"]; V = [k[:-2] for k in z.files if k.endswith("_q")]; fr = y > 0
for st, nm in (("pre", "before decoder"), ("q", "after decoder")):
    a = z[f"{V[0]}_{st}"]; fa = T.macro_f1(y, a)
    print(f"--- {nm}: {V[0]} macro {fa.mean():.4f}, right fraud rows {int(((a == y) & fr).sum())}, legit-labelled called fraud {int((~fr & (a > 0)).sum())}")
    for v in V[1:]:
        b = z[f"{v}_{st}"]; fb = T.macro_f1(y, b); fix, new = (b == y) & (a != y), (a == y) & (b != y)
        print(f"{v:9s} macro {fb.mean():.4f} ({fb.mean() - fa.mean():+.4f}) | changed {int((a != b).sum())} | corrected {int(fix.sum())} (fraud {int((fix & fr).sum())}, legit {int((fix & ~fr).sum())})"
              f" | new errors {int(new.sum())} (fraud {int((new & fr).sum())}, legit {int((new & ~fr).sum())})")
        print("          per-class F1 change: " + " ".join(f"{T.LABELS[k][:5]} {fb[k] - fa[k]:+.3f}" for k in range(1, T.K)))
