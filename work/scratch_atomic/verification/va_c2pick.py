"""Choice of F for mechanism c2 by the frozen rule: August macro-F1 after the decoder on N, mean of the two out-of-half scores,
ties -> smaller F. Reads the cached August decodes only. usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_c2pick.py"""
import numpy as np
import train as T
from va_score import Bench

for Bn, R in (("N", "C"), ("D", "A_C")):
    S = Bench(Bn); y = S.y["aug"]
    for v in (R, "vac2f3", "vac2f10", "vac2f30"):
        q = S.decode(v, "aug")[1]
        print(f"{Bn} August after decoder, mean of halves: {v:8s} {np.mean([T.macro_f1(y[S.half == h], q[S.half == h]).mean() for h in (0, 1)]):.5f}")
