"""Honest decode check: bias tuned on one OOF fold, scored on the other (no in-sample bias optimism).
usage (cwd=work): python oos.py TAG [TAG ...]
"""
import sys
import numpy as np
import polars as pl
from common import ROOT
from train import tune_bias, decode, macro_f1

for tag in sys.argv[1:]:
    P = np.load(ROOT / "oof" / f"{tag}.npy"); ids = pl.read_parquet(ROOT / "oof" / f"{tag}_ids.parquet")
    y, f = ids["y"].to_numpy(), ids["fold"].to_numpy()
    s = []
    for a, b in [(0, 1), (1, 0)]:
        bias = tune_bias(P[f == a], y[f == a])
        s.append(macro_f1(y[f == b], decode(P[f == b], bias)).mean())
    print(f"{tag:12s} Aug->Sep {s[0]:.4f}  Sep->Aug {s[1]:.4f}  mean {np.mean(s):.4f}", flush=True)
