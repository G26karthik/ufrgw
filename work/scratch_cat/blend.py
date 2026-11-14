"""0.65 reference + 0.35 arm, probability blend before the decoder, both months. Reads no label.
usage (cwd=work): python scratch_cat/blend.py N|D REF ARM OUT   -> scratch_gap/P<BM>_<OUT>_<aug|sep>.npy (refuses to overwrite)"""
import os, sys
import numpy as np

BM, REF, ARM, OUT = sys.argv[1:5]
for kind in ("aug", "sep"):
    f = f"scratch_gap/P{BM}_{OUT}_{kind}.npy"; assert not os.path.exists(f), f + " exists"
    a, b = np.load(f"scratch_gap/P{BM}_{REF}_{kind}.npy"), np.load(f"scratch_gap/P{BM}_{ARM}_{kind}.npy"); assert a.shape == b.shape
    P = (0.65 * a + 0.35 * b).astype(np.float32); assert abs(P.sum(1) - 1).max() < 1e-4
    np.save(f, P); print(f, P.shape)
