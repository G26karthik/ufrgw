"""Part 3 two-seed blend files (declared in PROGRESS.md 21:10): .65 C + .35 mean(e30 seed 0, e12 seed 1) and .65 C + .35 e30 seed 1."""
import numpy as np
G = "scratch_gap/PN_"
for k in ("aug", "sep"):
    c = np.load(G + f"C_{k}.npy"); a, b, d = (np.load(G + f"sTck{n}_{k}.npy") for n in ("0e30", "1e12", "1e30"))
    np.save(G + f"sTB2sel_{k}.npy", (0.65 * c + 0.175 * a + 0.175 * b).astype(np.float32)); np.save(G + f"sTB1e30_{k}.npy", (0.65 * c + 0.35 * d).astype(np.float32))
