"""Part 3: choose the TabM checkpoint on August by macro-F1 under the decoder (nb.aug_score: bias tuned on one
event-grouped half, other half scored, swapped). September labels are not read.
Rule (declared in PROGRESS.md before the run): TabM alone; keep epoch 6 (the log-loss choice) unless another epoch beats
it on the mean AND on both halves (the half-to-half spread of the difference).
The frozen w = .35 blend with LightGBM (notes/tabm_phfc.md) is scored for information only.
usage (cwd=work, PYTHONPATH=.): python scratch_stack/n14/ck_select.py SEED EP,EP,..."""
import sys
import numpy as np
from joblib import Parallel, delayed
from nb import aug_ctx, aug_score, split, load

SEED, GRID = sys.argv[1], [int(e) for e in sys.argv[2].split(",")]


def run(name, w):
    Da, ya, grp = aug_ctx(); P = load(name, "aug"); ll = float(-np.log(P[np.arange(len(ya)), ya] + 1e-12).mean())
    if w < 1:
        P = (w * P + (1 - w) * load("C", "aug")).astype(np.float32)
    return aug_score(P, Da, ya, split(grp, 2, 0)), ll


if __name__ == "__main__":
    names = ["C", f"TABMple{SEED}"] + [f"sTck{SEED}e{e}" for e in GRID]
    jobs = [(n, 1.0) for n in names] + [(n, 0.35) for n in names[1:]]
    res = dict(zip(jobs, Parallel(n_jobs=5)(delayed(run)(*j) for j in jobs)))
    for (n, w), ((f, q, h), ll) in res.items():
        print(f"{n:12s} {'alone' if w == 1 else 'blend .35'}: Aug forced {f:.4f} final {q:.4f} halves {h} log-loss {ll:.5f}", flush=True)
    a, b = np.load(f"scratch_gap/PN_TABMple{SEED}_aug.npy"), np.load(f"scratch_gap/PN_sTck{SEED}e{GRID[0]}_aug.npy")
    print(f"provenance: new epoch-{GRID[0]} vs existing TABMple{SEED} August probabilities: mean abs diff {np.abs(a - b).mean():.5f}, argmax differs on {int((a.argmax(1) != b.argmax(1)).sum())} rows")
    base = res[(f"sTck{SEED}e6", 1.0)][0]; pick = 6
    for e in GRID:
        r = res[(f"sTck{SEED}e{e}", 1.0)][0]
        if e != 6 and r[1] > res[(f"sTck{SEED}e{pick}", 1.0)][0][1] and all(x > y for x, y in zip(r[2], base[2])):
            pick = e
    print("checkpoint chosen on August:", pick)
