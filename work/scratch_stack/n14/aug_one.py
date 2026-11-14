"""August selection score (nb.aug_score) of saved benchmark-N probability files. usage: python scratch_stack/n14/aug_one.py NAME,NAME,..."""
import sys
from joblib import Parallel, delayed
from nb import aug_ctx, aug_score, split, load


def run(name):
    Da, ya, grp = aug_ctx(); return aug_score(load(name, "aug"), Da, ya, split(grp, 2, 0))


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    for n, (f, q, h) in zip(names, Parallel(n_jobs=5)(delayed(run)(n) for n in names)):
        print(f"{n:10s}: Aug forced {f:.4f} final {q:.4f} halves {h}", flush=True)
