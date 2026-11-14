"""Part 1: learn how to combine prediction sources on benchmark N. August labels only; September labels are never read.
Sources (<= 6): C (LightGBM, relabel-C), sXGB, sCAT, TABM (ple, 2 seeds), sTS (soft two-stage LightGBM),
sSUB (LightGBM without the burst and ring feature modules). RAW (LightGBM on raw labels) is left out: its August argmax differs from C
on 149 rows only (seed-level), and its September result is already known (+.0001, notes/gap_reset.md).
Every score is nb.aug_score: bias tuned on one event-grouped half of August, other half scored after the full decoder.
 1. each source alone.
 2. simple probability blends, declared set: C with each other source at weight .35 and .5; mean of the three GBDTs;
    mean of all six; greedy forward selection with replacement (start = best single source, at most 4 additions).
    The best of these on August is written as sBSEL.
 3. multinomial logistic regression on the 72 clipped log-probabilities (clip 1e-6, standardised), L2, C in {1e-4, 1e-3, 1e-2},
    5-fold event-grouped out-of-fold on August; legit rows downsampled to 20% with weight 5 for the fit. Chosen C -> sLR.
    Leave-one-source-out at the chosen C measures each source's incremental contribution; the fit without sTS is sLR5.
 4. shallow tree stacker (LightGBM, 4 leaves, 30 rounds, lr .05, init_score = log P of C, so trees learn corrections
    of the baseline from the 72 log-probabilities), same grouped folds -> sTREE.
August files of the stackers are their grouped out-of-fold predictions; September files come from the fit on all of August.
usage (cwd=work, PYTHONPATH=.): python scratch_stack/n14/combine.py  -> scratch_gap/PN_<sBSEL|sLR|sLR5|sTREE>_<aug|sep>.npy, combine.json"""
import functools, json, time
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from nb import aug_ctx, aug_score, split, load, K, G, HERE

SRC = ["C", "sXGB", "sCAT", "TABM", "sTS", "sSUB"]; NJ = 5


@functools.lru_cache(None)
def ctx():
    Da, ya, grp = aug_ctx(); n = len(ya)
    keep = (ya != 0) | (np.random.default_rng(0).random(n) < 0.2)
    return Da, ya, split(grp, 2, 0), split(grp, 5, 1), keep, np.where(ya == 0, 5.0, 1.0)


@functools.lru_cache(None)
def X(kind):
    return np.hstack([np.log(np.clip(load(s, kind), 1e-6, 1)) for s in SRC]).astype(np.float32)


def blend(w, kind):
    return (sum(v * load(k, kind) for k, v in w) / sum(v for _, v in w)).astype(np.float32)


def sc(P):
    Da, ya, half = ctx()[:3]; return aug_score(P, Da, ya, half)


def sc_blend(w):
    return sc(blend(w, "aug"))


def lr_fit(cols, C_, tr):
    _, ya, _, _, _, w = ctx(); Xa = X("aug")[:, cols]; mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-6
    m = LogisticRegression(C=C_, max_iter=300).fit((Xa[tr] - mu) / sd, ya[tr], sample_weight=w[tr]); assert list(m.classes_) == list(range(K))
    return lambda Z: m.predict_proba((Z[:, cols] - mu) / sd).astype(np.float32)


def tree_fit(cols, C_, tr):  # C_ unused; corrections on top of the baseline's log-probabilities (columns 0..11 = C)
    _, ya, _, _, _, w = ctx(); Xa = X("aug")
    m = lgb.train(dict(objective="multiclass", num_class=K, learning_rate=0.05, num_leaves=4, min_data_in_leaf=200, lambda_l2=10.0,
                       feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, num_threads=3, verbose=-1, seed=0),
                  lgb.Dataset(Xa[tr][:, cols], ya[tr], weight=w[tr], init_score=Xa[tr][:, :12].astype(np.float64)), 30)

    def pred(Z):
        r = m.predict(Z[:, cols], raw_score=True) + Z[:, :12]; r = np.exp(r - r.max(1, keepdims=True))
        return (r / r.sum(1, keepdims=True)).astype(np.float32)
    return pred


def oof(fit, drop=None, C_=None):
    _, ya, _, f5, keep, _ = ctx(); cols = [i for i in range(X("aug").shape[1]) if drop is None or i // 12 != SRC.index(drop)]
    O = np.zeros((len(ya), K), np.float32)
    for f in range(5):
        O[f5 == f] = fit(cols, C_, keep & (f5 != f))(X("aug")[f5 == f])
    return O, cols


def sc_oof(kind, drop, C_):
    return sc(oof(lr_fit if kind == "lr" else tree_fit, drop, C_)[0])


def save(name, Pa, Ps):
    for k, P in (("aug", Pa), ("sep", Ps)):
        assert P.shape[1] == K and abs(P.sum(1) - 1).max() < 1e-3; np.save(f"{G}/PN_{name}_{k}.npy", P.astype(np.float32))


if __name__ == "__main__":
    import combine as M  # workers unpickle the task functions by module reference, not from __main__
    ctx, X, blend, sc, sc_blend, lr_fit, tree_fit, oof, sc_oof = M.ctx, M.X, M.blend, M.sc, M.sc_blend, M.lr_fit, M.tree_fit, M.oof, M.sc_oof
    t0 = time.time(); R = {}; par = Parallel(n_jobs=NJ)
    ya = ctx()[1]; am = {s: load(s, "aug").argmax(1) for s in SRC}
    single = dict(zip(SRC, par(delayed(sc_blend)(((s, 1.0),)) for s in SRC)))
    for s in SRC:
        print(f"source {s:5s}: Aug forced {single[s][0]:.4f} final {single[s][1]:.4f} halves {single[s][2]} | argmax differs from C on {int((am[s] != am['C']).sum())} rows", flush=True)
    R["single"] = single
    # 2. declared simple blends
    cand = {f"C+{s}@{w}": (("C", 1 - w), (s, w)) for s in SRC[1:] for w in (0.35, 0.5)}
    cand["GBDT3"] = tuple((s, 1.0) for s in ("C", "sXGB", "sCAT")); cand["ALL6"] = tuple((s, 1.0) for s in SRC)
    res = dict(zip(cand, par(delayed(sc_blend)(w) for w in cand.values())))
    best = max(SRC, key=lambda s: single[s][1]); cnt = {best: 1}; cur = single[best][1]
    for step in range(4):
        tr = {s: tuple({**cnt, s: cnt.get(s, 0) + 1}.items()) for s in SRC}
        r = dict(zip(tr, par(delayed(sc_blend)(tuple((k, float(v)) for k, v in w)) for w in tr.values())))
        s = max(r, key=lambda k: r[k][1]); name = "greedy:" + "+".join(f"{k}x{v}" for k, v in tr[s])
        print(f"greedy step {step + 1}: add {s} -> {r[s][1]:.4f} (current {cur:.4f})", flush=True)
        if r[s][1] <= cur:
            break
        cnt, cur = dict(tr[s]), r[s][1]; cand[name] = tuple((k, float(v)) for k, v in tr[s]); res[name] = r[s]
    for k, v in res.items():
        print(f"blend {k:32s}: Aug forced {v[0]:.4f} final {v[1]:.4f} halves {v[2]} (vs C {v[1] - single['C'][1]:+.4f})", flush=True)
    bsel = max(res, key=lambda k: res[k][1]); R["blends"] = res; R["bsel"] = [bsel, cand[bsel]]
    print("best simple blend on August:", bsel, cand[bsel], flush=True)
    save("sBSEL", blend(cand[bsel], "aug"), blend(cand[bsel], "sep"))
    # 3. logistic-regression stacker
    grid = (1e-4, 1e-3, 1e-2); lr = dict(zip(grid, par(delayed(sc_oof)("lr", None, c_) for c_ in grid)))
    for c_, v in lr.items():
        print(f"LR stacker C={c_:g}: Aug forced {v[0]:.4f} final {v[1]:.4f} halves {v[2]} (vs C {v[1] - single['C'][1]:+.4f})", flush=True)
    cb = max(grid, key=lambda c_: lr[c_][1]); print("LR C chosen on August:", cb, flush=True)
    loso = dict(zip(SRC[1:], par(delayed(sc_oof)("lr", s, cb) for s in SRC[1:])))
    for s, v in loso.items():
        print(f"LR without {s:5s}: Aug final {v[1]:.4f} halves {v[2]} -> contribution of {s} {lr[cb][1] - v[1]:+.4f}", flush=True)
    R["lr"] = {str(k): v for k, v in lr.items()}; R["lr_C"] = cb; R["lr_loso"] = loso
    keep = ctx()[4]; allc = list(range(72))
    for name, drop in (("sLR", None), ("sLR5", "sTS")):
        O, cols = oof(lr_fit, drop, cb); save(name, O, lr_fit(cols, cb, keep)(X("sep")))
    # 4. shallow tree stacker on top of the baseline
    O, cols = oof(tree_fit); v = sc(O); R["tree"] = v
    print(f"tree stacker (4 leaves, 30 rounds, residual on C): Aug forced {v[0]:.4f} final {v[1]:.4f} halves {v[2]} (vs C {v[1] - single['C'][1]:+.4f})", flush=True)
    save("sTREE", O, tree_fit(cols, None, keep)(X("sep")))
    json.dump(R, open(f"{HERE}/combine.json", "w"), indent=1, default=str)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)
