"""Second-stage stacking on entity-aggregated stage-1 predictions (collective classification). notes/stack.md.

s1: leave-one-month-out (LOMO) tun_best models (matrix_m5dm, relabel C, train.LGB, 150 rounds, seeds 0..S-1), each
    fit on the other two of Jul/Aug/Sep (+ the 147 Oct train rows). Every Jul/Aug/Sep row gets probs from the model that
    never saw its month; the Oct/Nov tail (test) gets the mean of the three LOMO models, so train and test are both
    'unseen-month' predictions. -> scratch_stack/p1.npy (matrix row order), p1_ids.parquet, p1_tail_m{7,8,9}.npy, top.json
s2: features = own stage-1 probs + aggregates of OTHER rows' stage-1 probs sharing sender / device / P2P receiver /
    merchant within +-1h/1d/15d (pool = month x density half for train rows, all test rows for test, so every neighbour
    prob comes from a model that never saw that month and density matches Oct) + top raw feats by stage-1 gain.
    LGB evaluated on train.FOLDS (from scratch, or 'r' variants boosted from init_score = log stage-1 probs).
    Result: no gain over tun_best (notes/stack.md), so there is no full/test path.
usage (cwd=work): python stack.py s1 ; python stack.py s2 [--variants base,noagg,stack,rnoagg,rstack] [--checkpoints 30,75,150]
"""
import argparse, json, time
from types import SimpleNamespace
import numpy as np
import polars as pl
from common import ROOT, DATA, LABELS

K = len(LABELS)
S = ROOT / "scratch_stack"; S.mkdir(exist_ok=True)
MATRIX = ROOT / "feats_cache" / "matrix_m5dm.parquet"
ROUNDS = 150
WINS = {"1h": 3600, "1d": 86400, "15d": 15 * 86400}
NTOP = 30
c = pl.col


def args_ns(feats, threads, rounds):
    return SimpleNamespace(feats=feats, model="lgb", lr=0.05, threads=threads, rounds=rounds, legit_frac=0.2, cw_pow=0.0)


def s1(a):
    import train
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1) as ex: yf = ex.submit(train.fit_target, "C").result()
    D, feats = train.build_matrix([], "", str(MATRIX))
    D = D.join(yf, on="txn_id", how="left", maintain_order="left")
    A = args_ns(feats, a.threads, ROUNDS)
    mon = D["month"].to_numpy()
    tail = int(np.argmax(mon >= 10)); assert (mon[tail:] >= 10).all() and not D["is_test"][:tail].any()
    P = np.zeros((D.height, K), np.float32); imp = np.zeros(len(feats))
    for m in (7, 8, 9):
        idx = np.flatnonzero(mon == m); s0, n = int(idx[0]), len(idx); assert idx[-1] == s0 + n - 1
        Pt = np.zeros((D.height - tail, K), np.float32)
        for s in range(a.seeds):
            t = time.time()
            mdl, _ = train.fit_one(D, A, s, rounds=ROUNDS, rows=~c("is_test") & (c("month") != m))
            P[s0:s0 + n] += train.predict_chunked(train.lgb_pred, mdl, D.slice(s0, n), feats) / a.seeds
            Pt += train.predict_chunked(train.lgb_pred, mdl, D.slice(tail), feats) / a.seeds
            imp += mdl.feature_importance("gain")
            print(f"  lomo m{m} seed{s} ({time.time()-t:.0f}s)", flush=True)
        np.save(S / f"p1_tail_m{m}.npy", Pt); P[tail:] += Pt / 3
    np.save(S / "p1.npy", P)
    D.select("txn_id", "month", "y", "y_fit", "is_test").write_parquet(S / "p1_ids.parquet")
    (S / "top.json").write_text(json.dumps([feats[i] for i in np.argsort(-imp)[:NTOP]]))


# ---------------------------------------------------------------- stage-2 features
def _rmq_table(v):
    T = [v]
    while 2 ** len(T) <= len(v):
        h = 2 ** (len(T) - 1); p = T[-1]; T.append(np.maximum(p[:-h], p[h:]))
    return T


def _rmq(T, l, r):
    """max of v[l:r] per element; -inf where empty."""
    n = r - l; ok = n > 0; out = np.full(len(l), -np.inf, np.float32)
    k = np.floor(np.log2(np.maximum(n, 1))).astype(np.int64)
    for j in np.unique(k[ok]):
        m = ok & (k == j)
        out[m] = np.maximum(T[j][l[m]], T[j][r[m] - 2 ** j])
    return out


def window_aggs(code, pool, ts, P, name):
    """Aggregates over OTHER rows with the same (pool, code) within +-w. code -1 = no key -> NaN."""
    g = pool.astype(np.int64) * (code.max() + 2) + code
    comp = g * 2 ** 25 + (ts - ts.min())  # 2^25 s > data span + 15 d, so windows never cross groups
    o = np.argsort(comp, kind="stable"); cs_ = comp[o]
    pf = (1 - P[o, 0]).astype(np.float32)
    Q = np.column_stack([P[o, 1:], pf, pf > 0.5]).astype(np.float64)
    cs = np.vstack([np.zeros((1, Q.shape[1])), np.cumsum(Q, 0)])
    T = _rmq_table(pf); i = np.arange(len(o)); out = {}
    for wn, w in WINS.items():
        lo = np.searchsorted(cs_, cs_ - w, "left"); hi = np.searchsorted(cs_, cs_ + w, "right")
        n = (hi - lo - 1).astype(np.float32)
        sm = cs[hi] - cs[lo] - Q
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (sm / n[:, None]).astype(np.float32)
        mx = np.maximum(_rmq(T, lo, i), _rmq(T, i + 1, hi)); mx[~np.isfinite(mx)] = np.nan
        cols = {f"n": n, "pf_max": mx, "pf_mean": mean[:, 11], "n50": sm[:, 12].astype(np.float32)}
        if wn != "1h":
            cols.update({f"m_{LABELS[k]}": mean[:, k - 1] for k in range(1, K)})
        for cn, v in cols.items():
            u = np.empty_like(v); u[o] = v; u[code < 0] = np.nan
            out[f"{name}_{wn}_{cn}"] = u
    return out


def _selftest():
    rng = np.random.default_rng(0); n = 300
    code = rng.integers(-1, 5, n); pool = rng.integers(0, 2, n); ts = 1_700_000_000 + rng.integers(0, 40 * 86400, n)
    P = rng.dirichlet(np.ones(K), n).astype(np.float32)
    out = window_aggs(code, pool, ts, P, "k")
    for i in rng.integers(0, n, 40):
        for wn, w in WINS.items():
            nb = (code == code[i]) & (pool == pool[i]) & (np.abs(ts - ts[i]) <= w); nb[i] = False
            pf = 1 - P[nb, 0]
            exp = (np.nan, np.nan) if code[i] < 0 else (nb.sum(), pf.max() if nb.any() else np.nan)
            np.testing.assert_allclose([out[f"k_{wn}_n"][i], out[f"k_{wn}_pf_max"][i]], exp, rtol=1e-5)
            if code[i] >= 0 and nb.any() and wn != "1h":
                np.testing.assert_allclose(out[f"k_{wn}_m_{LABELS[3]}"][i], P[nb, 3].mean(), rtol=1e-4)


def minimal_tx(ids):
    cols = ["txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "device_id", "txn_type", "qr_vpa_mismatch"]
    raw = pl.concat([pl.read_parquet(DATA / f, columns=cols) for f in ("train.parquet", "test.parquet")])
    T = ids.join(raw, on="txn_id", how="left", maintain_order="left").with_columns(c("timestamp").dt.epoch("s").alias("ts"))
    assert T["ts"].null_count() == 0
    return T


def build_s2(T, P1):
    from density import halves
    h = halves(T).fill_null(0).to_numpy().astype(np.int64)
    pool = np.where(T["is_test"].to_numpy(), 0, 1 + 2 * T["month"].to_numpy().astype(np.int64) + h)
    ts = T["ts"].to_numpy().astype(np.int64)
    p2p = c("receiver_type") == "P2P"
    keys = {"snd": c("sender_id"), "dev": c("device_id"), "rcv": pl.when(p2p).then(c("receiver_id")),
            "mer": pl.when(~p2p).then(c("receiver_id"))}
    F = {f"own_{l}": P1[:, k] for k, l in enumerate(LABELS)}
    for kn, e in keys.items():
        code = T.select(e.rank("dense")).to_series().fill_null(0).to_numpy().astype(np.int64) - 1
        F.update(window_aggs(code, pool, ts, P1, kn)); print(f"  aggs {kn}", flush=True)
    top = json.loads((S / "top.json").read_text())
    R = pl.scan_parquet(MATRIX).select(["txn_id"] + top).collect()
    assert R["txn_id"].equals(T["txn_id"])
    return pl.DataFrame(F).hstack(R.drop("txn_id")).hstack(T.select("txn_id", "month", "y", "y_fit", "is_test"))


# ---------------------------------------------------------------- eval
def evaluate(name, P, y, yc, fold):
    from train import tune_bias, decode, macro_f1
    sep, aug = fold == 1, fold == 0
    b = tune_bias(P, y); pr = decode(P, b)
    bA = tune_bias(P[aug], y[aug]); pA = decode(P[sep], bA)
    r = dict(pooled=macro_f1(y, pr).mean(), sep=macro_f1(y[sep], pr[sep]).mean(), c_pooled=macro_f1(yc, pr).mean(),
             c_sep=macro_f1(yc[sep], pr[sep]).mean(), oos_sep=macro_f1(y[sep], pA).mean(), c_oos_sep=macro_f1(yc[sep], pA).mean())
    print(f"{name:28s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
    return r, b


def fit_pred(F, feats, rows, V, seeds, threads, checkpoints, resid=False):
    """resid: boost from init_score = log(own stage-1 probs), so trees only learn corrections."""
    import train, lightgbm as lgb
    A = args_ns(feats, threads, max(checkpoints))
    oi = [feats.index(f"own_{l}") for l in LABELS]
    init = lambda X: np.log(X[:, oi].astype(np.float64) + 1e-9) if resid else 0
    out = {r: np.zeros((V.height, K), np.float32) for r in checkpoints}
    for s in range(seeds):
        X, y, w = train.sample_fit(F, A, s, rows)
        p = dict(objective="multiclass", num_class=K, learning_rate=A.lr, **train.LGB, num_threads=threads, verbose=-1, seed=s)
        m = lgb.train(p, lgb.Dataset(X, y, weight=w, init_score=init(X) if resid else None), max(checkpoints)); del X
        X = V.select(feats).to_numpy()
        for r in checkpoints:
            z = m.predict(X, num_iteration=r, raw_score=True) + init(X)
            z = np.exp(z - z.max(1, keepdims=True)); out[r] += (z / z.sum(1, keepdims=True)).astype(np.float32) / seeds
        del X
    return out


def s2(a):
    import train, postproc
    from clean_labels import clean
    _selftest()
    ids = pl.read_parquet(S / "p1_ids.parquet"); P1 = np.load(S / "p1.npy")
    T = minimal_tx(ids)
    t = time.time(); F = build_s2(T, P1); print(f"stage-2 frame {F.shape} ({time.time()-t:.0f}s)", flush=True)
    yc_all = clean(T)
    allf = [x for x in F.columns if x not in ("txn_id", "month", "y", "y_fit", "is_test")]
    noagg = [x for x in allf if not x.startswith(("snd_", "dev_", "rcv_", "mer_"))]
    mon = F["month"].to_numpy(); tr = ~F["is_test"].to_numpy()
    vidx = np.concatenate([np.flatnonzero((mon == vm) & tr) for _, vm in train.FOLDS])
    fold = np.concatenate([np.full(int(((mon == vm) & tr).sum()), fi) for fi, (_, vm) in enumerate(train.FOLDS)])
    y = F["y"].to_numpy()[vidx].astype(np.int64); yc = yc_all.to_numpy()[vidx].astype(np.int64)
    ids_v = F.select("txn_id")[vidx]
    tb = pl.read_parquet(ROOT / "oof" / "tun_best_ids.parquet").with_row_index("r")
    tbr = ids_v.join(tb, on="txn_id", how="left", maintain_order="left")["r"].to_numpy()
    vs = a.variants.split(",")
    cand = {"tun_best": np.load(ROOT / "oof" / "tun_best.npy")[tbr], "lomo_s1": P1[vidx]} if "base" in vs else {}
    cps = [int(x) for x in a.checkpoints.split(",")]
    # variant: [r]esid (boost from stage-1 log-probs) + feature set (noagg = own probs + raw, stack = + aggregates)
    for vname in vs:
        if vname == "base": continue
        feats = noagg if vname.endswith("noagg") else allf
        Ps = {r: [] for r in cps}
        for fm, vm in train.FOLDS:
            vi = np.flatnonzero((mon == vm) & tr); V = F.slice(int(vi[0]), len(vi))  # months are contiguous (s1 asserts)
            o = fit_pred(F, feats, c("month").is_in(fm) & ~c("is_test"), V, a.seeds, a.threads, cps, vname.startswith("r"))
            for r in cps: Ps[r].append(o[r])
        for r in cps:
            cand[f"{vname}_r{r}"] = np.concatenate(Ps[r]); np.save(S / f"oof_{vname}_r{r}.npy", cand[f"{vname}_r{r}"])
    res = {}
    for nm, P in cand.items():
        res[nm], _ = evaluate(nm, P, y, yc, fold)
        res[nm + "+pp"], _ = evaluate(nm + "+pp", postproc.apply(P, ids_v, T), y, yc, fold)
    (S / f"results_{a.variants.replace(',', '_')}.json").write_text(json.dumps({k: {m: float(v) for m, v in r.items()} for k, r in res.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["s1", "s2"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--checkpoints", default="100,150,250")
    ap.add_argument("--variants", default="base,noagg,stack", help="base | [r]noagg | [r]stack (r = boost from stage-1)")
    a = ap.parse_args(); t0 = time.time()
    s1(a) if a.stage == "s1" else s2(a)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
