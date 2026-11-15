"""Second exploratory candidate (scratch_wt/DECLARED_sq35.md): raw blend .65 h6 + .35 wSQ6 on the D tables, and its controls.
  assess : PD_wSQ6 (six-seed mean), PD_SQ6b35 (candidate), PD_K1d05/10/15 (h6, fraud probabilities x e^d), PD_K2 (.65 h6 +
           .35 h6 x per-class ratio matched to wSQ6's mean class probabilities on August) for aug and sep.
  build  : oof/sq35.npy (+ _ids.parquet, rows in oof/h6_ids order) and preds/sq35_test.npy (sample order).
  counts : predicted non-legit rows, candidate / h6, on aug, sep, oct with the bias tuned on August (does the bias transfer).
Never overwrites. usage (cwd=work, PYTHONPATH=.): python scratch_wt/sq6.py assess|build|counts"""
import os, shutil, sys
import numpy as np
import polars as pl
import train as T

G = "scratch_gap"; c = pl.col; cmd = sys.argv[1]; ld = lambda n: np.load(f"{G}/{n}.npy").astype(np.float64)


def save(name, P):
    assert not os.path.exists(name), name
    assert np.isfinite(P).all() and abs(P.sum(1) - 1).max() < 1e-3; np.save(name, P.astype(np.float32)); print("wrote", name)


def wsq6(kind):
    P = np.mean([ld(f"PD_wSQ{s}_{kind}") for s in ("", "b2", "b4")], axis=0)
    assert P.shape == (pl.read_parquet(f"{G}/D_{kind}_rows.parquet", columns=["txn_id"]).height, T.K), kind
    return P


def shift(P, r):
    Q = P * r; return Q / Q.sum(1, keepdims=True)


def k2_ratio(H, W):
    """Per-class ratio r so that mean(shift(H, r)) == mean(W) on August (predictions only)."""
    r = np.ones(T.K)
    for _ in range(200): r *= W.mean(0) / shift(H, r).mean(0)
    assert abs(shift(H, r).mean(0) - W.mean(0)).max() < 1e-6
    return r / r[0]


def oct_wsq6():
    """October wSQ6 in sample_submission order (as scratch_mix/mix_blend.oct_pred)."""
    tid = pl.read_csv(T.SAMPLE).select(c("id").alias("txn_id"))
    ids = pl.read_parquet(f"{G}/D_f2.parquet", columns=["txn_id", "is_test"]).filter(c("is_test")).select("txn_id").with_row_index("i")
    j = tid.join(ids, on="txn_id", how="left", maintain_order="left"); assert j["i"].null_count() == 0 and j["i"].n_unique() == tid.height
    return np.mean([ld("PD_wSQo0_oct"), ld("PD_wSQo3_oct")], axis=0)[j["i"].to_numpy()]


if cmd == "assess":
    r = None
    for kind in ("aug", "sep"):
        H, W = ld(f"PD_h6_{kind}"), wsq6(kind); assert H.shape == W.shape
        if r is None: r = k2_ratio(H, W); np.save("scratch_wt/k2_ratio.npy", r); print("K2 per-class ratio (legit = 1):", np.round(r, 2).tolist())
        save(f"{G}/PD_wSQ6_{kind}.npy", W); save(f"{G}/PD_SQ6b35_{kind}.npy", .65 * H + .35 * W)
        for d in (0.5, 1.0, 1.5): save(f"{G}/PD_K1d{int(d * 10):02d}_{kind}.npy", shift(H, np.r_[1.0, np.full(T.K - 1, np.exp(d))]))
        S = shift(H, r); save(f"{G}/PD_K2_{kind}.npy", .65 * H + .35 * S)
        print(f"{kind}: mean fraud probability h6 {1 - H[:, 0].mean():.4f} | wSQ6 {1 - W[:, 0].mean():.4f} | h6 shifted (K2 source) {1 - S[:, 0].mean():.4f}"
              f" | candidate {1 - (.65 * H + .35 * W)[:, 0].mean():.4f} | K2 {1 - (.65 * H + .35 * S)[:, 0].mean():.4f}", flush=True)

if cmd == "build":
    tag = "sq35"
    for f in (f"oof/{tag}.npy", f"oof/{tag}_ids.parquet", f"preds/{tag}_test.npy"): assert not os.path.exists(f), f"{f} exists; never overwrite"
    ids = pl.read_parquet("oof/h6_ids.parquet"); V = np.zeros((ids.height, T.K), np.float32); seen = np.zeros(ids.height, bool)
    for kind, fold in (("aug", 0), ("sep", 1)):
        rows = pl.read_parquet(f"{G}/D_{kind}_rows.parquet", columns=["txn_id"]).with_row_index("i")
        j = ids.with_row_index("o").filter(c("fold") == fold).join(rows, on="txn_id", how="left", maintain_order="left")
        assert j["i"].null_count() == 0 and j.height == rows.height and j["i"].n_unique() == rows.height
        V[j["o"].to_numpy()] = ld(f"PD_SQ6b35_{kind}")[j["i"].to_numpy()]; seen[j["o"].to_numpy()] = True
    assert seen.all()
    H = np.load("oof/h6.npy"); print(f"validation blend vs h6: argmax differs on {int((V.argmax(1) != H.argmax(1)).sum())} of {len(V)} rows")
    H = np.load("preds/h6_test.npy").astype(np.float64); W = oct_wsq6(); Tt = .65 * H + .35 * W
    assert Tt.shape == H.shape and abs(Tt.sum(1) - 1).max() < 1e-3 and np.isfinite(Tt).all()
    print(f"test blend vs h6: argmax differs on {int((Tt.argmax(1) != H.argmax(1)).sum())} of {len(Tt)} rows | October mean fraud probability h6 {1 - H[:, 0].mean():.4f} wSQ6 {1 - W[:, 0].mean():.4f}")
    np.save(f"oof/{tag}.npy", V); shutil.copyfile("oof/h6_ids.parquet", f"oof/{tag}_ids.parquet"); np.save(f"preds/{tag}_test.npy", Tt.astype(np.float32)); print("written", tag)

if cmd == "counts":
    ya = pl.read_parquet(f"{G}/D_aug_rows.parquet")["y_true"].to_numpy().astype(np.int64)
    S = {"h6": (ld("PD_h6_aug"), ld("PD_h6_sep"), np.load("preds/h6_test.npy")), "sq35": (ld("PD_SQ6b35_aug"), ld("PD_SQ6b35_sep"), np.load("preds/sq35_test.npy"))}
    N = {}
    for k, (a, s, o) in S.items():
        b = T.tune_bias(a, ya); N[k] = [np.bincount(T.decode(x, b), minlength=T.K) for x in (a, s, o)]
        print(f"{k}: August bias {np.round(b[1:], 1).tolist()} {'CLIPPED' if b[1:].min() < -3.95 or b[1:].max() > 5.95 else 'inside grid'}")
    for i, kind in enumerate(("aug", "sep", "oct")):
        h, q = N["h6"][i], N["sq35"][i]
        print(f"{kind}: non-legit h6 {h[1:].sum()} sq35 {q[1:].sum()} ratio {q[1:].sum() / h[1:].sum():.4f} | per-class ratio {np.round(q[1:] / np.maximum(h[1:], 1), 3).tolist()}")
