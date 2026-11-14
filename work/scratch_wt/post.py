"""Helpers for the weight arms (cwd=work, PYTHONPATH=.). Nothing is overwritten.
check BM REF [SUF]      max abs difference between the harness control P<BM>_wC<SUF> and the stored reference P<BM>_<REF>
blend BM REF [SUF]      for wL1, wSQ: <arm>b35 = .65 REF + .35 arm (raw probabilities); <arm>p = arm divided by its per-class
                        arm/baseline weight ratio and renormalised (prior-matched; ratio from fit-row counts only: wL1 legit 1/5,
                        fraud 1; wSQ n_c^-0.5); <arm>pb35 = .65 REF + .35 <arm>p
bias BM VAR1,VAR2       August bias tuned on all August rows (raw and after takeover forcing); flags a value on the grid ends
table DEC REF VAR1,..   12-class precision / recall / F1 after the decoder, REF next to each variant (from a bench_eval dec file)"""
import os, sys
import numpy as np
import polars as pl
import train as T

c = pl.col; cmd = sys.argv[1]
MON = (("aug", 0), ("sep", 1))


def save(name, P):
    assert not os.path.exists(name), name
    np.save(name, P.astype(np.float32)); print("wrote", name)


if cmd == "check":
    B, REF, SUF = sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "")
    for m, _ in MON:
        a, b = np.load(f"scratch_gap/P{B}_wC{SUF}_{m}.npy"), np.load(f"scratch_gap/P{B}_{REF}_{m}.npy")
        print(f"{B} wC{SUF} vs {REF} {m}: shape {a.shape} max abs diff {np.abs(a - b).max():.3e} identical {np.array_equal(a, b)} argmax differs on {int((a.argmax(1) != b.argmax(1)).sum())} rows")

if cmd == "blend":
    B, REF, SUF = sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "")
    for m, fi in MON:
        y = pl.read_parquet(f"scratch_gap/{B}_f{fi}.parquet", columns=["is_test", "y_fit"]).filter(~c("is_test"))["y_fit"].to_numpy()
        n = np.bincount(y[y >= 0], minlength=T.K).astype(np.float64); R = np.load(f"scratch_gap/P{B}_{REF}_{m}.npy")
        ratio = {"wL1": np.r_[0.2, np.ones(T.K - 1)], "wSQ": n ** -0.5}
        for arm in ("wL1", "wSQ"):
            P = np.load(f"scratch_gap/P{B}_{arm}{SUF}_{m}.npy").astype(np.float64)
            Q = P / ratio[arm]; Q /= Q.sum(1, keepdims=True)
            print(f"{B} {arm}{SUF} {m}: mean fraud probability REF {1 - R[:, 0].mean():.4f} arm raw {1 - P[:, 0].mean():.4f} prior-matched {1 - Q[:, 0].mean():.4f}")
            save(f"scratch_gap/P{B}_{arm}b35{SUF}_{m}.npy", .65 * R + .35 * P)
            save(f"scratch_gap/P{B}_{arm}p{SUF}_{m}.npy", Q)
            save(f"scratch_gap/P{B}_{arm}pb35{SUF}_{m}.npy", .65 * R + .35 * Q)

if cmd == "bias":
    from common import LABEL2ID as L
    B = sys.argv[2]; Da = pl.read_parquet(f"scratch_gap/{B}_aug_rows.parquet"); ya = Da["y_true"].to_numpy().astype(np.int64)
    for v in sys.argv[3].split(","):
        P = np.load(f"scratch_gap/P{B}_{v}_aug.npy"); F = P.copy()
        for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):  # as bench_eval.force
            h = Da.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); F[h] = 0; F[h, k] = 1
        for name, X in (("raw", P), ("forced", F)):
            b = T.tune_bias(X, ya)
            print(f"{B} {v:10s} {name:6s} bias {np.round(b[1:], 1).tolist()} min {b[1:].min():.1f} max {b[1:].max():.1f} {'CLIPPED' if b[1:].min() < -3.95 or b[1:].max() > 5.95 else 'inside grid'}", flush=True)

if cmd == "table":
    z = np.load("scratch_gap/" + sys.argv[2]); y = z["y"]; R = sys.argv[3]
    def prf(q):
        tp = np.bincount(y[q == y], minlength=T.K); pr = np.bincount(q, minlength=T.K); n = np.bincount(y, minlength=T.K)
        return tp / np.maximum(pr, 1), tp / n, 2 * tp / (pr + n), pr
    a = prf(z[f"{R}_q"])
    for v in sys.argv[4].split(","):
        b = prf(z[f"{v}_q"])
        print(f"\n{sys.argv[2]}: {v} against {R}, September after decoder (P = precision, R = recall)")
        print(f"{'class':20s} {'n true':>7s} | {R + ' P':>8s} {R + ' R':>8s} {R + ' F1':>8s} {'n pred':>7s} | {v + ' P':>10s} {v + ' R':>10s} {v + ' F1':>10s} {'n pred':>7s} | {'dP':>6s} {'dR':>6s} {'dF1':>7s}")
        for k in range(T.K):
            print(f"{T.LABELS[k]:20s} {int((y == k).sum()):7d} | {a[0][k]:8.3f} {a[1][k]:8.3f} {a[2][k]:8.4f} {a[3][k]:7d} | {b[0][k]:10.3f} {b[1][k]:10.3f} {b[2][k]:10.4f} {b[3][k]:7d} |"
                  f" {b[0][k] - a[0][k]:+6.3f} {b[1][k] - a[1][k]:+6.3f} {b[2][k] - a[2][k]:+7.4f}")
        print(f"{'macro-F1':20s} {len(y):7d} | {'':8s} {'':8s} {a[2].mean():8.4f} {'':7s} | {'':10s} {'':10s} {b[2].mean():10.4f} {'':7s} | {'':6s} {'':6s} {b[2].mean() - a[2].mean():+7.4f}")
