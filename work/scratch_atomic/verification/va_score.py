"""Whole-held-month scoring of prediction files against a reference, before (`pre`) and after (`q`) the decoder rules.
August (freeze month): rows split into two event-grouped halves exactly as scratch_gap/aug_dec.py (rng 2026); the bias is
  tuned on one half and the other half is decoded (takeover forcing, device zones, bait rule), then the halves swap. The
  out-of-half labels of both halves are joined into one August decode, so row-level counts exist for August too.
September (reused development evidence): exactly scratch_gap/bench_eval.py (bias tuned on all August rows).
Per variant against REF: macro-F1, paired event bootstrap of the difference (200 Poisson draws, 5-95 %), 12-class precision /
recall / F1, and row + independent-event counts of: missed fraud recovered (REF legit, truth fraud, variant = truth), missed
fraud flagged with any fraud class, new false positives (REF legit, truth legit, variant fraud), false positives removed,
correct fraud damaged (REF = truth > 0, variant != truth), of which sent to legit. Events of fraud rows =
clean_labels.fraud_events(g=48 h) ids; events of legit rows = the same linking (sender, receiver key, device, 48 h) applied to
the row set itself. Prediction files: scratch_atomic/verification/P<B>_<v>_<m>.npy first, else scratch_gap/. Decodes are
cached in scratch_atomic/verification/dec/. Nothing outside scratch_atomic/verification is written.
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_score.py N|D REF VAR1,VAR2 [aug|sep|aug,sep]
       python scratch_atomic/verification/va_score.py summary N|D REF VAR1,VAR2 [aug|sep|aug,sep]   (one markdown row per variant / month / stage)
       python scratch_atomic/verification/va_score.py check     (control = stored reference; decode = stored bench_eval decode)"""
import os, sys
import numpy as np
import polars as pl
import train as T
from common import load_all, LABEL2ID as L
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES
from ts_judge_post import zones, override

c = pl.col; K = T.K; HERE = "scratch_atomic/verification"
os.makedirs(f"{HERE}/dec", exist_ok=True)


def keys():
    """Slim table of the labelled rows with the 48 h fraud event id (null on legit rows); cached."""
    f = f"{HERE}/keys.parquet"
    if not os.path.exists(f):
        tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
        tx.select("txn_id", "ts", "y", "sender_id", "device_id", "receiver_type", "receiver_id") \
          .join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left").write_parquet(f)
    return pl.read_parquet(f)


def path(B, v, m):
    p = f"{HERE}/P{B}_{v}_{m}.npy"
    return p if os.path.exists(p) else f"scratch_gap/P{B}_{v}_{m}.npy"


def force(P, D):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


class Bench:
    def __init__(self, B):
        self.B = B; KY = keys(); self.KY = KY
        self.D = {m: pl.read_parquet(f"scratch_gap/{B}_{m}_rows.parquet") for m in ("aug", "sep")}
        self.y = {m: d["y_true"].to_numpy().astype(np.int64) for m, d in self.D.items()}
        self.ev = {m: d.select("txn_id").join(KY.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
                   for m, d in self.D.items()}
        ev, n = self.ev["aug"], len(self.y["aug"])  # same draws as aug_dec.py
        rng = np.random.default_rng(2026); evh = rng.integers(0, 2, int(np.nanmax(ev)) + 1)
        self.half = np.where(np.isnan(ev), rng.integers(0, 2, n), evh[np.nan_to_num(ev).astype(np.int64)])
        self.rule = {m: d["ts_bait__rule"].fill_null(0).to_numpy() for m, d in self.D.items()}

    def decode(self, v, m):
        f = f"{HERE}/dec/{self.B}_{v}_{m}.npz"
        if os.path.exists(f) and os.path.getmtime(f) > max(os.path.getmtime(path(self.B, v, k)) for k in {"aug", m}):  # September decodes also use the August file (bias)
            z = np.load(f); return z["pre"], z["q"]
        Da, ya = self.D["aug"], self.y["aug"]; Pa = np.load(path(self.B, v, "aug")); Fa = force(Pa, Da)
        if m == "aug":
            pre, q = np.zeros(len(ya), np.int64), np.zeros(len(ya), np.int64)
            for a, b in ((0, 1), (1, 0)):
                A_, B_ = self.half == a, self.half == b
                pre[B_] = T.decode(Pa[B_], T.tune_bias(Pa[A_], ya[A_]))
                p1 = T.decode(Fa, T.tune_bias(Fa[A_], ya[A_])); z, _ = zones(Da, p1, TR_EDGES[0], TR_EDGES[-1])
                q[B_] = override(p1, z, self.rule["aug"])[B_]
        else:
            Ds = self.D["sep"]; Ps = np.load(path(self.B, v, "sep"))
            pre = T.decode(Ps, T.tune_bias(Pa, ya)); p1 = T.decode(force(Ps, Ds), T.tune_bias(Fa, ya))
            z, _ = zones(Ds, p1, TR_EDGES[0], TR_EDGES[-1]); q = override(p1, z, self.rule["sep"])
        np.savez(f, pre=pre, q=q); return pre, q

    def nev(self, m, mask):
        """Independent events among the masked rows: label events for fraud rows + 48 h key-linked components of legit rows."""
        if not mask.any(): return 0
        e = self.ev[m][mask]; fr = int(len(np.unique(e[~np.isnan(e)])))
        ids = self.D[m].select("txn_id").filter(pl.Series(mask & np.isnan(self.ev[m])))
        if ids.height == 0: return fr
        t = self.KY.join(ids, on="txn_id", how="semi").with_columns(pl.lit(1, pl.Int8).alias("y"))
        return fr + fraud_events(t, g=48 * 3600)[1]["ev"].n_unique()


def f1w(y, p, w):
    ok = p == y
    return 2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)


def prf(y, q):
    tp = np.bincount(y[q == y], minlength=K); pr = np.bincount(q, minlength=K); n = np.bincount(y, minlength=K)
    return tp / np.maximum(pr, 1), tp / np.maximum(n, 1), 2 * tp / np.maximum(pr + n, 1), pr


def report(Bn, R, VARS, months):
    S = Bench(Bn)
    for m in months:
        y, ev = S.y[m], S.ev[m]; n = len(y); ref = dict(zip(("pre", "q"), S.decode(R, m)))
        grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026)
        boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]
        print(f"\n=== benchmark {Bn}, {'August (out-of-half bias)' if m == 'aug' else 'September (August bias)'}: rows {n}, fraud rows {int((y > 0).sum())},"
              f" fraud events {len(np.unique(grp[grp >= 0]))}; reference {R} ===")
        for st in ("pre", "q"):
            a = ref[st]
            print(f"reference {R} {st}: macro-F1 {T.macro_f1(y, a).mean():.4f} | missed fraud rows {int(((y > 0) & (a == 0)).sum())} in {S.nev(m, (y > 0) & (a == 0))} events"
                  f" | legit called fraud {int(((y == 0) & (a > 0)).sum())}")
        if m == "aug":
            for st in ("pre", "q"):
                print(f"reference {R} {st} per half (aug_dec.py numbers): " + ", ".join(f"{T.macro_f1(y[S.half == h], ref[st][S.half == h]).mean():.4f}" for h in (1, 0)))
        for v in VARS:
            var = dict(zip(("pre", "q"), S.decode(v, m)))
            for st in ("pre", "q"):
                a, b = ref[st], var[st]; fa, fb = T.macro_f1(y, a), T.macro_f1(y, b); ch = a != b
                bd = np.quantile([f1w(y, b, w).mean() - f1w(y, a, w).mean() for w in boot], [.05, .95])
                hv = "" if m == "sep" else " | mean of halves " + f"{np.mean([T.macro_f1(y[S.half == h], b[S.half == h]).mean() for h in (0, 1)]):.4f}" \
                     + f" vs {np.mean([T.macro_f1(y[S.half == h], a[S.half == h]).mean() for h in (0, 1)]):.4f}"
                rec, anyf = (a == 0) & (y > 0) & (b == y), (a == 0) & (y > 0) & (b > 0)
                nfp, fpr = (a == 0) & (y == 0) & (b > 0), (a > 0) & (y == 0) & (b == 0)
                dmg = (a == y) & (y > 0) & (b != y); dl = dmg & (b == 0)
                print(f"\n{Bn} {m} {v} vs {R} [{st}] macro-F1 {fb.mean():.4f} vs {fa.mean():.4f} = {fb.mean() - fa.mean():+.4f} (event bootstrap 5-95 %: {bd[0]:+.4f}, {bd[1]:+.4f}){hv}")
                print(f"  rows changed {int(ch.sum())} corrected {int((ch & (b == y)).sum())} newly wrong {int((ch & (a == y)).sum())}"
                      f" | missed fraud recovered {int(rec.sum())} rows / {S.nev(m, rec)} events (flagged any fraud class {int(anyf.sum())} / {S.nev(m, anyf)})"
                      f" | new false positives {int(nfp.sum())} rows / {S.nev(m, nfp)} events (false positives removed {int(fpr.sum())} / {S.nev(m, fpr)})"
                      f" | correct fraud damaged {int(dmg.sum())} rows / {S.nev(m, dmg)} events (to legit {int(dl.sum())} / {S.nev(m, dl)})")
                pa, pb = prf(y, a), prf(y, b)
                print(f"  {'class':20s} {'n':>7s} | {'ref P':>6s} {'R':>6s} {'F1':>6s} {'npred':>7s} | {'var P':>6s} {'R':>6s} {'F1':>6s} {'npred':>7s} | {'dP':>6s} {'dR':>6s} {'dF1':>7s}")
                for k in range(K):
                    print(f"  {T.LABELS[k]:20s} {int((y == k).sum()):7d} | {pa[0][k]:6.3f} {pa[1][k]:6.3f} {pa[2][k]:6.3f} {pa[3][k]:7d} | {pb[0][k]:6.3f} {pb[1][k]:6.3f} {pb[2][k]:6.3f} {pb[3][k]:7d} |"
                          f" {pb[0][k] - pa[0][k]:+6.3f} {pb[1][k] - pa[1][k]:+6.3f} {pb[2][k] - pa[2][k]:+7.4f}")
            sys.stdout.flush()


def summary(Bn, R, VARS, months):
    """One markdown row per variant, month and stage, from the cached decodes (same definitions as report)."""
    S = Bench(Bn)
    print("| benchmark | month | stage | variant | reference | macro-F1 ref | macro-F1 variant | change (event bootstrap 5..95 %) | rows changed / corrected / newly wrong |"
          " missed fraud recovered rows / events | new false positives rows / events | false positives removed | correct fraud damaged rows / events (to legit) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in months:
        y, ev = S.y[m], S.ev[m]; n = len(y); ref = dict(zip(("pre", "q"), S.decode(R, m)))
        grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026)
        boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]
        for v in VARS:
            var = dict(zip(("pre", "q"), S.decode(v, m)))
            for st in ("pre", "q"):
                a, b = ref[st], var[st]; ch = a != b; fa, fb = T.macro_f1(y, a).mean(), T.macro_f1(y, b).mean()
                bd = np.quantile([f1w(y, b, w).mean() - f1w(y, a, w).mean() for w in boot], [.05, .95])
                rec, nfp, fpr = (a == 0) & (y > 0) & (b == y), (a == 0) & (y == 0) & (b > 0), (a > 0) & (y == 0) & (b == 0)
                dmg = (a == y) & (y > 0) & (b != y); dl = dmg & (b == 0)
                print(f"| {Bn} | {m} | {st} | {v} | {R} | {fa:.4f} | {fb:.4f} | {fb - fa:+.4f} ({bd[0]:+.4f}..{bd[1]:+.4f}) | {int(ch.sum())} / {int((ch & (b == y)).sum())} / {int((ch & (a == y)).sum())} |"
                      f" {int(rec.sum())} / {S.nev(m, rec)} | {int(nfp.sum())} / {S.nev(m, nfp)} | {int(fpr.sum())} / {S.nev(m, fpr)} | {int(dmg.sum())} / {S.nev(m, dmg)} ({int(dl.sum())} / {S.nev(m, dl)}) |", flush=True)


def check():
    for Bn, R, dec in (("N", "C", "dec_N_wt.npz"), ("D", "A_C", "dec_D_wt.npz")):
        for m in ("aug", "sep"):
            f = f"{HERE}/P{Bn}_vaC_{m}.npy"
            if os.path.exists(f):
                a, b = np.load(f), np.load(f"scratch_gap/P{Bn}_{R}_{m}.npy")
                print(f"{Bn} vaC vs {R} {m}: shape {a.shape} identical {np.array_equal(a, b)} max abs diff {np.abs(a - b).max():.3e}")
            else: print(f"{Bn} vaC {m}: control file not present yet")
        S = Bench(Bn); pre, q = S.decode(R, "sep"); z = np.load(f"scratch_gap/{dec}")
        print(f"{Bn} {R} September decode vs {dec}: y equal {np.array_equal(S.y['sep'], z['y'])}, pre identical {np.array_equal(pre, z[f'{R}_pre'])}, q identical {np.array_equal(q, z[f'{R}_q'])}")
        pre, q = S.decode(R, "aug"); ya = S.y["aug"]
        for st, p in (("before", pre), ("after", q)):
            h = [T.macro_f1(ya[S.half == b], p[S.half == b]).mean() for b in (1, 0)]
            print(f"{Bn} {R} August {st} decoder {np.mean(h):.4f} ({h[0]:.4f}, {h[1]:.4f})   (compare with scratch_gap/aug_dec_*.log)")


if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    elif sys.argv[1] == "summary": summary(sys.argv[2], sys.argv[3], sys.argv[4].split(","), (sys.argv[5] if len(sys.argv) > 5 else "aug,sep").split(","))
    else: report(sys.argv[1], sys.argv[2], sys.argv[3].split(","), (sys.argv[4] if len(sys.argv) > 4 else "aug,sep").split(","))
