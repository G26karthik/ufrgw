"""Agent A: the receiver's earlier fraud class as a LABEL-INFORMED input to the type decision (measurement only, no fits).
cwd=work, PYTHONPATH=.  python scratch_atomic/representation/14_hist_class.py > scratch_atomic/representation/14_out.txt
History = supplied labels of earlier months only (August rows: July; September rows: July + August), P2P rows, keyed on the
receiver account. hist_class = majority class of the receiver's earlier fraud-labelled rows (ties: none), hist_n = their count,
hist_share = majority share. Decodes: held_N.parquet (C_q) and held_D.parquet (h6_q) written by 01_missed.py.
Signature (never overridden) = takeover archetype, zone device (static edges + pool), bait rule 1 / 2, or final label != forced decode.
Rule R(n0, s0): decoded-fraud predicate rows without a signature take hist_class when hist_n >= n0 and hist_share >= s0.
(n0, s0) is chosen on AUGUST only by whole-month macro-F1 and applied once to September."""
import numpy as np
import polars as pl
import train as T
from common import load_all, LABELS
from feats.ts_event import zone_ids

c = pl.col; OUT = "scratch_atomic/representation/"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
tx = load_all()
tr = tx.filter(~c("is_test")).select("txn_id", "month", "y", "receiver_id", "receiver_type")


def history(months):
    h = tr.filter(c("month").is_in(months) & (c("y") > 0) & (c("receiver_type") == "P2P")).group_by("receiver_id", "y").len()
    h = h.sort("len", descending=True).group_by("receiver_id", maintain_order=True).agg(
        c("y").first().alias("top"), c("len").first().alias("ntop"), c("len").sum().alias("hist_n"), (c("len") == c("len").max()).sum().alias("nties"))
    return h.with_columns(pl.when(c("nties") == 1).then(c("top")).otherwise(0).cast(pl.Int64).alias("hist_class"), (c("ntop") / c("hist_n")).alias("hist_share")).select("receiver_id", "hist_class", "hist_n", "hist_share")


def f1(y, q): return T.macro_f1(y, q)


for B in ("N", "D"):
    H = pl.read_parquet(f"{OUT}held_{B}.parquet")
    res = {}
    for m, src, key in ((8, [7], "aug"), (9, [7, 8], "sep")):
        R = pl.read_parquet(f"scratch_gap/{B}_{key}_rows.parquet").select("txn_id", "receiver_type", "dn", "takeover__arch_sim", "takeover__arch_ato", "ts_bait__rule")
        D = H.filter(c("m") == m).join(R, on="txn_id", how="left").join(history(src), on="receiver_id", how="left")
        z = zone_ids(D["dn"].to_numpy(), np.zeros(D.height, bool), [2 ** 62] * 12)
        D = D.with_columns(pl.Series("zone", z.astype(np.int64))).with_columns(
            ((c("takeover__arch_sim").fill_nan(None) == 1).fill_null(False) | (c("takeover__arch_ato").fill_nan(None) == 1).fill_null(False) | (c("zone") > 0)
             | c("ts_bait__rule").fill_null(0).is_in([1, 2]) | (c("q") != c("p1"))).alias("sig"),
            ((c("receiver_type") == "P2P") & c("hist_n").is_not_null()).alias("pred"), c("hist_class").fill_null(0), c("hist_n").fill_null(0), c("hist_share").fill_null(0.0))
        y = D["y_true"].to_numpy().astype(np.int64); q = D["q"].to_numpy().astype(np.int64)
        P = D.filter(c("pred") & (c("q") > 0)).with_columns(
            (c("y_true") > 0).alias("fraud_lab"), (c("q") == c("y_true")).alias("dec_ok"), ((c("hist_class") > 0) & (c("hist_class") == c("y_true"))).alias("hist_ok"),
            pl.when(c("hist_n") >= 3).then(pl.lit("3+")).otherwise(c("hist_n").cast(pl.String)).alias("n_b"),
            pl.when(c("hist_share") >= .999).then(pl.lit("1.0")).when(c("hist_share") >= .6).then(pl.lit(".6-.9")).otherwise(pl.lit("<.6")).alias("s_b"))
        print(f"\n######## benchmark {B}, month {m} (history months {src}): held rows {D.height}, predicate rows {int(D['pred'].sum())}, decoded-fraud predicate rows {P.height}; "
              f"macro-F1 unchanged decode {f1(y, q).mean():.4f}")
        agg = [pl.len().alias("n"), c("fraud_lab").mean().round(3).alias("fraud_lab"), c("dec_ok").mean().round(3).alias("dec==label"), c("hist_ok").mean().round(3).alias("hist==label"),
               (c("q") == c("hist_class")).mean().round(3).alias("dec==hist"),
               ((c("q") != c("hist_class")) & (c("hist_class") > 0)).sum().alias("differ"),
               ((c("q") != c("hist_class")) & (c("hist_class") > 0) & c("dec_ok") & ~c("hist_ok")).sum().alias("dec_right_hist_wrong"),
               ((c("q") != c("hist_class")) & (c("hist_class") > 0) & ~c("dec_ok") & c("hist_ok")).sum().alias("dec_wrong_hist_right"),
               ((c("q") != c("hist_class")) & (c("hist_class") > 0) & ~c("dec_ok") & ~c("hist_ok")).sum().alias("both_wrong")]
        print("all decoded-fraud predicate rows:"); print(P.select(agg))
        hs_, ls_ = P["hist_class"].to_numpy(), P["y_true"].to_numpy(); n_ = len(hs_)
        print("   hist==label expected if hist_class were independent of the label (product of the two class distributions): %.3f; same for decoder class: %.3f" % (
            sum((hs_ == k).mean() * (ls_ == k).mean() for k in range(1, 12)), sum((P["q"].to_numpy() == k).mean() * (ls_ == k).mean() for k in range(1, 12))))
        for g in ("sig", "n_b", "s_b"):
            print(f"by {g}:"); print(P.group_by(g).agg(agg).sort(g))
        print("no-signature rows by hist_n x hist_share:"); print(P.filter(~c("sig")).group_by("n_b", "s_b").agg(agg).sort("n_b", "s_b"))
        res[m] = (D, y, q)
    # ---- rule R(n0, s0): chosen on August only
    def apply(D, q, n0, s0):
        k = D.select((c("pred") & (c("q") > 0) & ~c("sig") & (c("hist_class") > 0) & (c("hist_n") >= n0) & (c("hist_share") >= s0 - 1e-9)).alias("k"))["k"].to_numpy()
        q2 = q.copy(); q2[k] = D["hist_class"].to_numpy()[k]; return q2
    Da, ya, qa = res[8]; base = f1(ya, qa).mean(); best = None
    print(f"\n==== benchmark {B}: rule grid on AUGUST (macro-F1 unchanged {base:.4f})")
    for n0 in (1, 2, 3):
        for s0 in (.6, .8, 1.0):
            q2 = apply(Da, qa, n0, s0); v = f1(ya, q2).mean(); ch = q2 != qa
            print(f"  R({n0}, {s0}): macro-F1 {v:.4f} (change {v - base:+.4f}); rows changed {int(ch.sum())}, corrected {int((ch & (q2 == ya)).sum())}, newly wrong {int((ch & (qa == ya)).sum())}")
            if best is None or v > best[0] + 1e-9: best = (v, n0, s0)
    if best[0] <= base + 1e-9:
        print(f"  no setting is above the unchanged decode on August: stop, September not scored for the rule."); continue
    _, n0, s0 = best; print(f"  chosen on August: R({n0}, {s0})")
    for m in (8, 9):
        D, y, q = res[m]; q2 = apply(D, q, n0, s0); ch = q2 != q; a, b = f1(y, q), f1(y, q2)
        ev = D["ev"].to_numpy(); evc = ev[ch]; n_ev = len(set(evc[~np.isnan(evc.astype(float))].tolist())) + int(np.isnan(evc.astype(float)).sum())
        print(f"  month {m}: macro-F1 {a.mean():.4f} -> {b.mean():.4f} ({b.mean() - a.mean():+.4f}); rows changed {int(ch.sum())}, corrected {int((ch & (q2 == y)).sum())}, "
              f"newly wrong {int((ch & (q == y)).sum())}, wrong before and after {int((ch & (q != y) & (q2 != y)).sum())}; independent events changed {n_ev}")
        print("     per-class F1 change: " + ", ".join(f"{LABELS[k]} {b[k] - a[k]:+.4f}" for k in range(12) if abs(b[k] - a[k]) > 5e-5))
        print("     changed rows, from -> to (top):", pl.DataFrame({"f": q[ch], "t": q2[ch], "ok": (q2 == y)[ch]}).group_by("f", "t").agg(pl.len().alias("n"), c("ok").sum().alias("right_after")).sort("n", descending=True).head(8).rows())

# ---- October count (no labels inferred): P2P rows whose receiver holds >= 1 fraud label in July to September
hs = history([7, 8, 9])
te = tx.filter(c("is_test") & (c("receiver_type") == "P2P")).join(hs, on="receiver_id", how="inner")
print(f"\n==== October / test rows matching the predicate (history = July to September labels): {te.height} of {tx.filter(c('is_test')).height} test rows "
      f"({tx.filter(c('is_test') & (c('receiver_type') == 'P2P')).height} P2P); hist_n >= 2: {te.filter(c('hist_n') >= 2).height}; hist_n >= 3: {te.filter(c('hist_n') >= 3).height}; "
      f"distinct receivers {te['receiver_id'].n_unique()}")
