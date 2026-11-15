"""Audit instrument: how a label-free predicate or numeric column separates MISSED fraud from legit rows inside the rows the
reference pipeline decodes as legit (after the decoder, `_q`), on the whole held month, August and September separately.
Predicate (e.g. "x >= 3"): share flagged among missed fraud rows and among their independent events (fraud_events 48 h, an
  event counts once if any of its missed rows is flagged), share flagged among legit rows decoded legit (all of them, not a
  sample), the same legit share re-weighted to the missed-fraud distribution of the reference P(legit) (20 quantile bins of the
  missed rows = 'matched on baseline confidence'), lift without and with matching, and the precision a rule would have:
  fraud share among flagged decoded-legit rows against the base fraud share of decoded-legit rows. Flagged missed rows by class.
Numeric column (bare name): AUC missed fraud against legit, unmatched, and the missed-count-weighted mean of within-bin AUCs.
July has no held-out decode: all fraud-labelled rows against all legit rows (label-only stand-in), reported for reference.
The feature parquet must hold txn_id + columns for the scored rows (va_feat.py output, or an agent's all-rows file).
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_pred.py N|D FEATS.parquet[,FEATS_sep.parquet] "expr" ["expr" ...]"""
import sys
import numpy as np
import polars as pl
import train as T
from va_score import Bench, path, keys

c = pl.col


def auc(x, pos):
    ok = ~np.isnan(x); x, pos = x[ok], pos[ok]
    if pos.sum() == 0 or (~pos).sum() == 0: return np.nan
    r = pl.Series(x).rank().to_numpy()
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    Bn, files, exprs = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]; R = "C" if Bn == "N" else "A_C"; S = Bench(Bn)
    for i, m in enumerate(("jul", "aug", "sep")):
        F = pl.read_parquet(files[min(max(i - 1, 0), len(files) - 1)])
        if m == "jul":
            KY = keys().filter(pl.from_epoch("ts", "s").dt.month() == 7); ids = KY.select("txn_id")
            y = KY["y"].to_numpy().astype(np.int64); ev = KY["ev"].to_numpy().astype(float); pool = np.ones(len(y), bool); pl_ = None
        else:
            ids = S.D[m].select("txn_id"); y, ev = S.y[m], S.ev[m]; pool = S.decode(R, m)[1] == 0; pl_ = np.load(path(Bn, R, m))[:, 0].astype(np.float64)
        X = ids.join(F, on="txn_id", how="left", maintain_order="left").with_columns(pl.selectors.float().fill_nan(None))  # NaN >= k is true in polars
        miss, leg = pool & (y > 0), pool & (y == 0)
        print(f"\n=== {Bn} {m}: {'all rows (no decode exists)' if m == 'jul' else 'rows decoded legit by ' + R + ' after the decoder'} {int(pool.sum())},"
              f" {'fraud-labelled' if m == 'jul' else 'missed fraud'} rows {int(miss.sum())} in {len(np.unique(ev[miss]))} events, legit rows {int(leg.sum())},"
              f" base fraud share {miss.sum() / pool.sum():.4f}; feature rows missing {X.select(pl.all().exclude('txn_id').is_null().all()).row(0)} ===")
        if pl_ is not None:
            b = np.searchsorted(np.quantile(pl_[miss], np.linspace(0, 1, 21)[1:-1]), pl_); nb = b.max() + 1
            wm = np.bincount(b[miss], minlength=nb) / miss.sum(); nl = np.maximum(np.bincount(b[leg], minlength=nb), 1)
        for e in exprs:
            v = X.select(pl.sql_expr(e).alias("v"))["v"]
            if v.dtype == pl.Boolean:
                f = v.fill_null(False).to_numpy(); fm, fl = f & miss, f & leg
                line = (f"{e!r}: missed fraud flagged {int(fm.sum())} / {int(miss.sum())} rows = {fm.sum() / max(miss.sum(), 1):.4f}, events {len(np.unique(ev[fm]))} / {len(np.unique(ev[miss]))}"
                        f" | legit flagged {int(fl.sum())} / {int(leg.sum())} = {fl.sum() / max(leg.sum(), 1):.4f} | lift {(fm.sum() / max(miss.sum(), 1)) / max(fl.sum() / max(leg.sum(), 1), 1e-12):.2f}")
                if pl_ is not None:
                    rl = float((wm * np.bincount(b[fl], minlength=nb) / nl).sum())
                    line += f" | legit share matched on P(legit) {rl:.4f}, matched lift {(fm.sum() / max(miss.sum(), 1)) / max(rl, 1e-12):.2f}"
                line += f" | fraud share among flagged {fm.sum() / max((fm | fl).sum(), 1):.4f} (base {miss.sum() / pool.sum():.4f})"
                print(line); print("    flagged missed rows by class: " + " ".join(f"{T.LABELS[k][:6]} {int((fm & (y == k)).sum())}/{int((miss & (y == k)).sum())}" for k in range(1, T.K)))
            else:
                x = v.cast(pl.Float64).fill_null(float("nan")).to_numpy(); line = f"{e!r}: AUC missed fraud vs legit {auc(x[pool], miss[pool]):.4f}"
                if pl_ is not None:
                    a = [(auc(x[pool & (b == j)], miss[pool & (b == j)]), wm[j]) for j in range(nb)]; a = [(u, w) for u, w in a if not np.isnan(u)]
                    line += f" | matched on P(legit) (20 bins) {sum(u * w for u, w in a) / sum(w for _, w in a):.4f}"
                print(line + " | per class: " + " ".join(f"{T.LABELS[k][:6]} {auc(x[leg | (miss & (y == k))], (miss & (y == k))[leg | (miss & (y == k))]):.3f}" for k in range(1, T.K)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
