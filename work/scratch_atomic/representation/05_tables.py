"""Compact markdown tables from predicate_rates.csv (written by 01_missed.py). cwd=work."""
import polars as pl
c = pl.col
d = pl.read_csv("scratch_atomic/representation/predicate_rates.csv")
for B in ("N", "D"):
    w = d.filter(c("bench") == B).pivot(on="month", index="predicate", values=["missed_ev_n", "missed_ev_rate", "matched_ctl_rate", "all_legit_rate", "detected_ev_rate"])
    print(f"\nbenchmark {B}: predicate | missed events Aug n (rate) / matched ctl / all legit / detected | same for Sep")
    for r in w.sort("missed_ev_rate_8", descending=True).iter_rows(named=True):
        print(f"| {r['predicate']} | {r['missed_ev_n_8']} ({r['missed_ev_rate_8']:.4f}) | {r['matched_ctl_rate_8']:.4f} | {r['all_legit_rate_8']:.4f} | {r['detected_ev_rate_8']:.4f} "
              f"| {r['missed_ev_n_9']} ({r['missed_ev_rate_9']:.4f}) | {r['matched_ctl_rate_9']:.4f} | {r['all_legit_rate_9']:.4f} | {r['detected_ev_rate_9']:.4f} |")
