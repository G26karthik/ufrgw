"""Audit of agent A's section 8.5 (label-informed predicate: P2P receiver held a fraud label in an earlier month).
Whole held month, reference decode after the decoder. Uses only labels of months before the held month (August: July labels;
September: July + August labels), i.e. labels the fold's fit already has. Questions: (1) inside decoded-legit rows, is the fraud
share of predicate rows above the base share? (2) inside decoded-fraud rows, does the receiver's earlier majority class beat the
decoded class where they differ? usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_recur.py"""
import numpy as np
import polars as pl
from va_score import Bench

c = pl.col
for Bn, R in (("N", "C"), ("D", "A_C")):
    S = Bench(Bn); KY = S.KY.with_columns(pl.from_epoch("ts", "s").dt.month().alias("month"))
    for m, mv in (("aug", 8), ("sep", 9)):
        H = (KY.filter((c("month") < mv) & (c("y") > 0) & (c("receiver_type") == "P2P")).group_by("receiver_id", "y").len().sort("len", "y", descending=[True, False])
               .group_by("receiver_id", maintain_order=True).agg(c("y").first().alias("hist"), c("len").sum().alias("nh")))
        X = S.D[m].select("txn_id").join(KY.select("txn_id", "receiver_id", "receiver_type"), on="txn_id", how="left", maintain_order="left").join(H, on="receiver_id", how="left", maintain_order="left")
        p = ((X["receiver_type"] == "P2P") & X["hist"].is_not_null()).to_numpy(); hist = X["hist"].fill_null(0).to_numpy().astype(np.int64)
        y = S.y[m]; q = S.decode(R, m)[1]; dl, df = q == 0, q > 0
        print(f"{Bn} {m}: predicate rows {int(p.sum())}, fraud-labelled {int((p & (y > 0)).sum())} ({(p & (y > 0)).sum() / p.sum():.3f}; month base {(y > 0).mean():.4f})"
              f" | decoded legit: predicate rows {int((p & dl).sum())}, fraud-labelled {int((p & dl & (y > 0)).sum())} = {(p & dl & (y > 0)).sum() / max((p & dl).sum(), 1):.4f}"
              f" (base in decoded legit {(dl & (y > 0)).sum() / dl.sum():.4f}; P2P decoded-legit base {((X['receiver_type'] == 'P2P').to_numpy() & dl & (y > 0)).sum() / max(((X['receiver_type'] == 'P2P').to_numpy() & dl).sum(), 1):.4f})"
              f" | decoded fraud: predicate rows {int((p & df).sum())}, earlier class != decoded class on {int((p & df & (hist != q)).sum())}:"
              f" decoded right {int((p & df & (hist != q) & (q == y)).sum())}, earlier class right {int((p & df & (hist != q) & (hist == y)).sum())}, neither {int((p & df & (hist != q) & (q != y) & (hist != y)).sum())}")
