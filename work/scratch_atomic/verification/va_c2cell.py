"""Agent A's C2 QR cell (QR to merchant, qr_vpa_mismatch == 1, non-integer amount) inside the rows the reference decodes legit:
can the reference's own qr_swap probability rank the fraud-labelled rows of the cell? Top-k precision by P(qr_swap), August only
(freeze month), then September. usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_c2cell.py"""
import numpy as np
import polars as pl
from common import LABEL2ID as L
from va_score import Bench, path

F = pl.read_parquet("scratch_atomic/verification/feat/c2_ALL.parquet")
for Bn, R in (("N", "C"), ("D", "A_C")):
    S = Bench(Bn)
    for m in ("aug", "sep"):
        y = S.y[m]; q = S.decode(R, m)[1]; P = np.load(path(Bn, R, m))
        f = S.D[m].select("txn_id").join(F, on="txn_id", how="left", maintain_order="left")["va_c2__qr"].to_numpy() == 1
        g = f & (q == 0); o = np.argsort(-P[g, L["qr_swap"]]); yy = y[g][o]
        print(f"{Bn} {m}: cell rows decoded legit {int(g.sum())}, qr_swap-labelled {int((yy == L['qr_swap']).sum())}, any fraud label {int((yy > 0).sum())};"
              f" cell rows decoded qr_swap {int((f & (q == L['qr_swap'])).sum())} of which labelled qr_swap {int((f & (q == L['qr_swap']) & (y == L['qr_swap'])).sum())};"
              " top-k by P(qr_swap) -> qr_swap labels: " + ", ".join(f"k={k}: {int((yy[:k] == L['qr_swap']).sum())}" for k in (10, 20, 50, 100)))
