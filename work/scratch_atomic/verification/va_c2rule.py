"""Mechanism `c2` (agent A's C2, the only testable form A names): a lower qr_swap threshold inside the label-free cell
(txn_type == "QR") & (receiver_type == "MERCHANT") & (qr_vpa_mismatch == 1) & (amount != round(amount)).
Executable form: the reference probability of qr_swap is multiplied by F on the rows of the cell and the row is renormalised;
everything else (bias tuning, forcing, zones, bait rule) is the unchanged decoder. No model is fitted.
Writes scratch_atomic/verification/P<B>_vac2f<F>_<aug|sep>.npy from the stored references (N: PN_C, D: PD_A_C); never overwrites.
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_c2rule.py F [F ...]"""
import os, sys
import numpy as np
import polars as pl
from common import LABEL2ID as L

HERE = "scratch_atomic/verification"; C = pl.read_parquet(f"{HERE}/feat/c2_ALL.parquet").select("txn_id", "va_c2__qr")  # built by va_feat.py from mech_c2.py
for B, R in (("N", "C"), ("D", "A_C")):
    for m in ("aug", "sep"):
        cell = pl.read_parquet(f"scratch_gap/{B}_{m}_rows.parquet", columns=["txn_id"]).join(C, on="txn_id", how="left", maintain_order="left")["va_c2__qr"].to_numpy() == 1
        P = np.load(f"scratch_gap/P{B}_{R}_{m}.npy")
        for f in sys.argv[1:]:
            out = f"{HERE}/P{B}_vac2f{f}_{m}.npy"; assert not os.path.exists(out), out
            Q = P.astype(np.float64).copy(); Q[cell, L["qr_swap"]] *= float(f); Q /= Q.sum(1, keepdims=True)
            np.save(out, Q.astype(np.float32)); print(f"{out}: cell rows {int(cell.sum())} of {len(cell)}")
