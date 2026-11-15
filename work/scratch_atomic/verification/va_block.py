"""Decoded labels of the reference inside the device classes of notes/fresh_errors.md, whole held month, N and D.
Question: does the reference call legit rows of the legit fresh-phone region (device number >= last zone edge) fraud?
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_block.py"""
import numpy as np
import polars as pl
from feats.ts_event import TR_EDGES
from va_score import Bench

for Bn, R in (("N", "C"), ("D", "A_C")):
    S = Bench(Bn)
    for m in ("aug", "sep"):
        y = S.y[m]; dn = S.D[m]["dn"].to_numpy(); sn = S.D[m].select("txn_id").join(S.KY.select("txn_id", pl.col("sender_id").str.slice(2).cast(pl.Int64).alias("sn")), on="txn_id", how="left", maintain_order="left")["sn"].to_numpy()
        pre, q = S.decode(R, m)
        cls = np.select([dn == sn, dn < 420000, dn < TR_EDGES[0], dn < TR_EDGES[-1]], ["own", "base_other", "pool", "zone"], "fresh")
        print(f"\n{Bn} {m} (zone edges {TR_EDGES[0]}..{TR_EDGES[-1]}): device class | rows | fraud-labelled | decoded fraud pre / q | legit-labelled decoded fraud pre / q | fraud-labelled decoded legit q")
        for k in ("own", "base_other", "pool", "zone", "fresh"):
            g = cls == k
            print(f"  {k:10s} {int(g.sum()):7d} {int((g & (y > 0)).sum()):6d} | {int((g & (pre > 0)).sum()):5d} / {int((g & (q > 0)).sum()):5d} | {int((g & (y == 0) & (pre > 0)).sum()):4d} / {int((g & (y == 0) & (q > 0)).sum()):4d} | {int((g & (y > 0) & (q == 0)).sum()):6d}")
