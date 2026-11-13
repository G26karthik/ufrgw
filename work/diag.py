"""Leaderboard diagnostic subs: h6_ppq with every fraud-class bias shifted by DELTA (more/less fraud predicted).
The public LB response to DELTA tells whether clean test wants a more aggressive fraud operating point than noisy Sep.
usage (cwd=work): python diag.py TAG DELTA [DELTA ...]  -> subs/TAG_d<DELTA>.csv
"""
import sys
import numpy as np
import polars as pl
from common import ROOT, load_all
from postproc import apply
from train import tune_bias, decode, macro_f1, write_sub, SAMPLE
from clean_labels import clean

tag, deltas = sys.argv[1], [float(d) for d in sys.argv[2:]]
tx = load_all()
ids = pl.read_parquet(ROOT / "oof" / f"{tag}_ids.parquet")
y = ids["y"].to_numpy(); sep = ids["fold"].to_numpy() == 1
yc = ids.join(tx.select("txn_id", clean(tx).alias("yc")), on="txn_id", how="left", maintain_order="left")["yc"].to_numpy()
P = apply(np.load(ROOT / "oof" / f"{tag}.npy"), ids, tx, True)
T = apply(np.load(ROOT / "preds" / f"{tag}_test.npy"), pl.read_csv(SAMPLE).select(pl.col("id").alias("txn_id")), tx, True)
b = tune_bias(P, y)
for d in [0.0] + deltas:
    bd = b.copy(); bd[1:] += d
    pr = decode(P, bd); nt = int((decode(T, bd) != 0).sum())
    print(f"delta={d:+.1f} Sep noisy={macro_f1(y[sep], pr[sep]).mean():.4f} cleaned={macro_f1(yc[sep], pr[sep]).mean():.4f} "
          f"test fraud rows={nt}", flush=True)
    if d:
        write_sub(T, bd, f"{tag}_d{d:+.1f}")
