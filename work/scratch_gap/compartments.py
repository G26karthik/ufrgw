"""Error compartments of the final decode on the natural benchmark (Sep, supplied labels), and what an oracle fix of
each compartment alone would be worth. Structural support = relabel.support (the row shows its class's event signature).
Also prints the type precision of the final decode among fraud-labelled rows (input of budget.py scenario P).
usage (cwd=work, PYTHONPATH=.): python scratch_gap/compartments.py [N|D]"""
import sys
import numpy as np
import polars as pl
import relabel as R
import train as T
from common import load_all

c = pl.col; K = T.K; B = sys.argv[1] if len(sys.argv) > 1 else "N"
z = np.load(f"scratch_gap/dec_{B}.npz"); y, q = z["y"], z["C_q"]
ids = pl.read_parquet(f"scratch_gap/{B}_sep_rows.parquet").select("txn_id")
tx = load_all().join(ids, on="txn_id", how="semi")
f = R._feats(tx).with_columns(tx["y"].cast(pl.Int8).alias("y"))
sup = ids.join(f.select("txn_id", R.support(f).alias("s")), on="txn_id", how="left", maintain_order="left")["s"].to_numpy()
f1 = lambda p: T.macro_f1(y, p).mean()
comp = {"mistyped (fraud label, other fraud class predicted)": (y > 0) & (q > 0) & (q != y),
        "missed, with class signature (fraud label, predicted legit, support)": (y > 0) & (q == 0) & sup,
        "missed, no class signature (fraud label, predicted legit, no support)": (y > 0) & (q == 0) & ~sup,
        "legit label, fraud predicted": (y == 0) & (q > 0)}
print(f"benchmark {B}, final decode of the current recipe: macro {f1(q):.4f}; fraud-labelled rows {int((y > 0).sum())}, predicted fraud {int((q > 0).sum())}")
for name, m in comp.items():
    p = q.copy(); p[m] = y[m]
    print(f"  {name:70s} rows {int(m.sum()):6d}  oracle fix of this compartment alone: {f1(p):.4f} ({f1(p) - f1(q):+.4f})")
C = np.zeros((K, K), int); np.add.at(C, (y, q), 1)
print("type precision of the final decode among fraud-labelled rows (share of predicted-k fraud rows labelled k):")
print("  " + " ".join(f"{T.LABELS[k][:5]} {C[k, k] / max(C[1:, k].sum(), 1):.3f}" for k in range(1, K)))
m = comp["mistyped (fraud label, other fraud class predicted)"]
print(pl.DataFrame({"label": np.array(T.LABELS)[y[m]], "predicted": np.array(T.LABELS)[q[m]]}).group_by("label", "predicted").len().sort("len", descending=True).head(12))
m = comp["missed, with class signature (fraud label, predicted legit, support)"]
print("missed rows with a class signature, by label:", dict(pl.Series(np.array(T.LABELS)[y[m]]).value_counts().sort("count", descending=True).rows()))
