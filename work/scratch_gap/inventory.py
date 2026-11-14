"""Inventory for the historical ledger: saved validation predictions (oof/), test predictions (preds/) and submissions
(subs/, diff against dd_final). cwd=work. usage: python scratch_gap/inventory.py > scratch_gap/inventory.log"""
import glob, json, os
import numpy as np
import polars as pl

rows = []
for f in sorted(glob.glob("oof/*_meta.json")):
    m = json.load(open(f)); a = m.get("args", {}); tag = m.get("tag") or os.path.basename(f)[:-10]
    P = f"oof/{tag}.npy"
    rows.append((m.get("pooled_biased") or 0, tag, str(m.get("model") or a.get("model")), str(a.get("relabel", "")), a.get("seeds"), a.get("rounds"), m.get("n_feats"),
                 m.get("fold0_biased") or 0, m.get("fold1_biased") or 0, os.path.exists(P), os.path.exists(f"preds/{tag}_test.npy"),
                 os.path.basename(str(a.get("matrix", "")))[:36], os.path.getmtime(f)))
print(f"== {len(rows)} validation records (oof/*_meta.json), by pooled OOF macro (bias tuned on the pooled OOF)")
for r in sorted(rows, reverse=True):
    print(f"{r[1]:24s} {r[2]:5s} relabel={r[3]:5s} seeds={r[4]} rounds={r[5]} nf={r[6]} pooled={r[0]:.4f} aug={r[7]:.4f} sep={r[8]:.4f} oof={r[9]} test={r[10]} {r[11]}")
extra = [os.path.basename(f) for f in glob.glob("oof/*.npy") if not os.path.exists(f.replace(".npy", "_meta.json"))]
print("== oof arrays without a meta record:", extra)
base = pl.read_csv("subs/dd_final.csv")
print(f"== {len(glob.glob('subs/*.csv'))} submissions: rows differing from dd_final, and how")
out = []
for f in sorted(glob.glob("subs/*.csv"), key=os.path.getmtime):
    s = pl.read_csv(f)
    if s.height != base.height or not s["id"].equals(base["id"]):
        out.append((os.path.basename(f), -1, "", "")); continue
    d = base.with_columns(s["label"].alias("o")).filter(pl.col("label") != pl.col("o"))
    top = d.group_by("o", "label").len().sort("len", descending=True).head(3).rows()
    out.append((os.path.basename(f), d.height, int((s["label"] != "legit").sum()), " ".join(f"{a[:5]}>{b[:5]}:{n}" for a, b, n in top)))
for o in out:
    print(f"{o[0]:46s} diff={o[1]:6d} fraud_rows={o[2]} {o[3]}")
