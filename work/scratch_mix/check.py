"""Checks on the three built mixtures (read-only): ids / order / labels, hashes, per-class counts against the h6 files,
pairwise row differences, equality with any existing file in subs/. usage (cwd=work, PYTHONPATH=.): python scratch_mix/check.py"""
import glob, hashlib, os
import polars as pl
import train as T
from common import LABELS

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
c = pl.col; rd = lambda n: pl.read_csv(f"subs/{n}.csv"); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ids = pl.read_csv(T.SAMPLE)["id"]; W = [w for w in (50, 65, 80) if os.path.exists(f"subs/cmb{w}_dd_final.csv")]
F = {"dd_final": rd("dd_final")} | {f"cmb{w}": rd(f"cmb{w}_dd_final") for w in W}
B = {"h6": rd("ts_judge_h6")} | {f"cmb{w}": rd(f"ts_judge_cmb{w}") for w in W}
for k, d in (F | {"base_" + k: v for k, v in B.items()}).items():
    assert d.columns == ["id", "label"] and d["id"].equals(ids) and d["label"].is_in(LABELS).all() and d.height == 260582, k
print("ids, order, labels valid for", list(F), "and their bases")
cnt = lambda D: pl.DataFrame({"label": LABELS}).with_columns([pl.Series(k, [int((d["label"] == l).sum()) for l in LABELS]) for k, d in D.items()])
print("per-class counts before the later rules (ts_judge files):"); print(cnt(B))
print("per-class counts of the final files:"); print(cnt(F))
ks = list(F)
for i, a in enumerate(ks):
    for b in ks[i + 1:]: print(f"rows that differ {a} vs {b}: {int((F[a]['label'] != F[b]['label']).sum())}")
old = {sha(p): p for p in glob.glob("subs/*.csv") if not os.path.basename(p).startswith("cmb")}
for w in W:
    h = sha(f"subs/cmb{w}_dd_final.csv"); print(f"cmb{w}_dd_final.csv sha256 {h} | non-legit rows {int((F[f'cmb{w}']['label'] != 'legit').sum())} | equals an existing file: {old.get(h, 'no')}")
print("dd_final sha256", sha("subs/dd_final.csv"))
