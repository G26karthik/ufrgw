"""Checks on the second exploratory candidate (read-only): ids / order / labels, hash, per-class counts against dd_final and the
h6 base, changed rows by from -> to, overlap with the first exploratory file, rule layers that touch a changed row, equality
with any existing file in subs/. usage (cwd=work, PYTHONPATH=.): python scratch_wt/check_sq35.py"""
import glob, hashlib
import polars as pl
import train as T
from common import LABELS

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
c = pl.col; rd = lambda n: pl.read_csv(f"subs/{n}.csv"); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ids = pl.read_csv(T.SAMPLE)["id"]
F = {k: rd(k) for k in ("ts_judge_h6", "ts_judge_sq35", "dd_final", "sq35_dd_final", "cmb80_dd_final")}
for k, d in F.items(): assert d.columns == ["id", "label"] and d["id"].equals(ids) and d["label"].is_in(LABELS).all() and d.height == 260582, k
print("ids, order, labels valid:", list(F))
print(pl.DataFrame({"label": LABELS}).with_columns([pl.Series(k, [int((d["label"] == l).sum()) for l in LABELS]) for k, d in F.items()]))
print({k: int((d["label"] != "legit").sum()) for k, d in F.items()}, "non-legit rows")
a, b, m = F["dd_final"], F["sq35_dd_final"], F["cmb80_dd_final"]; ch = a["label"] != b["label"]
D = pl.DataFrame({"id": a["id"], "dd_final": a["label"], "sq35": b["label"], "cmb80": m["label"]}).filter(ch)
print(f"rows that differ from dd_final: {D.height} | of these also changed in cmb80: {int((D['cmb80'] != D['dd_final']).sum())} (same new label: {int((D['cmb80'] == D['sq35']).sum())})")
print(D.group_by("dd_final", "sq35").len().sort("len", descending=True))
print(f"base (before the later rules) rows that differ from the h6 base: {int((F['ts_judge_h6']['label'] != F['ts_judge_sq35']['label']).sum())}")
lay = pl.read_csv("notes/sq35_dd_final_layers.csv"); col = "id" if "id" in lay.columns else lay.columns[0]
print("rule-layer rows that are also changed rows:", lay.filter(c(col).is_in(D["id"].to_list())).height, "| layer file columns", lay.columns)
h = sha("subs/sq35_dd_final.csv"); old = {sha(p): p for p in glob.glob("subs/*.csv") if "sq35" not in p}
print("sq35_dd_final.csv sha256", h, "| equals an existing file:", old.get(h, "no")); print("dd_final sha256", sha("subs/dd_final.csv"))
