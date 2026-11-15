"""Agent A: text representation of every ORIGINAL CSV field (before prep.py parses to parquet), legit vs fraud-labelled.
cwd=work. python scratch_atomic/representation/03_rawtext.py > scratch_atomic/representation/03_out.txt
Pattern of a cell: every digit -> 9, leading digit run -> N (so '5000.0' -> 'N.9', '12.34' -> 'N.99', '' -> <null>).
A format variant that exists only on one generation path (e.g. '5000' vs '5000.0', 'nan' vs '') is lost by the float parse
in prep.py:10-13. Label-free trace = the pattern itself. Reported: every pattern per column with legit / fraud counts,
total-variation distance legit vs fraud per month, and the same for test.csv (no labels)."""
import polars as pl
from pathlib import Path
c = pl.col; RAW = Path("..") / "upi-fraud-rings"
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40)
tr = pl.read_csv(RAW / "train.csv", infer_schema=False); te = pl.read_csv(RAW / "test.csv", infer_schema=False)
print("train", tr.shape, "test", te.shape, "columns", tr.columns)
pat = lambda x: c(x).str.replace_all(r"\d", "9").str.replace(r"^9+", "N").fill_null("<null>")
tr = tr.with_columns((c("label") != "legit").alias("fraud"), c("timestamp").str.slice(5, 2).alias("mm"))
for x in [k for k in tr.columns if k not in ("label", "fraud", "mm")]:
    p = tr.select(pat(x).alias("p"), "fraud", "mm")
    g = p.group_by("p").agg((~c("fraud")).sum().alias("legit"), c("fraud").sum().alias("fraud")).sort("legit", descending=True)
    g = g.with_columns((c("legit") / c("legit").sum()).round(4).alias("legit_sh"), (c("fraud") / c("fraud").sum()).round(4).alias("fraud_sh"))
    tp = te.select(pat(x).alias("p")).group_by("p").len().with_columns((c("len") / c("len").sum()).round(4).alias("test_sh")).drop("len")
    g = g.join(tp, on="p", how="full", coalesce=True)
    tv = []
    for m in ("07", "08", "09"):
        q = p.filter(c("mm") == m).group_by("p").agg((~c("fraud")).sum().alias("l"), c("fraud").sum().alias("f"))
        tv.append(round(float((q["l"] / q["l"].sum() - q["f"] / q["f"].sum()).abs().sum() / 2), 4))
    print(f"\n== {x}: {g.height} patterns; TV(legit, fraud-labelled) Jul/Aug/Sep = {tv}")
    print(g.head(12))
for name in ("accounts", "merchants"):
    a = pl.read_csv(RAW / f"{name}.csv", infer_schema=False)
    print(f"\n#### {name} {a.shape}")
    for x in a.columns:
        print(x, a.select(pat(x).alias("p")).group_by("p").len().sort("len", descending=True).head(6).rows())
