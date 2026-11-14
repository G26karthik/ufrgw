"""Production dependency chain: run_all.ps1 step 11 builds final1 = ts_judge_c2s6 + art_acct + art_zonemax, but step 12
restarts from ts_judge_h6. Which final1 effects are absent from dd_final, already reproduced by later rules, or in conflict?
cwd=work. usage: python scratch_gap/chain.py"""
import polars as pl
from common import load_all

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(30)
c = pl.col
S = lambda n: pl.read_csv(f"subs/{n}.csv")["label"]
ids = pl.read_csv("subs/dd_final.csv")["id"]
d = pl.DataFrame({"txn_id": ids, "h6": S("ts_judge_h6"), "c2": S("ts_judge_c2s6"), "acct": S("art_acct"), "zmax": S("art_zonemax"), "final1": S("final1"),
                  "v3": S("v3_final"), "dd": S("dd_final")})
print("rows differing from ts_judge_h6:", {k: int((d[k] != d["h6"]).sum()) for k in ("c2", "acct", "zmax", "final1", "v3", "dd")})
x = d.filter(c("final1") != c("h6")).with_columns(
    pl.when(c("acct") != c("h6")).then(pl.lit("art_acct")).when(c("zmax") != c("h6")).then(pl.lit("art_zonemax")).otherwise(pl.lit("core2 model")).alias("source"),
    pl.when(c("dd") == c("final1")).then(pl.lit("same as final1 (reproduced)")).when(c("dd") == c("h6")).then(pl.lit("still ts_judge_h6 label (absent)"))
      .otherwise(pl.lit("third label (conflict)")).alias("in_dd_final"))
print("final1 rows that differ from ts_judge_h6, by source and by what dd_final says:")
print(x.group_by("source", "in_dd_final").len().sort("source", "in_dd_final"))
print(x.group_by("source", "h6", "final1", "dd").len().sort("source", "len", descending=[False, True]))
tx = load_all().filter(c("is_test")).select("txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "amount", "txn_type")
print(x.filter(c("in_dd_final") != "same as final1 (reproduced)").join(tx, on="txn_id").sort("source", "receiver_id", "timestamp")
       .select("txn_id", "source", "h6", "final1", "dd", "sender_id", "receiver_id", "device_id", "amount", "txn_type"))
x.write_csv("scratch_gap/chain_rows.csv")
