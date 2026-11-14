"""Read-only: every test row the pool-bait / stamp rule (dd_rules_both) selects on each mixture base, against its
expected-id guard. Rows selected here but not in EXPECT are hits the guard refused; rows in EXPECT not selected are
listed too. usage (cwd=work, PYTHONPATH=.): python scratch_mix/guard_rows.py"""
import polars as pl
import dd_rules_both as DD
from common import load_all, LABEL2ID as L

c = pl.col; tx = load_all(); fr = DD.frame(tx); exp = set().union(*DD.EXPECT.values())
for name in ("ts_judge_h6", "ts_judge_cmb50", "ts_judge_cmb65", "ts_judge_cmb80"):
    # the rule reads the label before its own layer = output of the EDGEFIX step; here: the chain up to EDGEFIX
    import alt_c2
    real = alt_c2.DD
    class Stop:
        @staticmethod
        def apply(s, tx): Stop.s = s; return s
    alt_c2.DD = Stop; alt_c2.chain(pl.read_csv(f"subs/{name}.csv"), tx); alt_c2.DD = real; s = Stop.s
    q = s.select(c("id").alias("txn_id"), c("label").replace_strict(L, return_dtype=pl.Int64).alias("q"))
    f, _ = DD.hits(fr.join(q, on="txn_id", how="left")); h = f.filter(c("in_P") | c("in_S"))
    got = set(h["txn_id"].to_list())
    print(name, "| selected", len(got), "| selected, not expected:", h.filter(~c("txn_id").is_in(list(exp))).select("txn_id", "in_P", "in_S", "sender_id", "amount").rows(),
          "| expected, not selected:", sorted(exp - got), flush=True)
