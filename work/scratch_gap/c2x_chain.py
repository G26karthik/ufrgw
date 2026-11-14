"""Isolate the c2s6 model: run the same later rule layers (run_all steps 12-17) on ts_judge_h6 and on ts_judge_c2s6.
The h6 run must reproduce subs/dd_final.csv exactly (self-check). The c2s6 run shows which October predictions change
when only the model changes. The production scripts assert base-specific row sets, so the rule functions are called
directly here; a layer that refuses the other base is reported, not forced.
Output: scratch_gap/c2x_dd_final.csv (NOT a candidate), scratch_gap/c2x_diff.csv. usage (cwd=work, PYTHONPATH=.): python scratch_gap/c2x_chain.py"""
import polars as pl
import v3_rules, sc_rules_collusion as RS, sc_rules_fc as FCB
import bc_rules_layering, bc_rules_velocity, bc_rules_mule, bc_rules_phishing, probe_rules, dd_rules_both as DD
from common import load_all, LABEL2ID as L

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(24)
c = pl.col; tx = load_all()


def dd(sub):  # dd_rules_both without its expected-id assertion
    q = sub.select(c("id").alias("txn_id"), c("label").replace_strict(L, return_dtype=pl.Int64).alias("q"))
    f, _ = DD.hits(DD.frame(tx).join(q, on="txn_id", how="left"))
    ids = f.filter(c("in_P") | c("in_S"))["txn_id"].implode()
    return sub.with_columns(pl.when(c("id").is_in(ids)).then(pl.lit("phishing_collect")).otherwise(c("label")).alias("label"))


LAYERS = [("v3 QRN+zprop", lambda s: v3_rules.apply_rules(s, tx)[0]), ("RS", lambda s: RS.apply(s, tx)), ("FCB", lambda s: FCB.apply(s, tx)),
          ("CZ", lambda s: bc_rules_layering.apply(s, tx)), ("LFM", lambda s: bc_rules_velocity.apply(s, tx)), ("X999", lambda s: bc_rules_phishing.apply(s, tx)),
          ("BANDV", lambda s: bc_rules_mule.apply(s, tx)), ("EDGEFIX", lambda s: probe_rules.apply(s, tx, "EDGEFIX")), ("dd pool+stamp", dd)]
out, logs = {}, {}
for base in ("ts_judge_h6", "ts_judge_c2s6"):
    s = pl.read_csv(f"subs/{base}.csv"); logs[base] = []
    for name, fn in LAYERS:
        try:
            n = fn(s).select("id", "label"); ch = s.with_columns(n["label"].alias("to")).filter(c("label") != c("to"))
            logs[base].append((name, ch.height, str(ch.group_by("label", "to").len().sort("len", descending=True).rows()), set(ch["id"]))); s = n
        except AssertionError as e:
            logs[base].append((name, -1, "refused: " + str(e)[:90], set()))
    out[base] = s
ref = pl.read_csv("subs/dd_final.csv")
print("h6 chain reproduces dd_final:", out["ts_judge_h6"]["label"].equals(ref["label"]))
for (n, a, da, ia), (_, b, db, ib) in zip(logs["ts_judge_h6"], logs["ts_judge_c2s6"]):
    print(f"{n:14s} h6 base: {a:3d} rows {da[:95]:95s} | c2s6 base: {b:3d} rows {db[:95]} | same row set: {ia == ib}")
A, B = pl.read_csv("subs/ts_judge_h6.csv")["label"], pl.read_csv("subs/ts_judge_c2s6.csv")["label"]
d = pl.DataFrame({"txn_id": ref["id"], "h6_model": A, "c2_model": B, "dd_final": ref["label"], "c2_chain": out["ts_judge_c2s6"]["label"]})
x = d.filter(c("dd_final") != c("c2_chain")).with_columns((c("h6_model") != c("c2_model")).alias("differs_before_rules"))
print(f"rows where the c2s6 model differs from h6 after zones+bait: {int((A != B).sum())}; after all later layers: {x.height}")
print(x.group_by("dd_final", "c2_chain", "differs_before_rules").len().sort("len", descending=True))
t = tx.filter(c("is_test")).select("txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "amount", "txn_type")
print(x.join(t, on="txn_id").sort("dd_final", "receiver_id").select("txn_id", "dd_final", "c2_chain", "differs_before_rules", "sender_id", "receiver_id", "amount", "txn_type"))
print("label counts c2 chain minus dd_final:", {k: v for k, v in ((k, int((out["ts_judge_c2s6"]["label"] == k).sum() - (ref["label"] == k).sum())) for k in L) if v})
x.write_csv("scratch_gap/c2x_diff.csv"); out["ts_judge_c2s6"].write_csv("scratch_gap/c2x_dd_final.csv")
