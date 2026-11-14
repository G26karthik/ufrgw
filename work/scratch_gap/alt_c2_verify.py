"""Checks on subs/alt_c2s6_dd_final.csv (built by alt_c2.py). Read-only; prints everything.
1 determinism: the driver on the second from-raw copy of the base (subs/ts_judge_r_repro_c2.csv) gives the same bytes
2 the 13 rows where the two models differ before the later rules, and what the rules do to them (fifth LFM row)
3 the 12 final rows: raw probabilities of both models, and the labels of other test rows that share receiver / sender / device
4 the 14 rows of the discarded art_acct / art_zonemax transplant: alternative must equal dd_final, not final1
5 September (benchmark D, after decoder): c2s6 vs h6 disagreements by transition type, who was right
usage (cwd=work, PYTHONPATH=.): python scratch_gap/alt_c2_verify.py"""
import numpy as np
import polars as pl
import train as T
import alt_c2
from common import load_all, LABELS, LABEL2ID as L

pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
c = pl.col; tx = load_all(); rd = lambda n: pl.read_csv(f"subs/{n}.csv")
alt, dd, h6b, c2b, fin1 = rd("alt_c2s6_dd_final"), rd("dd_final"), rd("ts_judge_h6"), rd("ts_judge_c2s6"), rd("final1")
assert all(x["id"].equals(dd["id"]) for x in (alt, h6b, c2b, fin1)) and alt["label"].is_in(LABELS).all() and alt.columns == dd.columns

print("== 1 determinism")
again, _ = alt_c2.chain(rd("ts_judge_r_repro_c2"), tx)
print("driver on subs/ts_judge_r_repro_c2.csv == subs/alt_c2s6_dd_final.csv:", alt_c2.csv_bytes(again) == open("subs/alt_c2s6_dd_final.csv", "rb").read())

A = pl.DataFrame({"id": dd["id"], "h6_before_rules": h6b["label"], "c2_before_rules": c2b["label"], "dd_final": dd["label"], "alt": alt["label"], "final1": fin1["label"]})
layers = pl.read_csv("notes/alt_c2s6_dd_final_layers.csv")
print("\n== 2 rows where the models differ before the later rules, and the layer (if any) that changed them in the c2s6 chain")
m = A.filter(c("h6_before_rules") != c("c2_before_rules")).join(layers.select("id", "layer", "from", "to"), on="id", how="left")
print(m.select("id", "h6_before_rules", "c2_before_rules", "layer", "dd_final", "alt", (c("dd_final") == c("alt")).alias("same_final")))
d12 = A.filter(c("dd_final") != c("alt"))
print("final differing rows touched by any layer in the c2s6 chain:", d12.join(layers, on="id", how="inner").height, "| differing rows:", d12.height)

print("\n== 3a raw model probabilities (before bias and rules) of the dd_final label and of the alternative label")
P = {k: np.load(f"preds/{k}_test.npy") for k in ("h6", "c2s6")}; pos = {t: i for i, t in enumerate(dd["id"])}
t = tx.filter(c("is_test")).select(c("txn_id").alias("id"), "ts", "sender_id", "receiver_id", "device_id", "amount", "txn_type", "receiver_type")
rows = []
for r in d12.iter_rows(named=True):
    i, a, b = pos[r["id"]], L[r["dd_final"]], L[r["alt"]]
    rows.append({"id": r["id"], "dd_final": r["dd_final"], "alt": r["alt"], "h6 p(dd)": round(float(P["h6"][i, a]), 3), "h6 p(alt)": round(float(P["h6"][i, b]), 3),
                 "c2 p(dd)": round(float(P["c2s6"][i, a]), 3), "c2 p(alt)": round(float(P["c2s6"][i, b]), 3)})
print(pl.DataFrame(rows).join(t.select("id", "sender_id", "receiver_id", "device_id", "amount", "txn_type"), on="id"))

print("\n== 3b other test rows sharing an entity with each differing row: label counts in dd_final (identical in the alternative unless listed)")
full = t.join(A.select("id", "dd_final", "alt"), on="id")
for r in d12.join(t, on="id").iter_rows(named=True):
    for key in ("receiver_id", "sender_id", "device_id"):
        o = full.filter(((c("receiver_id") == r[key]) | (c("sender_id") == r[key])) if key != "device_id" else (c("device_id") == r[key])).filter(c("id") != r["id"])
        fr = o.filter(c("dd_final") != "legit")
        if fr.height:
            print(f"{r['id']} {r['dd_final']}->{r['alt']} | {key} {r[key]}: {o.height} other rows, non-legit in dd_final {dict(fr['dd_final'].value_counts().rows())}"
                  f" | of them also changed: {o.filter(c('dd_final') != c('alt'))['id'].to_list()}")
    print(f"{r['id']} {r['dd_final']}->{r['alt']} | (rows listed above only where a neighbour is non-legit)")

print("\n== 4 discarded transplant rows (final1 != ts_judge_c2s6)")
tr = A.filter(c("final1") != c("c2_before_rules"))
print(tr.height, "rows | alternative == dd_final on", int((tr["alt"] == tr["dd_final"]).sum()), "| alternative == final1 on", int((tr["alt"] == tr["final1"]).sum()),
      "| dd_final == final1 on", int((tr["dd_final"] == tr["final1"]).sum()))
print(tr.group_by("c2_before_rules", "final1", "dd_final", "alt").len().sort("len", descending=True))

print("\n== 5 September, benchmark D, after decoder: rows where c2s6 and h6 disagree, by transition")
z = np.load("scratch_gap/dec_D_ens.npz"); y, a, b = z["y"], z["h6_q"], z["c2s6_q"]; ch = a != b
S = pl.DataFrame({"h6": [LABELS[k] for k in a[ch]], "c2s6": [LABELS[k] for k in b[ch]], "c2_right": b[ch] == y[ch], "h6_right": a[ch] == y[ch]})
S = S.group_by("h6", "c2s6").agg(pl.len().alias("rows"), c("c2_right").sum(), c("h6_right").sum()).with_columns((c("rows") - c("c2_right") - c("h6_right")).alias("both_wrong"))
oct_ = {(r["dd_final"], r["alt"]) for r in d12.iter_rows(named=True)}
S = S.with_columns(pl.struct("h6", "c2s6").map_elements(lambda s: (s["h6"], s["c2s6"]) in oct_, return_dtype=pl.Boolean).alias("in_october_diff")).sort("rows", descending=True)
print(S)
o = S.filter(c("in_october_diff"))
print("transitions present in October: rows", int(o["rows"].sum()), "c2s6 right", int(o["c2_right"].sum()), "h6 right", int(o["h6_right"].sum()), "both wrong", int(o["both_wrong"].sum()))
f = lambda p: T.macro_f1(y, p)
print("per-class F1 change on September (c2s6 - h6):", " ".join(f"{LABELS[k][:6]} {f(b)[k] - f(a)[k]:+.3f}" for k in range(1, T.K)), "| macro", round(float(f(b).mean() - f(a).mean()), 4))
print("September F1 of h6 by class (break-even precision for an added row = F1/2):", " ".join(f"{LABELS[k][:6]} {f(a)[k]:.2f}" for k in range(1, T.K)))
