"""Agent B task 1: 40 independent missed September events (benchmark N, seed 0) + 1 matched legit control each, local timelines.
Writes sample.parquet and timelines.txt. Run (cwd = work): PYTHONPATH=. python scratch_atomic/relationships/s2_sample_timelines.py
"""
import numpy as np, polars as pl
from common import load_all, LABELS
from clean_labels import fraud_events

D = "scratch_atomic/relationships/"
DAY = 86400
tx = load_all(); n = tx.height
t, F = fraud_events(tx, 48 * 3600)
rows = pl.read_parquet("scratch_gap/N_sep_rows.parquet", columns=["txn_id"]).with_columns(pl.Series("q", np.load("scratch_gap/dec_N_wt.npz")["C_q"]))
c = pl.col
x = (tx.select("txn_id", "ts", "timestamp", "month", "y", "sender_id", "receiver_id", "receiver_type", "device_id", "txn_type", "amount", "new_payee", "balance_before")
     .join(rows, on="txn_id", how="left").join(F.select("txn_id", "ev"), on="txn_id", how="left")
     .with_columns(pl.len().over("sender_id", "month").alias("snd_n")))
x = x.with_columns((c("month").cast(pl.Utf8) + "|" + c("txn_type") + "|" + c("receiver_type") + "|" + c("amount").cut([100, 300, 1000, 3000, 10000]).cast(pl.Utf8)
                    + "|" + c("snd_n").cut([1, 2, 4, 8]).cast(pl.Utf8)).alias("cell"))
miss = x.filter((c("y") > 0) & (c("q") == 0) & (c("month") == 9)).sample(fraction=1.0, shuffle=True, seed=0).unique("ev", keep="first", maintain_order=True)
S = miss.sample(40, seed=0)
ctl = (x.filter((c("y") == 0) & (c("month") == 9) & c("cell").is_in(S["cell"].implode())).sample(fraction=1.0, shuffle=True, seed=0)
       .with_columns(c("txn_id").cum_count().over("cell").alias("k")))
Sk = S.with_columns(c("txn_id").cum_count().over("cell").alias("k"))
C = Sk.select("cell", "k", c("txn_id").alias("case_txn")).join(ctl, on=["cell", "k"], how="left")
print("events sampled", S.height, "controls matched", C.filter(c("txn_id").is_not_null()).height)
pl.concat([S.with_columns(pl.lit("missed").alias("grp"), c("txn_id").alias("case_txn")).select("grp", "case_txn", "txn_id", "cell", "y", "ev"),
           C.filter(c("txn_id").is_not_null()).with_columns(pl.lit("control").alias("grp")).select("grp", "case_txn", "txn_id", "cell", "y", "ev")]).write_parquet(D + "sample.parquet")

own = lambda d: "AC" + d[3:]


def line(r, me):
    role = "OUT" if r["sender_id"] == me else "IN "
    cp = r["receiver_id"] if role == "OUT" else r["sender_id"]
    return (f"   {r['timestamp']:%m-%d %H:%M} {role} {cp:9s} {r['txn_type']:7s} {r['amount']:9.2f} dev={r['device_id']}"
            f"{'' if own(r['device_id']) == r['sender_id'] else '*'} np={r['new_payee']} y={r['y']}")


stats = []
with open(D + "timelines.txt", "w", encoding="utf-8") as f:
    for grp, df in (("MISSED", S), ("CONTROL", C.filter(c("txn_id").is_not_null()))):
        for r in df.iter_rows(named=True):
            f.write(f"\n=== {grp} {r['txn_id']} {LABELS[int(r['y'])]} {r['timestamp']} {r['sender_id']} -> {r['receiver_id']} {r['txn_type']} {r['amount']} dev={r['device_id']} cell={r['cell']}\n")
            st = {"grp": grp}
            for who, a in (("sender", r["sender_id"]), ("receiver", r["receiver_id"])):
                if not a.startswith("AC"): continue
                h = x.filter(((c("sender_id") == a) | (c("receiver_id") == a)) & ((c("ts") - r["ts"]).abs() <= 14 * DAY)).sort("ts")
                f.write(f"  {who} {a}: {h.height} events in +-14 d (both roles; * = device not owned by the sender)\n")
                for hr in h.iter_rows(named=True):
                    f.write(line(hr, a) + (" <<<" if hr["txn_id"] == r["txn_id"] else "") + "\n")
                st[who + "_n"] = h.height; st[who + "_in"] = (h["receiver_id"] == a).sum(); st[who + "_fraud_other"] = ((h["y"] > 0) & (h["txn_id"] != r["txn_id"])).sum()
            du = x.filter(c("device_id") == r["device_id"]).group_by("sender_id").agg(pl.len().alias("n"), c("ts").min().alias("first"))
            f.write(f"  device {r['device_id']} owner {own(r['device_id'])}: users {dict(zip(du['sender_id'], du['n']))}\n")
            st["dev_users"] = du.height; st["dev_foreign"] = own(r["device_id"]) != r["sender_id"]
            stats.append(st)
st = pl.DataFrame(stats)
print(st.group_by("grp").agg(pl.len(), pl.col("sender_n").mean(), pl.col("sender_in").mean(), pl.col("sender_fraud_other").mean(), pl.col("receiver_n").mean(),
                             pl.col("receiver_in").mean(), pl.col("receiver_fraud_other").mean(), pl.col("dev_users").mean(), pl.col("dev_foreign").mean()))
