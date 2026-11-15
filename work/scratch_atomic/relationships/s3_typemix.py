"""Agent B: are missed events compatible with the mechanism of their labelled class? Row-type mix of missed vs caught vs legit (Sep, N).
TV = total variation distance between the (txn_type, receiver_type, amount band, new_payee) distributions."""
import numpy as np, polars as pl
from common import load_all, LABELS
from clean_labels import fraud_events
tx = load_all(); c = pl.col
t, F = fraud_events(tx, 48 * 3600)
rows = pl.read_parquet("scratch_gap/N_sep_rows.parquet", columns=["txn_id"]).with_columns(pl.Series("q", np.load("scratch_gap/dec_N_wt.npz")["C_q"]))
x = (tx.filter(c("month") == 9).select("txn_id", "y", "txn_type", "receiver_type", "amount", "new_payee", (c("device_id").str.slice(3) != c("sender_id").str.slice(2)).alias("foreign_dev"))
     .join(rows, on="txn_id").join(F.select("txn_id", "ev"), on="txn_id", how="left")
     .with_columns((c("txn_type") + "|" + c("receiver_type") + "|" + pl.Series(np.digitize(tx.filter(c("month") == 9)["amount"].to_numpy(), [100, 300, 1000, 3000, 10000])).cast(pl.Utf8) + "|" + c("new_payee").cast(pl.Utf8)).alias("k")))
dist = lambda d: d.group_by("k").agg((pl.len() / d.height).alias("p"))
def tv(a, b):
    j = dist(a).join(dist(b), on="k", how="full", coalesce=True).fill_null(0); return float((j["p"] - j["p_right"]).abs().sum() / 2)
L = x.filter(c("y") == 0); out = []
for k in range(1, 12):
    m = x.filter((c("y") == k) & (c("q") == 0)).unique("ev"); g = x.filter((c("y") == k) & (c("q") > 0)).unique("ev")
    Ls = L.sample(m.height, seed=k)   # same-size legit sample: its TV to legit is the sampling-noise floor
    out.append({"cls": LABELS[k], "n_missed_ev": m.height, "n_caught_ev": g.height, "TV_missed_vs_legit": tv(m, L), "TV_legitsample_vs_legit(floor)": tv(Ls, L),
                "TV_caught_vs_legit": tv(g, L), "p2p_missed": (m["receiver_type"] == "P2P").mean(), "p2p_caught": (g["receiver_type"] == "P2P").mean(),
                "p2p_legit": (L["receiver_type"] == "P2P").mean(), "foreign_dev_missed": m["foreign_dev"].mean(), "foreign_dev_caught": g["foreign_dev"].mean()})
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(260)
o = pl.DataFrame(out).with_columns(pl.selectors.float().round(3)); print(o); o.write_csv("scratch_atomic/relationships/typemix.csv")
print("foreign_dev legit", L["foreign_dev"].mean())
