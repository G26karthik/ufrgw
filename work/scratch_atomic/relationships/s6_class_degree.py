"""Agent B: (1) per-class C5 / B4 on all fraud-labelled events with ring / burst conditioning;
(2) pattern c 'beyond absolute degree': inside a fixed recent 7 d count n7 of the sender, does the prior 21 d count nP matter?"""
import numpy as np, polars as pl
from common import load_all, LABELS
from clean_labels import fraud_events
D = "scratch_atomic/relationships/"; DAY = 86400; BIG = 10**10
tx = load_all(); n = tx.height
num = lambda c: tx[c].str.slice(2).cast(pl.Int64).to_numpy()
s, dn = num("sender_id"), num("device_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy(); r = np.where(p2p, num("receiver_id"), -1)
y = tx["y"].to_numpy().astype(float); mo = tx["month"].to_numpy(); ts = tx["ts"].to_numpy().astype(np.int64); amt = tx["amount"].to_numpy()
t, F = fraud_events(tx, 48 * 3600)
ev = np.full(n, -1); ev[F["i"].to_numpy()] = F["ev"].to_numpy(); evn = np.zeros(n); u, k = np.unique(ev[ev >= 0], return_counts=True); evn[ev >= 0] = k[np.searchsorted(u, ev[ev >= 0])]
rng = np.random.default_rng(0)
def one(mask):
    i = np.where(mask)[0]; i = i[rng.permutation(len(i))]; _, j = np.unique(ev[i], return_index=True); return np.sort(i[j])
POP = {f"allfraud_m{m}": one((y > 0) & (mo == m)) for m in (8, 9)}
kin = np.sort(r[p2p] * BIG + ts[p2p]); kout = np.sort(s * BIG + ts)
cnt = lambda keys, lo, hi: np.searchsorted(keys, s * BIG + hi, "left") - np.searchsorted(keys, s * BIG + lo, "left")
n7 = cnt(kin, ts - 7 * DAY, ts) + cnt(kout, ts - 7 * DAY, ts); nP = cnt(kin, ts - 28 * DAY, ts - 7 * DAY) + cnt(kout, ts - 28 * DAY, ts - 7 * DAY)
valid = ts - 28 * DAY >= ts.min()
R = tx.select("txn_id").join(pl.read_parquet(D + "rel_feats.parquet", columns=["txn_id", "rel__c_snd_act", "rel__b_co_to_rcv", "rel__b_rcv_to_co"]), on="txn_id", how="left", maintain_order="left")
EX = tx.select("txn_id").join(pl.read_parquet("scratch_gap/N_f1.parquet", columns=["txn_id", "flow__sin_n24h", "takeover__sess_n", "burst__snd_c1800", "ring__pair_n", "ring__s4_np_int", "burst__snd_ref7d", "burst__devrcv_c300", "burst__rs_n_device", "burst__ds_n_sender"]), on="txn_id", how="left", maintain_order="left")
b = lambda c, e: np.digitize(np.nan_to_num(EX[c].fill_null(float("nan")).to_numpy().astype(float), nan=-1), e).astype(np.int64)
g = lambda c: np.nan_to_num(R[c].to_numpy().astype(float), nan=0.0)
snd_n = tx.select(pl.len().over("sender_id", "month"))["len"].to_numpy()
cell = (mo.astype(np.int64) * 10**6 + tx["txn_type"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) * 10**4 + p2p * 10**3
        + np.digitize(amt, [100, 300, 1000, 3000, 10000]) * 10 + np.digitize(snd_n, [2, 3, 5, 9]))
M = {"C5_snd_act_up": g("rel__c_snd_act") > .5, "B4_co_rcv_edge": (g("rel__b_co_to_rcv") + g("rel__b_rcv_to_co")) > 0, "C8_nP_zero(no events in prior 21 d)": nP == 0}
CELLS = {"base": cell, "+sess_n,snd_c1800,ring__pair_n,ring__s4_np_int": cell * 10**4 + b("takeover__sess_n", [2, 3, 4]) * 1000 + b("burst__snd_c1800", [1, 2, 3]) * 100 + b("ring__pair_n", [2, 3, 5]) * 10 + b("ring__s4_np_int", [1, 2, 3]),
         "+devrcv_c300,rs_n_device,ds_n_sender": cell * 1000 + b("burst__devrcv_c300", [1, 2, 3]) * 100 + b("burst__rs_n_device", [0, 2, 3]) * 10 + b("burst__ds_n_sender", [0, 2, 3]),
         "+n7 exact (absolute recent degree, cap 8)": cell * 10 + np.minimum(n7, 8)}
legit = y == 0; out = []
for cn, cells in CELLS.items():
    for nm, m in M.items():
        v = valid if nm[0] == "C" else np.ones(n, bool)
        ctl = pl.DataFrame({"c": cells[legit & v], "m": m[legit & v].astype(float)}).group_by("c").agg(pl.col("m").mean().alias("rc"), pl.len().alias("nc"))
        for pn, idx in POP.items():
            idx = idx[v[idx]]
            d = pl.DataFrame({"c": cells[idx], "m": m[idx].astype(float), "y": y[idx], "multi": evn[idx] >= 2}).join(ctl, on="c").filter(pl.col("nc") >= 5)
            groups = [("all", d), ("single-row ev", d.filter(~pl.col("multi"))), ("multi-row ev", d.filter(pl.col("multi")))] + [(LABELS[int(kk[0])], gg) for kk, gg in d.group_by("y")]
            for cl, dd in groups:
                if dd.height < 30: continue
                df = (dd["m"] - dd["rc"]).to_numpy()
                out.append({"cond": cn, "motif": nm, "pop": pn, "cls": cl, "n_ev": dd.height, "fraud_pct": round(dd["m"].mean() * 100, 2), "ctrl_pct": round(dd["rc"].mean() * 100, 2),
                            "diff": round(df.mean() * 100, 2), "z": round(df.mean() / max(df.std(ddof=1) / np.sqrt(len(df)), 1e-12), 1)})
O = pl.DataFrame(out); O.write_csv(D + "class_degree.csv")
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(48)
O = O.with_columns((pl.col("diff").cast(pl.Utf8) + " (" + pl.col("z").cast(pl.Utf8) + ") n=" + pl.col("n_ev").cast(pl.Utf8)).alias("d_z"))
for nm in M:
    print("\n=====", nm); print(O.filter(pl.col("motif") == nm).pivot(on=["pop"], index=["cond", "cls"], values="d_z").sort("cond", "cls"))
