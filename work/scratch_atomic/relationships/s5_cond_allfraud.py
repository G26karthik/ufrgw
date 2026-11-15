"""Agent B (addendum): model-independent populations (all fraud-labelled events, own-device fraud events, multi-row events), real timestamps,
matched legit controls; conditioning on the closest existing burst / flow columns. Uses rel_feats.parquet (label-free)."""
import numpy as np, polars as pl
from common import load_all
from clean_labels import fraud_events
D = "scratch_atomic/relationships/"; DAY = 86400
tx = load_all(); n = tx.height
num = lambda c: tx[c].str.slice(2).cast(pl.Int64).to_numpy()
s, dn = num("sender_id"), num("device_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy()
y = tx["y"].to_numpy().astype(float); mo = tx["month"].to_numpy(); ts = tx["ts"].to_numpy(); amt = tx["amount"].to_numpy()
t, F = fraud_events(tx, 48 * 3600)
ev = np.full(n, -1); ev[F["i"].to_numpy()] = F["ev"].to_numpy(); evn = np.zeros(n); u, k = np.unique(ev[ev >= 0], return_counts=True); evn[ev >= 0] = k[np.searchsorted(u, ev[ev >= 0])]
rng = np.random.default_rng(0)
def one(mask):
    i = np.where(mask)[0]; i = i[rng.permutation(len(i))]; _, j = np.unique(ev[i], return_index=True); return np.sort(i[j])
POP = {}
for m in (7, 8, 9):
    POP[f"allfraud_m{m}"] = one((y > 0) & (mo == m)); POP[f"owndev_m{m}"] = one((y > 0) & (mo == m) & (dn == s))
    POP[f"owndev_single_m{m}"] = np.where((y > 0) & (mo == m) & (dn == s) & (evn == 1))[0]; POP[f"multi3_m{m}"] = one((y > 0) & (mo == m) & (evn >= 3))
print({a: len(b) for a, b in POP.items()})
R = tx.select("txn_id").join(pl.read_parquet(D + "rel_feats.parquet"), on="txn_id", how="left", maintain_order="left")
EX = tx.select("txn_id").join(pl.read_parquet("scratch_gap/N_f1.parquet", columns=["txn_id", "flow__sin_n24h", "takeover__sess_n", "burst__snd_c1800", "burst__ss_n", "scam__rcv_in_15d", "flow__rin_n12h"]), on="txn_id", how="left", maintain_order="left")
g = lambda c: np.nan_to_num(R["rel__" + c].to_numpy().astype(float), nan=0.0)
M = {"A1d_snd_prev_in_24h": (g("a_snd_prev_role") == 1) & (np.expm1(g("a_snd_prev_gap_log")) <= DAY), "A3_snd_in_run2": g("a_snd_in_run") >= 2,
     "A2_rcv_prev_out": R["rel__a_rcv_prev_role"].to_numpy() == 0, "C5_snd_act_up": g("c_snd_act") > .5, "C7_snd_out_up": g("c_snd_out_ratio") > .5,
     "C6_rcv_in_up": g("c_rcv_in_ratio") > .5, "C4_snd_first_out": g("c_snd_first_out") == 1, "C3_rcv_first_in": g("c_rcv_first_in") == 1,
     "C2_rcv_dshare_up": g("c_rcv_dshare") > .25, "B4_co_rcv_edge": (g("b_co_to_rcv") + g("b_rcv_to_co")) > 0, "B5_dev_co_before": g("b_dev_nco_before") >= 1}
snd_n = tx.select(pl.len().over("sender_id", "month"))["len"].to_numpy()
cell = (mo.astype(np.int64) * 10**6 + tx["txn_type"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) * 10**4 + p2p * 10**3
        + np.digitize(amt, [100, 300, 1000, 3000, 10000]) * 10 + np.digitize(snd_n, [2, 3, 5, 9]))
b = lambda c, e: np.digitize(np.nan_to_num(EX[c].fill_null(float("nan")).to_numpy().astype(float), nan=-1), e).astype(np.int64)
CELLS = {"base": cell, "+flow(sin_n24h,rin_n12h,rcv_in_15d)": cell * 1000 + b("flow__sin_n24h", [0, 1, 2]) * 100 + b("flow__rin_n12h", [0, 2, 4, 8]) * 10 + b("scam__rcv_in_15d", [0, 1, 2, 3, 6]),
         "+burst(sess_n,snd_c1800)": cell * 100 + b("takeover__sess_n", [2, 3, 4]) * 10 + b("burst__snd_c1800", [1, 2, 3])}
CELLS["+flow+burst"] = CELLS["+flow(sin_n24h,rin_n12h,rcv_in_15d)"] * 100 + CELLS["+burst(sess_n,snd_c1800)"] % 100
legit = y == 0; T0 = ts.min(); out = []
for cn, cells in CELLS.items():
    for nm, m in M.items():
        v = (ts - 28 * DAY >= T0) if nm.startswith("C") else np.ones(n, bool)
        ctl = pl.DataFrame({"c": cells[legit & v], "m": m[legit & v].astype(float)}).group_by("c").agg(pl.col("m").mean().alias("rc"), pl.len().alias("nc"))
        for pn, idx in POP.items():
            idx = idx[v[idx]]
            if len(idx) < 30: continue
            d = pl.DataFrame({"c": cells[idx], "m": m[idx].astype(float)}).join(ctl, on="c").filter(pl.col("nc") >= 5)
            df = (d["m"] - d["rc"]).to_numpy()
            out.append({"cond": cn, "motif": nm, "pop": pn, "n_ev": d.height, "n_drop": len(idx) - d.height, "fraud_pct": round(d["m"].mean() * 100, 2), "ctrl_pct": round(d["rc"].mean() * 100, 2),
                        "diff": round(df.mean() * 100, 2), "z": round(df.mean() / max(df.std(ddof=1) / np.sqrt(len(df)), 1e-12), 1)})
O = pl.DataFrame(out); O.write_csv(D + "cond_allfraud.csv")
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(50)
for cn in CELLS:
    print("\n=====", cn); print(O.filter(pl.col("cond") == cn).pivot(on="pop", index="motif", values=["diff", "z"]))
print(O.filter(pl.col("pop").is_in(["allfraud_m8", "owndev_m8", "owndev_single_m8", "multi3_m8"]) & pl.col("motif").is_in(["C5_snd_act_up", "A1d_snd_prev_in_24h", "C6_rcv_in_up"])).sort("motif", "pop", "cond"))
