"""Agent B: relational / ordering patterns a, b, c. Label-free features `feat(ts)`; labels used only for evaluation.
Run (cwd = work): PYTHONPATH=. python scratch_atomic/relationships/s1_patterns.py
Writes: rel_feats.parquet (all rows), results.csv, results_class.csv, sample.parquet, timelines.txt
"""
import numpy as np, polars as pl
from common import load_all, LABELS
from clean_labels import fraud_events

D = "scratch_atomic/relationships/"
DAY, BIG, PB = 86400, 10**10, 2_000_000
tx = load_all()
n = tx.height
num = lambda c: tx[c].str.slice(2).cast(pl.Int64).to_numpy()
s = num("sender_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy()
r = np.where(p2p, num("receiver_id"), 1_000_000 + num("receiver_id"))  # merchants get their own code range
dn = num("device_id")                                                   # DV0317348 = device of account AC317348
NACC = int(s.max()) + 1
ts0 = tx["ts"].to_numpy().astype(np.int64); T0 = int(ts0.min())
print("n", n, "NACC", NACC, "sec-device share", (dn != s).mean(), "owned-by-other share", ((dn != s) & (dn < NACC)).mean())


def feat(ts):
    """All columns use only events strictly earlier than the row's ts. No labels."""
    o = {}
    # long table: one line per (account, role); role 0 = sent, 1 = received (P2P only)
    la = np.concatenate([s, r[p2p]]); lt = np.concatenate([ts, ts[p2p]]); lr = np.concatenate([np.zeros(n, np.int8), np.ones(p2p.sum(), np.int8)])
    od = np.argsort(la * BIG + lt, kind="stable"); lk = (la * BIG + lt)[od]; la, lt, lr = la[od], lt[od], lr[od]

    def lastk(A, k=5):
        pos = np.searchsorted(lk, A * BIG + ts, "left"); R = []; gap = None
        for j in range(1, k + 1):
            q = np.maximum(pos - j, 0); ok = (pos - j >= 0) & (la[q] == A)
            R.append(np.where(ok, lr[q], 2))
            if j == 1: gap = np.where(ok, ts - lt[q], np.nan)
        return R, gap

    # ---- a. ordered role transitions
    R, gap = lastk(s)
    o["a_snd_prev_role"] = np.where(R[0] == 2, np.nan, R[0])          # 1 = sender's previous event was a receive
    o["a_snd_prev_gap_log"] = np.log1p(gap)
    run = np.zeros(n); alive = np.ones(n, bool)
    for x in R: alive &= x == 1; run += alive
    o["a_snd_in_run"] = run                                           # consecutive receives immediately before this send
    o["a_snd_str3"] = (R[0] + 3 * R[1] + 9 * R[2]).astype(float)       # ordered role string of last 3 events (2 = none)
    R, gap = lastk(r)
    nanp = lambda x: np.where(p2p, x, np.nan)
    o["a_rcv_prev_role"] = nanp(np.where(R[0] == 2, np.nan, R[0]))     # 0 = receiver's previous event was a send
    o["a_rcv_prev_gap_log"] = nanp(np.log1p(gap))
    run = np.zeros(n); alive = np.ones(n, bool)
    for x in R: alive &= x == 0; run += alive
    o["a_rcv_out_run"] = nanp(run)
    o["a_rcv_str3"] = nanp((R[0] + 3 * R[1] + 9 * R[2]).astype(float))

    # ---- c. in / out activity change relative to the account's own earlier window (7 d recent vs 21 d before that)
    kin = np.sort(r[p2p] * BIG + ts[p2p]); kout = np.sort(s * BIG + ts)
    cnt = lambda keys, A, lo, hi: (np.searchsorted(keys, A * BIG + hi, "left") - np.searchsorted(keys, A * BIG + lo, "left")).astype(float)
    valid = ts - 28 * DAY >= T0
    for nm, A, m in (("snd", s, np.ones(n, bool)), ("rcv", r, p2p)):
        i7, o7 = cnt(kin, A, ts - 7 * DAY, ts), cnt(kout, A, ts - 7 * DAY, ts)
        iP, oP = cnt(kin, A, ts - 28 * DAY, ts - 7 * DAY), cnt(kout, A, ts - 28 * DAY, ts - 7 * DAY)
        ok = valid & m
        with np.errstate(invalid="ignore", divide="ignore"):
            o[f"c_{nm}_dshare"] = np.where(ok, i7 / (i7 + o7) - iP / (iP + oP), np.nan)   # NaN when a window is empty
        o[f"c_{nm}_act"] = np.where(ok, np.log((i7 + o7 + 1) / ((iP + oP) / 3 + 1)), np.nan)
        o[f"c_{nm}_in_ratio"] = np.where(ok, np.log((i7 + 1) / (iP / 3 + 1)), np.nan)
        o[f"c_{nm}_out_ratio"] = np.where(ok, np.log((o7 + 1) / (oP / 3 + 1)), np.nan)
        if nm == "rcv": o["c_rcv_first_in"] = np.where(ok, ((i7 + iP) == 0) & ((o7 + oP) >= 1), np.nan)   # sender-only history, now receives
        else: o["c_snd_first_out"] = np.where(ok, ((o7 + oP) == 0) & ((i7 + iP) >= 1), np.nan)            # receiver-only history, now sends

    # ---- b. secondary device + directional counterparty edges
    sec = dn != s; own = np.where(sec & (dn < NACC), dn, -1)
    pk = s * PB + r; od = np.lexsort((ts, pk)); pks = pk[od]; f1 = np.r_[True, pks[1:] != pks[:-1]]; upk, uft = pks[f1], ts[od][f1]

    def edge(a, b):  # a -> b exists strictly before the row
        q = a * PB + b; pos = np.minimum(np.searchsorted(upk, q), len(upk) - 1)
        return (upk[pos] == q) & (uft[pos] < ts)
    hv = own >= 0
    o["b_sec"] = sec.astype(float)
    o["b_own_to_rcv"] = (hv & edge(own, r)).astype(float)
    o["b_rcv_to_own"] = (hv & p2p & edge(r, own)).astype(float)
    o["b_snd_to_own"] = (hv & edge(s, own)).astype(float)
    o["b_own_to_snd"] = (hv & edge(own, s)).astype(float)
    dk = dn * PB + s; od = np.lexsort((ts, dk)); dks = dk[od]; f1 = np.r_[True, dks[1:] != dks[:-1]]
    ud, uu, uf = dn[od][f1], s[od][f1], ts[od][f1]                     # unique (device, user) with first use
    k2 = np.sort(ud * BIG + uf)
    pos = np.minimum(np.searchsorted(dks[f1], dk), f1.sum() - 1); self_before = uf[pos] < ts
    nco = np.searchsorted(k2, dn * BIG + ts, "left") - np.searchsorted(k2, dn * BIG, "left") - self_before
    o["b_dev_nco_before"] = nco.astype(float)
    # co-users of the device (earlier, not the sender) that already paid this row's receiver / were paid by it
    rows = pl.DataFrame({"i": np.arange(n), "d": dn, "s": s, "r": r, "t": ts}).filter(pl.Series(nco > 0))
    du = pl.DataFrame({"d": ud, "u": uu, "fu": uf})
    j = rows.join(du, on="d").filter((pl.col("fu") < pl.col("t")) & (pl.col("u") != pl.col("s")))
    pr = pl.DataFrame({"k": upk, "fp": uft})
    for nm, key in (("b_co_to_rcv", pl.col("u") * PB + pl.col("r")), ("b_rcv_to_co", pl.col("r") * PB + pl.col("u"))):
        h = j.with_columns(key.alias("k")).join(pr, on="k").filter(pl.col("fp") < pl.col("t")).group_by("i").agg(pl.col("u").n_unique().alias("c"))
        a = np.zeros(n); a[h["i"].to_numpy()] = h["c"].to_numpy(); o[nm] = a
    return o


def motifs(o):
    f = lambda x: np.nan_to_num(x, nan=0.0)
    g24 = np.expm1(f(o["a_snd_prev_gap_log"])) <= DAY
    return {
        "A1_snd_prev_in": f(o["a_snd_prev_role"]) == 1, "A1d_snd_prev_in_24h": (f(o["a_snd_prev_role"]) == 1) & g24,
        "A2_rcv_prev_out": o["a_rcv_prev_role"] == 0, "A3_snd_in_run2": o["a_snd_in_run"] >= 2, "A4_rcv_out_run2": f(o["a_rcv_out_run"]) >= 2,
        "B1_sec": o["b_sec"] == 1, "B2_own_rcv_edge": (o["b_own_to_rcv"] + o["b_rcv_to_own"]) > 0,
        "B3_snd_own_edge": (o["b_snd_to_own"] + o["b_own_to_snd"]) > 0, "B4_co_rcv_edge": (o["b_co_to_rcv"] + o["b_rcv_to_co"]) > 0,
        "B5_dev_co_before": o["b_dev_nco_before"] >= 1,
        "C1_snd_dshare_up": f(o["c_snd_dshare"]) > .25, "C2_rcv_dshare_up": f(o["c_rcv_dshare"]) > .25, "C3_rcv_first_in": f(o["c_rcv_first_in"]) == 1,
        "C4_snd_first_out": f(o["c_snd_first_out"]) == 1, "C5_snd_act_up": f(o["c_snd_act"]) > .5, "C6_rcv_in_up": f(o["c_rcv_in_ratio"]) > .5,
        "C7_snd_out_up": f(o["c_snd_out_ratio"]) > .5,
    }


# ---------------- evaluation frame (labels used here only)
y = tx["y"].to_numpy().astype(float); month = tx["month"].to_numpy(); amt = tx["amount"].to_numpy()
t, F = fraud_events(tx, 48 * 3600)
assert (t["txn_id"] == tx["txn_id"]).all()
ev = np.full(n, -1); ev[F["i"].to_numpy()] = F["ev"].to_numpy()
evn = np.zeros(n); u, c = np.unique(ev[ev >= 0], return_counts=True); evn[ev >= 0] = c[np.searchsorted(u, ev[ev >= 0])]
rows = pl.read_parquet("scratch_gap/N_sep_rows.parquet", columns=["txn_id"]).with_columns(pl.Series("q", np.load("scratch_gap/dec_N_wt.npz")["C_q"]))
q = tx.select("txn_id").join(rows, on="txn_id", how="left", maintain_order="left")["q"].fill_null(-1).to_numpy()
rng = np.random.default_rng(0)


def one_per_event(mask):
    i = np.where(mask)[0]; i = i[rng.permutation(len(i))]
    _, k = np.unique(ev[i], return_index=True); return np.sort(i[k])


POP = {f"single_m{m}": np.where((y > 0) & (evn == 1) & (month == m))[0] for m in (7, 8, 9)}
POP["missed_m9"] = one_per_event((y > 0) & (q == 0) & (month == 9))
POP["caught_m9"] = one_per_event((y > 0) & (q > 0) & (month == 9))
POP["multi3_m8"] = one_per_event((y > 0) & (evn >= 3) & (month == 8))
for m_ in (7, 8, 9):   # addendum: model-independent populations (all fraud-labelled events; those on the sender's own device)
    POP[f"allfraud_m{m_}"] = one_per_event((y > 0) & (month == m_))
    POP[f"owndev_fraud_m{m_}"] = one_per_event((y > 0) & (month == m_) & (dn == s))
print({k: len(v) for k, v in POP.items()})
snd_n = tx.select(pl.len().over("sender_id", "month"))["len"].to_numpy()
cell = (month.astype(np.int64) * 10**6 + tx["txn_type"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) * 10**4
        + p2p * 10**3 + np.digitize(amt, [100, 300, 1000, 3000, 10000]) * 10 + np.digitize(snd_n, [2, 3, 5, 9]))
ex = tx.select("txn_id").join(pl.read_parquet("scratch_gap/N_f1.parquet", columns=["txn_id", "flow__sin_n24h", "scam__rcv_in_15d", "takeover__n_other_dev", "burst__ds_n_sender"]),
                              on="txn_id", how="left", maintain_order="left")
print("existing feature coverage", ex.null_count().row(0), "of", n)
b = lambda c, e: np.digitize(np.nan_to_num(ex[c].fill_null(float("nan")).to_numpy(), nan=-1), e).astype(np.int64)
cellc = cell * 1000 + b("flow__sin_n24h", [0, 1, 2]) * 100 + b("scam__rcv_in_15d", [0, 1, 2, 3, 6]) * 10 + b("takeover__n_other_dev", [0, 1, 2])
legit = (y == 0)
VALIDC = lambda ts: ts - 28 * DAY >= T0


def evaluate(M, ts, cells, tag, pops=POP, extra=None):
    out = []
    for nm, m in M.items():
        v = VALIDC(ts) if nm.startswith("C") else np.ones(n, bool)
        ctl = pl.DataFrame({"c": cells[legit & v], "m": m[legit & v].astype(float)}).group_by("c").agg(pl.col("m").mean().alias("rc"), pl.len().alias("nc"))
        for pn, idx in pops.items():
            idx = idx[v[idx]]
            if not len(idx): continue
            d = pl.DataFrame({"c": cells[idx], "m": m[idx].astype(float), "y": y[idx]}).join(ctl, on="c").filter(pl.col("nc") >= 5)
            for cl, dd in ([("all", d)] + ([(LABELS[int(k[0])], g) for k, g in d.group_by("y")] if extra else [])):
                if dd.height < 20: continue
                diff = (dd["m"] - dd["rc"]).to_numpy()
                out.append({"variant": tag, "motif": nm, "pop": pn, "cls": cl, "n_ev": dd.height, "rate_fraud": dd["m"].mean(), "rate_ctrl": dd["rc"].mean(),
                            "diff": diff.mean(), "z": diff.mean() / max(diff.std(ddof=1) / np.sqrt(len(diff)), 1e-12)})
    return out


def shuf(g, seed):
    rg = np.random.default_rng(seed); o1 = np.lexsort((rg.random(n), g)); o2 = np.argsort(g, kind="stable"); x = np.empty(n, np.int64); x[o1] = ts0[o2]; return x


o = feat(ts0); M = motifs(o)
assert all(len(v) == n for v in o.values())
pl.DataFrame({"txn_id": tx["txn_id"], **{"rel__" + k: v.astype(np.float32) for k, v in o.items()}}).write_parquet(D + "rel_feats.parquet")
assert tx["txn_id"].n_unique() == n
res = evaluate(M, ts0, cell, "real") + evaluate(M, ts0, cellc, "real_cond")
resc = evaluate(M, ts0, cell, "real", {k: POP[k] for k in ("missed_m9", "single_m8", "single_m7")}, extra=True)
for tag, g in (("shuf_day", ts0 // DAY), ("shuf_snd", s)):
    tsh = shuf(g, 1); res += evaluate(motifs(feat(tsh)), tsh, cell, tag)
R = pl.DataFrame(res); R.write_csv(D + "results.csv"); pl.DataFrame(resc).write_csv(D + "results_class.csv")
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
W = R.with_columns((pl.col("rate_fraud") * 100).round(2), (pl.col("rate_ctrl") * 100).round(2), (pl.col("diff") * 100).round(2), pl.col("z").round(1))
for v in ("real", "real_cond", "shuf_day", "shuf_snd"):
    print("\n=====", v, "(rates in %, diff in points, z = paired cell-standardised)")
    print(W.filter(pl.col("variant") == v).pivot(on="pop", index="motif", values=["diff", "z"]).sort("motif"))
print(W.filter((pl.col("variant") == "real") & pl.col("pop").is_in(["missed_m9", "single_m8", "caught_m9"])).select("motif", "pop", "n_ev", "rate_fraud", "rate_ctrl", "diff", "z").sort("motif", "pop"))
RC = pl.DataFrame(resc).filter((pl.col("cls") != "all") & (pl.col("z").abs() > 3)).with_columns((pl.col("diff") * 100).round(2), pl.col("z").round(1)).sort("motif", "cls", "pop")
print("\nper-class |z| > 3 (real ts):"); print(RC.select("motif", "cls", "pop", "n_ev", "diff", "z"))
