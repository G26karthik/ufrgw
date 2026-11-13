"""scam: phishing_collect, fake_cashback, qr_swap signatures (see work/notes/scam.md).

Generator signatures found in train:
- phishing: COLLECT to P2P, new_payee=1, amount = R-1 / R-10 / R-50 for R in {2000,5000,10000,25000,...};
  phishing receivers get several such collects over days. Also bait variant: fraudster sends Rs 1/2/5/10, victim pays
  a big (non-round) amount back within 3h.
- fake_cashback: fraudster sends 1-2 tiny (1/2/5/10) PAYs to victim, victim pays/approves big amount back 13 min..47 h later.
- qr_swap: QR txn whose payee is a P2P account (merchant_category still set) or a merchant QR with no mismatch check
  (qr_vpa_mismatch null) and round amount; fake payee receives several QR scans.
All counts are raw: density.run_feature_matched builds train on half-density halves, so train and Oct sit at the same
density (month-volume scaling doubled Oct fraud-event counts). Only window truncation at the data edges is corrected.
"""
import numpy as np
import polars as pl

P = "scam__"
DAY = 86400
ROUNDS = [100, 200, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000, 100000, 200000]


def _code(col: str) -> pl.Expr:
    # AC/DV ids -> int; MR merchants offset so they never collide with accounts
    c = pl.col(col)
    return pl.when(c.str.starts_with("MR")).then(c.str.slice(2).cast(pl.Int64) + 10_000_000).otherwise(c.str.slice(2).cast(pl.Int64))


def _wcount(ev_key, ev_ts, q_key, q_ts, before, after):
    """# events with same key and ts in [q_ts-before, q_ts+after]. Windows must stay < 2^32 - ts span (keys packed in high bits)."""
    ev = np.sort(ev_key.astype(np.int64) * (1 << 32) + ev_ts)
    q = q_key.astype(np.int64) * (1 << 32) + q_ts
    return (np.searchsorted(ev, q + after, "right") - np.searchsorted(ev, q - before, "left")).astype(np.float32)


def build(tx: pl.DataFrame) -> pl.DataFrame:
    a = pl.col("amount")
    # nearest round number at or above amount
    r_above = pl.lit(None, pl.Float64)
    for R in reversed(ROUNDS):
        r_above = pl.when(a <= R).then(pl.lit(float(R))).otherwise(r_above)
    d = tx.select(
        "txn_id", "ts", "amount", "txn_type", "receiver_type", "new_payee", "qr_vpa_mismatch", "month",
        "device_city", "rcv_home_city",
        _code("sender_id").alias("s"), _code("receiver_id").alias("r"),
        r_above.alias("R"),
    ).with_columns((pl.col("R") - a).alias("gap"))
    d = d.with_columns(
        (pl.col("gap").is_in([1.0, 10.0, 50.0]) & (pl.col("R") >= 500)).alias("under"),
        ((a <= 10) & (a == a.round(0))).alias("tiny"),
        (pl.col("receiver_type") == "P2P").alias("p2p"),
        (pl.col("txn_type") == "COLLECT").alias("collect"),
        (pl.col("txn_type") == "QR").alias("qr"),
    )
    s = d["s"].to_numpy(); r = d["r"].to_numpy(); ts = d["ts"].to_numpy().astype(np.int64)
    t0 = ts.min(); ts = ts - t0
    m = {c: d[c].to_numpy() for c in ["under", "tiny", "p2p", "collect", "qr", "new_payee"]}
    mm = d["qr_vpa_mismatch"].to_numpy()
    t_end = int(d.filter(pl.col("month") <= 10)["ts"].max()) - t0  # background stops at Oct 31; Nov rows are spill only

    def bg(ev_mask, ev_key, q_key, w):
        """Background-volume count minus self, divided by the share of the +-w window inside the data (only for
        background volume: fraud events are complete at the edges, Oct events spill into Nov)."""
        n = _wcount(ev_key[ev_mask], ts[ev_mask], q_key, ts, w, w) - ev_mask
        tq = np.minimum(ts, t_end)
        return (n / np.clip((np.minimum(tq + w, t_end) - np.maximum(tq - w, 0)) / (2 * w), 0.5, 1)).astype(np.float32)
    out = {}

    def ev(mask, key):
        return key[mask], ts[mask]

    # ---- phishing: under-round new-payee collects into the same receiver (event count, not volume) ----
    ph = m["under"] & m["collect"] & m["p2p"] & (m["new_payee"] == 1)
    out["rcv_phcol_15d"] = _wcount(*ev(ph, r), r, ts, 15 * DAY, 15 * DAY) - ph
    out["rcv_collect_15d"] = _wcount(*ev(m["collect"] & m["p2p"], r), r, ts, 15 * DAY, 15 * DAY) - (m["collect"] & m["p2p"])
    np_in = m["p2p"] & (m["new_payee"] == 1)
    out["rcv_newpayee_in_3d"] = bg(np_in, r, r, 3 * DAY)
    out["und_gap"] = np.where(m["under"], d["gap"].to_numpy(), -1).astype(np.float32)
    out["und_relgap"] = (d["gap"] / d["R"]).fill_null(1.0).to_numpy().astype(np.float32)
    BIG = 10**9  # whole data (all months) window: Oct rows see Jul-Sep history, more stable than +-15d at the data edge
    out["rcv_phcol_all"] = _wcount(*ev(ph, r), r, ts, BIG, BIG) - ph

    # ---- fake_cashback / bait: tiny P2P payments and reverse edges ----
    tiny_p2p = m["tiny"] & m["p2p"]
    out["snd_tiny_out_3d"] = _wcount(*ev(tiny_p2p, s), s, ts, 3 * DAY, 3 * DAY) - tiny_p2p
    out["rcv_tiny_in_3d"] = _wcount(*ev(tiny_p2p, r), r, ts, 3 * DAY, 3 * DAY) - tiny_p2p  # victim pinged
    # pair-level reverse edges via joins (tiny rows are few)
    base = d.select("txn_id", "s", "r", "ts", "amount", "p2p", "tiny").with_columns(pl.col("ts") - t0)
    tiny_df = base.filter(pl.col("tiny") & pl.col("p2p")).select(pl.col("s").alias("f"), pl.col("r").alias("v"), pl.col("ts").alias("tt"), pl.col("amount").alias("ta"))
    big_df = base.filter(~pl.col("tiny") & pl.col("p2p")).select(pl.col("txn_id").alias("bid"), pl.col("s").alias("v"), pl.col("r").alias("f"), pl.col("ts").alias("bt"), pl.col("amount").alias("ba"))
    # for a big row v->f: tiny f->v before it (within 72h)
    j = big_df.join(tiny_df, on=["f", "v"]).with_columns(((pl.col("bt") - pl.col("tt")) / 60).alias("dt")).filter((pl.col("dt") > 0) & (pl.col("dt") < 72 * 60))
    jb = j.group_by("bid").agg(pl.len().alias("bait_before_n"), pl.col("dt").min().alias("bait_before_min"), pl.col("ta").max().alias("bait_amt"))
    # for a tiny row f->v: big v->f after it
    tiny_id = base.filter(pl.col("tiny") & pl.col("p2p")).select(pl.col("txn_id").alias("tid"), pl.col("s").alias("f"), pl.col("r").alias("v"), pl.col("ts").alias("tt"))
    j2 = tiny_id.join(big_df, on=["f", "v"]).with_columns(((pl.col("bt") - pl.col("tt")) / 60).alias("dt")).filter((pl.col("dt") > 0) & (pl.col("dt") < 72 * 60))
    jt = j2.group_by("tid").agg(pl.len().alias("big_after_n"), pl.col("dt").min().alias("big_after_min"), pl.col("ba").max().log1p().alias("big_after_logamt"))
    # tiny rows in same pair (f->v) +-3d: bait count of the pair
    pair_tiny = tiny_id.join(tiny_id.select(pl.col("f"), pl.col("v"), pl.col("tt").alias("tt2")), on=["f", "v"]).filter((pl.col("tt") - pl.col("tt2")).abs() < 3 * DAY).group_by("tid").agg((pl.len() - 1).alias("pair_tiny_n"))
    # any reverse edge ever (P2P), excluding tiny/big logic
    pairs = base.filter(pl.col("p2p")).select("s", "r").unique()
    rev = pairs.join(pairs, left_on=["s", "r"], right_on=["r", "s"]).select("s", "r").with_columns(pl.lit(1, pl.Int8).alias("rev_pair"))

    # ---- qr_swap: QR inflows to the same payee ----
    qr = m["qr"]
    qr_p2p = qr & m["p2p"]
    out["rcv_qr_p2p_in_15d"] = _wcount(*ev(qr_p2p, r), r, ts, 15 * DAY, 15 * DAY) - qr_p2p
    mm1 = qr & (mm == 1)
    qr_in = _wcount(*ev(qr, r), r, ts, 15 * DAY, 15 * DAY) - qr
    out["rcv_qr_mm1_frac_15d"] = (_wcount(*ev(mm1, r), r, ts, 15 * DAY, 15 * DAY) - mm1) / np.maximum(qr_in, 1)
    out["rcv_qr_p2p_in_all"] = _wcount(*ev(qr_p2p, r), r, ts, BIG, BIG) - qr_p2p
    mmn = qr & ~m["p2p"] & np.isnan(mm)
    out["rcv_qr_mmnull_15d"] = _wcount(*ev(mmn, r), r, ts, 15 * DAY, 15 * DAY) - mmn
    out["snd_qr_mmnull_15d"] = _wcount(*ev(mmn, s), s, ts, 15 * DAY, 15 * DAY) - mmn
    # pair repetition (merchant collusion lookalike) and refund history (fraud-event counts: raw, no edge correction)
    _, inv = np.unique(s * 20_000_000 + r, return_inverse=True)  # rank-encode pair so it packs with ts
    out["pair_n_15d"] = _wcount(inv, ts, inv, ts, 15 * DAY, 15 * DAY) - 1
    refund = (d["txn_type"] == "REFUND").to_numpy()
    out["snd_refund_15d"] = _wcount(*ev(refund, s), s, ts, 15 * DAY, 15 * DAY) - refund
    in_raw = _wcount(r, ts, r, ts, 15 * DAY, 15 * DAY) - 1
    out["rcv_in_15d"] = bg(np.ones(len(ts), bool), r, r, 15 * DAY)
    # fake/small merchants receive round 1000/2000/2500/5000/10000 payments (qr_swap / collusion / refund lookalikes)
    rnd = np.isin(d["amount"].to_numpy(), [1000.0, 2000.0, 2500.0, 5000.0, 10000.0])
    rcv_rnd = _wcount(*ev(rnd, r), r, ts, 15 * DAY, 15 * DAY) - rnd
    out["rcv_round_frac_15d"] = rcv_rnd / np.maximum(in_raw, 1)

    res = d.select("txn_id", "s", "r", "device_city", "rcv_home_city", "p2p", "qr").with_columns(
        **{k: pl.Series(v) for k, v in out.items()})
    res = (res.join(jb, left_on="txn_id", right_on="bid", how="left")
              .join(jt, left_on="txn_id", right_on="tid", how="left")
              .join(pair_tiny, left_on="txn_id", right_on="tid", how="left")
              .join(rev, on=["s", "r"], how="left"))
    res = res.with_columns(
        pl.col("bait_before_n", "big_after_n", "pair_tiny_n", "rev_pair").fill_null(0),
        (pl.col("qr") & pl.col("p2p") & (pl.col("device_city") == pl.col("rcv_home_city"))).cast(pl.Int8).alias("qrp2p_payee_local"),
    ).drop("s", "r", "device_city", "rcv_home_city", "p2p", "qr")
    return res.rename({c: P + c for c in res.columns if c != "txn_id"})
