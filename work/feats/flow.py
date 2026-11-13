"""Money-flow features for mule_account and layering_chain (see notes/flow.md).

Generator signatures (train, verified):
- mule: P2P receiver gets ~14 (5-24) inbound PAYs from distinct new_payee senders, amounts multiples of 10 (500-5000),
  spread over ~3-18h, then forwards 1-4 integer amounts (sum ~ 0.8-0.9x inflow) to other P2P accounts 30-150 min later.
- layering: P2P hops A->B->C..., 3-6 rows, integer amounts, each hop 0.96-0.995x previous, 2 min-5 h apart.
Only structure is used (no labels). Windows are symmetric or short so Oct (half month) is not truncated much.
"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

P = "flow__"
BIG = 10**10  # key = entity_code * BIG + ts  (ts < 1e10)
H = 3600


def _win(keys_sorted, cum, q_lo, q_hi):
    """sum of values with key in [q_lo, q_hi] given sorted keys and prefix sums cum (len n+1)."""
    lo = np.searchsorted(keys_sorted, q_lo, "left"); hi = np.searchsorted(keys_sorted, q_hi, "right")
    return cum[hi] - cum[lo]


def _prefix(v):
    return np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])


def build(tx: pl.DataFrame) -> pl.DataFrame:
    t = tx.select("txn_id", "sender_id", "receiver_id", "receiver_type", "ts", "amount", "new_payee").with_row_index("i")
    p2p_mask = (t["receiver_type"] == "P2P").to_numpy()
    # shared account coding for sender & receiver
    codes = pl.concat([t["sender_id"], t["receiver_id"]]).unique().sort()
    cmap = pl.DataFrame({"id": codes, "c": np.arange(len(codes), dtype=np.int64)})
    t = t.join(cmap.rename({"id": "sender_id", "c": "sc"}), on="sender_id", how="left") \
         .join(cmap.rename({"id": "receiver_id", "c": "rc"}), on="receiver_id", how="left").sort("i")
    ts = t["ts"].to_numpy().astype(np.int64); amt = t["amount"].to_numpy()
    sc = t["sc"].to_numpy(); rc = t["rc"].to_numpy()
    npay = t["new_payee"].fill_null(0).to_numpy().astype(float)
    r10 = (np.round(amt * 100) % 1000 == 0).astype(float)  # multiple of 10 rupees
    n = len(ts)
    out = {}

    # ---- P2P inbound index keyed by receiver
    pi = np.where(p2p_mask)[0]
    kin = rc[pi] * BIG + ts[pi]; o = np.argsort(kin, kind="stable"); kin = kin[o]; pin = pi[o]
    c_one = _prefix(np.ones(len(pin))); c_np = _prefix(npay[pin]); c_r10 = _prefix(r10[pin]); c_amt = _prefix(amt[pin])
    # ---- P2P outbound index keyed by sender
    kout = sc[pi] * BIG + ts[pi]; o2 = np.argsort(kout, kind="stable"); kout = kout[o2]; pout = pi[o2]
    co_one = _prefix(np.ones(len(pout))); co_amt = _prefix(amt[pout])

    def nanp2p(x):
        x = x.astype(np.float64); x[~p2p_mask] = np.nan; return x

    # A. receiver-side burst (this txn's receiver), symmetric windows
    qr = rc * BIG + ts
    for w in (3, 12):
        cnt = _win(kin, c_one, qr - w * H, qr + w * H)
        out[f"rin_n{w}h"] = nanp2p(cnt)
        if w == 12:
            out["rin_np_frac12h"] = nanp2p(_win(kin, c_np, qr - w * H, qr + w * H) / np.maximum(cnt, 1))
            out["rin_r10_frac12h"] = nanp2p(_win(kin, c_r10, qr - w * H, qr + w * H) / np.maximum(cnt, 1))
            in_sum12 = _win(kin, c_amt, qr - w * H, qr + w * H)
    # same sender->receiver pair repeats in +-12h (velocity bot / repeat payer, not a mule victim)
    pid = t.select(pl.struct("sc", "rc").rank("dense").cast(pl.Int64))["sc"].to_numpy()
    kp = pid[pi] * BIG + ts[pi]; kp.sort(); qp = pid * BIG + ts
    out["pair_n12h"] = nanp2p(_win(kp, _prefix(np.ones(len(kp))), qp - 12 * H, qp + 12 * H))
    # receiver forwards money on: P2P outbound from receiver within (t, t+6h]
    rout_n6 = _win(kout, co_one, qr + 1, qr + 6 * H); rout_s6 = _win(kout, co_amt, qr + 1, qr + 6 * H)
    out["rout_n6h"] = nanp2p(rout_n6)
    out["rout_ratio_in12"] = nanp2p(rout_s6 / np.maximum(in_sum12, 1))
    # B. sender-side: inbound to sender before this txn (mule forward / chain hop)
    qs = sc * BIG + ts
    sin_n24 = _win(kin, c_one, qs - 24 * H, qs - 1); sin_s24 = _win(kin, c_amt, qs - 24 * H, qs - 1)
    out["sin_n24h"] = nanp2p(sin_n24)
    out["sin_amt_ratio24"] = nanp2p(np.where(sin_s24 > 0, amt / np.maximum(sin_s24, 1), np.nan))
    out["sin_r10_frac24"] = nanp2p(np.where(sin_n24 > 0, _win(kin, c_r10, qs - 24 * H, qs - 1) / np.maximum(sin_n24, 1), np.nan))

    # C. layering hops: pairs (A->B at t1, B->C at t2), 0<t2-t1<=6h, ratio a2/a1 in [0.9,1)
    # middle account B must not be a burst receiver (else mule inflow->forward makes spurious edges)
    p = t.with_columns(pl.Series("rin3", out["rin_n3h"])).filter(pl.col("receiver_type") == "P2P").select("i", "sc", "rc", "ts", "amount", "rin3")
    pr = p.join(p, left_on="rc", right_on="sc", suffix="2").filter(
        (pl.col("ts2") > pl.col("ts")) & (pl.col("ts2") - pl.col("ts") <= 6 * H)).with_columns(
        (pl.col("amount2") / pl.col("amount")).alias("ratio"), ((pl.col("ts2") - pl.col("ts")) / 60).alias("gap"))
    e = pr.filter((pl.col("ratio") >= 0.9) & (pl.col("ratio") < 1.0) & (pl.col("rin3") <= 3))
    i1 = e["i"].to_numpy(); i2 = e["i2"].to_numpy()
    nxt = e.sort("gap").group_by("i").agg(pl.col("ratio").first().alias("nr"), pl.col("gap").first().alias("ng"))
    prv = e.sort("gap").group_by("i2").agg(pl.col("ratio").first().alias("pr"), pl.col("gap").first().alias("pg"))
    for df, col, keys in ((nxt, "i", ("nr", "ng")), (prv, "i2", ("pr", "pg"))):
        idx = df[col].to_numpy()
        for k in keys:
            a = np.full(n, np.nan); a[idx] = df[k].to_numpy(); out[f"hop_{k}"] = a
    A = coo_matrix((np.ones(len(i1)), (i1, i2)), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    inE = np.zeros(n, bool); inE[i1] = True; inE[i2] = True
    cdf = pl.DataFrame({"i": np.arange(n), "comp": comp, "ts": ts, "sin": sin_n24, "rin": out["rin_n3h"]}).filter(pl.Series(inE))
    cagg = cdf.group_by("comp").agg(pl.len().alias("csize"), ((pl.col("ts").max() - pl.col("ts").min()) / 60).alias("cspan"),
                                    pl.col("sin").max().alias("csin"), pl.col("rin").max().alias("crin"))
    cdf = cdf.join(cagg, on="comp").with_columns(pl.col("ts").rank("ordinal").over("comp").alias("cpos"))
    ci = cdf["i"].to_numpy()
    for k in ("csize", "cspan", "csin", "crin", "cpos"):
        a = np.full(n, np.nan); a[ci] = cdf[k].to_numpy().astype(float); out[f"chain_{k}"] = a
    out["chain_size"] = np.where(p2p_mask, np.nan_to_num(out.pop("chain_csize"), nan=1.0), np.nan)
    return pl.DataFrame({"txn_id": t["txn_id"], **{P + k: v for k, v in out.items()}})
