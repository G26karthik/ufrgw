"""Mechanism `relcx`: agent B's pattern c, sender side (activity of the sender in the last 7 d against the sender's own earlier
window), with one change so that the July fit rows of fold 0 carry the columns: B's prior window is [t-28 d, t-7 d) and B sets the
columns to NaN before data start + 28 d (29 July). Here the prior window is the part of [t-28 d, t-7 d) that lies inside the
data, its counts are scaled to 7 d by its real length, and the columns are NaN only before data start + 14 d. For every row
at or after data start + 28 d (all of August, September, October) the five columns equal B's `rel__c_snd_*` exactly (asserted
by `python scratch_atomic/verification/mech_relcx.py`). Only rows strictly earlier than the row are counted. No labels are read."""
import numpy as np
import polars as pl

DAY, BIG = 86400, 10**10


def build(tx):
    num = lambda k: tx[k].str.slice(2).cast(pl.Int64).to_numpy()
    s = num("sender_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy(); ts = tx["ts"].to_numpy().astype(np.int64); T0 = int(ts.min())
    r = np.where(p2p, num("receiver_id"), 1_000_000 + num("receiver_id"))
    kin = np.sort(r[p2p] * BIG + ts[p2p]); kout = np.sort(s * BIG + ts)
    cnt = lambda keys, lo, hi: (np.searchsorted(keys, s * BIG + hi, "left") - np.searchsorted(keys, s * BIG + lo, "left")).astype(float)
    i7, o7 = cnt(kin, ts - 7 * DAY, ts), cnt(kout, ts - 7 * DAY, ts)
    iP, oP = cnt(kin, ts - 28 * DAY, ts - 7 * DAY), cnt(kout, ts - 28 * DAY, ts - 7 * DAY)
    ok = ts - 14 * DAY >= T0; k = np.maximum(np.minimum(21 * DAY, ts - 7 * DAY - T0), 1) / (7 * DAY)  # prior window length in units of 7 d (3 when complete)
    with np.errstate(invalid="ignore", divide="ignore"):
        o = {"dshare": np.where(ok, i7 / (i7 + o7) - iP / (iP + oP), np.nan)}
    o["act"] = np.where(ok, np.log((i7 + o7 + 1) / ((iP + oP) / k + 1)), np.nan)
    o["in_ratio"] = np.where(ok, np.log((i7 + 1) / (iP / k + 1)), np.nan)
    o["out_ratio"] = np.where(ok, np.log((o7 + 1) / (oP / k + 1)), np.nan)
    o["first_out"] = np.where(ok, ((o7 + oP) == 0) & ((i7 + iP) >= 1), np.nan)
    return pl.DataFrame({"txn_id": tx["txn_id"], **{"relcx__snd_" + a: v.astype(np.float32) for a, v in o.items()}})


if __name__ == "__main__":  # self-check against B's columns on the all-rows frame
    from common import load_all
    tx = load_all(); F = build(tx); B = tx.select("txn_id").join(pl.read_parquet("scratch_atomic/relationships/rel_feats.parquet"), on="txn_id", how="left", maintain_order="left")
    late = (tx["ts"].to_numpy() - 28 * DAY >= int(tx["ts"].min()))
    for a in ("dshare", "act", "in_ratio", "out_ratio", "first_out"):
        x, y = F["relcx__snd_" + a].to_numpy(), B["rel__c_snd_" + a].to_numpy()
        assert np.array_equal(x[late], y[late], equal_nan=True), a
        print(f"{a}: equal to B's column on all {int(late.sum())} rows from data start + 28 d; extra non-NaN rows before that: {int((~np.isnan(x[~late])).sum())} (B: {int((~np.isnan(y[~late])).sum())})")
