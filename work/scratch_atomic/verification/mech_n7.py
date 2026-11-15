"""Mechanism `n7` (agent B's final candidate, code of scratch_atomic/relationships/s8_n7.py as a build function): backward 7-day
counts of the sender. n7__snd_n7 = events of the sender in [t-7 d, t) in both roles (sent anything, received P2P), n7__snd_out7 = sent,
n7__snd_in7 = received; NaN before data start + 7 d. Only rows strictly earlier than the row are counted. No labels are read."""
import numpy as np
import polars as pl

DAY, BIG = 86400, 10**10


def build(tx):
    num = lambda k: tx[k].str.slice(2).cast(pl.Int64).to_numpy()
    s = num("sender_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy(); r = np.where(p2p, num("receiver_id"), -1)
    ts = tx["ts"].to_numpy().astype(np.int64)
    kin = np.sort(r[p2p] * BIG + ts[p2p]); kout = np.sort(s * BIG + ts)
    cnt = lambda keys: (np.searchsorted(keys, s * BIG + ts, "left") - np.searchsorted(keys, s * BIG + ts - 7 * DAY, "left")).astype(np.float32)
    i7, o7 = cnt(kin), cnt(kout); ok = ts - 7 * DAY >= ts.min()
    return pl.DataFrame({"txn_id": tx["txn_id"], "n7__snd_n7": np.where(ok, i7 + o7, np.nan).astype(np.float32),
                         "n7__snd_out7": np.where(ok, o7, np.nan).astype(np.float32), "n7__snd_in7": np.where(ok, i7, np.nan).astype(np.float32)})
