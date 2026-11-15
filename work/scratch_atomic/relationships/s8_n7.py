"""Agent B: the plain backward 7 d sender counts that survive in the joint test (label-free). Writes rel_n7.parquet for all rows.
rel__snd_n7 = sender's events in [t-7d, t) in both roles; rel__snd_out7 / rel__snd_in7 = sent / received parts; NaN before data start + 7 d."""
import numpy as np, polars as pl
from common import load_all
DAY, BIG = 86400, 10**10
tx = load_all(); n = tx.height
num = lambda c: tx[c].str.slice(2).cast(pl.Int64).to_numpy()
s = num("sender_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy(); r = np.where(p2p, num("receiver_id"), -1)
ts = tx["ts"].to_numpy().astype(np.int64)
kin = np.sort(r[p2p] * BIG + ts[p2p]); kout = np.sort(s * BIG + ts)
cnt = lambda keys: (np.searchsorted(keys, s * BIG + ts, "left") - np.searchsorted(keys, s * BIG + ts - 7 * DAY, "left")).astype(np.float32)
i7, o7 = cnt(kin), cnt(kout); ok = ts - 7 * DAY >= ts.min()
f = pl.DataFrame({"txn_id": tx["txn_id"], "rel__snd_n7": np.where(ok, i7 + o7, np.nan), "rel__snd_out7": np.where(ok, o7, np.nan), "rel__snd_in7": np.where(ok, i7, np.nan)})
assert f.height == 1824642 and f["txn_id"].n_unique() == f.height
f.write_parquet("scratch_atomic/relationships/rel_n7.parquet")
print(f.group_by(tx["month"]).agg(pl.col("rel__snd_n7").mean(), pl.col("rel__snd_out7").mean()).sort("month"))
