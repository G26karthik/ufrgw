"""burst: velocity_bot (device bursts) + refund_abuse (repeated REFUND claims) features.

Generator signatures (see notes/burst.md): velocity_bot = one device, ~40 integer-amount payments
(median ~245) to 1-3 ecommerce/gaming merchants in ~8 min, often many distinct sender accounts.
refund_abuse = one sender, 3-8 REFUND rows over ~4-9 days, mostly to 1-2 ecommerce/electronics merchants.
Only symmetric windows / event-intrinsic stats or ratios: Oct volume per day is half of Jul-Sep, so raw
activity counts of busy entities (merchants) and long-window sender counts drift; event-intrinsic counts do not.
"""
from datetime import datetime
import numpy as np
import polars as pl

P = "burst__"


def _win(tx, key, w, flag=None):
    """Rows with same key within +-w seconds (incl. self); if flag given, count flagged rows."""
    codes = tx[key].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    ts = tx["ts"].to_numpy().astype(np.int64)
    z = codes * 10**10 + ts
    o = np.argsort(z, kind="stable"); zs = z[o]
    lo = np.searchsorted(zs, zs - w, "left"); hi = np.searchsorted(zs, zs + w, "right")
    if flag is None:
        r = hi - lo
    else:
        cs = np.concatenate([[0], np.cumsum(flag[o])])
        r = cs[hi] - cs[lo]
    out = np.empty_like(r); out[o] = r
    return out


def _loo(num, den, own):
    """Share among the OTHER rows in the window (self removed); NaN when the row is alone."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1, (num - own) / (den - 1), np.nan)


assert np.allclose(_loo(np.array([3, 1]), np.array([4, 1]), np.array([1, 1])), [2 / 3, np.nan], equal_nan=True)


def _sessions(t, key, gap, pre, other=None, min_n=3):
    """Split each key's rows into sessions (gap > `gap` s starts a new one); broadcast session stats to rows."""
    t = t.sort(key, "ts")
    d = pl.col("ts").diff().over(key)
    t = t.with_columns((d.fill_null(10**9) > gap).cum_sum().alias("_s"), d.alias("_gp"), (-pl.col("ts").diff(-1).over(key)).alias("_gn"))
    a = pl.col("amount")
    ex = [pl.col(other).n_unique().over("_s").alias(pre + "n_" + other.split("_")[0])] if other else []
    t = t.with_columns(
        *ex,
        pl.len().over("_s").alias(pre + "n"),
        pl.col("receiver_id").n_unique().over("_s").alias(pre + "nrcv"),
        ((pl.col("ts").max() - pl.col("ts").min()).over("_s") / 60).alias(pre + "dur_min"),
        ((a * 100).round() % 100 == 0).mean().over("_s").alias(pre + "int_frac"),
        # coarse amount band, not the median: a per-event continuous stat acts as an event ID and overfits
        (a <= 500).mean().over("_s").alias(pre + "small_frac"),
        pl.col("_cat").mean().over("_s").alias(pre + "cat_frac"),
        pl.min_horizontal(pl.col("_gp"), pl.col("_gn")).clip(upper_bound=3600).fill_null(3600).alias(pre + "min_gap"),
    )
    # session stats only where a session exists; for singletons they would just copy (drifting) row values
    stats = [c for c in t.columns if c.startswith(pre) and c not in (pre + "n", pre + "min_gap")]
    t = t.with_columns(pl.when(pl.col(pre + "n") >= min_n).then(pl.col(c)).alias(c) for c in stats)
    return t.select("txn_id", pl.col(f"^{pre}.*$"))


def build(tx: pl.DataFrame) -> pl.DataFrame:
    tx = tx.select("txn_id", "ts", "sender_id", "device_id", "receiver_id", "amount", "txn_type", "merchant_category", "receiver_type")
    ref = (tx["txn_type"] == "REFUND").to_numpy().astype(np.int64)
    intamt = ((tx["amount"] * 100).round() % 100 == 0).to_numpy().astype(np.int64)
    cols = {}
    for w in (60, 300, 1800):
        cols[f"{P}dev_c{w}"] = _win(tx, "device_id", w)
        cols[f"{P}snd_c{w}"] = _win(tx, "sender_id", w)
    for d in (3, 7, 14):
        cols[f"{P}snd_ref{d}d"] = _win(tx, "sender_id", d * 86400, ref)
    cols[f"{P}dev_ref7d"] = _win(tx, "device_id", 7 * 86400, ref)
    snd7 = _win(tx, "sender_id", 7 * 86400)
    rc300 = _win(tx, "receiver_id", 300); rc1d = _win(tx, "receiver_id", 86400)
    # shares are leave-self-out: with self included, half density (Oct) pulls every share toward the row's own flag
    # (legit REFUND rows: snd_ref_share7d +19%, rcv_ref_share1d +18% on 50%-thinned Sep); self-out is unbiased under thinning
    cols[f"{P}snd_ref_share7d"] = _loo(cols[f"{P}snd_ref7d"], snd7, ref)
    cols[f"{P}rcv_int_share300"] = _loo(_win(tx, "receiver_id", 300, intamt), rc300, intamt)
    cols[f"{P}rcv_ref_share1d"] = _loo(_win(tx, "receiver_id", 86400, ref), rc1d, ref)
    ri1d = _win(tx, "receiver_id", 86400, intamt)
    cols[f"{P}rcv_int_share1d"] = _loo(ri1d, rc1d, intamt)
    # P2P only: merchant in-counts scale with volume (Oct = half), P2P in-counts are stable Sep vs Oct (mean 0.79 both)
    cols[f"{P}p2p_int_in1d"] = np.where(tx["receiver_type"].to_numpy() == "P2P", np.minimum(ri1d, 10), np.nan)
    # merchants: integer-amount in-flow today vs the merchant's own +-15d volume (both scale with volume -> Oct-stable)
    # +-15d window is clipped at the data edges (Jul 1 / Nov 1) far more than +-1d: rescale by the spans actually covered,
    # else late-Oct rows inflate up to 1.75x; rows >= 15d from both edges keep factor 1.
    # Fixed edges, not ts min/max: a few dozen spill-over test rows run to Nov 9 (clipped -> they get the Oct 31 factor)
    lo, hi = pl.Series([datetime(2026, 7, 1), datetime(2026, 11, 1)]).dt.epoch("s").to_list()
    t = np.clip(tx["ts"].to_numpy().astype(np.int64), lo, hi - 1)
    cov = lambda w: np.minimum(t + w, hi) - np.maximum(t - w, lo)
    edge = cov(15 * 86400) / cov(86400) / 15
    cols[f"{P}mer_int_burst"] = np.where(tx["receiver_type"].to_numpy() == "MERCHANT", _loo(ri1d, _win(tx, "receiver_id", 15 * 86400), intamt) * edge, np.nan)
    tx2 = tx.with_columns(pl.concat_str("sender_id", "receiver_id").alias("_pair"))
    cols[f"{P}pair_ref14d"] = _win(tx2, "_pair", 14 * 86400, ref)
    # bot bursts hit one merchant from ONE device; mules receive from many devices -> separates the two
    tx3 = tx.with_columns(pl.concat_str("device_id", "receiver_id").alias("_dr"))
    cols[f"{P}devrcv_c300"] = _win(tx3, "_dr", 300)
    cols[f"{P}devrcv_share300"] = _loo(cols[f"{P}devrcv_c300"], rc300, 1)
    f = tx.select("txn_id").with_columns(**{k: pl.Series(v) for k, v in cols.items()})

    # device / sender bursts (velocity_bot): session = activity with gaps <= 10 min
    t = tx.with_columns(pl.col("merchant_category").is_in(["ecommerce", "gaming"]).fill_null(False).cast(pl.Float64).alias("_cat"))
    f = f.join(_sessions(t, "device_id", 600, P + "ds_", "sender_id"), on="txn_id")
    f = f.join(_sessions(t, "sender_id", 600, P + "ss_", "device_id"), on="txn_id")
    # refund episodes (refund_abuse): sender's REFUND rows with gaps <= 4 days; null on non-refund rows
    r = tx.filter(pl.col("txn_type") == "REFUND").with_columns(
        pl.col("merchant_category").is_in(["ecommerce", "electronics"]).fill_null(False).cast(pl.Float64).alias("_cat"))
    f = f.join(_sessions(r, "sender_id", 4 * 86400, P + "rs_", "device_id", min_n=2).drop(P + "rs_min_gap"), on="txn_id", how="left")
    return f
