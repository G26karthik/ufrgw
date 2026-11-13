"""Density-matched feature building (see notes/drift.md, "Oct density").

Oct (test) has ~half the legit rows per day of Jul-Sep, and ~half the fraud events, but every fraud event is complete
(velocity bursts keep full size and 8-9 s gaps; P2P fan-in bursts are not thinned). Window/graph features built on
full-density train therefore differ from the same features on test (adversarial AUC 0.70-0.89 on burst/ring/scam).

Fix: split TRAIN rows into two halves the way Oct was sampled: legit rows 50/50 at random, fraud events (label-based
union-find, 48 h) whole into one half. Build each module once per half (that half + all test rows) and give every row
the features from its own half. Test rows take their features from half 0 (they see a half-density Sep, as Oct does).
Measured on Sep (scratch_drift/eval_octlike.py, base+burst+ring+scam): Oct-density Sep scored 0.4363 with the full-density
fit vs 0.4527 with the density-matched fit (tuned macro-F1; full-density reference 0.4559).

Labels are used only to decide which half a train row goes to; no label value reaches a feature.
usage: from density import run_feature_matched; f = run_feature_matched("burst")   # cached as feats_cache/burst__dm.parquet
"""
import importlib
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import CACHE, ROOT, load_all
from clean_labels import fraud_events, clean

SEED = 7
MLINK_MAX_VOL = 1000  # merchants with < this many rows (all months): collusion/refund/qr bursts sit at small merchants


def _merchant_roots(tx: pl.DataFrame, F: pl.DataFrame, gap: int) -> np.ndarray:
    """Per event id: smallest event id of its component after also linking fraud rows on the same small merchant
    (consecutive within `gap` s). fraud_events links by sender / sender|merchant / device only, so a burst paid by
    several senders (merchant_collusion ~17 rows / 4 senders / 45 h, refund ~5/3, qr ~3/2) is split across halves."""
    vol = tx.group_by("receiver_id").len()
    M = (F.join(tx.select("txn_id", "receiver_id", "receiver_type"), on="txn_id").join(vol, on="receiver_id")
          .filter((pl.col("receiver_type") == "MERCHANT") & (pl.col("len") < MLINK_MAX_VOL)).sort("receiver_id", "ts")
          .with_columns(pl.col("ev").shift(-1).alias("ej"), pl.col("ts").shift(-1).alias("tj"), pl.col("receiver_id").shift(-1).alias("rj"))
          .filter((pl.col("receiver_id") == pl.col("rj")) & (pl.col("tj") - pl.col("ts") <= gap)))
    n = int(F["ev"].max()) + 1
    _, comp = connected_components(coo_matrix((np.ones(M.height), (M["ev"].to_numpy(), M["ej"].to_numpy())), shape=(n, n)), directed=False)
    root = np.full(comp.max() + 1, n)
    np.minimum.at(root, comp, np.arange(n))
    return root[comp]


def halves(tx: pl.DataFrame, seed: int = SEED, mlink: int = 0) -> pl.Series:
    """0/1 per train row (fraud events kept whole), null for test rows. Aligned with tx.
    mlink > 0: also keep small-merchant fraud bursts whole (gap mlink seconds); same random draws, so mlink=0 is unchanged."""
    tr = tx.filter(~pl.col("is_test"))
    # cleaned labels: legit-labelled rows inside fraud events follow their event, so a row's own noisy label does not pick its half
    _, F = fraud_events(tr.with_columns(clean(tr).alias("y")), g=48 * 3600)
    rng = np.random.default_rng(seed)
    ev_half = rng.integers(0, 2, int(F["ev"].max()) + 1)
    if mlink:
        ev_half = ev_half[_merchant_roots(tx, F, mlink)]
    e = tx.select("txn_id").join(F.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"]
    h = np.where(e.is_null().to_numpy(), rng.integers(0, 2, tx.height), ev_half[e.fill_null(0).to_numpy()])
    return pl.Series("half", h, dtype=pl.Int8).set(tx["is_test"], None)


def run_feature_matched(name: str, tx: pl.DataFrame | None = None, force: bool = False, mlink: int = 0) -> pl.DataFrame:
    """mlink > 0 -> halves(mlink=...) and cache feats_cache/<name>__dm2.parquet (scratch_halves/: +.001 pooled, +.002 OOS)."""
    out = CACHE / f"{name}__{'dm2' if mlink else 'dm'}.parquet"
    if out.exists() and out.stat().st_mtime > max((ROOT / p).stat().st_mtime for p in (f"feats/{name}.py", "density.py", "clean_labels.py")) and not force:
        return pl.read_parquet(out)
    tx = load_all() if tx is None else tx
    h = halves(tx, mlink=mlink)
    mod = importlib.import_module(f"feats.{name}")
    parts = []
    for k in (0, 1):
        sub = tx.filter((h == k).fill_null(False) | tx["is_test"])
        f = mod.build(sub)
        own = sub.filter(~pl.col("is_test") | pl.lit(k == 0)).select("txn_id")  # test rows only from half 0
        parts.append(f.join(own, on="txn_id", how="semi"))
    f = pl.concat(parts)
    assert f.height == tx.height and f["txn_id"].n_unique() == tx.height, f"{name}: matched build must cover every row once"
    f.write_parquet(out)
    return f


if __name__ == "__main__":
    import sys
    tx = load_all()
    h = halves(tx)
    assert h.null_count() == int(tx["is_test"].sum())
    print("half sizes:", h.value_counts().sort("half").to_dicts())
    for name in sys.argv[1:]:
        f = run_feature_matched(name, tx, force=True)
        print(name, f.shape)
