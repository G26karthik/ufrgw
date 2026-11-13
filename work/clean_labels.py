"""Label-noise cleaning for TRAIN rows (see notes/drift.md).

Noise model found in train (per row, not per event): ~10-12% of fraud rows are labelled legit, ~4% carry a random other
fraud type (uniform over the other 10). Recipe:
  1. Link fraud-labelled train rows into events: consecutive rows on the same key within G seconds, keys = sender_id,
     receiver account (P2P) or sender|merchant pair, device_id. Union-find (connected components).
  2. Events with >= MIN_SZ rows and a majority type holding >= MIN_SHARE: minority fraud rows -> majority type.
  3. Legit-labelled train rows inside an event's [start, end] that share its receiver key (mule_account, velocity_bot,
     merchant_collusion events) or its device (velocity_bot events) -> event majority type.
Uses labels, so it is a training-time target transform only; never a feature. Test rows get null.
"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

G = 6 * 3600
MIN_SZ, MIN_SHARE = 3, 0.6
RK_TYPES = [1, 2, 11]   # mule_account, velocity_bot, merchant_collusion: legit rows on the event's receiver key are mostly flipped fraud
DEV_TYPES = [2]          # velocity_bot: legit rows on the bot device during the burst


def _rk():
    return pl.when(pl.col("receiver_type") == "P2P").then(pl.col("receiver_id")).otherwise(pl.col("sender_id") + "|" + pl.col("receiver_id"))


def fraud_events(tx: pl.DataFrame, g: int = G) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label-based event ids. Returns (t: all rows with row index i, F: fraud-labelled rows with column ev)."""
    t = tx.select("txn_id", "ts", "y", "sender_id", "device_id", _rk().alias("rk")).with_row_index("i")
    n = t.height
    F = t.filter(pl.col("y") > 0)
    ent = pl.concat([F.select("i", "ts", pl.col("sender_id").alias("e")), F.select("i", "ts", pl.col("rk").alias("e")),
                     F.select("i", "ts", ("D" + pl.col("device_id")).alias("e"))]).sort("e", "ts")
    ed = ent.with_columns(pl.col("i").shift(-1).alias("j"), pl.col("ts").shift(-1).alias("tj"), pl.col("e").shift(-1).alias("ej")) \
            .filter((pl.col("e") == pl.col("ej")) & (pl.col("tj") - pl.col("ts") <= g) & (pl.col("i") != pl.col("j")))
    a, b = ed["i"].to_numpy().astype(np.int64), ed["j"].to_numpy().astype(np.int64)
    _, comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    return t, F.with_columns(pl.Series("ev", comp[F["i"].to_numpy()]))


def clean(tx: pl.DataFrame, g: int = G, relabel_legit: bool = True, relabel_types: bool = True) -> pl.Series:
    t, F = fraud_events(tx, g)
    C = (F.group_by("ev", "y").len().sort("len", "y", descending=[True, False])
          .group_by("ev", maintain_order=True).agg(pl.col("y").first().alias("maj"), pl.col("len").first().alias("nmaj"), pl.col("len").sum().alias("sz"))
          .join(F.group_by("ev").agg(pl.col("ts").min().alias("s"), pl.col("ts").max().alias("t")), on="ev")
          .filter((pl.col("sz") >= MIN_SZ) & (pl.col("nmaj") >= MIN_SHARE * pl.col("sz"))))
    y = t["y"].to_numpy().astype(np.float64).copy()   # NaN for test
    if relabel_types:
        R = F.join(C.select("ev", "maj"), on="ev").filter(pl.col("y") != pl.col("maj"))
        y[R["i"].to_numpy()] = R["maj"].to_numpy()
    if relabel_legit:
        L = t.filter(pl.col("y") == 0).select("i", "ts", "rk", "device_id")
        evk = F.join(C.select("ev", "maj", "s", "t"), on="ev")
        hits = []
        for lk, types in [("rk", RK_TYPES), ("device_id", DEV_TYPES)]:
            keys = evk.filter(pl.col("maj").is_in(types)).select("ev", "maj", "s", "t", pl.col(lk).alias("k")).unique()
            hits.append(L.join(keys, left_on=lk, right_on="k").filter(pl.col("ts").is_between(pl.col("s"), pl.col("t"))).select("i", "maj"))
        H = pl.concat(hits).unique("i", keep="first")
        y[H["i"].to_numpy()] = H["maj"].to_numpy()
    return pl.Series("y_clean", y).fill_nan(None).cast(pl.Int8)


if __name__ == "__main__":
    from common import load_all
    tx = load_all()
    yc = clean(tx)
    y = tx["y"]
    assert yc.len() == tx.height and yc.null_count() == y.null_count()
    ch = (yc != y).fill_null(False)
    print("changed rows:", int(ch.sum()), "of", int(y.is_not_null().sum()))
    print(pl.DataFrame({"y": y, "yc": yc}).filter(ch).group_by("y", "yc").len().sort("len", descending=True).head(30))
