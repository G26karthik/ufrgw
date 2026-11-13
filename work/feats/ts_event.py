"""Generator device-id zones (notes/typesplit_event.md).

The data generator allocates fresh device ids in blocks, one block per fraud type, in a fixed order:
velocity bot pool (DV0420000-0422999, reused every month), then per-type zones for the train period, then single-use legit
devices; the Oct test period repeats the per-type zones right after the last train id, then its own legit singles.
A row on a device inside a zone is that zone's type (train Sep: 85% label==zone, 11% legit = label noise, 4% other type).
Train zone edges were fit on Jul+Aug labels; the test zones are the same layout scaled to the test block.
"""
import numpy as np
import polars as pl
from common import LABEL2ID

TYPES = ["mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover", "phishing_collect", "qr_swap",
         "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
TR_EDGES = [423000, 423658, 424156, 424628, 425039, 426769, 427165, 427848, 428059, 428203, 428304, 428455]  # zone k = [e[k], e[k+1])
POOL = (420000, 422999)  # velocity bot devices, all velocity_bot in train


def test_block(tx: pl.DataFrame):
    """(lo, hi) of the test-period fraud zones: first device id never seen in train .. start of the final gap-free id run."""
    d = tx.select("is_test", pl.col("device_id").str.slice(2).cast(pl.Int64).alias("dnum"))
    lo = int(d.filter(~pl.col("is_test"))["dnum"].max()) + 1
    used = np.sort(d.filter(pl.col("dnum") >= lo)["dnum"].unique().to_numpy())
    hi = int(used[np.nonzero(np.diff(used) > 1)[0][-1] + 1])
    return lo, hi


def edges_for(lo, hi):
    L = np.array(TR_EDGES) - TR_EDGES[0]
    return np.round(lo + L / L[-1] * (hi - lo)).astype(int).tolist()


def zone_ids(dnum: np.ndarray, is_test: np.ndarray, te_edges) -> np.ndarray:
    """label id of the device's zone, 0 = none."""
    z = np.zeros(len(dnum), np.int8)
    z[(dnum >= POOL[0]) & (dnum <= POOL[1])] = LABEL2ID["velocity_bot"]
    for e, msk in ((TR_EDGES, ~is_test | (dnum < te_edges[0])), (te_edges, is_test & (dnum >= te_edges[0]))):
        k = np.searchsorted(e, dnum, side="right") - 1
        ok = msk & (k >= 0) & (k < len(TYPES))
        z[ok] = [LABEL2ID[TYPES[i]] for i in k[ok]]
    return z


def build(tx: pl.DataFrame) -> pl.DataFrame:
    dnum = tx["device_id"].str.slice(2).cast(pl.Int64).to_numpy()
    z = zone_ids(dnum, tx["is_test"].to_numpy(), edges_for(*test_block(tx)))
    return tx.select("txn_id").with_columns(pl.Series("ts_event__zone", z, pl.Float32),
                                            pl.Series("ts_event__gen_dev", (z > 0).astype(np.float32)))
