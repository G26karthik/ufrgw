"""ts_victim: device-number allocation blocks (notes/typesplit_victim.md). Label-free.

The generator gives a fresh phone (device_id DV0<n>, n >= 420000 = above every account's own phone DV0<account n>) to a
fraud row from a per-class counter block, in class order mule | velocity | layering | sim_swap | account_takeover | phishing |
qr | refund | fake_cashback | loan | collusion, and only after all fraud blocks to legit rows (a dense run: every number used).
Train (Jul-Sep) and test (Oct) are numbered separately (train fraud region 423000-428454, test 473632-~474545), with
block widths proportional to class volume. sim_swap block is dense (gap ~1.4) and account_takeover sparse (gap ~3.8).
Features (NaN outside the region):
  blkpos   position of the device number inside its period's fraud region, 0..1 (transfers: same block proportions)
  gapden   mean gap between consecutive used device numbers in a +-4 device window (sim ~1.4 vs ATO ~3.8)
  ext      1 = device is a fresh allocated phone (n >= 420000), 0 = an account's own phone
"""
import numpy as np
import polars as pl

c = pl.col
DENSE = 40  # a run of this many consecutive used numbers = legit allocation (fraud blocks use ~30-70% of numbers)


def region(d: np.ndarray) -> tuple[int, int]:
    """[lo, hi) of the single-sender fraud region in sorted unique device numbers d of one period."""
    brk = np.r_[0, np.flatnonzero(np.diff(d) != 1) + 1]
    lens = np.diff(np.r_[brk, len(d)])
    hi = d[brk[np.argmax(lens >= DENSE)]]                      # first long dense run starts the legit block
    return int(d[0]), int(hi)


def build(tx: pl.DataFrame) -> pl.DataFrame:
    P = "ts_victim__"
    t = tx.select("txn_id", "sender_id", "device_id", "is_test", "month").with_columns(
        c("device_id").str.slice(2).cast(pl.Int64).alias("dn"), c("sender_id").str.slice(2).cast(pl.Int64).alias("sn"))
    t = t.with_columns((c("dn") >= 420000).alias("ext"))
    # a device belongs to the period (train / test) where it is first seen; velocity bot phones are shared by many senders
    dv = (t.filter(c("ext")).group_by("dn").agg(c("is_test").min().alias("per"), c("sender_id").n_unique().alias("ns"))
          .filter(c("ns") == 1).sort("dn"))
    parts = []
    for per in (False, True):
        d = dv.filter(c("per") == per)["dn"].to_numpy()  # region starts at the period's first single-sender fresh phone
        lo, hi = region(d)
        r = d[(d >= lo) & (d < hi)]
        g = np.diff(r, prepend=r[0] - 1).astype(float)
        gd = np.convolve(np.pad(g, 4, mode="edge"), np.ones(9) / 9, mode="valid")
        parts.append(pl.DataFrame({"dn": r, "blkpos": (r - lo) / (hi - lo), "gapden": gd}))
        print(f"ts_victim: period test={per} region [{lo}, {hi}) devices {len(r)}")
    B = pl.concat(parts)
    out = t.join(B, on="dn", how="left", maintain_order="left")
    return out.select("txn_id", c("blkpos").cast(pl.Float32).alias(P + "blkpos"), c("gapden").cast(pl.Float32).alias(P + "gapden"),
                      c("ext").cast(pl.Float32).alias(P + "ext"))
