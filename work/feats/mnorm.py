"""mnorm: ring__r36_excess without its hidden month tag (see notes/monthsep.md).

ring__r36_excess = r36_n / (1 + 3 * cnt_month / days_in_month). Integer counts over a month-specific divisor put every
quiet receiver on a per-month lattice (31% of Sep rows sit at 1/(1+3/30), 31% of Jul/Aug/Oct rows at 1/(1+3/31)), so the
feature alone tags Sep vs other months (adversarial AUC .91 -> .59 when dropped). Here the divisor is a constant 30.4375
days for every month. Outputs the raw parts too so the matrix step can check it reproduces ring's value.
Build density-matched (density.run_feature_matched("mnorm")), like ring.
"""
from datetime import datetime

import polars as pl
from feats.ring import _sym_roll

P = "mnorm__"


def build(tx: pl.DataFrame) -> pl.DataFrame:
    df = tx.select("txn_id", "timestamp", "receiver_id")
    rcv = _sym_roll(df, "receiver_id", "36h", [pl.len().alias(P + "r36_n")])
    mts = pl.col("timestamp").clip(upper_bound=datetime(2026, 10, 31, 23, 59, 59))  # Nov spill rows use Oct, as in ring
    m = df.select("txn_id", pl.len().over("receiver_id", mts.dt.month()).alias(P + "mcnt"), mts.dt.month_end().dt.day().alias(P + "dim"))
    out = m.join(rcv, on="txn_id").with_columns((pl.col(P + "r36_n") / (1 + 3 * pl.col(P + "mcnt") / 30.4375)).alias(P + "r36x"))
    return out.with_columns(pl.col(f"^{P}.*$").cast(pl.Float32))
