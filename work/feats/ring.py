"""ring: loan_app_extortion + merchant_collusion structure features.

Generator signatures (see notes/ring.md):
- loan_app: one victim pays 2-6 NEW payees (mostly 'lending' merchants, some P2P) within ~1 week,
  20:00-24:00, whole-rupee amounts in the thousands, balance_before often == 1.1*amount.
- merchant_collusion: a small merchant gets a burst of ~15-35 payments in 1-3 days from 2-5 accounts
  (each paying many times), whole-rupee amounts (often 1000/2000/2500/5000/10000).
"""
from datetime import datetime

import polars as pl

P = "ring__"
GAP_S = 12 * 3600  # receiver burst split: new segment after a 12h silence


def _sym_roll(df, key, half, aggs):
    """Aggregates over rows of the same `key` with timestamp in (t-half, t+half]."""
    d = df.sort(key, "timestamp")
    r = d.rolling(index_column="timestamp", period=f"{2 * half}", offset=f"-{half}", group_by=key).agg(aggs)
    assert (r[key] == d[key]).all()
    return pl.concat([d.select("txn_id"), r.drop(key, "timestamp")], how="horizontal")


def build(tx: pl.DataFrame) -> pl.DataFrame:
    a = pl.col("amount")
    h = pl.col("timestamp").dt.hour()
    df = tx.select(
        "txn_id", "timestamp", "ts", "sender_id", "receiver_id", "amount", "new_payee",
        (a % 1 == 0).cast(pl.Int8).alias("int_amt"),
        (a % 500 == 0).cast(pl.Int8).alias("r500"),
        ((pl.col("balance_before") - a * 1.1).abs() < 0.011).cast(pl.Int8).alias("b11"),
        (h >= 20).cast(pl.Int8).alias("eve"),
        (pl.col("merchant_category") == "lending").fill_null(False).cast(pl.Int8).alias("lend"),
        (pl.col("txn_type") == "REFUND").cast(pl.Int8).alias("refund"),
    ).with_columns(
        # loan-like row: new payee, whole rupees >= 1000, evening
        ((pl.col("new_payee") == 1) & (pl.col("int_amt") == 1) & (a >= 1000)).cast(pl.Int8).alias("np_int"),
    ).with_columns((pl.col("np_int") * pl.col("eve")).alias("np_int_eve"))

    # ---- sender side (loan victims): symmetric +-4d window
    snd = _sym_roll(df, "sender_id", "4d", [
        pl.col("np_int").sum().alias(P + "s4_np_int"),
        pl.col("np_int_eve").sum().alias(P + "s4_np_int_eve"),
        pl.col("lend").sum().alias(P + "s4_lend"),
        pl.col("receiver_id").filter(pl.col("np_int") == 1).n_unique().alias(P + "s4_np_int_nrcv"),
        pl.col("b11").sum().alias(P + "s4_b11"),
    ])
    snd10 = _sym_roll(df, "sender_id", "10d", [
        pl.col("np_int_eve").sum().alias(P + "s10_np_int_eve"),
        pl.col("lend").sum().alias(P + "s10_lend"),
    ])
    # colluding accounts pay only their merchant: distinct int-amount receivers of the sender in +-2d
    snd2 = _sym_roll(df, "sender_id", "2d", [
        pl.col("receiver_id").n_unique().alias(P + "s2_nrcv"),
        pl.col("int_amt").sum().alias(P + "s2_int"),
    ])
    # receiver +-36h rolling (gap-free view of bursts)
    rcv = _sym_roll(df, "receiver_id", "36h", [
        pl.len().alias(P + "r36_n"),
        pl.col("int_amt").sum().alias(P + "r36_int"),
        pl.col("r500").sum().alias(P + "r500_36"),
    ])

    # ---- receiver bursts (collusion): gap-based segments per receiver
    d = df.sort("receiver_id", "ts").with_columns(
        ((pl.col("ts").diff().over("receiver_id").fill_null(GAP_S + 1) > GAP_S).cum_sum()).alias("seg"))
    rtot = pl.len().over("receiver_id")
    seg = d.with_columns(
        pl.len().over("seg").alias(P + "seg_n"),
        pl.col("sender_id").n_unique().over("seg").alias(P + "seg_nsnd"),
        ((pl.col("ts").max() - pl.col("ts").min()).over("seg") / 3600).alias(P + "seg_dur_h"),
        pl.col("int_amt").mean().over("seg").alias(P + "seg_int"),
        pl.col("r500").mean().over("seg").alias(P + "seg_r500"),
        pl.col("b11").mean().over("seg").alias(P + "seg_b11"),
        pl.col("np_int_eve").mean().over("seg").alias(P + "seg_np_int_eve"),
        rtot.alias(P + "rcv_n"),
        (pl.len().over("seg") / rtot).alias(P + "seg_frac"),
        pl.len().over("seg", "sender_id").alias(P + "pair_seg_n"),
        pl.len().over("receiver_id", "sender_id").alias(P + "pair_n"),
        # receiver-level loan-likeness: share of incoming rows that look like loan payments
        pl.col("np_int_eve").mean().over("receiver_id").alias(P + "rcv_np_int_eve"),
        pl.col("sender_id").n_unique().over("receiver_id").alias(P + "rcv_nsnd"),
        pl.col("refund").mean().over("seg").alias(P + "seg_refund"),
        ((pl.col("ts").max() - pl.col("ts").min()).over("seg", "sender_id") / 3600).alias(P + "pair_seg_dur_h"),
    ).with_columns(
        # senders with >=2 rows in the segment, and share of segment rows from the top sender
        (pl.col("sender_id").filter(pl.col(P + "pair_seg_n") >= 2).n_unique().over("seg")).alias(P + "seg_nrep_snd"),
        (pl.col(P + "pair_seg_n").max().over("seg") / pl.col(P + "seg_n")).alias(P + "seg_top1"),
        # burst vs receiver's own average daily rate (data spans ~123 days)
        (pl.col(P + "seg_n") / (1 + pl.col(P + "rcv_n") / 123
                                * (pl.col(P + "seg_dur_h") / 24).clip(lower_bound=1))).alias(P + "seg_excess"),
        (pl.col(P + "seg_n") / pl.col(P + "seg_nsnd")).alias(P + "seg_n_per_snd"),
        (pl.col(P + "pair_seg_n") / pl.col(P + "seg_n")).alias(P + "pair_seg_share"),
    ).select("txn_id", pl.col(f"^{P}.*$"))

    row = df.select("txn_id", pl.col("b11").alias(P + "b11"), pl.col("np_int_eve").alias(P + "np_int_eve"),
                    pl.col("lend").alias(P + "lend"))
    out = row.join(snd, on="txn_id").join(seg, on="txn_id")
    for t in (snd10, snd2, rcv):
        out = out.join(t, on="txn_id")
    # Oct volume is ~half of Jul-Sep: express receiver window counts relative to the receiver's own same-month rate
    # the 46 Nov test rows (Oct events running past month end) use Oct's bucket, not a tiny Nov one
    mts = pl.col("timestamp").clip(upper_bound=datetime(2026, 10, 31, 23, 59, 59))
    mrate = df.select("txn_id", (pl.len().over("receiver_id", mts.dt.month())
                                 / mts.dt.month_end().dt.day()).alias("mrate"))
    out = out.join(mrate, on="txn_id").with_columns(
        (pl.col(P + "r36_int") / pl.col(P + "r36_n")).alias(P + "r36_int_share"),
        (pl.col(P + "r36_n") / (1 + 3 * pl.col("mrate"))).alias(P + "r36_excess"),
    )
    return out.drop("mrate", P + "r36_n", P + "r36_int")
