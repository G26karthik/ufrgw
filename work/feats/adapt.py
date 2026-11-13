"""adapt: features robust to the October ring adaptations (see notes/adapt.md).

Measured adaptation (Oct test vs Jul-Sep train): qr_swap rings blank `qr_vpa_mismatch` on QR rows.
- QR to a P2P payee with merchant_category set (82% qr_swap in noisy train, i.e. ~all qr_swap clean) shows null mismatch
  on 140/309 Oct rows; train has 0 such nulls. mm=0 and mm=1 shrank alike, so the flag is blanked regardless of value.
- Merchant QR with null mismatch and a non-integer amount: 0 rows in train, 17 in Oct (tiny merchants, deg ~5).
In train, merchant-QR null exists only with integer round amounts (collusion/refund lookalikes) and keeps its meaning.
Row-level only: identical under density matching. Labels never read.
"""
import polars as pl

P = "adapt__"


def build(tx: pl.DataFrame) -> pl.DataFrame:
    a, mm = pl.col("amount"), pl.col("qr_vpa_mismatch")
    qr = pl.col("txn_type") == "QR"
    p2p = pl.col("receiver_type") == "P2P"
    unseen_null = qr & mm.is_null() & (p2p | (a != a.round()))  # null patterns that never occur in train
    return tx.select(
        "txn_id",
        (qr & p2p & pl.col("merchant_category").is_not_null()).cast(pl.Int8).alias(P + "qr_p2p_cat"),
        pl.when(unseen_null).then(1.0).otherwise(mm.cast(pl.Float64)).alias(P + "qr_mm"),
    )
