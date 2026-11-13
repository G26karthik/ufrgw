"""qr_swap post-rules (notes/bc_qr.md): none accepted, so apply() is the identity.

Kept so a combiner can import every bc_rules_<key>.apply uniformly. qr is structurally saturated on sc_final:
every Oct row in a hard qr cell (P2P-QR any flag, merchant-QR null flag, round-set merchant segment n<=3) is decoded qr,
the blanked-flag cells are already taken by qr_null / QRN, and qr victims and fake payees are one-shot (linked rows ~0% qr).
The one sig-vs-zone conflict row (P2P-QR mm=1 on a collusion-zone device) has 0 train support and break-even P(qr) ~.71.
"""
import polars as pl


def apply(sub_df: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    return sub_df
