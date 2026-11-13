"""loan_app_extortion post-rules (notes/sc_loan.md): none accepted, so apply() is the identity.

Kept so a combiner can import every sc_rules_<key>.apply uniformly. The loan core (victim paying >=2 lending merchants,
new payee, int >= 1000, evening) is decoded loan on 83/83 test rows. Oct signature rates match train, and the loan/phishing
under-round COLLECT split is flat (.035 loan on every key). No label-free rule clears break-even.
"""
import polars as pl


def apply(sub_df: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    return sub_df
