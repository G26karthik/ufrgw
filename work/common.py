"""Shared loader. Every feature module and the trainer use load_all() so rows/joins are identical.

Feature module contract (work/feats/<name>.py):
    def build(tx: pl.DataFrame) -> pl.DataFrame
        tx = load_all() output (train+test, label null on test, never read `label` inside build).
        return: txn_id + feature columns, every feature column prefixed '<name>__'.
Cached by run_feature(name) to work/feats_cache/<name>.parquet; the cache is rebuilt when feats/<name>.py is newer.
"""
from pathlib import Path
import os
import importlib
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = Path(os.environ.get("UPI_CACHE") or ROOT / "feats_cache")  # UPI_CACHE: build caches elsewhere (reproduction check)
CACHE.mkdir(parents=True, exist_ok=True)

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
          "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_all() -> pl.DataFrame:
    tr = pl.read_parquet(DATA / "train.parquet").with_columns(pl.lit(False).alias("is_test"))
    te = pl.read_parquet(DATA / "test.parquet").with_columns(pl.lit(None, pl.String).alias("label"), pl.lit(True).alias("is_test"))
    tx = pl.concat([tr, te.select(tr.columns)])
    acc = pl.read_parquet(DATA / "accounts.parquet").with_columns(pl.col("opened_on").str.to_date())
    mer = pl.read_parquet(DATA / "merchants.parquet").with_columns(pl.col("onboarded_on").str.to_date())
    tx = tx.join(acc.rename({c: f"snd_{c}" for c in acc.columns if c != "account_id"}),
                 left_on="sender_id", right_on="account_id", how="left")
    # receiver may be an account (P2P) or merchant
    tx = tx.join(acc.rename({c: f"rcv_{c}" for c in acc.columns if c != "account_id"}),
                 left_on="receiver_id", right_on="account_id", how="left")
    tx = tx.join(mer.rename({c: f"mer_{c}" for c in mer.columns if c != "merchant_id"}),
                 left_on="receiver_id", right_on="merchant_id", how="left")
    tx = tx.with_columns(
        pl.col("timestamp").dt.epoch("s").alias("ts"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("label").replace_strict(LABEL2ID, default=None, return_dtype=pl.Int8).alias("y"),
    )
    return tx.sort("ts")


def run_feature(name: str, tx: pl.DataFrame | None = None, force: bool = False) -> pl.DataFrame:
    out = CACHE / f"{name}.parquet"
    if out.exists() and out.stat().st_mtime > (ROOT / "feats" / f"{name}.py").stat().st_mtime and not force:
        return pl.read_parquet(out)
    tx = load_all() if tx is None else tx
    mod = importlib.import_module(f"feats.{name}")
    f = mod.build(tx)
    assert f.height == tx.height and f["txn_id"].n_unique() == tx.height, f"{name}: must return one row per txn"
    bad = [c for c in f.columns if c != "txn_id" and not c.startswith(f"{name}__")]
    assert not bad, f"{name}: unprefixed columns {bad}"
    f.write_parquet(out)
    return f
