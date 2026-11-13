"""Shared yardstick: base row features + given feature modules -> LightGBM, fit Jul-Aug, score Sep.

usage: python quick_eval.py [module ...] [--threads N] [--drop col,col]
Modules are density-matched (train.build_matrix -> density.run_feature_matched). Prints per-class F1 and macro-F1
(argmax and per-class-bias-tuned with train.tune_bias). Same protocol for everyone so numbers compare.
"""
import argparse, time
from functools import cache
import numpy as np
import polars as pl
import lightgbm as lgb
from common import DATA, LABELS

CAT_COLS = ["receiver_type", "txn_type", "merchant_category", "app", "snd_kyc_level", "snd_os", "snd_bank_code",
            "rcv_kyc_level", "mer_category"]


@cache
def cat_dtypes() -> dict:
    """pl.Enum per CAT_COLS column: sorted uniques of the raw tables + 'NA', so codes never depend on the frame."""
    t = pl.concat([pl.read_parquet(DATA / f, columns=CAT_COLS[:4]) for f in ("train.parquet", "test.parquet")])
    acc, mer = pl.read_parquet(DATA / "accounts.parquet"), pl.read_parquet(DATA / "merchants.parquet")
    src = {**{c: t[c] for c in CAT_COLS[:4]}, "snd_kyc_level": acc["kyc_level"], "snd_os": acc["os"],
           "snd_bank_code": acc["bank_code"], "rcv_kyc_level": acc["kyc_level"], "mer_category": mer["category"]}
    return {c: pl.Enum(sorted(set(s.drop_nulls()) | {"NA"})) for c, s in src.items()}


def base_features(tx: pl.DataFrame) -> pl.DataFrame:
    a = pl.col("amount"); E = cat_dtypes()
    # excluded on purpose (Oct drift, notes/drift.md): ip_risk_score, app_version; account/merchant ages and weekday drift by design with no signal
    return tx.select(
        "txn_id",
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()).alias("minute_of_day"),
        a, a.log1p().alias("log_amount"),
        ((a * 100).round() % 100).alias("cents"),
        (a % 100 == 0).cast(pl.Int8).alias("round100"),
        (a % 1000 == 0).cast(pl.Int8).alias("round1000"),
        "sim_age_hours", "device_age_days", "new_payee", "pin_retries",
        "login_to_txn_sec", "balance_before", "qr_vpa_mismatch",
        (a / pl.col("balance_before")).alias("amt_share_bal"),
        (pl.col("device_city") != pl.col("snd_home_city")).cast(pl.Int8).alias("away_from_home"),
        (pl.col("receiver_type") == "P2P").cast(pl.Int8).alias("is_p2p"),
        (pl.col("mer_city") == pl.col("device_city")).cast(pl.Int8).alias("mer_same_city"),
        *[pl.col(c).fill_null("NA").cast(E[c]).to_physical().cast(pl.Int32).alias(c) for c in CAT_COLS],
    )


def main():
    from train import build_matrix, tune_bias, decode, macro_f1  # lazy: train imports base_features from here
    ap = argparse.ArgumentParser(); ap.add_argument("modules", nargs="*"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--drop", default=""); ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args(); t0 = time.time()
    D, feats = build_matrix(a.modules, a.drop)
    tr = D.filter(pl.col("month").is_in([7, 8]) & ~pl.col("is_test"))
    va = D.filter((pl.col("month") == 9) & ~pl.col("is_test"))
    rng = np.random.default_rng(0)
    ytr = tr["y"].to_numpy(); keep = (ytr != 0) | (rng.random(len(ytr)) < 0.2)  # legit downsample x5
    w = np.where(ytr[keep] == 0, 5.0, 1.0)
    Xtr = tr.select(feats).to_numpy().astype(np.float32)[keep]
    params = dict(objective="multiclass", num_class=len(LABELS), learning_rate=0.08, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr[keep], weight=w, feature_name=feats), a.rounds)
    P = m.predict(va.select(feats).to_numpy().astype(np.float32)); yv = va["y"].to_numpy().astype(np.int64)
    f_arg = macro_f1(yv, P.argmax(1))
    b = tune_bias(P, yv)
    f_t = macro_f1(yv, decode(P, b))
    print(f"modules={a.modules} n_feats={len(feats)} secs={time.time()-t0:.0f}")
    for i, l in enumerate(LABELS):
        print(f"  {l:20s} argmaxF1={f_arg[i]:.3f} tunedF1={f_t[i]:.3f}")
    print(f"MACRO argmax={f_arg.mean():.4f} tuned={f_t.mean():.4f}")
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:30]
    print("top gain:", ", ".join(f"{n}={g:.0f}" for g, n in imp))


if __name__ == "__main__":
    main()
