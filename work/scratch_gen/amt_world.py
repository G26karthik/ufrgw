"""Mechanism 2: October legit amounts are x1.17 (notes/drift.md item 5); fraud signature amounts are not.
Build an "Oct-like" copy of the train months: amount x1.17 on legit-labelled rows that do not look like fraud
(integer amounts stay integer and multiples of 10 stay multiples of 10: real Oct legit P2P
integers are 99.6% multiples of 10, as in Sep; scratch_gen/amt_check.py), every other field unchanged, then the same feature extraction as make_matrix.py.
Rows that look like fraud (legit label, model says fraud = flipped-label core rows) keep their amount, as core rows do in Oct.
Look: arm A probabilities for Aug and Sep (scratch_gen/P_A_*.npy); July from a model fit on Aug.
usage (cwd=work, PYTHONPATH=.): python scratch_gen/amt_world.py  -> scratch_gen/matrix_amt.parquet"""
import importlib, numpy as np, polars as pl, lightgbm as lgb
import train as T
from common import load_all
from density import halves
from make_matrix import MODULES
from quick_eval import base_features
c = pl.col; S = 1.17
tx = load_all()
D = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet"); feats = [k for k in D.columns if k not in ("txn_id", "month", "y", "is_test")]
# measured shift in the unlabeled October rows (all rows, no labels): amount and balance medians vs Sep
print(tx.filter(c("month").is_in([9, 10])).group_by("month", "is_test").agg(pl.len(), c("amount").median().alias("amt_med"), c("balance_before").median().alias("bal_med"),
      (c("amount") / c("balance_before")).median().alias("share_med"), (c("amount") % 1 == 0).mean().alias("int")).sort("month"))
aug = D.filter((c("month") == 8) & ~c("is_test")); jul = D.filter((c("month") == 7) & ~c("is_test"))
ya = aug["y"].to_numpy().astype(np.int32); keep = (ya != 0) | (np.random.default_rng(0).random(len(ya)) < .2)
m = lgb.train(dict(objective="multiclass", num_class=T.K, learning_rate=.05, **T.LGB, num_threads=8, verbose=-1, seed=0),
              lgb.Dataset(aug.select(feats).to_numpy()[keep], ya[keep], weight=np.where(ya[keep] == 0, 5., 1.)), 150)
look = pl.concat([pl.DataFrame({"txn_id": jul["txn_id"], "fraud_look": T.predict_chunked(T.lgb_pred, m, jul, feats).argmax(1) != 0}),
                  *[pl.DataFrame({"txn_id": pl.read_parquet(f"scratch_gen/ids_{k}.parquet")["txn_id"], "fraud_look": np.load(f"scratch_gen/P_A_{k}.npy").argmax(1) != 0}) for k in ("aug", "sep")]])
tx = tx.join(look, on="txn_id", how="left", maintain_order="left")
sc = ~c("is_test") & (c("y") == 0) & ~c("fraud_look").fill_null(False); a = c("amount")
print("scaled rows by month:", tx.filter(sc).group_by("month").len().sort("month").rows(), "| legit-labelled, fraud look, kept:", tx.filter(~c("is_test") & (c("y") == 0) & c("fraud_look")).height)
MAXA = tx["amount"].max(); print("amount cap (max in train and test):", MAXA, "| rows capped:", tx.filter(sc & (a * S > MAXA)).height)  # no amount above the observed limit
txs = tx.with_columns(pl.when(sc).then(pl.when(a % 10 == 0).then((a * S / 10).round(0) * 10).when(a % 1 == 0).then((a * S).round(0)).otherwise((a * S).round(2)).clip(upper_bound=MAXA)).otherwise(a).alias("amount")).drop("fraud_look")
X = base_features(txs)
for name, ml in MODULES:
    h = halves(txs, mlink=ml); mod = importlib.import_module(f"feats.{name}"); parts = []
    for k in (0, 1):
        sub = txs.filter((h == k).fill_null(False) | txs["is_test"])
        parts.append(mod.build(sub).join(sub.filter(~c("is_test") | pl.lit(k == 0)).select("txn_id"), on="txn_id", how="semi"))
    X = X.join(pl.concat(parts), on="txn_id", how="left", maintain_order="left"); print(name, X.shape, flush=True)
X = X.join(importlib.import_module("feats.pairs").build(txs), on="txn_id", how="left", maintain_order="left")
M = txs.select("txn_id", "month", "y", "is_test").join(X, on="txn_id", how="left", maintain_order="left").with_columns([c(k).cast(pl.Float32) for k in feats])
assert M.columns == D.columns and M["txn_id"].equals(D["txn_id"])
M.write_parquet("scratch_gen/matrix_amt.parquet"); print("written", M.shape)
