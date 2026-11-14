"""Fidelity of generated rows: can a classifier tell generated rows from their real source rows on the 190 model features?
Also saves the source rows (scratch_gen/src_rows.parquet). cwd=work, PYTHONPATH=."""
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import load_all
from gen_events import source_events
S = source_events(load_all()).select("txn_id", "src_ev", "fam", "month")
S.write_parquet("scratch_gen/src_rows.parquet")
D = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet").join(S.select("txn_id", "src_ev"), on="txn_id")
G = pl.read_parquet("scratch_gen/gen_v2.parquet")
feats = [k for k in D.columns if k not in ("txn_id", "month", "y", "is_test", "src_ev")]
X = np.vstack([D.select(feats).to_numpy(), G.select(feats).to_numpy()]); y = np.r_[np.zeros(D.height), np.ones(G.height)]
grp = np.r_[D["src_ev"].to_numpy(), G["src_ev"].to_numpy()]
p = np.zeros(len(y))
for tr, va in GroupKFold(5).split(X, y, grp):
    m = lgb.train(dict(objective="binary", learning_rate=.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=.5, verbose=-1, num_threads=6), lgb.Dataset(X[tr], y[tr]), 200)
    p[va] = m.predict(X[va])
print("rows real", D.height, "generated", G.height, "| real-vs-generated AUC (5-fold, grouped by source event):", round(roc_auc_score(y, p), 4))
m = lgb.train(dict(objective="binary", learning_rate=.05, num_leaves=15, min_data_in_leaf=20, verbose=-1, num_threads=6), lgb.Dataset(X, y, feature_name=feats), 200)
print("top gain:", sorted(zip(m.feature_importance("gain").round(0), feats), reverse=True)[:12])
for k in [f for _, f in sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:6]]:
    print(f"  {k}: real mean {np.nanmean(D[k].to_numpy()):.3f} null {D[k].is_nan().mean() + D[k].null_count()/D.height:.3f} | gen mean {np.nanmean(G[k].to_numpy()):.3f} null {G[k].is_nan().mean() + G[k].null_count()/G.height:.3f}")
