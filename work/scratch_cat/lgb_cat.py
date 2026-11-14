"""Categorical handling, LightGBM matched pair. Recipe = scratch_gap/numrep.py arm A_C (train.LGB, 150 rounds, lr .05,
legit rows downsampled to 20% with weight 5, seeds 0 and 1, target y_fit), num_threads=4.
  cLGBnum : numpy Float32 input, categorical_feature="auto" (the reference recipe, refit here)
  cLGBcat : same rows / weights / target / seeds / column order; the nine quick_eval.CAT_COLS are int32 columns of a
            pandas frame and are declared with categorical_feature=CAT_COLS. Codes are the stored Enum codes (same integers
            at fit and at inference, 'NA' is its own code). No pandas category dtype, no re-coding, no new ordering.
Every booster is inspected (feature_infos, pandas_categorical, split decision types and gain share on the nine columns).
usage (cwd=work, PYTHONPATH=.): python scratch_cat/lgb_cat.py cLGBnum|cLGBcat N|D
-> scratch_gap/P<BM>_<arm>_<aug|sep>.npy (refuses to overwrite), scratch_cat/models/<BM>_<arm>_f<fold>_s<seed>.txt"""
import json, os, sys, time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import train as T
from quick_eval import CAT_COLS

c = pl.col; ARM, BM = sys.argv[1], sys.argv[2]; SEEDS, ROUNDS, LF, TH = 2, 150, 0.2, 4
META = ("txn_id", "month", "is_test", "y_fit"); HERE = os.path.dirname(os.path.abspath(__file__))
assert ARM in ("cLGBnum", "cLGBcat") and BM in ("N", "D")
os.makedirs(f"{HERE}/models", exist_ok=True)


def frame(X, feats):
    """numeric arm: Float32 numpy (as numrep.py). categorical arm: pandas frame, same column order, nine columns int32."""
    if ARM == "cLGBnum": return X.select(feats).to_numpy()
    d = pd.DataFrame({k: X[k].to_numpy() for k in feats})
    for k in CAT_COLS:
        v = d[k].to_numpy(); assert not np.isnan(v).any() and (v == np.round(v)).all() and v.min() >= 0, k
        d[k] = v.astype(np.int32)
    assert list(d.columns) == feats
    return d


def inspect(m, tag):
    d = m.dump_model(); info = d["feature_infos"]; S = m.trees_to_dataframe(); S = S[S["split_feature"].notna()]; tot = S["split_gain"].sum()
    out = {"pandas_categorical": m.pandas_categorical, "splits_total": int(len(S)), "decision_types_all": S["decision_type"].value_counts().to_dict(), "cols": {}}
    for k in CAT_COLS:
        s = S[S["split_feature"] == k]
        out["cols"][k] = {"feature_info": info.get(k), "splits": int(len(s)), "decision_types": s["decision_type"].value_counts().to_dict(),
                          "gain_share": round(float(s["split_gain"].sum() / tot), 5)}
    out["nine_cols_splits"] = sum(v["splits"] for v in out["cols"].values()); out["nine_cols_gain_share"] = round(sum(v["gain_share"] for v in out["cols"].values()), 5)
    json.dump(out, open(f"{HERE}/models/{tag}_inspect.json", "w"), indent=1)
    print(f"INSPECT {tag}: pandas_categorical={out['pandas_categorical']} splits={out['splits_total']} all decision types={out['decision_types_all']} "
          f"nine columns: splits={out['nine_cols_splits']} gain share={out['nine_cols_gain_share']}")
    for k, v in out["cols"].items(): print(f"   {k:18s} {v['feature_info']} splits={v['splits']} {v['decision_types']} gain={v['gain_share']}")
    sys.stdout.flush()


for fi, (fm, vm) in enumerate(T.FOLDS):
    kind = "aug" if vm == 8 else "sep"; out = f"scratch_gap/P{BM}_{ARM}_{kind}.npy"; assert not os.path.exists(out), out + " exists"
    D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
    fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
    assert V["txn_id"].equals(pl.read_parquet(f"scratch_gap/{BM}_{kind}_rows.parquet", columns=["txn_id"])["txn_id"]), "held rows not aligned"
    params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=TH, verbose=-1)
    Xv = frame(V, feats); P = np.zeros((V.height, T.K), np.float32); t = time.time()
    for s in range(SEEDS):
        keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
        ds = lgb.Dataset(frame(fit.filter(pl.Series(keep)), feats), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32),
                         feature_name=feats, categorical_feature=CAT_COLS if ARM == "cLGBcat" else "auto")
        m = lgb.train({**params, "seed": s}, ds, ROUNDS)
        P += m.predict(Xv).astype(np.float32) / SEEDS
        tag = f"{BM}_{ARM}_f{fi}_s{s}"; m.save_model(f"{HERE}/models/{tag}.txt"); inspect(m, tag)
        print(f"{ARM} {BM} fold{fi} seed{s}: fit rows {int(keep.sum())} ({time.time() - t:.0f}s)", flush=True)
    np.save(out, P)
    ref = f"scratch_gap/P{BM}_{'C' if BM == 'N' else 'A_C'}_{kind}.npy"; R = np.load(ref)
    print(f"{ARM} {BM} fold{fi} saved {out} | max abs diff vs {ref}: {np.abs(P - R).max():.3e}, argmax rows differing {int((P.argmax(1) != R.argmax(1)).sum())}", flush=True)
print(ARM, BM, "done", flush=True)
