"""Categorical handling, CatBoost matched pair. Settings = scratch_stack/n14/gbdt_n.py `cat` (MultiClass, depth 8, lr .1,
l2 3, 1 seed = 0, same fit rows / target / weights as the LightGBM recipe), identical in both arms:
  cCBnum : all 190 columns numeric
  cCBcat : the nine quick_eval.CAT_COLS passed as cat_features (int32 columns, stored Enum codes, 'NA' is its own code)
Both arms get the same pandas frame (181 Float32 columns + nine int32 columns, same order); only cat_features differs.
usage (cwd=work, PYTHONPATH=.): python scratch_cat/cb_cat.py cCBnum|cCBcat N|D gpu ITERATIONS  |  cCBnumCPU|cCBcatCPU N|D cpu ITERATIONS
-> scratch_gap/P<BM>_<arm>_<aug|sep>.npy (refuses to overwrite)"""
import json, os, sys, time
import numpy as np
import pandas as pd
import polars as pl
import train as T
from catboost import CatBoostClassifier
from quick_eval import CAT_COLS

c = pl.col; ARM, BM, DEV, ITERS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); SEED, LF, TH, K = int(os.environ.get("SEED0", 0)), 0.2, 4, T.K
TAG = ARM + (f"s{SEED}" if SEED else "")  # replicate pair with another seed: env SEED0=1 -> P<BM>_<arm>s1_*
META = ("txn_id", "month", "is_test", "y_fit"); HERE = os.path.dirname(os.path.abspath(__file__))
assert ARM in ("cCBnum", "cCBcat", "cCBnumCPU", "cCBcatCPU") and BM in ("N", "D") and DEV in ("gpu", "cpu") and (DEV == "cpu") == ARM.endswith("CPU")
CAT = ARM.startswith("cCBcat")  # ...CPU arms: same pair fitted on CPU (GPU not available), fewer iterations


def frame(X, feats):
    d = pd.DataFrame({k: X[k].to_numpy() for k in feats})
    for k in CAT_COLS:
        v = d[k].to_numpy(); assert not np.isnan(v).any() and (v == np.round(v)).all() and v.min() >= 0, k
        d[k] = v.astype(np.int32)
    assert list(d.columns) == feats
    return d


for fi, (fm, vm) in enumerate(T.FOLDS):
    kind = "aug" if vm == 8 else "sep"; out = f"scratch_gap/P{BM}_{TAG}_{kind}.npy"; assert not os.path.exists(out), out + " exists"
    D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; assert len(feats) == 190
    fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
    assert V["txn_id"].equals(pl.read_parquet(f"scratch_gap/{BM}_{kind}_rows.parquet", columns=["txn_id"])["txn_id"]), "held rows not aligned"
    keep = ((y != 0) | (np.random.default_rng(SEED).random(len(y)) < LF)) & (y >= 0); t = time.time()
    # gpu_ram_part .5: the card is shared (default .95 would reserve it); it does not change the model
    m = CatBoostClassifier(loss_function="MultiClass", iterations=ITERS, learning_rate=0.1, depth=8, l2_leaf_reg=3, random_seed=SEED, thread_count=TH,
                           verbose=250, allow_writing_files=False, **(dict(task_type="GPU", devices="0", gpu_ram_part=0.5) if DEV == "gpu" else {}))
    m.fit(frame(fit.filter(pl.Series(keep)), feats), y[keep], sample_weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32),
          cat_features=CAT_COLS if CAT else None)
    assert [int(k) for k in m.classes_] == list(range(K))
    P = m.predict_proba(frame(V, feats)).astype(np.float32); assert abs(P.sum(1) - 1).max() < 1e-4
    np.save(out, P)
    ap = m.get_all_params(); keys = ("task_type", "iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "boosting_type", "bootstrap_type", "one_hot_max_size",
                                     "simple_ctr", "combinations_ctr", "max_ctr_complexity", "ctr_target_border_count", "counter_calc_method", "ctr_leaf_count_limit", "grow_policy")
    info = {"params": {k: ap.get(k) for k in keys}, "cat_feature_indices": m.get_cat_feature_indices(), "fit_rows": int(keep.sum()), "secs": round(time.time() - t)}
    imp = None
    try: imp = dict(zip(feats, m.get_feature_importance()))
    except Exception as e: print("feature importance not available:", repr(e)[:200])
    if imp: info["nine_cols_importance_pct"] = {k: round(float(imp[k]), 3) for k in CAT_COLS}; info["nine_cols_importance_sum_pct"] = round(float(sum(imp[k] for k in CAT_COLS)), 3)
    json.dump(info, open(f"{HERE}/{BM}_{TAG}_f{fi}_info.json", "w"), indent=1, default=str)
    print(f"{ARM} {BM} fold{fi}: {json.dumps(info, default=str)}", flush=True)
    if kind == "aug":  # September labels are not read here
        ya = pl.read_parquet(f"scratch_gap/{BM}_aug_rows.parquet")["y_true"].to_numpy().astype(np.int64); C = np.load(f"scratch_gap/P{BM}_{'C' if BM == 'N' else 'A_C'}_aug.npy")
        print(f"{ARM} Aug: macro argmax {T.macro_f1(ya, P.argmax(1)).mean():.4f} (LightGBM C {T.macro_f1(ya, C.argmax(1)).mean():.4f}), "
              f"log-loss {-np.log(P[np.arange(len(ya)), ya] + 1e-12).mean():.5f} (C {-np.log(C[np.arange(len(ya)), ya] + 1e-12).mean():.5f})", flush=True)
print(ARM, BM, "done", flush=True)
