"""Benchmark-D fold tables for TabM, and saved historical validation predictions aligned to the benchmark-D row order.
D_f0.parquet: Jul fit rows + Aug held rows; D_f1.parquet: Jul+Aug fit + Sep held (production matrix, relabel-C target on fit rows).
PD_<tag>_<aug|sep>.npy: oof/<tag>.npy re-ordered by txn_id to scratch_gap/D_<aug|sep>_rows.parquet.
Checks per saved model: every benchmark row present once, 12 columns, rows sum to 1, argmax macro-F1 on its own fold.
Provenance of the saved OOFs: Aug rows predicted by a July fit, Sep rows by a July+August fit (train.py run_val); their
features come from the label-informed density split, so they belong to benchmark D, not N.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/d_data.py"""
import numpy as np
import polars as pl
import train as T

c = pl.col
TAGS = ["h6", "c2s6", "ls_halves", "aug3btta", "xgbC", "gpu_cat"]
rows = {k: pl.read_parquet(f"scratch_gap/D_{k}_rows.parquet").select("txn_id", "y_true") for k in ("aug", "sep")}
for tag in TAGS:
    P = np.load(f"oof/{tag}.npy"); ids = pl.read_parquet(f"oof/{tag}_ids.parquet").with_row_index("i")
    assert P.shape == (ids.height, T.K) and abs(P.sum(1) - 1).max() < 1e-3, tag
    for k, fold in (("aug", 0), ("sep", 1)):
        j = rows[k].join(ids.filter(c("fold") == fold).select("txn_id", "i", "y"), on="txn_id", how="left", maintain_order="left")
        assert j["i"].null_count() == 0 and j["i"].n_unique() == j.height and (j["y"] == j["y_true"]).all(), (tag, k)
        Q = P[j["i"].to_numpy()]; np.save(f"scratch_gap/PD_{tag}_{k}.npy", Q)
        print(f"{tag:10s} {k}: rows {len(Q)} argmax macro {T.macro_f1(j['y_true'].to_numpy().astype(np.int64), Q.argmax(1)).mean():.4f}", flush=True)
M = pl.read_parquet("feats_cache/matrix_m5dm2_pairsfd.parquet")
yf = T.fit_target("C")
M = M.join(yf, on="txn_id", how="left", maintain_order="left")
feats = [k for k in M.columns if k not in ("txn_id", "month", "y", "is_test", "y_fit")]
for fi, (fm, vm) in enumerate(T.FOLDS):
    D = M.filter(~c("is_test") & c("month").is_in(fm + [vm])).with_columns((c("month") == vm).alias("is_test"))
    D = D.with_columns(pl.when(c("is_test")).then(None).otherwise(c("y_fit")).alias("y_fit")).select(["txn_id", "month", "is_test", "y_fit"] + feats)
    D.write_parquet(f"scratch_gap/D_f{fi}.parquet"); print("written D fold", fi, D.shape, flush=True)
