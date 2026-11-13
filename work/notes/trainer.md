# Trainer / blender usage

All commands from `work/` with `PYTHONUTF8=1 POLARS_MAX_THREADS=2` and `..\.venv\Scripts\python.exe`.

## train.py

```
python train.py --modules a,b,c --mode val  --threads 4 --seeds 1 --tag NAME [--label-clean]
python train.py --modules a,b,c --mode full --threads 4 --seeds 3 --tag NAME
```

- Features: `quick_eval.base_features` + each module via `density.run_feature_matched` (density-matched, cached as `feats_cache/<module>__dm.parquet`, rebuilt when `feats/<module>.py`, `density.py` or `clean_labels.py` is newer), all cast to float32. base_features leaves out ip_risk_score, app_version, account/merchant ages and weekday (Oct drift, notes/drift.md); categorical codes are fixed `pl.Enum` codes, identical on any subset. `--drop col,col` removes more columns (every name must exist).
- `--mode val`: temporal folds fit Jul -> val Aug, fit Jul+Aug -> val Sep. Early stopping (multi_logloss) on the legit-downsampled val month. Prints per-fold per-class F1 (argmax), then tunes per-class log-prob biases on the pooled OOF and prints pooled + per-fold scores with that bias.
  Writes `oof/NAME.npy` (float32 probs, rows = Aug then Sep, ts order), `oof/NAME_ids.parquet` (txn_id, y, fold), `oof/NAME_meta.json` (best iters, bias, scores, args).
- `--mode full`: reads `oof/NAME_meta.json` (runs val first if missing) and asserts its args (modules, model, lr, legit_frac, cw_pow, label_clean, clean_thr, clean_rounds, drop, max_rounds, patience) and n_feats match the current run. Fits on all train rows (Jul-Sep + the 147 Oct train rows), rounds = mean best iter * 1.1, averages `--seeds` models. Writes `preds/NAME_test.npy` (rows in `sample_submission.csv` order) and `subs/NAME.csv` (validated: 260582 rows, ids equal sample order, labels in `common.LABELS`).
- Same `--tag` must be used for val and full; full reuses the val biases and iterations, so keep the other flags identical.
- Imbalance: legit downsampled to `--legit-frac` (0.2) with weight 1/frac; `--cw-pow p` adds class weights (n_largest_fraud / n_c)^p.
- `--label-clean`: on each fit set, 2-fold random cross-fit (`--clean-rounds` 200), drop rows whose own label gets p < `--clean-thr` (0.03). Val labels are never touched.
- Model: `--model lgb` (default) or `xgb` (`--device cuda` for GPU). Add a model by putting `(fit, pred)` into `MODELS`.
- Knobs: `--lr 0.05 --max-rounds 3000 --patience 100`.
- RAM: ~1 GB for base features; grows ~7 MB per feature column (float32 x 1.82M rows).

## blend.py

```
python blend.py --tags a,b,c --weights 1,1,2 [--geo] --out NAME [--val-only]
```
Weighted mean of probs (or of log-probs with `--geo`) for OOF and test, re-tunes biases on the pooled OOF, writes `oof/NAME*`, `preds/NAME_test.npy`, `subs/NAME.csv`. All tags must come from train.py val runs (same OOF rows); unless `--val-only`, each `preds/<tag>_test.npy` must be newer than `oof/<tag>.npy`.

## Decoding

`train.tune_bias`: coordinate ascent over classes 1..11 (legit fixed 0), grid -4..6 step 0.1, score curve smoothed with a 5-point moving average before argmax, up to 4 passes. Uses a fast numpy macro-F1. quick_eval.py uses the same tuner.
