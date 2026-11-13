# tuning: LightGBM training scheme on the m5dm features (tuning agent, 13 Nov)

Code: `work/train.py` (owner). Scratch: `work/scratch_tuning/`:
- `lib.py`: harness on memmapped month arrays, with checkpoints scored from one fit.
- `exp.py`, `lane.sh`: runners.
- `rescore.py`: scores saved P files and their means.
- `results.jsonl`, `log_*.txt`: raw results.
- `P/<name>_r<N>.npy`: pooled OOF probs per checkpoint, in the same row order as `oof/m5dm.npy`.

## Result
**tag `tun_best`**: m5dm features + LGB `num_leaves 31, min_data_in_leaf 200, feature_fraction .5, lambda_l2 10` + relabel C fit target + fixed 150 rounds at lr .05, no early stopping. 3 seeds.

| | pooled biased | Sep biased | clean pooled | clean Sep | pooled argmax |
|---|---|---|---|---|---|
| m5dm (old params, ES ~30 it) | .4574 | .4650 | .4700 | .4772 | .4533 |
| **tun_best** (3 seeds, train.py) | **.4719** | **.4793** | **.4842** | **.4916** | .4665 |

That is +.0145 on pooled, +.014 on Sep and +.014 to +.0145 on the clean proxy.
- The gain splits into about +.011 from the LGB params and fixed rounds, and about +.003 from relabel C.
- Artifacts: `oof/tun_best.npy` (+ `_ids`, `_meta`), `preds/tun_best_test.npy`, `subs/tun_best.csv`. The sub predicts 3637 fraud rows.
- Logs: `notes/train_tun_best_val.log` and `notes/train_tun_best_full.log`. A rerun reproduced the same numbers exactly.

Reproduce (cwd=work, `PYTHONUTF8=1 POLARS_MAX_THREADS=3`):
```
python train.py --modules flow,burst,takeover,scam,ring --matrix feats_cache/matrix_m5dm.parquet --relabel C --rounds 150 --mode val  --threads 3 --seeds 3 --tag tun_best
python train.py --modules flow,burst,takeover,scam,ring --matrix feats_cache/matrix_m5dm.parquet --relabel C --rounds 150 --mode full --threads 3 --seeds 3 --tag tun_best
```
Time: val 4.5 min and full 3 min on 3 threads. full fits 165 rounds (150 x 1.1).

## Protocol
- Folds are the same as train.py: Jul->Aug and Jul+Aug->Sep. Legit is downsampled to .2 with weight 5, using the same rng draw as train.py.
- Val is the full month. Bias is tuned on the noisy pooled OOF (a vectorised `tune_bias`, checked to give identical biases).
- The SAME bias decodes noisy pooled, noisy Sep, clean pooled and clean Sep. Clean = `clean_labels.clean` on train rows.
- Every row below is 1 seed (seed 0) at lr .05 unless noted.
- **Seed noise is about ±.0013 pooled.** Example: the r1 config at seed 0 scores .4697 and at seed 1 scores .4671.
- Columns are: pooled biased / Sep / clean pooled / clean Sep.

## (c) Tree params: lambda_l2 is the knob
| config | rounds | pooled | Sep | cPool | cSep |
|---|---|---|---|---|---|
| old: nl63 md40 ff.8 l2 1 | 100 | .4582 | .4662 | .4705 | .4781 |
| nl31 md200 | 100 | .4612 | .4698 | .4733 | .4817 |
| ff.5 (l2 1) | 200 | .4608 | .4683 | .4731 | .4806 |
| ff.5 l2 10 | 100 | .4677 | .4748 | .4799 | .4868 |
| ff.3 l2 10 | 100 | .4685 | .4754 | .4804 | .4873 |
| ff.5 l2 50 | 200 | .4679 | .4759 | .4802 | .4879 |
| **ff.5 l2 10 nl31 md200 ("r1")** | 100 | **.4697** | **.4768** | **.4819** | **.4888** |
| ff.3 l2 10 nl31 md200 | 100 | .4689 | .4750 | .4812 | .4872 |
| ff.5 l2 30 nl15 md400 | 100 | .4652 | .4747 | .4774 | .4870 |
| r1, 2 seeds averaged | 100 | .4690 | .4761 | .4814 | .4883 |

- l2 10-50 gives +.009 over l2 1, on every metric. Smaller trees add about +.001 to +.002.
- feature_fraction .3 vs .5 is within noise.
- nl15/md400 is too much. It underfits pooled.

## (a)/(b) Rounds, lr, early stopping
- Old params, lr .05, pooled / Sep:

  | rounds | 30 | 60 | 100 | 200 | 400 | 800 |
  |---|---|---|---|---|---|---|
  | pooled | .4573 | .4580 | .4582 | .4589 | .4598 | .4605 |
  | Sep | .4647 | .4656 | .4662 | .4656 | .4651 | .4638 |

  Pooled creeps up (fold0 has less data and likes more rounds), but Sep peaks at about 100.
- With r1 + C, at rounds 50/100/150/250: pooled .4699/.4714/.4719/.4715 and Sep .4764/.4785/.4794/.4786. The curve is flat from 100 to 250. **Fixed 150 rounds.**
- lr .02 gives r150 .4689/.4762 and r250 .4690/.4763. That equals lr .05 r100 (.4697/.4768) at 2.5x the cost. Keep .05.
- (b) Macro-F1 early stopping is not worth it. The F1-vs-rounds curve is flat to within ±.001 from 100 to 250 rounds, so a fixed round count does the same job with less code. Logloss early stopping (the train.py default) is still available with `--rounds 0`.

## (d) legit_frac / cw_pow (old params): noise level
- lf .5: r30 .4603/.4675; r100 .4587/.4663.
- lf .1: r200 .4598/.4662.
- cw_pow .5: r100 .4613/.4652 (Sep is worse).
- With r1 params, cw .5 gives r50 .4674/.4755 and r100 .4674/.4747, against .4677/.4748 without. Neutral. Keep lf .2 and cw 0.

## (e) Training-target cleaning
| target (r1 params unless noted) | rounds | pooled | Sep | cPool | cSep |
|---|---|---|---|---|---|
| raw y | 100 | .4697 | .4768 | .4819 | .4888 |
| clean_labels.clean (old params) | 100 | .4595 (+.001) | .4665 | .4717 | .4787 |
| clean_labels.clean (ff.5 l2 10) | 100 | .4679 (+.0002) | .4748 | .4801 | .4867 |
| **relabel C** (notes/relabel.md) | 150 | **.4719** | **.4794** | **.4842** | **.4917** |
| relabel C, seed 1 | 150 | .4710 | .4784 | .4832 | .4905 |
| relabel C + old `--label-clean` (cross-fit drop p<.03) | 150 | .4706 | .4768 | .4828 | .4892 |

- C stacks with the new params at about +.003 on every metric. clean_labels is neutral.
- The cross-fit `--label-clean` drops about 11 % of fraud rows and loses .0013 to .0026 on every metric. **It is deleted from train.py.**

## (f) Seeds / mixing
- 2 seeds: +.000 to +.0005 pooled.
- Mixing 5 different param variants gives .4692, the same as 2 seeds.
- Seed averaging mainly reduces variance. full uses 3 seeds.

## train.py changes
- `LGB` params are now nl31 md200 ff.5 bf.8 l2 10. They are stored in `_meta.json` as `"lgb"`. `--mode full` refuses a val meta made with other lgb params, so old tags need a val rerun.
- `--rounds N`: fixed N rounds in val with no early stopping; full uses 1.1N. 0 (the default) keeps the old logloss early stopping.
- `--relabel VARIANT`: the fit target is `relabel.relabel(tx, VARIANT)` (from the relabel agent); -1 rows are dropped.
  - Val early stopping, OOF scoring and bias tuning keep the raw y.
  - The relabel runs in a child process, so its load_all heap does not stay in the trainer.
  - It is computed once on all train rows. That is leak-free for C, which uses only a row's own label plus features. It is NOT leak-free for A/B, which read neighbours' labels across the fold boundary.
- `--matrix PATH`: read the cached feature matrix instead of building it. The matrix must be in load_all row order, which holds for `matrix_m5dm`.
- `sample_fit(F, a, seed, rows=None, col="y_fit")` / `fit_one(F, a, seed, V=None, rounds=None, rows=None)`:
  - Old positional calls still work. scratch_adapt uses `fit_one(F, a, s, V)`.
  - When `y_fit` is missing, `y` is used.
  - Only the sampled rows are copied, so full no longer copies the whole train frame.
- Deleted: `label_clean`, `--label-clean`, `--clean-thr`, `--clean-rounds` (they lose; see (e)).

## Risks / notes
- RAM: train.py with `--matrix` holds about 3 GB of commit. The matrix alone is 1.6 GB and `import train` (lightgbm etc.) is about 1.1 GB. That is above the 2.5 GB brief. A build without `--matrix` is larger.
- The pooled-OOF bias is tuned in-sample on about 1M rows. All variants were compared under the identical decoder, so the deltas are fair; the absolute levels are a little optimistic.
- The lgb gain (+.011) agrees with the GPU agent's xgb at fixed rounds (.4682) and catboost (.4659). This model is now level with or slightly above those, and +C puts it ahead.
- Case-insensitive FS gotcha: `yc*.npy` and `yC*.npy` collide on Windows. Scratch files are now `yclean*` and `yrelC*`.

## Next ideas
- Blend tun_best with the GPU xgb/cat models after retraining them with relabel C. Expect only +.001, because all use the same features.
- Try l2 10 + C on xgb (`reg_lambda`); the GPU agent tried only l2 5 with depth 10.
- The bigger levers are features and post-processing (event smoothing), not trainer knobs. Remaining trainer knobs are all within ±.002 (seed noise).
