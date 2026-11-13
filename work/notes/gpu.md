# GPU models (XGBoost cuda, CatBoost GPU) on matrix_m5dm

Code: `work/models_gpu.py` (cwd=work, `PYTHONUTF8=1 POLARS_MAX_THREADS=3`).
- Same folds, legit downsample and bias tuning as train.py. The keep mask uses the same rng draw over the same row order, so the fit rows match train.py exactly.
- Rows come from `feats_cache/matrix_m5dm.parquet`. Its row order equals train.py's D, and val rows match `oof/m5dm_ids.parquet`. Verified.
- Rounds are fixed, with no early stopping.
- val writes `oof/<tag>.npy`, `_ids.parquet` and `_meta.json`; blend.py accepts them. full writes `preds/<tag>_test.npy` and `subs/<tag>.csv`.

```
python models_gpu.py --model xgb --tag gpu_xgb --mode val  --rounds 600 --ckpt 50,100,200,300,400   # lr .05 depth 8 mcw 5 l2 1
python models_gpu.py --model xgb --tag gpu_xgb --mode full --rounds 600 --seeds 3
python models_gpu.py --model cat --tag gpu_cat --mode val  --rounds 2000 --lr 0.1 --depth 8 --l2 3 --ckpt 250,500,1000,1500
python models_gpu.py --model cat --tag gpu_cat --mode full --rounds 2000 --lr 0.1 --depth 8 --l2 3 --seeds 3
```
Eval script: `scratch_gpu/ev.py "tagA:w+tagB:w" ...`.
- Bias is tuned on noisy pooled OOF. That same bias is applied to noisy y and to the cleaned-proxy y (`clean_labels.clean`, 752 OOF rows changed, cached in `scratch_gpu/yclean_oof.parquet`).
- tune_bias takes about 2 min per call on 1.04M rows. It dominates run time.

## Results (1 seed in val; argmax / biased)

| tag / blend | noisy pooled | noisy Sep | clean pooled | clean Sep |
|---|---|---|---|---|
| m5dm (lgb, ES ~30 it) | .4533 / .4574 | .4599 / .4650 | .4657 / .4700 | .4721 / .4772 |
| **gpu_xgb** d8 lr.05 600r | .4645 / **.4682** | .4693 / .4731 | .4766 / .4804 | .4813 / .4849 |
| gpu_xgb @50r | .4552 / .4688 | .4620 / .4758 | .4674 / .4809 | .4738 / .4877 |
| gpu_xgb @100r | .4635 / .4682 | .4701 / .4744 | .4755 / .4803 | .4820 / .4865 |
| gpu_xgb @200r | .4644 / .4677 | .4700 / .4725 | .4765 / .4798 | .4820 / .4846 |
| gpu_xgb @400r | .4647 / .4683 | .4697 / .4729 | .4770 / .4806 | .4818 / .4849 |
| gpu_xgb_d6 400r | .4635 / .4681 | .4694 / .4741 | | |
| gpu_xgb_d10 mcw20 l2 5 300r | .4616 / .4659 | .4676 / .4725 | | |
| gpu_xgb_lf5 (legit_frac .5) 300r | .4631 / .4669 | .4694 / .4720 | | |
| **gpu_cat** d8 lr.1 l2 3 2000r | .4632 / .4659 | .4658 / .4712 | .4754 / .4780 | .4777 / .4832 |
| gpu_cat_d10 2000r | .4635 / .4650 | .4667 / .4691 | | |
| m5dm + gpu_xgb@50 | .4574 / .4654 | .4637 / .4723 | .4697 / .4779 | .4759 / .4846 |
| m5dm + gpu_xgb@100 | .4601 / .4670 | .4665 / .4729 | .4723 / .4794 | .4787 / .4851 |
| gpu_xgb + gpu_cat | .4660 / .4683 | .4703 / .4733 | .4783 / .4804 | .4823 / .4852 |
| **gpu_xgb:2 + gpu_cat:1** | .4665 / **.4689** | .4710 / .4742 | .4786 / .4810 | .4829 / .4861 |
| gpu_xgb:3 + gpu_cat:1 | .4667 / .4681 | .4716 / .4733 | .4788 / .4802 | .4835 / .4851 |
| m5dm + gpu_xgb + gpu_cat | .4650 / .4688 | .4695 / .4746 | .4772 / .4810 | .4817 / .4866 |
| gpu_xgb:2 + gpu_cat:1 + m5dm:1 | .4653 / .4689 | .4703 / .4739 | .4775 / .4811 | .4825 / .4860 |
| xgb + xgb_d6 + cat | .4660 / .4689 | .4705 / .4737 | .4782 / .4811 | .4826 / .4858 |
| all 5 GPU variants | .4662 / .4691 | .4708 / .4746 | .4783 / .4812 | .4830 / .4865 |

## Findings
1. **XGB and CatBoost both beat m5dm lgb by about +.011 pooled.** This holds on argmax (.4645 against .4533), on biased (.4682 against .4574) and on the clean proxy (.4804 against .4700).
   - The gain comes from sim_swap (+.04), ATO (+.02 to .03), phishing (+.02), refund (+.01) and merchant_collusion (+.02).
   - Likely cause: lgb early-stops at ~30 iterations on noisy val logloss and underfits. With fixed rounds, F1 is flat from 50 to 600 xgb rounds at lr .05.
   - Suggestion for the train.py owner: run lgb with fixed rounds (300+) and stop early stopping on logloss.
2. **Rounds do not matter much past ~100** (xgb lr .05). Bias-decoded F1 is flat; argmax F1 rises slightly up to ~200. CatBoost lr .1 keeps improving slowly up to ~1500.
3. **Depth and regularisation are flat or worse.** d6 is equal to d8. d10 with heavy regularisation is -.002. CatBoost d10 is -.001. legit_frac .5 is -.001.
4. **Blending adds almost nothing**, +.0007 at most. All models use the same 182 features and are highly correlated.
   - Adding m5dm (lgb) to the GPU blend neither helps nor hurts.
   - More gains need new features or new labels (cleaning or pseudo-labels), not new learners.
   - Between-variant spread is about ±.002, which is seed-noise level.
5. Recommended blend: `gpu_xgb:2,gpu_cat:1`, or with m5dm:1. Both give .4689 pooled biased and .481 on the clean proxy. Full fits use 3 seeds. See the blend.py numbers below.
6. MLP skipped: torch is not installed in the venv, and given finding 4 a same-feature learner is unlikely to add diversity.

## blend.py runs (bias re-tuned on pooled OOF; test = 3-seed full fits)
| out | tags:weights | pooled argmax / biased | fold0 biased | Sep biased | sub |
|---|---|---|---|---|---|
| gpu_blend | gpu_xgb:2, gpu_cat:1 | .4665 / **.4689** | .4635 | .4742 | subs/gpu_blend.csv |
| gpu_blend_m5 | gpu_xgb:2, gpu_cat:1, m5dm:1 | .4653 / .4689 | .4638 | .4739 | subs/gpu_blend_m5.csv |
| (single) gpu_xgb | | .4645 / .4682 | .4633 | .4731 | subs/gpu_xgb.csv |
| (single) gpu_cat | | .4632 / .4659 | .4606 | .4712 | subs/gpu_cat.csv |

Clean proxy for gpu_blend: pooled .4786/.4810, Sep .4829/.4861. m5dm gets .4657/.4700 and .4721/.4772.
- Upload candidate: `subs/gpu_blend.csv`. It scores +.0115 pooled biased and +.011 clean pooled over m5dm.
- Test label counts are close to m5dm. qr_swap drops 463 -> 381, phishing rises 463 -> 513.
- Reproduce with `python blend.py --tags gpu_xgb,gpu_cat --weights 2,1 --out gpu_blend`, after the val and full commands above.
