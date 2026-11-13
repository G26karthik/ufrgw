# Review of agent `gpu` (XGBoost/CatBoost on matrix_m5dm)

Scripts: `work/scratch_gpurev/c1.py` (static checks, log `c1.log`) and `rr.py` (GPU rerun of gpu_xgb fold1 plus QR-evasion test, log `rr.log`).

## Verdict
- The OOF gain is real. Rows are aligned, the rerun reproduces exactly, and no leakage was found.
- The recommended upload `subs/gpu_blend.csv` has a concrete Oct regression on the evaded QR rows. That regression probably cancels most of the gain.
- Do not upload gpu_blend as is. Refit on adapt features first (see Recommendation).

## Confirmed
- **Rerun.** gpu_xgb fold1 (Jul+Aug -> Sep, seed 0, 600 rounds) matches the saved OOF exactly (max |dP| 0.0000). Sep scores .4693 argmax / .4731 biased, identical to the claim.
- **Alignment.** The OOF ids and y of gpu_xgb, gpu_cat and gpu_blend equal m5dm_ids and the matrix y. The blend OOF and test equal (2·xgb + cat)/3.
  - subs/gpu_blend.csv equals decode(preds, meta bias).
  - Test argmax agreement with m5dm is .9990, the same as OOF agreement (.9990). Test row order is fine.
- **Leakage.** None found in models_gpu.py. No feature column carries a label. y is used only for fit rows and bias tuning.
  - density.halves uses labels, but m5dm shares that, so it does not bias this comparison.
- **Statistical support.** Day-block bootstrap (200 resamples) of gpu_blend minus m5dm, each model with its own pooled bias:
  - noisy pooled +.0116 [+.009, +.014]
  - noisy Sep +.0093 [+.006, +.012]
  - clean pooled +.0110
  - clean Sep +.0090
  - P(diff <= 0) is 0 in all four.
- **Half-legit Sep (Oct density).** The gain holds: blend .4793 vs m5dm .4699 on noisy labels, .4879 vs .4789 on clean.

## Problems
1. **QR evasion: the GPU models label most of the evaded Oct rows legit.** These are the 140 test P2P-QR rows with null qr_vpa_mismatch (notes/adapt.md).

   | sub | null-mm P2P-QR rows labelled qr_swap |
   |---|---|
   | m5dm | 109/140 |
   | lgb base refit (adapt agent) | 124/140 |
   | gpu_xgb | 45/140 |
   | gpu_cat | 10/140 |
   | **gpu_blend** | **29/140** |

   - Sep simulation (45% of P2P-QR-cat rows blanked, protocol of scratch_adapt/ev.py): xgb qr F1 falls .486 -> .422 (-.064). Predicted qr on the hit rows falls from 287 to 119.
   - lgb loses only -.021 (.484 -> .463).
   - Rough Oct estimate: about 80 lost qr_swap true positives, a qr F1 drop of about .14, and about -.012 macro. That is the same size as the claimed +.011 gain.
   - Likely cause: xgb and CatBoost send missing values down a learned or Min branch, and in train null mismatch means non-QR. lgb happened to route these rows toward qr.
2. **In-sample bias optimism hides a worse blend.** With the bias tuned on Aug and scored on Sep:
   - m5dm .4627, gpu_xgb **.4719**, gpu_blend .4685.
   - The blend is below xgb alone, and below its own Sep argmax (.4710). The xgb:2 + cat:1 choice is not supported over xgb alone.
   - The honest Sep gain over m5dm is +.009 for xgb alone and +.006 for the blend, against the claimed +.009.
3. **The clean proxy is weak.** clean_labels changes only 752 of 1.04M OOF rows. "Clean" numbers are therefore noisy numbers plus a constant, not independent evidence about the clean Oct labels.
4. **Per-class trade-off.** On the Sep clean proxy, fake_cashback is -.023 vs m5dm. Every other class is +.002 to +.029.

## Recommendation
- Refit gpu_xgb on the m5dm matrix joined with feats_cache/adapt__dm.parquet, with raw qr_vpa_mismatch dropped. This is the adapt agent's evasion-proof recipe.
  - Check that the null P2P-QR test rows get qr_swap >= ~109/140.
  - Upload xgb alone. Drop cat: it is the worst at evasion and adds nothing under honest bias.
- The quick alternative needs no refit: set qr_vpa_mismatch=1 on the null P2P-QR test rows before predicting. Train has 0 such rows, so this is the same recipe as adapt__qr_mm.
- Until either fix is in, m5dm and gpu_blend are about even on Oct (estimate).
