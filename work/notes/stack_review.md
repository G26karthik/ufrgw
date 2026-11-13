# stack review (adversarial, 13 Nov)

Verdict: the "no win, keep tun_best_ppq" conclusion in `stack.md` holds and is stronger than stated. The +.0005 pooled for rstack r30 is seed luck.

## Reproduced from saved OOF (`scratch_stackrev/check.py`)
- Rows align with `oof/tun_best_ids.parquet`: same txn_id set, y and fold.
- Every number in the stack.md table matches to 4 decimals:
  - tun_best: .4719 / .4793 / .4842 / .4916 / oos .4787 / .4909
  - lomo_s1: .4722 / .4791 / .4845 / .4914 / .4773 / .4896
  - rstack_r30: .4724 / .4796 / .4848 / .4920 / .4762 / .4885
- Reverse oos decode (bias tuned on Sep, scored on Aug): tun_best .4634, lomo_s1 .4637, rstack_r30 .4648, rnoagg_r30 .4643.
  - The Aug fold is confounded: its stage-1 model was fit on Jul+Sep, so it saw future data.
  - Mean of both oos directions: tun_best .4711, rstack .4705, lomo_s1 .4705.
- Paired bootstrap on Sep rows (200 resamples, 90% CI), rstack_r30 minus each baseline:

  | baseline | pooled-bias Sep | oos Sep |
  |---|---|---|
  | tun_best | +.0002 [-.0008, +.0012] | -.0026 [-.0045, -.0006], significant loss |
  | lomo_s1 | +.0005 [-.0006, +.0014] | -.0011 [-.0028, +.0009] |

## Fresh seeds (`scratch_stackrev/reseed.py`, seeds 3-5, 5.5 min)
Columns: pooled / Sep / c_pooled / c_Sep / oos Sep / c_oos Sep.

| run | pooled | Sep | c_pooled | c_Sep | oos Sep | c_oos Sep |
|---|---|---|---|---|---|---|
| rstack_r30, seeds 3-5 | .4721 | .4784 | .4843 | .4907 | .4755 | .4877 |
| rstack_r30, 6 seeds | .4721 | .4790 | .4844 | .4914 | .4751 | .4875 |
| rnoagg_r30, 6 seeds | .4722 | .4790 | .4845 | .4913 | .4741 | .4862 |

- With seeds 3-5, rstack falls below tun_best on Sep (.4784 vs .4793).
- At 6 seeds the aggregate gain over rnoagg drops from the claimed +.0027 to +.0010 oos and -.0001 pooled.
- r30 was also picked from 6 checkpoint/variant combinations on these same folds, which adds selection optimism.

## Leakage / protocol notes
None of these can turn the verdict into a win, because each one inflates stack scores.
- Stage-2 training rows carry stage-1 probs from LOMO models that were trained on the validation month. Examples:
  - For the Aug fold, Jul features come from an Aug+Sep model.
  - For the Sep fold, Jul and Aug features come from models that include Sep.
  - This is not a row-level leak into validation features. It still breaks nested cross-fit hygiene.
- LOMO stage 1 also trains on the 147 Oct train rows, which tun_best's folds do not use. The effect is negligible.
- No test path, artifacts or sub exist, so there is no Oct transfer risk.
