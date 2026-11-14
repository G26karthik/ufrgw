# scratch_wt: training influence of legit versus fraud classes (weights only)

Started 2026-11-14 23:27 IST. No new fits after 05:30 IST 15 Nov. Report by 06:00.
Rules: 4 threads, no GPU, nothing overwritten, prediction files only `scratch_gap/PN_w*` / `PD_w*`, decode files `scratch_gap/dec_<B>_wt*.npz`.

## Arms (script `scratch_wt/wt.py`, same harness as `scratch_gap/numrep.py::run` arm A_C, num_threads 4 instead of 8)
- `wC`  baseline weights (legit 5, fraud 1): harness control, compared with the stored reference.
- `wL1` every sampled row weight 1.
- `wSQ` baseline weights x n_c^-0.5, n_c = class counts of all fit rows of the fold (before legit downsampling).
- Each arm rescaled so total weight = baseline total on the same sampled rows (asserted after float32 cast, logged).
- Fit-row counts, fold 0 (July): [507552, 3422, 2834, 2325, 1409, 1437, 2280, 2421, 1041, 888, 678, 778];
  fold 1 (July+August): [1015059, 6870, 5446, 4636, 2803, 2769, 4534, 4910, 2156, 1699, 1338, 1697]. Same for N and D. No y_fit < 0 rows.

## Prior coverage found (notes/tuning.md section d; all single seed, pooled OOF / Sep before the decoder, older setups)
- old params r100: baseline .4582/.4662; cw_pow .5 .4613/.4652; lf .5 r100 .4587/.4663 (r30 .4603/.4675); lf .1 r200 .4598/.4662.
- r1 params: cw .5 r50 .4674/.4755, r100 .4674/.4747, against .4677/.4748 without. "Neutral. Keep lf .2 and cw 0."
- notes/gpu.md: gpu_xgb_lf5 (legit_frac .5) 300r .4631/.4669, .4694/.4720, "-.001".
- notes/research.md:72 is a literature note, not a measurement. notes/generator_experiment.md B05/B1 weight source-event rows, not classes.
- cw_pow = (n_largest_fraud / n_c)^p on fraud only, legit multiplier 1, no total rescale. legit_frac changes keep weight 1/frac.
- New here: wL1 keeps the rows and drops the correction (legit share of weight ~96% -> ~84%). wSQ has the same fraud-to-fraud
  ratios as cw_pow .5 but also lowers legit relative to fraud (~12x) and rescales the total. Scored after the decoder, 2 seeds, on N and D.

## Steps
- [done 00:29] 23:32 fits N then D: wC wL1 wSQ seeds 0,1 -> `scratch_wt/fit_N.log`, `fit_D.log`
- [done 23:47] N control: `PN_wC` is bit-identical to `PN_C` for August and September at 4 threads (`check_N.log`). Reference on N = `C`.
  One arm takes about 15 min (fold0 276 s, fold1 607 s). All six fits expected by about 01:05.
- [done 00:06] N fits wL1, wSQ seeds 0,1 -> `scratch_gap/PN_wL1_{aug,sep}.npy`, `PN_wSQ_{aug,sep}.npy`. Weight totals (fit_N.log):
  baseline total = arm total on every fold and seed (527563.0 / 527893.0 / 1054373.0 / 1054193.0). Legit share of total weight:
  C .963, wL1 .839, wSQ .614. Per-row weights: wL1 4.356 all classes; wSQ legit 3.19, fraud 7.8 (mule) to 17.4 (loan_app_extortion).
- [done 00:08] N blends (`blend_N.log`): `PN_wL1b35`, `PN_wSQb35` = .65 C + .35 arm (raw, the specified reading);
  `PN_wL1p`, `PN_wSQp` = arm divided by its per-class weight ratio and renormalised (prior-matched, fit counts only);
  `PN_wL1pb35`, `PN_wSQpb35` = .65 C + .35 prior-matched arm. Mean fraud probability on August: C .0370, wL1 raw .1193,
  wSQ raw .2793, prior-matched .0356 / .0338, so the raw blend is not a 35% source in prior terms.
- [done 00:29] D fits wC, wL1, wSQ seeds 0,1 -> `PD_wC`, `PD_wL1`, `PD_wSQ` (`fit_D.log`; totals and legit shares equal to N).
  `PD_wC` is bit-identical to `PD_A_C` (`check_D.log`). Reference on D = `A_C`. D blends built (`blend_D.log`), same names with PD_.
- [done 00:44] August only (`bias_N.log`, `bias_D.log`, `aug_dec_N_{1,2}.log`, `aug_dec_D_{1..4}.log`).
  Bias clip check (all August rows, raw and forced, C / wL1 / wSQ / both raw blends, N and D): every tuned bias is inside
  the grid (most negative: wSQ -2.6 on N, -2.8 on D; grid floor -4). So the single-arm reading uses raw probabilities.
  August after decoder (mean of two halves; change vs reference):
  | variant | N | D |
  |---|---|---|
  | reference (C / A_C) | .4978 | .5004 |
  | wL1 | .4979 (+.0001) | .5004 (+.0000) |
  | wSQ | .4973 (-.0005) | .4989 (-.0015) |
  | wL1b35 (raw blend) | .4980 (+.0002) | .5003 (-.0001) |
  | wSQb35 (raw blend) | .4985 (+.0007) | .5010 (+.0006) |
  | wL1pb35 (prior-matched blend) | .4983 (+.0005) | .5001 (-.0003) |
  | wSQpb35 (prior-matched blend) | .4983 (+.0005) | .5002 (-.0002) |
  | wL1p / wSQp (prior-matched alone, August only) | .4964 / .4971 | .5004 / .4986 |
  The two August halves differ by about .010 for every source; these changes are far inside that.

## FREEZE 2026-11-15 00:44 IST (written before any September score of these arms was computed or read)
(Header first said 00:46 by estimate; the file was saved at 00:44:03 by the clock and the September scoring started right after.)
- Frozen: arms wL1 and wSQ exactly as fitted (seeds 0,1), blend weights .65 reference + .35 arm (given by the task). Nothing is chosen on September.
- September is scored once per benchmark for: reference, wL1, wSQ (raw), wL1b35, wSQb35 (raw blend = the specified reading),
  wL1pb35, wSQpb35 (prior-matched blend = declared second reading; it adds two September looks per benchmark).
  wL1p / wSQp alone are NOT scored on September (no bias clipping, they are a re-parametrisation of the raw arm's bias).
- Commands: `bench_eval.py N C,wL1,wSQ,wL1b35,wSQb35,wL1pb35,wSQpb35 _wt` and `bench_eval.py D A_C,<same> _wt`.
- Seeds 2,3 trigger (N only, reference `A_Cb2`): an arm is refitted with seeds 2,3 if, on N seeds 0,1, the single arm or its
  raw blend has August after-decoder change > 0 AND September after-decoder change > 0. On August that leaves wL1, wL1b35 and
  wSQb35 eligible (wSQ alone is negative on August); September decides.
- Status rule per variant (form of `scratch_gap/numrep_rule.txt`): "supported" only if the September after-decoder change is
  positive with the paired event-bootstrap 5%..95% interval above zero on N seeds 0,1 AND in both confirmations (N seeds 2,3; D seeds 0,1)
  and the August after-decoder change is >= 0 in each. First reading above zero plus one of two confirmations = "promising but unconfirmed".
  Interval entirely below zero on N = "measurably worse". Everything else = "no detected difference".
  Seed floor on N about +-.002; September has been inspected more than 40 times before this run.

## After the freeze
- [done 00:55] September once: `eval_N_wt.log`, `eval_D_wt.log` -> `scratch_gap/dec_N_wt.npz`, `dec_D_wt.npz`;
  row changes `diff_N.log`, `diff_D.log`; 12-class precision/recall `table_N.log`, `table_D.log`.
  September, seeds 0,1 (before decoder / after decoder, change vs reference, 5%..95% interval of the after-decoder change;
  after-decoder rows changed / corrected / newly wrong; missed fraud recovered / new legit called fraud):
  | variant | N before | N after | N change (interval) | N rows | N fraud rec / legit FP | D before | D after | D change (interval) | D rows | D fraud rec / legit FP |
  |---|---|---|---|---|---|---|---|---|---|---|
  | reference | .4815 | .5070 | | | | .4824 | .5078 | | | |
  | wL1 | .4842 | .5092 | +.0022 (+.0010..+.0033) | 121/51/57 | 32 / 41 | .4812 | .5068 | -.0010 (-.0024..+.0004) | 151/51/88 | 24 / 57 |
  | wSQ | .4832 | .5083 | +.0013 (+.0002..+.0023) | 135/46/77 | 28 / 56 | .4818 | .5074 | -.0004 (-.0015..+.0007) | 260/64/178 | 42 / 160 |
  | wL1b35 | .4841 | .5093 | +.0023 (+.0012..+.0034) | 84/41/36 | 28 / 27 | .4828 | .5083 | +.0005 (-.0003..+.0012) | 53/22/24 | 11 / 16 |
  | wSQb35 | .4834 | .5086 | +.0016 (+.0006..+.0024) | 60/27/27 | 18 / 16 | .4827 | .5082 | +.0004 (-.0003..+.0009) | 71/26/37 | 16 / 31 |
  | wL1pb35 | .4823 | .5074 | +.0004 (-.0002..+.0011) | 38/14/22 | 9 / 16 | .4824 | .5079 | +.0001 (-.0008..+.0008) | 41/18/17 | 8 / 5 |
  | wSQpb35 | .4822 | .5075 | +.0005 (-.0004..+.0013) | 48/17/25 | 7 / 12 | .4821 | .5075 | -.0003 (-.0010..+.0004) | 45/16/24 | 4 / 10 |
- Trigger met on N for wL1 (August +.0001, September +.0022) and for wSQ through its raw blend (August +.0007, September +.0016).
- [done 01:07] N seeds 2,3 fits of wL1 and wSQ -> `PN_wL1b2`, `PN_wSQb2` (`fit_Nb2.log`).
- [done 01:17] seeds 2,3 on N, reference `A_Cb2`: `blend_Nb2.log`, `aug_dec_Nb2_{1,2,3}.log`, `eval_N_wt_b2.log`
  (-> `scratch_gap/dec_N_wt_b2.npz`), `diff_Nb2.log`, `table_Nb2.log`.
  | variant (seeds 2,3) | August after | Sep before | Sep after | change (interval) | rows chg/corr/wrong | fraud rec / legit FP |
  |---|---|---|---|---|---|---|
  | A_Cb2 | .4992 | .4821 | .5072 | | | |
  | wL1b2 | .4984 (-.0008) | .4827 | .5081 | +.0009 (-.0013..+.0029) | 153/63/69 | 20 / 27 |
  | wSQb2 | .4971 (-.0021) | .4832 | .5084 | +.0012 (-.0010..+.0029) | 170/55/95 | 23 / 59 |
  | wL1b35b2 | .4987 (-.0005) | .4825 | .5079 | +.0007 (-.0003..+.0017) | 59/31/19 | 7 / 3 |
  | wSQb35b2 | .4993 (+.0001) | .4830 | .5085 | +.0013 (+.0001..+.0024) | 74/38/25 | 10 / 5 |
  | wL1pb35b2 | .4993 (+.0001) | .4813 | .5067 | -.0004 (-.0014..+.0005) | 59/27/24 | 2 / 1 |
  | wSQpb35b2 | .4990 (-.0002) | .4813 | .5066 | -.0006 (-.0023..+.0006) | 72/24/38 | 4 / 5 |

## Status by the declared rule (01:18)
- wL1 alone: no detected difference (N seeds 0,1 +.0022 above zero; seeds 2,3 +.0009 across zero with August -.0008; D -.0010 across zero). 0 of 2.
- wSQ alone: no detected difference (N +.0013 above zero but August -.0005; seeds 2,3 +.0012 across zero, August -.0021; D -.0004). 0 of 2.
- .65 C + .35 wL1 (raw): no detected difference by the rule (N +.0023 above zero; seeds 2,3 +.0007 across zero; D +.0005 across zero). 0 of 2; all three point estimates positive.
- .65 C + .35 wSQ (raw): promising but unconfirmed (N +.0016 above zero; seeds 2,3 +.0013 (+.0001..+.0024), August +.0001; D +.0004 (-.0003..+.0009)). 1 of 2.
- prior-matched blends: no detected difference (N +.0004 / +.0005 across zero; seeds 2,3 -.0004 / -.0006; D +.0001 / -.0003).
- No fits remain. Nothing in work/subs, oof, preds, feats_cache touched. Final report sent to the caller.
- ALL STEPS DONE 01:20. Nothing is running, nothing is queued. 18 September comparisons were read (6 variants x 3 runs).
- Notes for the reader: (1) seeds 2,3 velocity_bot F1 falls for the single arms (wL1 -.0057, wSQ -.0080; reference .5117),
  a class that did not move with seeds 0,1 or on D. (2) wL1p against wL1 on N August (.4964 against .4979) is the same model
  with the bias re-parametrised: about .0015 of decoder-tuning noise that the bootstrap interval does not cover.
  (3) The raw blend carries 3x to 8x the fraud probability mass of C, so it is not a 35% source; the prior-matched blend is.
