# Learning how to combine h6 and wSQ6 (15 Nov 2026, 08:01 IST)

Question: does a learned combination beat the fixed .65 / .35 blend (`notes/sq35_submission.md`)?
Answer: no learned selector beat the fixed blend on August or September, on N or on D. The fixed .35 weight is at the
August optimum of the global-weight grid. No additional file was exported. `subs/dd_final.csv` and all candidates are unchanged.

## Was this already implemented?
No. Closest existing code: the fixed blend (`scratch_wt/sq6.py`), declared equal-share mixtures
(`scratch_mix/mix_blend.py`) and a tree stacker over other sources (`scratch_stack/n14`, see `notes/stack.md`). None
learns a weight for h6 and wSQ6. New code: `scratch_wt/learn.py`; rules written before any score:
`scratch_wt/DECLARED_learn.md` (07:33 IST, amended 07:39 IST). Fitted selectors: `scratch_wt/learn_D.json`,
`learn_N.json`; fit logs `learn_D2.log`, `learn_N2.log`.

## Method
- Inputs: saved out-of-sample predictions only, no model retrained. D: h6 and wSQ6 (six seeds). N: four-seed means of
  the N-trained reference and of the N-trained wSQ arm (reference `C4`). N and D are fitted, selected and read separately.
- 1. Global weight w in {.10, .20, .35, .50, .65, .80, 1.00}, selected on August.
- 2. Class-specific weights w_k, weighted multinomial log-loss with a free per-class offset, penalty on the spread of
  the weights (lambda 1e-4, 1e-3, 1e-2).
- 3. Row gate: weight = sigmoid(a + theta . z) on 10 label-free columns (fraud log-odds and top-class share of both
  models, top-class agreement, log amount, log device age, log SIM age, log login-to-transaction seconds, QR stamp
  present), same loss, penalty on theta, same lambda grid.
- Cross-fitting: August is split into the two event-grouped halves of `aug_dec.py`; a selector fitted on one half
  predicts the other; the bias is tuned on one half and the other half is scored after the full decoder. September:
  selector fitted on all August, bias from the cross-fitted August output, `bench_eval.py` once.
- A first run exported the mixture without the fitted offset (scale differs between fits); those files (`CS*`, `GT*`)
  were never scored. The scored selectors (`CW*`, `RG*`) export natural-prior probabilities on one scale.

## August after the decoder (selection)
| variant | D (h6 .5002) | N (C4 .4985) |
|---|---|---|
| global .10 | .5006 | .4987 |
| global .20 | .5007 | .4988 |
| global .35 = fixed blend | .5007 | .4990 |
| global .50 | .5000 | .4987 |
| global .65 | .4998 | .4978 |
| global .80 | .4995 | .4980 |
| global 1.00 (wSQ alone) | .4984 | .4965 |
| class weights, lambda 1e-4 / 1e-3 / 1e-2 | .4961 / .4977 / .4998 | .4977 / .4987 / .4985 |
| row gate, lambda 1e-4 / 1e-3 / 1e-2 | .4996 / .5003 / .5001 | .4987 / .4982 / .4984 |

- Picks by the declared rule (highest; within .0003 the smaller weight or the larger lambda): global .10, class weights
  1e-2, row gate 1e-2, on both benchmarks. Overall pick: global .10 (D .5006, N .4987).
- The highest August value on both benchmarks is the fixed .35 blend (shared with .20 on D). The tie rule moved the
  pick to .10; it was written before the scores and is kept.
- Export rule: the pick must exceed the fixed blend on August D by more than .0005. Not met (.5006 against .5007).

## September after the decoder (reused development evidence; 5%..95%)
| variant | D against h6 (.5090) | D against the fixed blend | N against C4 (.5076) | N against the fixed blend |
|---|---|---|---|---|
| fixed blend .35 | +.0002 (-.0003..+.0009) | reference | +.0008 (+.0001..+.0015) | reference |
| global .10 | +.0000 (-.0004..+.0005) | -.0002 (-.0010..+.0003) | -.0002 (-.0009..+.0004) | -.0011 (-.0017..-.0005) |
| class weights | -.0014 (-.0022..-.0005) | -.0016 (-.0026..-.0008) | +.0000 (-.0008..+.0007) | -.0008 (-.0017..+.0001) |
| row gate | -.0003 (-.0008..+.0002) | -.0005 (-.0014..+.0002) | -.0001 (-.0010..+.0005) | -.0010 (-.0019..-.0002) |

Rows changed / corrected / newly wrong against the reference: D fixed blend 47 / 17 / 23, global .10 29 / 8 / 19, class
weights 35 / 6 / 27, row gate 17 / 6 / 9; N fixed blend 38 / 19 / 15, global .10 26 / 9 / 15, class weights 42 / 15 / 24,
row gate 37 / 11 / 21.

## Status
- Fixed .35 blend against the reference: promising but unconfirmed, as before. New here: with four seeds on N its
  September interval is above zero (+.0001..+.0015); on D it is not.
- Global weight .10 against the fixed blend: no detected difference on D, measurably worse on N.
- Class-specific weights: measurably worse on D (against h6 and against the fixed blend); no detected difference on N.
- Row gate: no detected difference on D; measurably worse than the fixed blend on N.
- No selector is supported. None was exported.

## What the fits show
- Log-loss puts the weight of wSQ6 near .07 for every class (D and N); the August score after the decoder prefers .20
  to .35. The two criteria disagree: log-loss rewards calibration, the decoder retunes the bias anyway.
- Class weights at weak penalty spread between .04 and .21 and score worst; they shrink to a single weight as the
  penalty grows. Gate coefficients change sign between the two August halves at weak penalty (fraud log-odds of h6:
  +.66 against -.72). Both agree with control K2: wSQ6 acts like a per-class rescale of h6, so the weights are barely
  identifiable and carry no stable row-level information.
- Limits: selectors learn on August-fold predictions (July-only models) and are applied to September-fold models; one
  loss, one row weighting and one gate column set were tried; a selector trained directly on the after-decoder score
  was not tried. A narrow test: it does not show that no learned combination can help.
- The two public scores reported at 07:31 IST (cmb80 .5484, sq35 .5494) were not used for any setting, pick or row.

## Reproduce (cwd = work, `PYTHONPATH=. PYTHONUTF8=1`)
```
python scratch_wt/learn.py D ; python scratch_wt/learn.py N
python scratch_gap/aug_dec.py D h6,GW10,GW20,GW35,GW50,GW65,GW80,GW100,CWa,CWb,CWc,RGa,RGb,RGc   # N: C4 first
python scratch_gap/bench_eval.py D h6,GW35,GW10,CWc,RGc _lrn ; python scratch_gap/bench_eval.py D GW35,GW10,CWc,RGc _lrnf
python scratch_gap/bench_eval.py N C4,GW35,GW10,CWc,RGc _lrn ; python scratch_gap/bench_eval.py N GW35,GW10,CWc,RGc _lrnf
```
Logs: `scratch_wt/learn_aug_*.log`, `learn_sep_*.log`, `learn_diff_*.log`.
