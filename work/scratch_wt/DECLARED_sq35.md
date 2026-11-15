# Second exploratory candidate (raw blend .65 h6 + .35 wSQ): rules declared 15 Nov 2026, 07:06 IST, before any score of this recipe exists

Known before this note: the two-seed harness readings in `notes/assumption_tests.md` (C). No score of the six-seed recipe,
of either control, or of any October file exists yet. The four fits were launched at 07:02 IST and are still running.

## Recipe (fixed)
- `wSQ6`: `scratch_wt/wt.py` arm `wSQ` (baseline weights x n_c^-0.5, total weight rescaled to the baseline total) on the
  production density-matched matrix (D tables), seed numbers 0 to 5, same keep-mask formula as `train.py`
  (`default_rng(seed)`), 150 rounds, learning rate .05, `T.LGB`. August / September: mean of `PD_wSQ` (seeds 0,1),
  `PD_wSQb2` (2,3), `PD_wSQb4` (4,5). October: `scratch_wt/wt_oct.py` on fold 2, seeds 0 to 5.
- Candidate `SQ6b35` = .65 h6 + .35 wSQ6, raw probabilities, blended before the decoder. Weight .65 / .35 is not searched.
- Reference = h6 (six seeds), the anchor of dd_final.

## Assessment order
1. August D after the decoder (`aug_dec.py D`), reference h6 .5002.
2. Gate. "Clearly worse" = August D below h6 by more than .0010, or September D interval (5%..95%) entirely below zero.
   Clearly worse: report and stop, no file. Otherwise: export, also when the interval includes zero.
3. September D once (`bench_eval.py D`), descriptive.
- N: no new fit. The existing N readings (seeds 0,1 and 2,3; N-trained two-seed versions, reference C) are reported
  separately; they are not the shipped file.
- The public score of the first exploratory upload changes nothing here.

## Controls (no new model; August only)
- K1, scalar fraud-against-legit shift: h6 with every fraud probability multiplied by e^d and renormalised,
  d in {0.5, 1.0, 1.5}; d picked on August D after the decoder (highest; tie: smallest d). Expectation written now: about
  zero, because `tune_bias` already tunes a per-class bias on August and a scalar shift only moves its starting point.
- K2, scale-matched mixture: `.65 h6 + .35 h6s`, where h6s = h6 x per-class ratio, renormalised; the ratio is fitted so
  that the mean class probabilities of h6s on August equal those of wSQ6 on August (predictions only, no labels). The same
  ratio is used on September. Same probability-scale mixture as the candidate, without the weighted model.
- Reading: candidate minus K2 on September (paired) = what the weighted model adds beyond the scale change.

## Build (only if the gate passes)
Tag `sq35`: blended validation and test probabilities -> unmodified `postproc.py sq35 --qr-null` -> `ts_judge_post.py sq35`
-> `scratch_mix/chain_fb.py` (accepted rule chain, steps 12 to 17, no step 11, declared fallback if a guard refuses the
base) -> `subs/sq35_dd_final.csv`. Checks: ids / order / labels, tuned bias inside the grid, zone edges against h6's,
per-class counts against dd_final, predicted non-legit ratio candidate / h6 on August, September and October (does the
August bias transfer to October), rule rows per layer, whether a rule touches a changed row.
No automatic upload. `subs/dd_final.csv` is not touched.
