# Learned combination of h6 and wSQ6: rules declared 15 Nov 2026, 07:33 IST, before any score of these selectors exists

Not implemented before this note: the repository holds the fixed .65 / .35 blend (`sq6.py`), declared equal-share mixtures
(`scratch_mix/mix_blend.py`) and a tree stacker over other sources (`scratch_stack/n14`, excluded earlier). No selector
for h6 and wSQ6 exists. Code: `scratch_wt/learn.py`. No model is retrained; only saved out-of-sample predictions are used.

Known before this note: the fixed blend's readings (`notes/sq35_submission.md`) and two public scores reported by the user
at 07:31 IST (cmb80 .5484, sq35 .5494, dd_final .5505). The public scores are not used: not for any setting, not for the
export rule, not for any row.

## Inputs
- D: H = `PD_h6`, W = `PD_wSQ6` (six seeds each). Reference h6; fixed blend `SQ6b35`.
- N: H = mean of `PN_C` and `PN_A_Cb2` (seeds 0-3), W = mean of `PN_wSQ` and `PN_wSQb2` (seeds 0-3), N-trained versions.
  Reference `C4`; fixed blend = `GW35` on N. N and D are fitted, selected and reported separately.

## Three selectors
1. Global weight, selected on validation: P = (1 - w) H + w W, w in {.10, .20, .35, .50, .65, .80, 1.00}. No fit.
2. Class-specific weights: mix_k = (1 - w_k) H_k + w_k W_k, renormalised; w_k = sigmoid(a_k). Fitted by weighted
   multinomial log-loss of log(mix_k) + b_k (b = free per-class offset, legit fixed at 0; it absorbs the prior change and
   is not exported: the decoder tunes its own bias). Penalty lambda x sum_k (a_k - mean a)^2, lambda in {1e-4, 1e-3, 1e-2}.
3. Row gate: mix = (1 - g) H + g W, g = sigmoid(a + theta . z), same loss and offsets, penalty lambda x |theta|^2, same
   lambda grid. z (10 standardised columns, all label-free): fraud log-odds of H and of W, share of the top fraud class
   within the fraud mass for H and for W, whether the two top fraud classes agree; log amount, log device age, log SIM
   age, log login-to-transaction seconds, QR stamp present.
- Loss row weight for 2 and 3: (class count)^-0.5, mean 1 (fixed, not searched). L-BFGS, start at w = .35.

## Cross-fitting and selection
- August rows are split into the two event-grouped halves of `aug_dec.py` (fraud events whole). A selector fitted on one
  half predicts the other half; `aug_dec.py` then tunes the bias on one half and scores the other after the decoder
  (forcing, zones, bait rule). No row is scored with a selector or a bias that saw its label.
- Settings are chosen on August after the decoder: per selector the highest value; within .0003 the smaller weight (1) or
  the larger lambda (2, 3). Overall pick: the highest of the three; within .0003 the simpler one (1 before 2 before 3).
- September: selector fitted on all August rows with the chosen setting, bias tuned on the cross-fitted August output,
  `bench_eval.py` once. Reported as reused development evidence, against h6 / C4 and against the fixed .35 blend (paired).

## Export rule (D decides, N must not contradict)
An additional candidate file is built only if the overall pick on D
- is above the fixed blend on August D by more than .0003, and
- is not below the fixed blend on September D (point estimate >= 0 in the paired comparison), and
- on N is not measurably worse than the fixed blend (September interval not entirely below zero).
Otherwise: report, no file. If built: selector cross-fitted over event-grouped halves of August + September for the bias
tuning, fitted on August + September for October, then the unmodified decoder and rule chain. No automatic upload.

## Amendment, 15 Nov 2026, 07:39 IST, before any score of a selector exists
- A first run wrote class-weight and gate files (`CSa..c`, `GTa..c`) that export the mixture without the fitted offset.
  The weights are barely identifiable against the offset, so two fits can sit on different probability scales, and the
  bias tuned on one half's rows would be applied to rows from another fit. Those files are not scored. The selectors
  now export mix_k x exp(b_k) / (loss class weight), renormalised: natural-prior probabilities, equal to H at weight 0.
  New names `CWa..c` (class weights) and `RGa..c` (row gate). The global-weight files are unchanged.
- Seen in the first run's fit log (training loss only, no score): fitted weights near .07 for every class.
- Export margin raised: the overall pick must be above the fixed blend on August D by more than .0005 (control K1, a pure
  rescale of h6, moved August by up to .0004).
- Expectation written now: control K2 showed the weighted model acts like a per-class rescale of h6, so all three
  selectors should land near the fixed blend and the fitted per-class weights should not be read as meaningful.
- Limit: selectors learn on August-fold predictions (July-only models) and are applied to September-fold models.
- September is read once for the three per-selector picks, against h6 / C4 and against the fixed blend.
