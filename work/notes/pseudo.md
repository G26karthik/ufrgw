# pseudo: transductive adaptation by pseudo-labelling and domain weights (pseudo agent, 13 Nov)

Code: `work/pseudo.py` (`prep | sim VAR --seeds | score VAR | full VAR`). Scratch: `work/scratch_pseudo/`.
- `lab.parquet`: event keys, relabel-C fit target and the clean_labels proxy.
- `P_<variant>_s<seed>.npy`: OOF probs in the `oof/tun_best_ids` row order.
- `sim*.log`, `score1.log`, `evdirect.py` / `evdirect.log`.

## Verdict: no variant wins. Nothing was applied to October.
- Self-training on the model's own target-month predictions never beats the plain refit. Every pseudo variant scores the same as tun_best or worse, on every metric.
- Adversarial domain weighting loses clearly.
- `preds/pseudo_test.npy` and `subs/pseudo.csv` were **not** produced. tun_best_ppq stays the best.

## Protocol (simulation)
- Folds are the same as train.py: fit Jul -> target Aug, and fit Jul+Aug -> target Sep. Both folds are treated, so the pooled OOF is in the same format as tun_best.
- The target month is unlabeled. Its pseudo labels are the tun_best OOF probs, decoded with the bias tuned on the **other** fold, so no target label is used.
- Refit = tun_best config: m5dm matrix, relabel C fit target, LGB params from train.LGB, 150 rounds, legit_frac .2. The fit is on the fit-month rows plus the pseudo rows. Pseudo rows also get legit_frac sampling.
- Columns:
  - pooled / sep: biased macro-F1, with the bias tuned on the pooled OOF.
  - cPool / cSep: clean_labels proxy.
  - oosSep / oosCSep: bias tuned on the Aug fold only, scored on Sep.
- `base s0` reproduces the tun_best seed-0 numbers exactly (.4719/.4794/.4842/.4917). All variants are seed 0, so the comparison is paired against `base s0`.

| variant | pooled | sep | cPool | cSep | oosSep | oosCSep |
|---|---|---|---|---|---|---|
| tun_best (3 seeds) | .4719 | .4793 | .4842 | .4916 | .4787 | .4909 |
| **base s0** (no pseudo) | **.4719** | **.4794** | **.4842** | **.4917** | **.4783** | **.4906** |
| frall: every decoded-fraud row | .4717 | .4794 | .4839 | .4916 | .4780 | .4903 |
| fr50: decoded fraud, P>=.5 | .4716 | .4786 | .4838 | .4909 | .4770 | .4893 |
| fr80: P>=.8 | .4699 | .4773 | .4821 | .4896 | .4760 | .4881 |
| frq50: top 50% per decoded class (per-class threshold) | .4712 | .4784 | .4835 | .4907 | .4771 | .4893 |
| fr50leg: fr50 + most-legit 50% of rows as legit | .4713 | .4779 | .4835 | .4902 | .4760 | .4884 |
| ev: event majority (clean_labels.clean on decoded labels) | .4713 | .4792 | .4836 | .4915 | .4777 | .4900 |
| ev50: event majority + P>=.5 | .4715 | .4793 | .4838 | .4917 | .4773 | .4895 |
| dw: adversarial importance weights, no pseudo rows | .4673 | .4720 | .4795 | .4841 | .4626 | .4749 |
| evdirect: event majority as a post-rule, no refit | .4704 | .4777 | .4840 | .4913 | .4776 | .4912 |

## Findings
- **Pseudo labels: the more confident the selection, the worse the score.**
  - frall -.0002, fr50 -.0003, frq50 -.0007, fr80 -.0020 pooled.
  - Confident rows are the ones the model already gets right. Adding them shifts the fit-set class balance toward easy core rows and teaches nothing new.
  - The pseudo-label accuracy against noisy true y is .75-.80, about .9 after allowing for the 16% label noise. So the losses do not come from bad pseudo labels.
  - Adding target-month legit rows (fr50leg) costs -.0006 pooled and -.0023 out-of-sample.
  - P[legit] never exceeds .99 (max .989), so a probability threshold on legit picks no rows. A rank share is used instead.
- **Event-level pseudo labels change almost nothing.** clean() on the decoded target month moves only 38 (Aug) and 65 (Sep) rows: the predictions are already event-coherent, as notes/postproc.md found.
  - The rows it does move become *less* correct on noisy y: accuracy .71 -> .05 on Aug and .49 -> .17 on Sep.
  - As a direct post-rule, evdirect loses .0015 pooled.
- Weight .5 (fr50w05) and round 2 (`NAME_r2`) were not run. Every round-1 parent loses, so neither can help.
- **Domain weighting loses -.0046 pooled and -.016 out-of-sample.**
  - The adversarial AUC is .78 for Jul vs Aug and **.97 for Jul+Aug vs Sep**.
  - The drivers are receiver-window features: `scam__rcv_in_15d`, `scam__rcv_qr_mm1_frac_15d`, `scam__rcv_round_frac_15d`, `ring__seg_n`, `ring__seg_nsnd`, `ring__r36_excess`, `ring__rcv_n`, `ring__rcv_nsnd`, `ring__r36_int_share`.
  - Their monthly medians are identical (for example rcv_n 31/30/30/31 for Jul/Aug/Sep/Oct). The classifier is separating months by fine per-receiver (merchant identity x month volume) patterns, which have nothing to do with fraud.
  - The weights are extreme: on fold 1, median .18 and max 33. The effective sample size collapses.

## Why it cannot work here
- 63% of fraud rows are camouflage (notes/camo.md). The model predicts them legit with P ~.98, so no pseudo label can recover them.
- The core rows are already fit from 1M+ labelled rows. Re-adding the model's own core decisions for the target month adds no information.
- The density-matched features already removed the main covariate shift (notes/drift.md). The only real Oct adaptation is the QR-null evasion, which postproc --qr-null handles.
- The remaining Sep->Oct changes are prior shifts: velocity runs at about 1/3, QR is down. Pseudo-labelling would only reinforce the model's own Sep-tuned prior.

## Not checked / caveats
- The simulation cannot show Oct-specific drift: legit amounts x1.17, and ip/app, which are dropped anyway. A pseudo refit on Oct would only see that drift through the model's own decisions.
- `pseudo.py full` was written but **never run**, because no variant passed the gate.
- Seed noise is about ±.0013 pooled. Only seed 0 was run. Every variant is paired at or below base on all six metrics, so more seeds cannot turn any of them into a win.

## Next ideas (not pseudo)
- The .97 month-separability of ring/scam receiver-window features suggests per-receiver month drift. A feature agent could check whether `ring__rcv_n`, `ring__r36_excess` and the `scam__rcv_*_15d` features also separate Sep from Oct. If so, a rank or within-month normalisation might help, or dropping them, to be tested with `train.py --drop`.
