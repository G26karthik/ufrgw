# Combined exploratory submission: rules declared 14 Nov 2026, 23:35 IST, before any mixture score exists

## Deviation from the brief, stated first
The brief asks for test predictions from each source's actual feature pipeline. Five alternative sources were developed on
benchmark N (features at full density). October has about half the row density of July to September. `density.py`
records, for an older four-module LightGBM, that a full-density fit scored .4363 on a September thinned to October density
against .4527 for the density-matched fit. A full-density fit of these sources would therefore predict October rows that
do not look like its training rows. Every alternative source is refitted on the production density-matched matrix
(`feats_cache/matrix_m5dm2_pairsfd.parquet`, the matrix h6 is trained on), with the same recipe otherwise. Consequences:
- The N results of these sources (two-stage blend +.0017, binning +.0011, ...) belong to N-trained versions that are not
  shipped. They are not evidence for this file.
- The shipped versions are assessed on benchmark D, which is a label-informed density stress test.
- For the October fit itself, labels are used only to place train rows in a half; no test label exists.

## Pool (one averaged probability source per configuration)
| # | source | recipe on the density-matched matrix | validation files (D) | test fit |
|---|---|---|---|---|
| anchor | h6 | production LightGBM, 6 seeds | `PD_h6` | `preds/h6_test.npy` |
| 1 | c2s6 | production LightGBM + core2, 6 seeds | `PD_c2s6` | `preds/c2s6_test.npy` |
| 2 | sTS4 | soft two-stage, seeds 0-3 | mean of `PD_sTS`, `PD_sTSb2` | 4 seeds |
| 3 | TabM | ple, seeds 0 and 1; per seed the checkpoint from {6, 12, 20, 30} with the best August score after the decoder (fold 0 of D); the same epoch number for fold 1 and for the full fit | new | 2 seeds |
| 4 | kA | 190 + 50 neighbour columns (same space and bank design as `scratch_knn`), seeds 0 and 1 | new | 2 seeds |
| 5 | TabICLv2 | context 20,000 legit + up to 2,500 per fraud class, seed 0, ensemble 1, prior-corrected | new | 1 context |
| 6 | BSEL | 1023 bins on the 9 row-level numeric columns, seeds 0 and 1 | `PD_A_BSEL` | 2 seeds |
Excluded: exact-value arm, tree stacker (as instructed).

## Mixtures
Anchor weight a in {0.50, 0.65, 0.80}; every alternative gets (1 - a) / (number of alternatives). Probabilities are
blended before the decoder.

## Selector
- D August after the decoder (`aug_dec.py`: bias tuned on one event-grouped half, other half scored after forcing, zones
  and bait rule, halves swapped). The highest of the three wins. If two are within .0003, the higher anchor weight wins.
- Side reading, not used for selection: the same three anchor weights on benchmark N with the N-trained versions
  (anchor `PN_C`; five alternatives, because c2s6 equals the anchor on N).
- September is read after the selection and reported descriptively. It has been inspected many times.

## Cut-off
A source whose D validation or test predictions are not complete by 03:15 IST is dropped from the assessment and from
the file; the remaining alternatives share the weight equally.

## Build
- Blended validation probabilities (August from fold 0, September from fold 1; same sources and weights as the test
  blend) and blended test probabilities are written under a new tag. Then the unmodified `postproc.py --qr-null`,
  `ts_judge_post.py`, and the rule driver `alt_c2.py` (steps 12 to 17; no step 11).
- Checks before the rules: every row has every source, rows sum to 1, per-class predicted counts against
  `ts_judge_h6`, zone edges against h6's.
- No production assertion is relaxed. If an assertion inside a rule layer refuses the new base, that layer's accepted
  rows are taken from the dd_final chain instead (row id and target as produced by `alt_c2.chain` on
  `subs/ts_judge_h6.csv`), applied in the same order, and every row whose current label differs from the rule's recorded
  source label is listed as a conflict.
- No upload if the candidate equals dd_final or any earlier upload, or if the three mixtures give the same file.
