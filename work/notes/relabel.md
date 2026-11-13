# relabel: training-target relabelling (relabel agent, 13 Nov)

Code: `work/relabel.py` -> `relabel(tx, variant) -> pl.Series` (train labels, -1 = drop row, null on test). Variant = letters applied in order A,B,C,D,E.
Scratch: `work/scratch_relabel/` (labels.py, labels2.py, ev.py, score.py, proxy2.py, dcheck.py, full.py; logs `ev_*.log`, `score_*.log`).

## Verdict
- **Use C (takeover archetype type fix, type only). It is the only robust gain: +0.006 on every metric, 3 seeds.**
- A (clean_labels) and B (flow mode C) are neutral, within ±0.001. D (ring collusion) and E (drop generic labels) lose.
- Dropping 1-row-session sim/ATO labels to legit (the second half of the takeover proposal) gives nothing (-0.002 at 1 seed). relabel.py's C is therefore type-only.

## Protocol
- Model: train.py val protocol on `feats_cache/matrix_m5dm.parquet` (182 feats). LGB lr .05, legit_frac .2, ES on the noisy legit-downsampled val month, folds Jul->Aug and Jul+Aug->Sep, 3 threads.
- The legit subsample is a fixed per-row draw shared by all variants, so the fit sets differ only by the relabel.
- Relabel is computed on the fit-month rows only, so val labels never leak.
- Validation labels are untouched. Bias is `tune_bias` on the noisy pooled OOF, and the SAME bias decodes every proxy.
- Columns:
  - pooledB / sepB: noisy labels.
  - cleanB / sepCleanB: `clean_labels.clean` proxy (the official proxy).
  - proxy2: clean + B + C-type + D applied to val. This is circular for B/C/D; it only shows the upside if those hypotheses hold.
  - proxy3: proxy2 with unsupported ("generic") fraud labels set to legit. This is the "generic labels are noise" world.
- The OOF row order equals `oof/m5dm_ids.parquet`, so the OOFs are blendable.

## Results, 3 seeds (`score_s3a.log`, `score_s3b.log`)
| variant | pooledArg | pooledB | sepB | cleanB | sepCleanB | proxy2 |
|---|---|---|---|---|---|---|
| none (m5dm) | .4584 | .4616 | .4685 | .4739 | .4806 | .4856 |
| **C** | **.4635** | .4677 | .4741 | .4800 | .4864 | .4971 |
| AC | .4602 | .4670 | .4732 | .4793 | .4853 | .4973 |
| BC | .4617 | .4682 | .4755 | .4805 | .4878 | .5011 |
| ABC | .4600 | .4686 | .4759 | .4808 | .4881 | .4991 |

Per-class change, C vs none (pooledB / sepB / cleanB):
- sim_swap: +.051 / +.045 / +.050.
- account_takeover: +.017 / +.025 / +.018.
- Every other class is within ±.006 (noise).
- best_iter rises from ~28 to ~35. Cleaner sim/ATO targets let the model train longer before val logloss turns.

BC and ABC beat C by only +.0005 to +.0009 pooled. Collusion and refund swing ±.01 between seeds. B's own effect is consistent but tiny: mule +.003, layering +.002. Optional; a coin-flip.

## Results, 1 seed (`score_combo.log`, `score_c.log`; old C = type + singleton->legit)
| variant | pooledB | sepB | cleanB | sepCleanB | notes |
|---|---|---|---|---|---|
| none | .4592 | .4684 | .4712 | .4802 | |
| A | .4585 | .4631 | .4707 | .4751 | collusion -.013, sim -.007 |
| B | .4578 | .4624 | .4701 | .4744 | mule +.003, layering +.003 |
| C (old) | .4646 | .4716 | .4768 | .4838 | sim +.051, ATO +.022 |
| Ct = C type only | .4663 | .4745 | .4784 | .4865 | equal to old C or better |
| Cs = singleton->legit only | .4576 | .4645 | .4699 | .4765 | no gain |
| D | .4503 | .4573 | .4620 | .4687 | qr -.019, refund -.032, collusion -.048 (proxy2: collusion +.111) |
| E | .4362 | .4438 | .4480 | .4553 | layering -.118, phishing -.028, qr -.027, refund -.034 |
| AB | .4599 | .4678 | .4719 | .4793 | |
| ABC (old) | .4651 | .4731 | .4772 | .4853 | |
| ABCD | .4565 | .4628 | .4685 | .4748 | |

## Rules (rows changed on Jul-Sep train)
- A: 1165 rows. legit->velocity 462, legit->mule 273, then scattered type fixes.
- B: 1009 rows. mule->layering 353 (rows in flow chains), legit->mule 345 (burst rule), legit->layering 311.
- C: 750 rows. ATO->sim 380 and sim->ATO 370, by archetype (`takeover__arch_sim` / `arch_ato`, full-density cache).
- D: 913 rows. qr->collusion 413, refund->collusion 319, legit->collusion 181.
- E: drops 37k fraud rows (~60 % of fraud labels) that fail a per-class support rule (`relabel.support`). Example rules: sess_n>=2 for sim/ATO, chain_size>=2 for layering, txn_type for qr/refund/phishing.

## D rejected without an LB A/B (`dcheck.py`)
This settles the ring note's H1 vs H2 question. Take round-amount small-merchant segments and group them by majority label:
- qr_swap: 154 segments, median 3 rows, 2 senders, p90 3 rows, 0 % mixed labels.
- refund_abuse: 73 segments, median 5 rows, 3 senders, p90 8 rows.
- merchant_collusion: 46 segments, median 14.5 rows, 4 senders, p90 24 rows, 33 % mixed.

If H2 held, the qr- and refund-labelled segments would be misfiled collusion events and would share collusion's size distribution. They are systematically much smaller, so they are generator variants of qr_swap and refund_abuse (H1). Training on D also loses 0.009 on noisy val.

## Generic ("camouflage") fraud rows: are they fraud in the clean test?
- Under "generic labels are noise" (proxy3), every model scores about 0.80 macro. Under "they are real camouflage fraud" (noisy and clean proxies), scores are about 0.47.
- The organiser's note says "0.40 is excellent, nobody expected to reach 0.50". That only makes sense if camouflage fraud is in the clean test, so I assume it is.
- **The first LB upload settles this at once:** public ~0.47-0.50 means camouflage; ~0.75+ means noise.
- Under camouflage, the noisy and clean-proxy metrics are the right targets. E-type dropping only removes learnable signal: many "generic" rows are partly predictable from row features, e.g. layering -.118.
- C, being type-only, does not change how many rows are predicted fraud. Its test sub predicts 3727 fraud rows vs 3666 for m5dm.
- If the LB shows the noise world, revisit E and the bias: proxy3 for E is still worse than C (.8095 vs .8228 at 1 seed). Biases tuned on noisy labels would also under-call there.

## Artifacts (variant C, 3 seeds)
- `scratch_relabel/oof_C_s3.npy`: OOF, same rows and order as `oof/m5dm.npy`, which makes it blendable.
- `scratch_relabel/relabelC_test.npy`: test probs, sample_submission order. Full fit uses all train rows with C labels, 38 rounds (mean best_iter 34.5 x 1.1), 3 seeds.
- `scratch_relabel/relabelC_bias.json`: bias tuned on the pooled noisy OOF.
- `scratch_relabel/relabelC.csv`: validated, 260582 rows. It agrees with subs/m5dm.csv on 99.94 % of rows; sim_swap goes 257 -> 288.
- Baseline OOFs: `oof_none_s3.npy`. Also AC/BC/ABC `_s3` and all 1-seed variants `_s1`.

## Integration (for the train.py owner)
C uses only a row's own label plus takeover cache features, so it can be applied once to all train rows.
Change the fit target only; ES and val keep the raw `y`. For example, in `build_matrix`, add the target column:
`D = D.with_columns(relabel(tx, "C").alias("y_fit"))`
Then make `sample_fit` read `y_fit` on the fit set, while `fit_one` keeps calling `sample_fit(V, ...)` with `y` for early stopping.
Leave the bias tuning on noisy y as it is now.
