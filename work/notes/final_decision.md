# Final decision report (15 Nov 2026, 03:15 IST)

## Decision
- `subs/dd_final.csv` stays the reference file (public .5505, SHA-256 `8e04353d6f52fe86ab9e44b5dd33d7adec9ba26df36f566d0bcfd68d49d829cd`). Unchanged.
- One exploratory upload was authorised and prepared: `subs/cmb80_dd_final.csv`
  (SHA-256 `cff9925e5503b644bd16701916bea693d4942bf3c43b853a913fced4ea82c3e3`, 15 rows differ from dd_final). No uploader
  script exists in the repository, so the file was handed over for manual upload at 03:00 IST.
- Its public score, once known, is recorded below as one observation. No further upload or row change follows from it.
- The platform ranks each team on its best public upload. If the exploratory file scores above .5505 on the public
  part, it becomes the file ranked on the private part.

Public score of `cmb80_dd_final.csv`: not yet reported.

## How to read every number here
- Benchmark N: labels hidden, features at full density. Benchmark D: the production density-matched matrix, a
  label-informed density stress test and not independent ground truth. N and D are reported separately. "Fails D" does
  not mean "cannot generalise".
- Variants are chosen or frozen on August. September is read afterwards. September has been inspected more than 40
  times; the intervals cover event sampling for one comparison and do not correct for that reuse.
- Status words: supported / promising but unconfirmed / no detected difference / measurably worse / not evaluated adequately.
- No score ceiling is claimed. A narrow test that fails does not show that its whole method family fails.

## Track 1: combined submission (`notes/combined_submission.md`, rules in `scratch_mix/DECLARED.md`)
- Deviation from the brief: the six alternative sources were refitted on the density-matched production matrix, because
  October has about half the row density. Their N results belong to versions that are not shipped.
- Pool: h6 anchor + c2s6, four-seed soft two-stage, TabM (2 seeds, epoch 30 chosen on August), neighbour-feature model,
  TabICLv2, higher-binning model. Exact-value arm and tree stacker excluded. All sources finished before the cut-off.
- Three declared mixtures, probabilities blended before the decoder:

| anchor weight | August D (selector) | September D | August N (side) | September N (side) |
|---|---|---|---|---|
| reference | .5002 | .5090 | .4978 | .5070 |
| .50 | .5001 | -.0010 (-.0017..-.0003) | .4985 | +.0008 (+.0001..+.0015) |
| .65 | .5002 | -.0000 (-.0006..+.0006) | .4985 | +.0004 (-.0003..+.0013) |
| .80 (chosen) | .5003 | +.0001 (-.0004..+.0007) | .4985 | +.0005 (-.0001..+.0012) |

- Status of the chosen mixture: no detected difference on D (the shipped recipe). N side reading positive, not confirmed.
- Rule chain: all accepted layers applied, no discarded bundle. Five rule rows did not fire because the model already
  gives the rule's label. The pool-bait guard refused the base over `TX1647c469`; the declared fallback applied its 13
  accepted rows; that one row stays legit (1 of the 15 differences).
- Gains do not need to "survive the rules": none of the 15 changed rows is touched by a rule layer.

## Track 2: shared assumptions (`notes/assumption_tests.md`)
| assumption | status |
|---|---|
| A. detection depends on the non-QR stamp | supported that stamped rows stay detected without it (.995 -> .991); the saturation claim is not evaluated adequately |
| B. the nine categorical columns need categorical handling | no detected difference (LightGBM and CatBoost, N and D); models today treat them as ordered numbers |
| C. legit rows carry too much training weight | no detected difference alone; raw blend of the square-root arm promising but unconfirmed (N yes, D no) |

No candidate file came from Track 2.

## Earlier candidates still on file
- `subs/alt_c2s6_dd_final.csv` (12 rows from dd_final): no detected difference, not uploaded (`notes/alt_c2s6.md`).
- All evening tests of 14 Nov: `notes/final_comparison.md`.

## Reproduce
- dd_final: `run_all.ps1` (steps 1 to 17).
- Exploratory file: `scratch_mix/mix_data.py`, `mix_lgb.py`, `mix_knn.py`, `mix_icl.py`, `gpu_chain.sh` (source
  predictions), then `bash scratch_mix/build_all.sh`.

## Not done
- No source was scored alone on D for the neighbour model and TabICLv2 (density-matched versions).
- The three Track 2 tests each ran one or two declared variants; untested variants are listed in the note.
- Which changed rows fall in the public part is unknown.
