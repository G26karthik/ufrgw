# Alternative for review: c2s6 model under the dd_final rule chain (14 Nov 2026, 23:15 IST)

Status: prepared for review. Not a validated improvement. Not uploaded. `subs/dd_final.csv` is unchanged
(SHA-256 `8e04353d6f52fe86ab9e44b5dd33d7adec9ba26df36f566d0bcfd68d49d829cd`).

| | file | SHA-256 | rows | non-legit rows |
|---|---|---|---|---|
| current | `subs/dd_final.csv` | `8e04353d...29cd` | 260,582 | 3,984 |
| alternative | `subs/alt_c2s6_dd_final.csv` | `b7c293e54611dbb2a36e5a32e798f770f3e164e8b842a95051929285fe44345e` | 260,582 | 3,980 |

## What it is
- The only change is the model: c2s6 (the six modules plus `core2`) in place of h6. Base file `subs/ts_judge_c2s6.csv`
  (run_all step 10: model, post-processing, device zones, bait rule).
- Every accepted later layer of dd_final is applied (run_all steps 12 to 17): v3 (QRN + zone propagation), RS, FCB, CZ,
  LFM, X999, BANDV, EDGEFIX, pool-bait + stamp.
- Step 11 is left out: the art_acct / art_zonemax transplant that made `final1` (public .5435, below .5443).

## How to regenerate
```
.\run_all.ps1 -Tag X                      # steps 1-10 write work/subs/ts_judge_X_c2.csv
cd work; python alt_c2.py subs/ts_judge_X_c2.csv X_alt_c2_dd_final     # PYTHONPATH=. PYTHONUTF8=1
```
`alt_c2.py` with no arguments rebuilds `subs/alt_c2s6_dd_final.csv` from `subs/ts_judge_c2s6.csv` (it refuses to overwrite).

## Reproducibility checks (all passed)
- Base: `ts_judge_c2s6.csv` (`16e42ef4...97ba`) is byte-identical to the file made by the clean from-raw run of 14 Nov
  (`ts_judge_repro_c2.csv`) and to an earlier from-raw run (`ts_judge_r_repro_c2.csv`).
- Driver self-check: `alt_c2.py` on `ts_judge_h6.csv` gives the bytes of `dd_final.csv`. It writes nothing otherwise.
- Determinism: the driver on the second copy of the base gives the same bytes as the alternative.
- A second implementation (`scratch_gap/c2x_chain.py`, layers applied one after another) gives the same bytes.
- Format: `validate_sub.py`: same ids and order as dd_final, valid labels, 12 rows changed (`logs/validate_alt_c2s6.log`).

## Deviation from the pipeline as written
- `sc_final.py` and `bc_final.py` cannot run on this base: they gate on row counts measured on h6. `bc_final.py` requires
  LFM to change exactly 4 rows; here it changes 5. `alt_c2.py` calls the same rule functions in the same composition
  (CZ, LFM, X999 each on the step-13 output, diffs unioned, overlaps must agree) without those count gates.
- Kept: every assertion inside the rule functions (train-side purity checks, expected ids of the pool-bait and stamp
  rows, EDGEFIX row checks), the RS / FCB disjointness check, the clash check.
- The fifth LFM row, `TX2927ddfc`: h6 decodes it mule_account, c2s6 decodes it layering_chain, LFM moves it to
  mule_account. The alternative therefore carries dd_final's label on it. This firing was not part of the original rule
  review. The row (sender AC070731, receiver AC291816) shares no sender, receiver or device with the 12 changed rows.

## Rule interactions
| layer | rows changed, h6 base | rows changed, c2s6 base | same rows and targets |
|---|---|---|---|
| v3 (QRN + zone propagation) | 28 | 28 | yes |
| RS | 3 | 3 | yes |
| FCB | 5 | 5 | yes |
| CZ | 10 | 10 | yes |
| LFM | 4 | 5 | the same 4 + `TX2927ddfc` |
| X999 | 1 | 1 | yes |
| BANDV | 10 | 10 | yes |
| EDGEFIX | 3 | 3 | yes |
| pool-bait + stamp | 13 | 13 | yes |

- Zone edges found on the decode are identical for both bases (both edge sets, `logs/alt_c2s6.log`).
- The models differ on 13 rows before the later rules. LFM removes one difference (`TX2927ddfc`). 12 remain.
- None of the 12 remaining rows is touched by any layer in either chain. They are pure model disagreements.
- Transplant rows of the discarded bundle (14 rows where `final1` differs from `ts_judge_c2s6`): the alternative equals
  dd_final on all 14. It equals `final1` on 8 of them, the same 8 that dd_final already has through BANDV.

## The 12 changed rows
Probabilities are raw model outputs before bias and rules. "Neighbours" = other October rows that share the receiver or
sender, with their non-legit labels (identical in both files).

| id | dd_final | alternative | h6 p(dd) / p(alt) | c2s6 p(dd) / p(alt) | neighbours |
|---|---|---|---|---|---|
| TX6a3ada77 | phishing_collect | legit | .320 / .616 | .212 / .709 | none non-legit |
| TX10754001 | phishing_collect | legit | .362 / .574 | .271 / .639 | none non-legit |
| TX9cba4eb3 | phishing_collect | legit | .353 / .571 | .261 / .649 | none non-legit |
| TX1d444be7 | phishing_collect | legit | .310 / .658 | .293 / .670 | none non-legit |
| TXf0208897 | phishing_collect | fake_cashback | .534 / .251 | .512 / .273 | receiver: 1 phishing; sender: 6 refund, 1 qr, 1 phishing |
| TX96e28c86 | phishing_collect | fake_cashback | .595 / .291 | .530 / .361 | receiver: 1 fake_cashback; sender: 1 fake_cashback |
| TX7755c193 | mule_account | layering_chain | .417 / .325 | .311 / .411 | receiver AC392297: 5 layering in the same burst; these two are its only mule rows |
| TX78dc5f40 | mule_account | layering_chain | .404 / .324 | .300 / .442 | receiver AC392297: 5 layering in the same burst; these two are its only mule rows |
| TXf591f8e7 | legit | layering_chain | .741 / .226 | .618 / .338 | receiver AC151804: 3 layering |
| TXffb4ef65 | velocity_bot | mule_account | .346 / .448 | .334 / .476 | receiver AC083650: 21 mule |
| TX1bb042e8 | fake_cashback | phishing_collect | .336 / .595 | .308 / .627 | receiver: 1 phishing; sender: 1 phishing |
| TXb05075a8 | refund_abuse | legit | .710 / .273 | .468 / .491 | merchant: 1 refund; sender: 2 refund |

- Every flip is a margin case: the two models' probabilities differ by .01 to .24 on these rows, near the decode
  threshold. The models also carry different tuned biases (h6 / c2s6: mule .4 / .7, layering .6 / .8, refund .3 / .0;
  phishing .8 and fake_cashback 1.5 in both), so some flips come from the bias and some from the probability shift.
- No rule layer touches any row at AC392297 in either chain.
- Neighbours: 6 flips agree with other rows of the same entity (both AC392297 rows, AC151804, AC083650, TX96e28c86,
  TX1bb042e8), 2 disagree (TXb05075a8, and TXf0208897 by its receiver), 4 have no non-legit neighbour. This is an
  observation on unlabelled rows. No row was kept or dropped on it; the alternative is the complete model swap.
- Full row list with both pre-rule labels: `notes/alt_c2s6_dd_final_diff.csv`. Rows changed by each layer:
  `notes/alt_c2s6_dd_final_layers.csv`. All checks: `scratch_gap/alt_c2_verify.py`, `alt_c2_verify.log`.

## Evidence
- September, benchmark D, c2s6 against h6 with the same decoder: +.0008 (+.0000..+.0017); with the four generic later
  layers +.0008 (+.0001..+.0017). August after the decoder: +.0006 (.5008 against .5002).
- On the September rows where the models differ: c2s6 right 41, h6 right 51, both wrong 8. The macro gain comes from
  loan_app_extortion (+.006 F1). No October change involves that class.
- September by the transition types that occur in October (after decoder):

| h6 -> c2s6 | September rows | c2s6 right | h6 right | both wrong | October rows |
|---|---|---|---|---|---|
| phishing_collect -> legit | 8 | 6 | 2 | 0 | 4 |
| legit -> layering_chain | 47 | 15 | 29 | 3 | 1 |
| fake_cashback -> phishing_collect | 8 | 3 | 1 | 4 | 1 |
| mule_account -> layering_chain | 3 | 0 | 2 | 1 | 2 |
| refund_abuse -> legit | 1 | 0 | 1 | 0 | 1 |
| phishing_collect -> fake_cashback | 0 | | | | 2 |
| velocity_bot -> mule_account | 0 | | | | 1 |
| all seven | 67 | 24 | 35 | 8 | 12 |

- Weighted by October's mix, the September rates give about 3.7 rows right for c2s6 and 4.1 for h6 over the 9 October
  rows that have a September analogue. Three rows have none. The pooled 24 against 35 is dominated by legit ->
  layering_chain (47 rows), which is 1 of the 12 October rows.
- Scenario from `validate_sub.py` (a truth model, not known test truth; whole test set): +.0011 if all 12 new labels are
  right, -.0012 if none is, break-even at 53% right. Its break-even assumes that a wrong new label means the old label
  was right; in September 8 of the 67 rows were wrong under both models.

## Uncertainty
- The supporting evidence is D only. D is a label-informed density stress test, and benchmark N cannot test this swap
  (N builds every feature at full density, which removes the difference between the two models).
- The +.0008 has a lower bound at zero, rests on one class, and September has been inspected many times. The interval
  covers September event sampling for one comparison and does not correct for that reuse.
- The September gain and the October changes are in different classes. On the transition types that do occur in October,
  September gives no usable signal either way (about 3.7 against 4.1 rows of 9; 4 of 7 types have 0 to 3 September rows).
- Public evidence: none for these 12 rows alone. `final1` held them together with 14 rule rows and scored -.0008 against
  `ts_judge_h6`.
- Size: the whole-test effect is bounded near +-.0012. About 30% of rows are public, so roughly 4 of the 12 rows would
  move the public score. Ranking uses the best public upload, so an upload is not a free probe: a public score above
  .5505 on those few rows would make this file the one ranked on the private part, whatever the other rows do.

## Not checked
- The two-argument form of `alt_c2.py` (another tag's base file) was not run as written; the default form and the import
  path on a second base copy were.
- Which of the 12 rows fall in the public part is unknown.

## Reading
No detected difference for these 12 rows: no usable evidence for or against. The file is a faithful, reproducible model swap, ready if the reviewer wants it,
but the evidence above does not support replacing dd_final with it.
