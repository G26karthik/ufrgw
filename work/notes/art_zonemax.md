# art_zonemax: how much more can the device-zone exploit give? (13 Nov)

Scratch: `work/scratch_art_zonemax/`
- `prep.py`: caches h6 post probs, decodes, zones and labels
- `zm.py`: shared helpers. It reproduces ts_judge_h6 exactly: Sep .5096 / .5224 / OOS .5082, and the test labels match the file (asserted).
- `a1`-`a12`: analyses, logs next to them
- `edp.py`: DP variants
- `post.py`: candidate

Run from cwd=work with `PYTHONPATH=.;scratch_art_zonemax`.

**Verdict: the zone exploit is saturated.** ts_judge_h6 is within .0006 (Sep) of the oracle-edge ceiling. No angle gives a material test gain.

## Results per angle
| # | angle | result |
|---|---|---|
| 1 | Propagate the zone type to off-zone sibling rows (same sender, same receiver, zone receiver = row sender, zone sender = row receiver; windows 10 min to 48 h) | **Null** (a2). h6/judge already decode these siblings as well as the zone would. The best net gain is <=6 rows in Aug, and it does not replicate in Sep. Wide windows and the receiver link are heavily negative (popular receivers). |
| 2 | Devices just outside the zones; legit fresh-run boundary | **Null** (a3, a9). The train legit-fresh range has fraud rate .0007, and h6 predicts 0 fraud there (train and test). The test_block "final gap-free run" rule gives exactly 428455 on train, so the boundary is exact. Base-range "other owner" devices (dn != sender): fraud rate equals own-phone (3.4%), and class mix is flat. Account and merchant ids carry no class blocks (a8). Zone device ids are not ordered by time (monotone steps .51). No device is reused across train/test zones. |
| 3 | Velocity pool reuse | Already maxed. 269 test pool rows: h6 already decodes 100% as fraud and the judge sets velocity. Train pool purity .87, the rest is the label-noise rate. |
| 4 | Edge estimation, small zones (fc, collusion) | **Ceiling is tiny.** Label-fit (oracle) edges on Sep: .5102 / .5229 / .5088, i.e. +.0006 over the judge. Test-like simulation (judge DP on 50% device subsamples of ONE month of evidence; Oct is about half a train month): mean Sep .5094-.5101, sd .0005 (a10). Edge noise costs <=.001 on test. An emission-matrix DP (a7) is worse: it moves sim/ATO by about 16 ids. |
| 4b | Mule-vs-velocity bias in the DP | **Real but test-neutral.** h6 never decodes velocity-zone rows as velocity: 85% legit, and the rest mostly mule (a6). So mule preds drag the mule/vel edge into the velocity zone: train pooled edge +49 ids, Aug-only +21. Counting a mule pred as evidence for mule+velocity cuts the test-like edge error 76->18 (Aug) and 63->4 (Sep) ids (a11). Test edges move 473749->473747 and 473835->473834: **1 test row**. |
| 5 | Camouflage rows (p_legit>.9) on zone devices | Already captured by the judge. Sep: 362 of 1011 zone rows have p_legit>.9, and 84% of them carry the zone label. Test: 211 of 564 zone rows; the judge already flips them. |
| 6 | Zone position as model features | Not re-run. The train block is one contiguous block for Jul-Sep, so "per-month edges" are the same edges. ts_comb (judge notes) already trained zone features with label-free DP edges: model+rule .5089 vs rule-on-h6 .5096, a tie. Post-rule stays. |
| 7 | Re-tune the class bias after the zone override (zone rows one-hot) | Null: OOS .5082 (a12). |

## Remaining Sep errors after the judge (a9)
- Recall on zone rows is .90-1.00 for every class except velocity (.78, from the train pooled-edge bias; the mv DP fixes it) and loan (.68, 19 rows).
- Everything else is own-phone rows: recall .01-.40, with no zone information.

## Candidate
`subs/art_zonemax.csv` (scratch_art_zonemax/post.py) is ts_judge_h6 with the mule+velocity DP grouping.

| | Sep noisy | Sep cleaned | OOS Aug->Sep |
|---|---|---|---|
| ts_judge_h6 | .5096 | .5224 | .5082 |
| art_zonemax | .5100 | .5227 | .5086 |

- Rows changed vs subs/ts_judge_h6.csv: **1** (mule_account -> velocity_bot).
- Expected test impact is about 0: the gain is almost all in how the train edges are estimated.
- Not worth an upload slot. Use it as the pipeline default if the zone rule is rebuilt on new preds.

## Note
Re-running `ts_judge_post.py h6` rewrites `subs/ts_judge_h6.csv`. It was re-run once here (14:50). The output is deterministic: same edges, same scores, same 299 changed rows.
