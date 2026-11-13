# typesplit judge (13 Nov): verdicts on the 4 hunters + combined candidate

Scripts: `work/scratch_ts_judge/j1.py` (re-measure), `j2.py` (zone purity), `mk_matrix.py` (model matrix),
`work/ts_judge_post.py` (combined post-rule, writes `subs/ts_judge_<tag>.csv`). Logs in `work/scratch_ts_judge/`.

## Verdicts
| hunter | claim (Sep event acc) | judge re-measure | verdict |
|---|---|---|---|
| event (device-id zones, Jul+Aug label edges) | .836 | .856 events; Sep noisy .4855 -> .5099, cleaned .4979 -> .5227 (528 rows) | CONFIRMED |
| victim (same leak, label-free cores) | .848 | .848; full blk .5051 / .5180 (483 rows) | CONFIRMED, weaker than event's contiguous zones |
| ring | null | not re-run (no rule) | null accepted |
| bait (ts_bait rule 1/2) | +.0002 macro | included in combined; +-.000 | real but tiny |

Same leak found independently by 2 hunters: fresh device ids (DV0 >= 420000) are handed out per fraud class in a fixed
class order. A row on a zone device is that zone's type (Sep label purity .74-.93 per zone, legit share ~.12 = label noise).

## Is the test-edge estimate safe? (key risk)
Transductive simulation: run the label-free DP (`ts_event_post.dp_edges`) on Sep devices + h6 Sep OOF preds only (what we do
on test). Recovered edges are within 1-2 ids of the label-fit ones; Sep score is identical (.5101 noisy / .5228 cleaned).
Test sim|ATO edge 473987 is found by both independent methods (DP length prior; victim's gap-density changepoint).
Mule/velocity edge: victim cores put 473749-473782 in mule; event DP says velocity. On Sep the DP recovers the true edge
(423657 vs 423658), so the DP is used.

## sim/ATO
Event accuracy .856 out-of-time (DEV archetype ~.99, SIM archetype own-phone events ~.70, no rule). 90% target is NOT
reachable: own-phone events carry no label-free signal (all hunters, ~150 aggregates). sim F1 cleaned .409 -> .499,
ATO .396 -> .442.

## Combined candidate (h6 + zone override + bait rule; `ts_judge_post.py h6`)
| | Sep noisy | Sep cleaned | OOS Aug->Sep |
|---|---|---|---|
| h6_ppq | .4855 | .4979 | .4841 |
| ts_judge_h6 | .5096 | .5224 | .5082 |
Test rows changed vs subs/h6_ppq.csv: 299 (73 ATO->sim, ~200 legit->zone type, 2 bait ph->fc).

Oct consistency: every test zone's h6 non-legit preds agree with the zone (sim/ATO pooled 1.00, layering 1.00,
velocity .99, qr .89, mule .83), except tiny fc (12 rows, .33) and collusion (12 rows, .50) zones.

## Model test (zone as features): `feats_cache/matrix_ts_comb.parquet` = m5dm2_pairsfd + ts_comb__zone/gen_dev
(label-free DP edges from h6 preds) + ts_victim__* + ts_bait__rule; train.py --relabel C --rounds 150 --seeds 3 --tag ts_comb
| | pooled biased | Sep (pooled bias) | OOS Aug->Sep / mean |
|---|---|---|---|
| h3 (same recipe, 3 seeds) | .4789 | .4850 | .4831 / .4776 |
| ts_comb | .4957 | .5001 | .4984 / .4931 |
| ts_comb_ppq (postproc --qr-null) | .4958 | .5004 (cleaned .5126) | - |
| ts_comb + zone/bait post-rule (`subs/ts_judge_ts_comb.csv`) | - | .5089 (cleaned .5216) | .5076 |
| h6 + zone/bait post-rule (`subs/ts_judge_h6.csv`) | - | .5096 (cleaned .5224) | .5082 |
The model learns the non-sim/ATO zones (+.017 pooled), but relabel C erases the sim/ATO zone signal, so the post-rule is
still needed. Model + rule ties rule-on-h6 (-.0007, 3 vs 6 seeds). Recommended: `subs/ts_judge_h6.csv` (299 rows vs
h6_ppq). ts_judge_ts_comb.csv changes 384. Test small-zone edges move by up to ~13 ids between the two models' preds
(phishing/qr/refund/fc), so edges there are uncertain at about +-10 ids.

## Risk
Exploits a generator artefact (sequential device ids). Valid columns of test.csv, but organizers may call it a leak.
