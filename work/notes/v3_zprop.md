# v3_zprop: zone-type propagation on design-ambiguous pairs (13 Nov)

Scratch: `work/scratch_v3_zprop/` (run from cwd=work with `PYTHONPATH=.;scratch_art_zonemax;scratch_v3_zprop`; reuses the art_zonemax cache).
- `a1_bait`: (a) bait pairs vs zone rows of f / v
- `a2_pairs` (+ `.csv`): sweep over group x link x window
- `a3_mv`: (b) receiver detail
- `a4_rule`: rules + label-only purity
- `a5_diag`: per-rule table and legit-decoded control
- `a6_edge`: source-device edge distance
- `a7_fanin`: fan-in host rate per zone
- `post.py`: the candidate

## Rule
An off-zone row whose judge decode is inside a confusable group takes the zone type of linked zone rows. Conditions: the zone type is in the same group, it is unique, and it differs from the decode. Windows were chosen on Aug and scored on Sep.
- **R1 fan-in:** decode in {mule, vel, lay}, P2P receiver. A zone row into the same receiver within ±48 h.
- **R2 takeover:** decode in {sim, ATO}. A zone row with the same sender within ±1 h.
- **R3 chain hop:** decode in {mule, vel, lay}. A zone row whose receiver is this row's sender (R3a), or whose sender is this row's receiver (R3b), within ±1 h.
- **Guard:** a zone device inside the test mule|vel band [473749, 473783) is never a source. That is where the DP and the victim cores disagree.

Label-only purity, y == zone type, measured on off-zone rows labelled in the group with no decode filter (Jul / Aug / Sep):

| rule | rows (Jul / Aug / Sep) | purity (Jul / Aug / Sep) |
|---|---|---|
| R1 | 439 / 420 / 378 | .99 / .94 / .95 |
| R2 | 41 / 31 / 30 | 1.0 |
| R3a | 9 / 18 / 15 | 1.0 |
| R3b | 13 / 15 / 12 | 1.0 |

The model already decodes most of these rows correctly, so few rows change. Legit-decoded linked rows are about 92-100% legit (a5), so they are excluded. Fan-in majority label = host zone type in 103/103 mule, 8/8 velocity and 24/24 layering cases.

## Gate (judge = ts_judge_h6: Sep noisy .5096 / cleaned .5224 / OOS .5082)
| component | Sep noisy / cleaned / OOS | Aug cleaned delta | rows changed Aug (ok) / Sep (ok) | test rows |
|---|---|---|---|---|
| R1 | .5098 / .5226 / .5084 | +.0025 | 65 (64) / 5 (4) | 15 without the guard, 0 with it |
| R2 | .5100 / .5228 / .5086 | +.0006 | 8 (6) / 5 (4) | 4 sim->ATO (2 senders) |
| R3 | .5097 / .5224 / .5083 | +.0003 | 8 (8) / 1 (1) | 4 lay->mule (4 distinct sender-receiver pairs over 22 days) |
| **R2+R3 = v3_zprop** | **.5101 / .5228 / .5087** | +.0009 | 16 (14) / 6 (5) | **8** |
| all | .5102 / .5230 / .5088 | +.0034 | 81 (78) / 11 (9) | 23 without the guard |

"ok" = cleaned label equals the new type. The non-ok changed rows are cleaned-legit, i.e. noise.

## Per question
- **(a) phishing vs fake_cashback bait pairs:** null for test.
  - Within ±1 d, the only zone rows linked to f or v are the event's own rows, which the judge already overrides. Beyond the event, f's zone rows are mostly velocity-pool sends, which carry no ph/fc information.
  - Test ambiguous (rule-3) events with a ph/fc zone link outside themselves: 0.
  - The ph/fc/loan propagation via SR/RS ±6 h is validated (Aug 8 rows net +7, Sep 3 net +3, Sep +.0003) but changes 0 test rows.
- **(b) mule vs velocity fan-ins:** the mechanism is real (R1). Velocity-labelled fan-ins with one fresh velocity-zone sender row: Aug 57/57 rows fixed.
  - On test, both source devices (473749, 473782) sit in the disputed band, so R1 is a bet on the edge.
  - Fan-in host rate per zone (a7, train label-fit zones): mule .29 (103/350 devices), velocity .045 (8/176), layering .11, others ~0.
  - Test: mule zone 17/79, band 2/11, rest of velocity 0/17. The band looks mule-like, with a likelihood ratio of about 3:1 for all-mule vs all-velocity on hosting alone.
  - **The edge stays unresolved.** On hosting alone the odds are about 3:1 for mule.
  - Train check (a8): the train stretch that the DP calls mule but the labels call velocity, [423658, 423707), has 3/21 hosts (.14). The interior of the velocity zone has .032. Velocity devices next to the mule edge therefore host fan-ins at an intermediate rate, and the test band's .18 fits that as well.
  - So hosting does not localise this edge. Do not move the 11 band rows on it.
  - It is also not independent of the decodes: the host at 473782 is decoded mule, and velocity fan-ins decode mule 15/16.
  - Guarded R1 gives 0 test rows (measured, a8). Unguarded R1 gives 15 (11 mule->vel, 4 lay->vel), all conditional on the band being velocity.
  - If the final edges put the band in the mule zone, R1 gives 4 lay->mule instead. Rebuild both `q` and `z` on the final edges before re-running `zprop_hits`.
- **(c) own-phone sim/ATO linked to zone takeover events (R2):** same-sender link within 1 h only. Receiver links are negative (Aug -2, Sep -2 at 48 h).
  - Test: 4 rows sim->ATO, all sourced from the ATO zone: devices 474007 (2 rows, 20 ids past the sim|ATO edge 473987) and 474132 (2 rows, 145 ids in).
  - Both estimators agree on that edge.
- **(d) mule vs layering chain hop (R3):** 4 test rows lay->mule. The sources lie below 473749, so they hold whichever way the band resolves.

## Candidate
`subs/v3_zprop.csv` = `subs/ts_judge_h6_qrm.csv` + 8 rows:
- layering_chain -> mule_account: 4 (R3a 2, R3b 2)
- sim_swap -> account_takeover: 4 (R2)

Expected test gain: Sep +.0004/+.0005 (6 Sep rows). Oct classes are about half Sep size, so the per-row value is roughly 2x. Expected test gain is about +.0005 to +.001 macro, with about 2-3 of the rows public. Too small to see on its own on the LB, so stack it in the final combined candidate rather than spend an upload.

Combiner: `post.zprop_hits(Z, rules, band)` takes the zones (column z) from any edges. If the final edges move the mule|vel band, re-run it there and drop the band guard where the estimators then agree.
