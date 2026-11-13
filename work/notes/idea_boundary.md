# idea_boundary: event-start-month split (13 Nov) - NULL, no sub

Scripts: `work/scratch_idea_boundary/b1.py` .. `b5.py`. Judge decode cached in `scratch_idea_boundary/judge_h6.pkl`
(`ts_judge_post.get("h6")`; asserted equal to `subs/ts_judge_h6.csv`).

## Facts
- Train has 147 Oct-dated rows (to Oct 10 18:03): 129 fraud, 18 legit. They are spill-over of Sep-started events
  (refund 74, mule 24, collusion 11, loan 8). Test has 46 Nov rows. So the start-month split holds.
- Train legit rows stop at Sep 30. Legit background is continuous across the file boundary.

## How big is the effect (upper bound)
Link = same sender_id / rk (P2P receiver or sender|merchant) / device_id, |dt| <= 48 h, to a train row that is fraud
(cleaned) or Oct-dated.
- Test: 105 rows linked. 103 already decode legit in ts_judge_h6. 1 is on a test zone device (zone owns it).
  Only 2 rows are off-zone and decoded fraud: TXfcc7bf19 (layering) and TXb2571aaa (fake_cashback).
- Adding receiver_id (merchant) as a key: 4426 rows linked, still only 3 off-zone fraud decodes.
- So any correction (feature recompute or post-rule) can move at most 2-3 test rows. A 6-seed retrain on
  boundary-corrected features is bounded by the same rows and was skipped.

## Sep simulation (Aug file = Aug-started label events; their 68 Sep-dated rows dropped from scoring)
Rule: off-zone decoded-fraud row linked to an other-file fraud row -> legit.
| keys / event gap | fires | true legit | Sep noisy | cleaned | OOS |
|---|---|---|---|---|---|
| judge (baseline, Sep minus 68 rows) | - | - | .50721 | .51995 | .50581 |
| sender, rk, device / 48 h | 4 | 4 | .50733 | .52007 | .50593 |
| + receiver_id / 48 h | 27 | 5 | .50518 | .51791 | .50378 |
| sender, rk, device / events 24 h | 13 | 2 | -.0009 vs its baseline | -.0009 | -.0009 |
| sender, rk, device / events 6 h | 25 | 2 | -.0016 | -.0016 | -.0016 |

The 48 h row's 4/4 precision is a construction artefact: the same 48 h union-find that defines the Aug events
absorbs every linked fraud row into the Aug event, so only legit rows remain to fire on. With a finer event
definition, linked rows from separate events are mostly true fraud (23/25, 11/13). Keys are shared across events
(layering intermediaries, refund abusers who are also bait victims, collusion merchants). The rule hurts.

## Test rows by eye
- TXfcc7bf19: AC181111 -> AC400450 29151, then AC400450 -> AC012367 28990 (0.994x) 13 min later. That is a new
  layering hop through an intermediary that was in a Sep chain. Likely real layering.
- TXb2571aaa: Rs 5 bait to AC085371, payback 7488 from AC085371 85 min later. That is a textbook fake_cashback,
  on an account that is also a Sep refund abuser. Likely real fake_cashback.
Both flips would most likely be wrong.

## Verdict
Null. Do not use. The model does not over-call early-Oct test rows from Sep neighbours (103/105 linked rows are
already legit). No subs/idea_boundary.csv was written.
