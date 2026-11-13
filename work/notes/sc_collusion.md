# sc_collusion: merchant_collusion on top of v3_final (13 Nov)

Scratch: `work/scratch_sc_collusion/` (a0 frame: all rows + judge OOF decode on Aug/Sep, v3_final decode + zones on test, h6 probs; a1-a7 analyses with `.log`).
Rule: `work/sc_rules_collusion.py` (`apply(sub_df, tx)`). Candidate: `subs/sc_collusion.csv` = v3_final + 3 rows collusion -> refund.

## (a) Test rows decoded collusion in v3_final: 85
- 71 rows: the 3 Oct round-set bursts (MR014892 28, MR010487 23, MR002433 20; 3-5 senders, 23-47 h). 71/71 decoded collusion, 0 split.
- 12 rows: collusion zone [474529, 474552) override. 11 are camouflage-looking (P2P / small amounts, p_legit .97); 1 is a burst row.
- 3 rows: one 8-row round-set segment, 2 senders (7+1 rows), 18 h, fuel merchant of degree 8. Decode 5 refund + 3 collusion (p_ref .51-.54, p_coll .31-.34).
- 71 + 12 + 3 = 86 > 85: one zone row (474529) is also a MR010487 burst row, so the zone adds 11 new rows.
- h6 argmax gives 71 collusion rows. The 3 split rows have h6 argmax refund (p_ref .51-.54 > p_coll .31-.34); the judge's collusion bias flips them. Own devices, no zone row in that segment.

## (b) Oct structure
- Segment counts (round set, 24 h gap, whole data): n>=9 Oct 4 vs 7/14/14 per train month; n 6-8 Oct 11 vs 10/15/9. The size shift does not point to smaller collusion bursts:
  - The 3 Oct bursts are 20-28 rows, which is the normal train size.
  - Zone devices: train bursts carry coll-zone rows at about 3% (15/504), which predicts about 2.3 among the 76 Oct n 4-9 rows. Observed 0 coll-zone and 4 refund-zone.
  - Durations are 8-20 h, the refund shape.
- Burst neighbourhood (a4): rows at the burst merchant within 48 h outside the segment, burst senders' other rows within 48 h, and burst sender to merchant outside the window. Train labels are legit at base rate; test has 3 / 1 / 1 such rows. Null.
- Camouflage link (a5): non-burst rows of burst senders have a collusion rate of 46 per 1e4 vs 12.5 base. That is 3.7x, but only .5%. Burst merchants' other rows have 0 collusion labels. Null.
- Variant B (whole 1000-9999): Oct n>=6 segments exist only at big merchants (degree 6k-37k), all decoded legit, train label legit. Null.
- Merchant-QR with null mm: every Oct row is already decoded fraud (0 legit). The burst rows split qr 33 / refund 25 / coll 26, following the segment size. Null.
- Cross-split: MR014643 (Sep-started burst) lives in train, plus 2 off-segment test rows. Nothing to fix.
- Partial decodes in Sep (legit-decoded rows inside collusion bursts, 15) do not occur in Oct (0).

## (c) Rule RS: decoded collusion inside a round-set merchant segment of 4-8 rows -> refund_abuse
- Label-only purity, chosen on Jul+Aug and checked on Sep. Collusion labels in the cell: Jul 0/121, Aug 0/142, Sep 0/108. Refund labels .83 / .81 / .76.
- Sender-heavy rows (rps>=4, the model's collusion cue) are also 0 collusion: 0/30, 0/29, 0/17.
- Collusion appears only from n=10 up (n 10-11: 25/32).
- The judge OOF decodes 0 Aug/Sep rows into the cell, so there is no Sep score delta to measure. Evidence is label-only.
- Test: 3 rows change.
- Expected macro gain. Oct coll T is about 423 (Sep 828 x .51), P 85, TP about 73.
  - Collusion F1 .2862 -> .2879 (3 FP removed): +.00015 macro.
  - Refund: T about 549, P 273, F1 .5645. With 3 rows at precision about .8, F1 goes to .5682: +.0003 macro.
  - Total about +.0005 if the segment is refund.
  - If it is collusion: collusion F1 drops .2862 -> .2772 and refund gains 3 FP: about -.0009. Break-even P(refund) is about .64.
- Support check (a8, all train segments; split chosen on Jul+Aug, holds on Sep):
  - The hit shape is n=8, ns=2, rows per sender 7+1 (top share .88), 18.1 h.
  - Collusion segments: n>=10 in 36/36, duration >=19.7 h in 36/36, top share <=.70 in 36/36. The hit is outside all three.
  - Refund segments: n 4-8, duration up to 40 h. Top share reaches .83 (n=6, 5+1 rows, Jul and Sep).
  - Max rows per sender is at most 5 in refund segments, so the hit's 7 is outside refund support too. That is mechanically tied to n<=8.
  - Rough posterior from the n and duration likelihoods and the segment prior (63 refund vs 36 collusion): P(refund) about .9. EV about +.0004.
- About 1 of the 3 rows is expected to be public, so the LB cannot see this. Stack it, do not spend an upload on it.

## Verdict
Collusion is structurally saturated in Oct: every burst row is decoded collusion and every zone row is overridden. RS is the only defensible change: EV about +.0004, +.0005 if right, -.0009 if wrong. It reverts the bias flip to h6 argmax in a cell with 0 train collusion labels. The remaining collusion recall (about 83% of true rows) is camouflage. No collusion lever reaches the +.004 target.
