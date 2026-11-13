# sc_refund: refund_abuse class sweep on v3_final (13 Nov)

Scratch: `work/scratch_sc_refund/s1..s17.py` (+ `.log`); cwd=work, `PYTHONPATH=.`. Rule: `work/sc_rules_refund.py`. Candidate: `subs/sc_refund.csv` = v3_final + 3 rows.

## Verdict
The refund class is saturated on structure. The only lever is 3 rows (collusion -> refund) in one small round-amount segment.
Expected gain is about +.0002 to +.0004 test macro. Refund cannot supply the +.004 target.

## (a) Test refund decodes (v3_final: 273 rows)
| source | struct | rows |
|---|---|---|
| model | REFUND episode n>=2 (sender, 4 d gap) | 181 |
| model | round-amount merchant segment (pairs def.) QR 40 / PAY 29 | 69 |
| zone override | REFUND ep 8, other 10 (model said legit), rseg 4 | 22 |
| model | REFUND singleton | 1 |
| rule (v3 QRN/zprop) | - | 0 |
Sep judge precision (cleaned): REFUND ep .84-.87, rseg .83-.87, zone .81-1.0.
Sep refund: true 1074, pred 487, TP 415. 622 of the 659 FN are non-core camouflage (`p_legit > .9`, no structure).

## (b) Oct structure checks (all null except the segment below)
- **REFUND episodes.** Every test episode with n>=3 is decoded refund (185 rows). n=2: 6 rows; the model's OOF precision there is 7/12, no gain. Singletons: train refund rate .2-.5%.
  No episode spans train and test, so features are not inflated by Sep-started train-Oct rows.
- **Spread-out claimers.** Senders with >=2 REFUND rows but no episode: test has none of the refund-like cell (tot 3, one receiver: .92 train).
- **Shared-device and receiver-linked REFUND singletons.** All legit in train.
- **Episode senders' other rows.** Label mix in train is legit/velocity, 0 refund. The abuser does not camouflage.
- **Nov spillover.** All 36 Nov REFUND rows are in Oct-started episodes (n 4-8), every row decoded refund. Other Nov rows match the train-Oct spillover mix.
- **Round segments by size** (test decode is pure in every segment but one):
  - n 1-3: all qr. Train refund share in n=3 is 8%; no cell reaches .27.
  - n 4-8: refund.
  - n>=20: collusion.
  - Segment counts per size fit Oct volume (refund 4-8: 11 vs ~12 expected; the n=3 excess is qr).
- **Rows near segments.** Rows at the same merchant within 6 h / 24 h but outside the segment are legit in train.
- **Refund zone** `[474462, 474496)`. The qr|refund edge is pinned by REFUND rows.
  The refund|fc gap (474496-474504, 7 rows now fc) stays: zone-row volume is ~3.1% of class rows. That gives expected refund ~21 / fc ~15 zone rows.
  The current split is 22 / 12; moving the gap to refund gives 29 / 5.

## (c) Rule SEGR (`sc_rules_refund.py`)
Rule: round-amount merchant segment (pairs definition, built per split) with 4<=n<=8, row decoded merchant_collusion -> refund_abuse.
- **Train labels.** Segments with 4<=n<=8 contain 0 collusion rows in every month (Jul+Aug: 45 refund-majority segments, Sep: 20). n=2/3 segments hold 2+2 collusion rows.
  Row-level refund precision in 4-8 segments: Jul .83, Aug .81 (Jul+Aug .82), Sep .76 (the rest is mostly legit noise).
- **Test hit.** 3 rows in the one mixed segment: n=8, 2 senders, one sender with 7 rows, 18 h. The model splits it 5 refund / 3 collusion.
  Raw h6 probs for the 3 rows are p_ref .51-.54 vs p_coll .31-.34. The collusion bias tips them. Segment mean is .60 / .24.
- **Concentration check** (advisor flag: pmax 7 is outside refund's train range, max 5). Train marginals (segments):

  | class | segments | n 7-8 | pmax/n >= .75 | dur <= 20 h | 2 senders |
  |---|---|---|---|---|---|
  | collusion | 36 | 0 | 0 | 1 | 6 |
  | refund | 65 | 19 | 11 | 52 | 28 |

  Jul+Aug and Sep agree. Truncating collusion segments to their first 18 h never gives pmax>=6 below 13 rows.
  High absolute pmax in collusion comes from size. Concentration is higher in refund.
  For 2 senders and 8 rows, the 7+1 split has about 6% chance under an even split.
  Under that even-split model the 6+1 split is about 11%, so both Oct segments together are under 1%: an unexplained Oct oddity.
  Train 2-sender collusion segments have pmax share .62-.70, and size is 65:0 for refund. So P(event = refund) is about .8-.9.
- **The other concentrated test segment** (n=7, 6 rows from one sender) is already all refund. Moving the whole cell (12 refund rows) to collusion: s16 gives -.0012 at P(refund)=.8, -.0018 at .9.
- **OOF.** The rule fires on 0 Aug/Sep rows, so there is no validation score.
  The looser "segment-majority" variant has no support: Sep 4 rows, 1 correct, -.00014. It is not used.

## Expected gain (s16: Sep-calibrated per-class P/TP/T scaled to test)
Break-even is P(event refund) of about .62. At row purity .8: P .8 gives +.00029, .85 gives +.00037, .9 gives +.00045.
Best estimate: +.0002 to +.0004 test macro.
Conflict note: SEGR reverts any collusion decode in a 4-8 round segment. If sc_collusion moves that cell the other way, the two conflict and the orchestrator must pick. About 1 of the 3 rows lands in public, so the LB will not measure it. Bundle it; do not spend an upload on it alone.

## Disclosure
s12 and one inline check printed row-level probabilities, device ids and zone status for this single test segment, after the segment-level evidence had already selected it.
The rule was defined on the size cell before those prints, and the prints did not change it.
