# ring module: loan_app_extortion + merchant_collusion

Module: `work/feats/ring.py` (35 features, `ring__*`; build takes about 25 s; no label use). Scratch scripts are in `work/scratch_ring/`.

## Eval (quick_eval.py ring, fit Jul-Aug, score Sep)
| | macro argmax | macro tuned | loan argmax | collusion argmax |
|---|---|---|---|---|
| baseline (row features) | 0.3577 | 0.3773 | 0.345 | 0.053 |
| ring v1 (sender ±4d, receiver 12h-gap segments) | 0.4238 | 0.4358 | 0.386 | 0.309 |
| ring v3 (final: + ±10d/±2d sender, ±36h receiver, repeat-sender, excess, volume-normalised) | 0.4356 | 0.4432 | 0.397 | 0.363 |
| v3 + seg share of {1000,2000,2500,5000,10000} (dropped) | 0.4149 | 0.4281 | 0.395 | 0.267 |

The module also lifts other classes: mule .222->.357, layering .353->.467, refund .327->.485, qr .428->.475, ATO .290->.330.
Small classes are noisy: adding one near-duplicate feature moved collusion by 0.1 and velocity by 0.08. Treat a difference under ±0.03 per class as noise.

## Generator signatures
### loan_app_extortion (core events = about 29% of the train labels)
- One victim (sender) makes 2-6 payments over 1-7 days. Each payment goes to a DIFFERENT, NEW payee (new_payee=1 in 98%).
- 89% of payees are `merchant_category == lending` merchants (607 such merchants, shared with legit traffic); the rest are P2P accounts.
- Amounts are whole rupees (99%), usually thousands (median 7.5k, up to 15k); 92% are sent 20:00-24:00.
- balance_before == 1.1*amount exactly in 56% of rows. This drain artefact also appears in collusion, phishing and fake_cashback.
- Receiver-side repetition is absent: a lending merchant does not get a burst. The signal is on the SENDER.
- About 1416 of 1991 loan labels are isolated rows with no structure (evening 20%, int 28%, lending 1%). They are label noise and cannot be predicted.
  Core-pattern recall in Sep is about 161/166, so Sep F1 for loan is capped near 0.45 by the noise.
- Confusers: ATO and sim_swap drains also give "new payee + whole amount + evening" (133 ATO, 65 sim_swap rows). Those drains go to P2P from a new device or new SIM, so they almost never involve lending.
  About 116 legit-labelled rows match the core pattern (68 of them go to lending merchants): these look like unreported loan fraud.
- Oct check (loan-like senders): evening share .64 vs .58 in Sep; lending share of merchant rows .83 vs .92; P2P share .44 vs .50; b11 .39 vs .32. Stable.
  The predicted loan rate is 4.0/10k rows in Oct vs 3.9/10k in Sep.

### merchant_collusion (core events = about 45% of the train labels)
- A small merchant (fewer than about 150 rows over the whole data) gets a burst of 8-35 payments over 0.5-3 days from 2-5 accounts. Each account pays 2-12 times.
- Variant A (about 85%): amounts are exactly 1000/2000/2500/5000/10000. Variant B: random whole-rupee amounts of 1000-9999.
- balance_before == 1.1*amount in about 33% of rows. QR and PAY are each about 50%; new_payee is mostly 0.
- The accounts come from different home cities and banks. There are no P2P edges among ring accounts and no device sharing; the "merchant pays back" hypothesis is not supported.
- A merchant can host 2 bursts weeks apart with different accounts.
- The other about 1400 collusion labels are isolated P2P or big-merchant rows: noise.
- Confusers: velocity_bot (same small-merchant repeat pattern, but 1 phone, minutes apart, gaming/ecommerce) and refund_abuse (repeat sender at a merchant).

## Label noise finding: please read
Small-merchant bursts of round amounts (the exact {1000,2000,2500,5000,10000} set, non-REFUND, qr_vpa_mismatch != 1) appear in these train labels:
merchant_collusion 529 rows, **qr_swap 413**, **refund_abuse 319**, legit 181, and 7 or fewer in every other class.
- Row-level signatures are identical across the three labels: 100% set amounts, b11 ≈ .28-.33, QR share ≈ .5, evening ≈ .2, mismatch = 0.
- Real qr_swap has mismatch=1 (1160 rows). Real refund_abuse is txn_type REFUND (950 rows).
- The qr- and refund-labelled bursts show no REFUND activity and no mismatch rows nearby.
- They differ only in size: qr bursts have about 2.4 senders and 4.6 h, refund about 3 senders and 10 h, collusion about 4 senders, 35 h and about 6 rows per pair.
- The same merchant often hosts bursts with different labels (qr+refund 16 merchants, coll+qr 9, coll+refund 4).

There are two hypotheses:
- H2: these are collusion events filed under the wrong type, and test labels (clean) would call them collusion.
- H1: the generator deliberately builds smaller qr_swap and refund lookalike bursts.

Experiment (`scratch_ring/relabel_exp.py`): relabel those rows (labels legit/qr/refund) to merchant_collusion in train.
| train labels | Sep eval labels | macro | collusion | qr | refund |
|---|---|---|---|---|---|
| original | original | 0.4353 | .362 | .475 | .485 |
| original | relabelled | 0.4222 | .335 | .412 | .417 |
| relabelled | original | 0.4199 | .296 | .414 | .425 |
| relabelled | relabelled | 0.4491 | .597 | .434 | .453 |

Recommendation: settle H1 vs H2 with ONE public-LB A/B (same model, train labels relabelled vs not).
In Oct the trained model sends small round bursts (8 rows, 3-4 senders, 8-20 h) mostly to refund_abuse.

## Drift and Oct notes
- **Oct volume is half of a train month** (8.4k rows/day vs 17k; 148k senders vs 215k). Fraud events keep their full size: collusion-like bursts average 21 rows in Oct vs 20-23 in train, and loan-like victims make 3.2 payments in both.
  Any per-entity count that includes legit traffic (sender activity, receiver window counts, in-month rates) drops by about 2x in Oct.
  This module dropped raw s4_n/s2_n/r36_n. It uses r36_excess (window count / receiver same-month daily rate) and seg_excess (segment rows / whole-data rate), and both are stable Sep vs Oct (0.98 vs 0.98, 1.14 vs 0.99).
- 46 rows are dated Nov 1-9. They are spill-over of Oct fraud events (b11, lending and np_int_eve are all about 0.13, versus 0.002 overall).
- Predicted class rate per 10k rows, Sep vs Oct: velocity_bot 22.0 -> 10.5 (halved; worth a look by the velocity owner), collusion 4.9 -> 3.2, the rest stable.

## Features (most useful)
- Sender side, loan: `s4_np_int_eve`, `s4_np_int`, `s4_np_int_nrcv`, `s4_lend`, `s10_np_int_eve`, `s10_lend`, `s4_b11`, and row-level `np_int_eve`, `lend`, `b11`.
  np_int means new_payee & whole amount >= 1000; eve means hour >= 20.
- Receiver side, collusion: 12 h-gap segments per receiver with `seg_n`, `seg_nsnd`, `seg_nrep_snd`, `seg_top1`, `seg_n_per_snd`, `seg_int`, `seg_r500`, `seg_b11`, `seg_refund`, `seg_dur_h`, `seg_excess`, `pair_seg_n`, `pair_seg_dur_h`.
  Also ±36 h rolling `r36_excess`, `r36_int_share`, `r500_36`, plus `s2_nrcv` (colluders pay only their merchant).

## Next ideas
1. LB A/B on the collusion relabel (above). This is potentially the largest single gain for collusion, qr and refund.
2. More train-label cleaning: drop isolated loan/collusion-labelled rows that do not match any structure (about 70% of loan labels and 55% of collusion labels) or downweight them. This is a precision win on clean test labels; it cannot be measured on noisy Sep.
3. Merge segments with a 24 h gap; MR013817's burst was split 5+20 by the 12 h gap.
