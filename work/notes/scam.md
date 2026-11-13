# scam module: phishing_collect, fake_cashback, qr_swap

Module: `work/feats/scam.py` (cache `feats_cache/scam.parquet`, 26 features, builds in about 25 s).
Scratch analysis is in `work/scratch_scam/`. `err.py` breaks errors down by segment, `generic_auc.py` tests learnability, `drift.py` and `oct.py` check Oct.

## Eval (quick_eval.py scam --threads 2; fit Jul-Aug, score Sep)
| version | macro argmax | macro tuned | phishing | qr_swap | fake_cashback |
|---|---|---|---|---|---|
| baseline (row only) | 0.3577 | 0.3773 | .400 | .428 | .112 |
| v1 | 0.4427 | 0.4496 | .437 | .462 | .347 |
| v2 (+ round-amount payee) | 0.4432 | 0.4495 | .434 | .464 | .349 |
| v3 FINAL (drift-safer: fractions, all-time counts) | 0.4401 | 0.4486 | .435 | .466 | .345 |

The v1-v3 differences are within run-to-run noise. sim_swap moves between .25 and .31 even though no feature targets it.
Side gains from v1 onwards: merchant_collusion .053 -> .36-.38 (pair_n_15d), refund_abuse .33 -> .46 (snd_refund_15d), mule .22 -> .36 and layering .35 -> .48 (rcv_in_15d, rcv_newpayee_in_3d). Another module may already cover these.

## Generator signatures (train, Jul-Sep)
**Most rows labelled with these classes are "generic".** They look like random legit rows: about 62% of phishing, 72% of fake_cashback and 71% of qr_swap.
- Their txn_type and receiver mix matches legit (QR->MERCHANT, PAY->P2P, PAY->MERCHANT).
- They show no temporal burst, and fewer than 11% share an entity with a core row.
- One-vs-legit AUC on generic rows only (fit Jul-Aug, score Sep): phishing 0.52, fake_cashback 0.50, qr_swap 0.57 (round amounts only). For comparison: sim_swap 0.68, mule 0.66.
- **These rows cannot be learned.** On the core segments the model already gets 80-99% recall.
- Test labels are clean. If these rows are a noise artefact, test fraud is mostly core rows. Then precision on core rows matters more, and Sep-tuned biases (fit on noisy labels) may under-predict these classes.

**phishing_collect core**
- COLLECT to P2P with new_payee=1 (0.999 for phishing, 0.39 for legit look-alikes).
- Amount is R-1, R-10 or R-50 for R in {2000, 5000, 10000, 25000}. Top amounts: 1999, 4999, 9999, 1950, 1990, 4950, 24999...
- About 1470 such rows per 3 months are phishing; 931 are labelled legit (about 360 of those are new_payee=1, probably unreported). 156 are labelled fake_cashback and 70 loan_app (type noise).
- Receivers repeat: 9% of core rows have another one within ±15 days.
- **Bait variant:** the fraudster sends exactly 1 tiny PAY (1/2/5/10). The victim pays back a big (median about 11k), integer, non-round amount 8-178 min later, always under 3 h, and always to a new payee.
- **Unanswered bait:** about 530 tiny 1/2/5/10 PAYs labelled phishing get no payback. The model catches about 85% of them through amount plus snd_tiny_out_3d / pair_tiny_n.

**fake_cashback core**
- The fraudster sends 1 or 2 tiny PAYs (1/2/5/10, roughly uniform) to the victim.
- The victim then pays back a big amount (PAY 56%, COLLECT 44%, median about 10.7k, integer, never under-round) 13 min to 47 h later (median 2 h, 32% over 3 h).
- A reverse edge almost never happens in legit: only 0.03% of legit P2P rows have a reverse pair, against 47% of fake_cashback P2P rows.
- Separating it from bait-phishing: nbait=2 or dt>180 min means fake_cashback. With nbait=1 and dt<=180 the two cannot be told apart (about 90 fake_cashback vs 130 phishing per 3 months).

**qr_swap core**
- **QR to a P2P account with merchant_category still set:** about 730 per month, 82% qr_swap. qr_vpa_mismatch=1 on about 59% of them.
- **QR to a merchant with qr_vpa_mismatch null and a round amount** (1000/2000/2500/5000/10000), going to small merchants (median in-degree 14). These are shared with merchant_collusion and refund_abuse; collusion merchants have more repeated pairs and a higher round fraction.
- **QR to a merchant with mismatch=1:** qr_swap only when the merchant is tiny (in-degree 7 vs 1558 for legit).
- The hint "payee in shop city" does not hold: device_city == rcv_home_city in only 7% of cases.
- "Many customers to the same fake payee" is weak: 83% of fake payees get exactly 1 QR row.

## Drift (Oct vs Sep)
- **Oct volume:** Oct is a full month with half the rows (261k vs 510k). Events stay intact: the share of paired bait rows is the same. There are simply fewer events and fewer per-entity repeats.
- **Repeat counts:**
  - rcv_phcol / rcv_qr_p2p repeat rates drop: 0.09 -> 0.05 and 0.08 -> 0.03. I left them unscaled (conservative) and added all-time versions (`*_all`) that let Oct rows see Jul-Sep history.
  - Background-volume counts (rcv_in_15d, rcv_newpayee_in_3d, pair_n_15d, snd_refund_15d) are multiplied by the month-volume scale (about 2.0 for Oct).
- **Dropped:**
  - Raw rcv_round_15d: nonzero share was 6.3% in Aug vs 0.6% in Sep, driven by big merchants. Only the fraction is kept.
  - rcv_qr_mm1_15d: volume-driven, replaced by a fraction.
  - Nearest-event distance: did not transfer to Oct.
- **Signatures stable in Oct:**
  - Under-round collect amounts are the same: 4999, 1999, 9999, 1950, ...
  - Bait-pair dt median is 103 vs 117 min, and the share over 180 min is 0.23 in both months.
  - QR P2P mismatch rate is 0.63 vs 0.59.
  - The 1/2/5/10 bait share is 0.95 vs 0.93.
- **'cents':** the integer-amount share per txn_type is identical in Jul, Aug, Sep and Oct (PAY .42, COLLECT .72, QR .013). The artefact comes from the amount generator: fraud rows are integer about 50% of the time vs 26% for legit. It is stable in aggregate.

## Next ideas
1. Make bait pairs consistent at event level: both rows should get the same label, for example by averaging model probabilities over each pair in post-processing.
2. Post-process: the core rules are near-deterministic. Check whether forcing the label on QR->P2P rows and new-payee under-round COLLECTs beats argmax on test-like (clean) data.
3. Merchant-level graph for the round-amount small merchants: separate qr_swap, merchant_collusion and refund_abuse by sender repetition and refund history per merchant (shared with the collusion owner).
