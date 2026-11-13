# fresh_columns: rebuilding the generator one column at a time (13 Nov)

Scratch: `work/scratch_fresh_columns/` (c0 builds frame.parquet = load_all + zone + h6 OOF/test probs + ts_judge_h6 labels;
c1-c13 analyses). Groups: L = legit label; camo = fraud label and h6 p_legit >= .9; core = fraud label and p_legit < .9 (Aug+Sep OOF).

## Verdict
This lens finds no new lever. There are two code paths, and h6 already handles the one we can see:
- **Legit path.** Every row is redrawn independently. Camouflage rows come from this path and match legit to 3 decimals on
  every check, month by month.
- **Fraud path.** One shared "finalize" step stamps every fraud-path row with the same marks, whatever the class.
  h6 already catches ~99.6% of these rows in test and makes almost no false positives on legit-path rows.
All the headroom left is (a) camouflage, which cannot be recovered, and (b) type swaps at event level inside the fraud path,
which carry identical marks.

## 1. How the legit path makes each column (camo is identical to L, month-controlled; c1, c2, c4)
| check | L Aug / Sep | camo Aug / Sep | core |
|---|---|---|---|
| new_payee=1 though the pair was seen before | .0106 / .0153 | .0103 / .0147 | .029 |
| new_payee=0 on the pair's first row in the data | .717 / .690 | .719 / .694 | .345 |
| device_age < observed (sender, device) history - 1 d | .0666 / .1224 | .0658 / .1230 | .05-.10 |
| app_version lower than the sender's previous row | .457 / .456 | .455 / .460 | .63 |
| away from home city | .068 | .070 / .064 | .12 |
| sender_account_age_days - round((ts - opened_on)/day): 0 (else -1) | .431 | .427 / .434 | .47 |
- new_payee by pair rank (0, 1, 2, 3+): L .232 / .150 / .171 / .190; camo .231 / .145 / .171 / .193.
- device_age / account_age: the quantiles match on home and on non-home devices. device_age is capped at the account age.
  The cap explains why consecutive rows often repeat device_age (dev_eq .03-.13). It is not state.
- Consecutive rows (same sender, device or receiver, within 6 h), every class including core: balance chain
  (b = prev_b - prev_amt) .000-.007; sim_age continuity .000-.005; device_age continuity .00-.07 (the cap). balance, ip,
  sim_age and device_age are never copied. login is copied only for velocity (.09-.15, the scripted bot).
- There are no impossible combinations in any class: txn before merchant onboarding, before the receiver or sender account
  opened, device_age > account_age, merchant_category != merchants.csv category, P2P rows with a category. The one exception
  is qr_swap's known QR-to-P2P signature.
- Camo selection is not tied to same-class events in time. AUC of the distance to the nearest same-class core row, camo vs L:
  .49-.51 for all 11 classes (c3). Per-class chi2 for camo vs L, all p > .01 except two: refund txn_type (REFUND rows the
  model only half-catches) and loan weekday (p .0005, n=890, one test in 88).
- Float precision is identical in L / camo / test-legit: sim_age and device_age have 1 decimal, ip 3, balance 2 (c9).
- Row order of test.csv and sample_submission (the same order): AUC .503 against predicted fraud, .506 against zone rows.
  Neighbour-fraud rate .0149 vs base .0152 (c5).

## 2. The fraud path's stamp (c9, c10, c12)
- **qr_vpa_mismatch = 0 on non-QR rows**, drawn per row with p = .352, flat over class (.33-.38), txn_type, receiver_type
  and month. Not per event: only 17-21% of entity groups are all-same.
  - Legit path: about 1.3e-4.
  - Train labels on these rows are fraud .87-.88, which is exactly 1 minus the noise rate. So in clean labels they are
    certain fraud.
- QR rows with a null mismatch: same purity (.87-.90) in train.
- Integer amount: ~1.0 for 9 classes on the fraud path; qr .49, refund .28. Legit path .26.
- app_version 3.x: .61-.66 on the fraud path, .15 on the legit path. Dead in Oct: every Oct row is 5.x.
- ip_risk_score mean .41 on the fraud path.
- **Swapped event types share the stamp exactly** (c12). Inside the h6-confused pairs, the fp rate, integer share, v3 share
  and ip are the same for both labels:
  - mule/vel .354/.345
  - mule/lay .353/.347
  - ph/fc .347/.375
  - sim/ATO .39/.315 (row-level, ~1000 rows)
  The swap is a relabel of one template, not a second generator. Again: cannot be separated.

## 3. Label-free audit of test decoding with the stamp (c11, c13)
- Recall on the fraud path: 1134 test rows carry the non-QR fp stamp, and ts_judge_h6 decodes 5 of them as legit (99.6%).
  The stamp is an independent coin per row, so recall on the fraud path is about 99.6%.
- Precision against the legit path: estimated fraud-path share of decoded-fraud non-QR rows = fp rate / .352.
  - Sep h6: .91-1.20 for every class.
  - Test: .79-1.13. The lows are qr .79 and sim .83; the judge put zone-camo rows there on purpose.
  - Sep errors on decoded-fraud rows are type swaps plus the ~10% noisy legit labels, not legit-path rows.
- Per true class (Sep, non-QR), the fp rate is .15-.19, so 43-55% of a class's non-QR rows are fraud-path. The rest is
  camouflage. This fits the known ~63% camo share overall.

## 4. Pitfall seen
mule vs velocity on h6-confused P2P rows: the receiver account number gives row AUC .671 (c6). At event level (c8) the
receivers are 18 velocity vs 185 mule, Mann-Whitney p = .11. The row AUC came from event clustering, so it is null.
Score any row-level "artefact" at event level.

## Opportunities (all small; none recommended for an upload slot)
1. Stop spending effort on core detection or camouflage. The stamp shows that detection of fraud-path rows is saturated in
   test, both recall and precision.
2. 25 test rows decoded legit carry the stamp: 20 are QR-to-merchant with a null mismatch, 5 are non-QR with 0. The same
   kind of row in Sep (fp and h6 argmax legit, n=103) is only 36% fraud, with mixed types. No rule. Expected gain < +.0005.
3. Clean the training noise with the stamp: about 1,000 legit-labelled train rows with the stamp are certain fraud-path.
   Drop them from the legit class, or give them h6's top fraud type. Expected +0 to +.001; h6 already decodes 96% of these
   rows as fraud. To validate: a train.py relabel variant, scored on Sep cleaned and OOS.
