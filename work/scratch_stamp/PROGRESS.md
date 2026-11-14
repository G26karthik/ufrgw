# scratch_stamp: stamp-dependence counterfactual (A1) and one training variant (A2)

Started 14 Nov 23:27 IST. Deadlines: no new fits after 05:30 IST 15 Nov, report by 06:00.
Run: from work/, `PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4 ../.venv/Scripts/python.exe scratch_stamp/<script>`; LightGBM num_threads=4.

## Declared before any A1 result (23:29 IST)
- A2 trigger: after masking the stamp on held-month stamped non-QR rows, the share of FRAUD-LABELLED masked rows decoded
  non-legit (August-tuned bias, before the decoder overrides) falls by more than 20 percentage points in September
  (fold 1). August (fold 0) is reported next to it; the trigger is the September number.
- If met, the A2 variant is chosen and declared here before fitting.

## A2 variant, declared before any A1 prediction result was read (23:33 IST)
- Variant `m50`: controlled stamp masking on FIT rows. Every stamped non-QR fit row (txn_type != QR, mismatch == 0), whatever
  its label, is masked (mismatch -> null) with probability .5, mask seed 7 (independent of model seeds 0,1). Share fixed at
  .5; no August selection between shares (August halves differ by .010, too noisy to pick). Labels, recipe, seeds unchanged.
- Why this and not the stamp-excluded view: the rebuild shows that only the base column `qr_vpa_mismatch` changes when
  non-QR 0 becomes null (the scam features are QR-gated, pairs treats 0 and null alike), so fit-row masking is exactly
  consistent with a full rebuild. Dropping the column and its derived features would also remove the QR-row mismatch
  signal for qr_swap, which confounds the stamp with a real class feature.
- Blend: .65 C + .35 m50 (probabilities, before the decoder), fixed.
- Evaluation: natural held rows, no masking. August via aug_dec.py, then FREEZE line, then September once via bench_eval.py
  with a unique suffix.

## Code facts (read, not yet measured)
- Features that read `qr_vpa_mismatch`: base `qr_vpa_mismatch` (quick_eval.base_features, raw value);
  feats/scam.py: `scam__rcv_qr_mm1_frac_15d` (QR rows with mm==1), `scam__rcv_qr_mmnull_15d`, `scam__snd_qr_mmnull_15d`
  (merchant-QR rows with null mm); feats/pairs.py: row filter `mm.fill_null(0) != 1` (0 and null are treated alike).
  feats/adapt.py reads it but is not in the N build. relabel.py reads it only for variant D (not used by C).
- Expected: masking non-QR mm 0 -> null changes only the base column. To be verified by a full rebuild.

## Steps
1. [done] s1_facts.py -> s1_facts.log. Facts (train rows, load_all):
   - `qr_vpa_mismatch` on non-QR rows: null on 1,001,100 rows, 0.0 on 7,454 (the stamp), 1.0 on 24. Ordinary unstamped
     non-QR representation = null. QR rows: 0.0 / 1.0, null on 678 merchant-QR rows (QR stamp); P2P-QR never null in train.
   - Stamped non-QR rows: Jul 2600 (fraud-labelled 2284, legit-labelled 316), Aug 2415 (2135 / 280), Sep 2391 (2080 / 311),
     Oct-dated 48. By type: PAY-P2P 4415, PAY-MERCHANT 1821, COLLECT-P2P 841, REFUND-MERCHANT 377.
   - Share of fraud-labelled non-QR rows that carry the stamp: Aug 2135/13839 = .154, Sep 2080/13404 = .155
     (by class .14-.20; qr_swap .05, collusion .09). So 85% of fraud-labelled non-QR rows are unstamped.
2. [done 23:47] a1.py -> a1.log. Outputs: P_orig_/P_mask_<aug|sep>.npy, mask_<aug|sep>.npy, model_f<fold>_s<seed>.txt.
   - Full rebuild of the masked world, both folds: only `qr_vpa_mismatch` differs, exactly on the masked rows
     (Aug 2415, Sep 2391); the other 189 columns are identical on all rows (fit and held).
   - Self-check: unmasked predictions equal PN_C_aug.npy and PN_C_sep.npy exactly (max abs diff 0, 4 threads).
   - Rows not masked with any change in P: 0 in both folds (no derived feature carries the change).
   - Gain share of `qr_vpa_mismatch`: fold 0 .017/.018 (rank 13 of 190), fold 1 .024/.032 (rank 9/8).
3. [done 23:50] a1_report.py -> a1_report.log. Counterfactual diagnostic only: not a generalisation score, not an
   estimate of population recall. Before -> after masking, same boosters, bias tuned on unmasked August
   (August rows: bias from the other event-grouped half):

   | masked rows | n | decoded non-legit | mean P(fraud) | P(fraud) >= .5 | decoded = label |
   |---|---|---|---|---|---|
   | Sep fraud-labelled | 2080 | .995 -> .991 | .905 -> .893 | .992 -> .989 | .857 -> .854 |
   | Sep legit-labelled | 311 | .990 -> .990 | .908 -> .895 | .994 -> .994 | .010 -> .010 |
   | Aug fraud-labelled | 2135 | .994 -> .990 | .906 -> .892 | .993 -> .984 | .844 -> .839 |
   | Aug legit-labelled | 280 | .989 -> .986 | .900 -> .884 | .986 -> .979 | .011 -> .014 |

   - Sep label changes on masked fraud-labelled rows: 11 of 2080 (8 fraud -> legit: phishing 6, sim 1, fake_cashback 1;
     3 type -> type). Aug: 13 of 2135 (9 fraud -> legit: phishing 5, mule 2, fake_cashback 1, sim 1; 4 type -> type).
   - Largest class effect: phishing_collect, Sep decoded non-legit .984 -> .957, mean P(fraud) .862 -> .808 (n 258);
     Aug .990 -> .972, .859 -> .812 (n 286). Every other class moves by at most .011 in decoded non-legit.
   - Rows not masked: P changed on 0, decoded label changed on 0, after-decoder label changed on 0 (zone edges identical).
   - Sep after the full decoder, masked fraud-labelled: decoded non-legit .995 -> .991, decoded = label .875 -> .872.
   - Natural context (no masking): unstamped non-QR fraud-labelled rows decoded non-legit Aug .328 (n 11704), Sep .333
     (n 11324); unstamped non-QR legit-labelled .002.

## A2 decision (23:52 IST)
- Trigger NOT met: September detection of masked stamped fraud-labelled rows fell by 0.4 percentage points (.995 -> .991),
  threshold was more than 20. August: 0.4 points. A2 is not run. `a2.py` and `a2_report.py` are written (variant m50 as
  declared) but were NOT executed; no PN_m* / PD_m* file was written; September was not read for any variant.

## Step 6: earlier experiments vs this intervention
- ls_stamp.md: changed the FIT TARGET (dropped / relabelled the 992 legit-labelled stamped fit rows). Features untouched;
  the stamp stayed visible at fit and at prediction. It tests label cleaning, not reliance on the feature. Not equivalent.
- adapt.md: blanked `qr_vpa_mismatch` on a random 45% of P2P-QR-with-category held rows (scratch_adapt/ev.py line 34-37),
  QR rows only. Its "drop raw qr_vpa_mismatch" arm kept `adapt__qr_mm`, which passes the raw value through on non-QR rows
  (feats/adapt.py: `otherwise(mm)`), so the non-QR stamp stayed visible to that model. Not equivalent.
- postproc.py --qr-null: decoder rule (P2P-QR with null mismatch -> qr_swap). QR rows only, no model input changes.
  Not equivalent.
- dd_rules_stamp.py / notes/dd_stamp.md: decoder rule that USES the stamp on legit-decoded rows. Not a masking test.
- notes/ls_halves.md ("masked" there = excluding moved Sep rows from scoring) is unrelated.
- So non-QR stamp masking at prediction time is not covered by the items checked (the six above). Other notes were not searched.

## Status and limits
- Supported (N, both folds, recipe C, seeds 0,1): stamped non-QR held rows stay detected without the stamp column.
- Not evaluated adequately: whether detection of fraud-path rows is saturated. Masked rows keep every other mark they
  carry; A1 says nothing about the two thirds of unstamped fraud-labelled non-QR rows that are decoded legit.
- Not tested: masking the merchant-QR null stamp (Aug 238, Sep 210 rows); fit-time masking or a stamp-excluded view (A2);
  benchmark D; any model other than LightGBM recipe C; other seeds.
- No interval computed (8 of 2080 Sep rows, 9 of 2135 Aug rows flipped to legit). September was read once, for this
  diagnostic only; no variant was scored or selected.
- Gain share of `qr_vpa_mismatch` (.017-.032) includes its use on QR rows; it is not a measure of the stamp.
- A1 arrays stay in scratch_stamp/ (not scratch_gap/), so no tool scores a counterfactual by mistake.
- Done. Final report sent Sat Nov 14 23:53:23 IST 2026.

