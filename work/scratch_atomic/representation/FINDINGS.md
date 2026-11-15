# Agent A (representation) findings

Updated 08:23 IST 15 Nov (clock from `date`). All scripts have finished; section 7 lists what was not reached; section 8 is the follow-up placement test; section 9 is the receiver-history type rule.
Scripts: `work/scratch_atomic/representation/`, run with cwd = `work`, `PYTHONPATH=.`. September is reused evidence.

## 0. Short answer for agent C
- No original field, join, text representation or raw-history aggregate was found that separates the missed
  fraud-labelled events from matched legit rows. Status: **no detected difference** (23 predicates, 20 numeric raw fields,
  31 raw-history aggregates, raw CSV text formats, one small diagnostic fit on original fields only). This repeats
  `camo.md` / `fresh_columns.md` with a different design (independent events, 5 matched controls each, benchmarks N and D,
  bias margin, and a cohort defined without any model output) and reaches the same reading.
- New and measured: a **count of fraud-path rows among the fraud-labelled rows that the final decoder calls legit**. The trace is
  label-free and model-free; the count uses the decode, and its baseline is the trace rate on legit-labelled decoded-legit rows of
  the same month, so it cannot see fraud-path rows that carry a legit label.
  Benchmark N: August about 146, September about 111 rows; benchmark D: 165 and 155 (2 SE about +-80), out of about
  12,000 / 11,600 missed fraud-labelled rows per month. Repeatable in **qr_swap** (N: August 20 trace rows against 10.2
  expected, September 26 against 9.5; D: 21 / 10.3 and 26 / 9.7) and in August phishing (27 / 9.1; September 13 / 8.6).
  Status: **supported** as a measurement; as a lever **promising but unconfirmed**, small.
- Nothing here is ready as an October feature. C1 gives a scoring tool (fraud-path rows lost by any decode, independent of
  label noise and of the model). C2 names the cells where the lost fraud-path rows sit.

## 1. Missed-fraud groups (task 1): `01_missed.py`, output `01_out.txt` (N and D; `01_out_v2.txt` is an identical rerun)
Missed row = supplied label > 0 and final decoder label = 0. September from `dec_N_wt.npz` (`C_q`) / `dec_D_sq6.npz` (`h6_q`);
the script re-derives the September forced decode and asserts at least 99.99% agreement with the dec file. August = `aug_dec.py`-style decode
(bias tuned on the other event-grouped half). Macro-F1 after decoder: N Aug .4980 / Sep .5070; D Aug .5004 / Sep .5090.

| benchmark, month | fraud rows | missed rows | events (48 h) | events with a missed row | fully missed events |
|---|---|---|---|---|---|
| N Aug | 19,345 | 12,014 | 15,016 | 11,912 | 11,811 |
| N Sep | 18,661 | 11,596 | 14,400 | 11,476 | 11,395 |
| D Aug | 19,345 | 12,026 | 15,016 | 11,929 | 11,837 |
| D Sep | 18,661 | 11,624 | 14,400 | 11,506 | 11,430 |

- Missed events are singletons: 12,014 rows in 11,912 events (N Aug).
- Margin after bias (best fraud log-prob + bias minus legit log-prob), N Aug missed rows: borderline (> -1) 71, mid (-3..-1) 468,
  confident legit (< -3) 11,475 (95.5%). Per class the confident share is 93-97%. Borderline misses are 0.6% of all misses.
- Row miss rate per class, N Aug: mule .665, velocity .594, layering .555, sim .619, ATO .609, phishing .616, qr .623,
  refund .594, fake_cashback .661, loan .672, collusion .690.
- Signatures: fraud rows with any signature (takeover archetype, bait rule > 0, `qr_vpa_mismatch == 1`, zone device) have miss rate
  .019 (n 3,123) against .737 without (n 16,222), N Aug; Sep .016 / .743. Missed rows that carry a signature: Aug 60
  (bait 30, mismatch-1 30, archetype 0, zone 0), Sep 51 (bait 18, mismatch-1 33).
- Context, N Aug (fraud rows, miss rate): QR-MERCHANT 4,840 .905; PAY-P2P 7,693 .510; PAY-MERCHANT 4,255 .669;
  COLLECT-P2P 1,104 .362; AUTOPAY-MERCHANT 396 .967; REFUND-MERCHANT 390 .215; QR-P2P 666 .000.
- New device (`device_age_days < 3`): 644 fraud rows, miss rate .13.

## 2. Sample and matched controls (task 2)
- `missed_events_{N,D}_m{8,9}.parquet`: one row per missed event (first missed row), all original and joined fields, decode,
  margin. `detected_events_*`: one row per event with no missed row. `controls_*`: 5 legit-labelled rows per missed event,
  exact match on month, txn_type, receiver_type, amount band (cuts 100, 500, 1000, 2000, 5000, 10000) and sender activity band
  (rows of the sender in the month, cuts 2, 5, 10, 20), seed 2026: 59,560 controls for 11,912 events (N Aug); 1 event unmatched in Sep.
- `sample_{B}_m{m}_events.parquet`: 300 missed + 150 detected events (seed 2026); `sample_*_controls.parquet`: their controls.
- `held_{N,D}.parquet`: final decode, margin, event id for every held row of August and September.

## 3. Missed events against matched controls on original fields (tasks 3, 5): `01_missed.py`, `05_tables.py`, `predicate_rates.csv`
Rates are per independent event (one row each). Benchmark N; D is within +-.006 on every line (`05_tables.py` prints both).

| predicate (label-free; exact code in `01_missed.py`, dict PRED) | Aug missed n (rate) | Aug matched ctl | Aug all legit | Aug detected | Sep missed n (rate) | Sep matched ctl | Sep all legit | Sep detected |
|---|---|---|---|---|---|---|---|---|
| `app_version < 4` | 1,812 (.1521) | .1497 | .1507 | .5635 | 1,776 (.1548) | .1511 | .1505 | .5735 |
| `ip_risk_score >= .441` (July all-row q95) | 559 (.0469) | .0452 | .0443 | .4298 | 502 (.0437) | .0445 | .0446 | .4244 |
| ip minus sender's month median `>= .196` | 571 (.0479) | .0458 | .0471 | .2703 | 551 (.0480) | .0453 | .0463 | .2558 |
| app_version below sender's month max | 6,518 (.5472) | .5381 | .5445 | .6505 | 6,115 (.5329) | .5369 | .5377 | .6583 |
| device city differs from sender's previous row | 1,389 (.1166) | .1138 | .1129 | .1401 | 1,356 (.1182) | .1222 | .1217 | .1477 |
| device used by >= 2 senders (whole data) | 857 (.0719) | .0741 | .0741 | .0673 | 860 (.0749) | .0710 | .0737 | .0752 |
| receiver's first row in the data | 2,107 (.1769) | .1753 | .1771 | .2639 | 1,254 (.1093) | .1107 | .1112 | .1269 |
| balance jump up > 5% against previous row | 5,681 (.4769) | .4720 | .4763 | .5306 | 5,869 (.5114) | .5127 | .5114 | .5893 |
| same bank, sender and receiver | 357 (.0300) | .0304 | .0300 | .0606 | 369 (.0322) | .0311 | .0305 | .0609 |
| amount > balance | 4 (.0003) | .0001 | .0001 | .0180 | 0 (.0000) | .0002 | .0001 | .0239 |
| sender's first row on this device, not the sender's first row (`sd_first`) | 67 (.0056) | .0309 | .0307 | .1753 | 51 (.0044) | .0322 | .0316 | .1881 |

- July (no held-out predictions; all 14,892 fraud events against all legit): `app_version < 4` .2399 / .1505, ip .1266 / .0443, `sd_first` .0301 / .0246.
- Every line except `sd_first`: missed = matched controls = all legit, while detected events differ strongly. Status: **no detected difference**.
- Numeric raw fields, AUC missed against matched controls, N Aug: app_version .501, sender_account_age .500, receiver account
  age .495, merchant age .498, timestamp second .504, senders per device .499, cities per device .508, balance jump .500,
  gap to sender's previous row .498 (detected against controls: app_version .291, device_age .422, cities per device .441).
  Per class (11 classes, N Aug) every AUC for ip, ip deviation, app_version, account age, balance jump and gap to the previous row is inside .472-.529.
- `sd_first` is the one difference (5.5x lower than controls, both months, N and D). Reading: a fresh phone on a fraud row
  comes from the class device blocks (`ts_victim.py:3-7`), which the decoder zones catch, so no missed row has one; legit rows
  get fresh single-use phones from the dense run after the fraud blocks. `art_order.md` already measured it (21 fraud labels
  in 29.9k rows on legit fresh phones; the model predicts legit there). The complement (not `sd_first`) has lift 1.03: not a lever.
  The same effect shows as `device_age_days` AUC .513 / .515 and `takeover__dev_excl` .486 / .486 in `02_trace.py`.
- Timestamp resolution: every timestamp is a whole second (`usec == 0` on 100% of rows); nothing is lost by `dt.epoch("s")`.
- Known drift (not ranked): `app_version < 4` is 0% in October; ip drifts (`drift.md`). `rcv_first_seen`, `dev_shared`, `sd_first`
  are whole-data counts and depend on density.

## 4. Vector columns and raw-history aggregates (tasks 3, 5): `02_trace.py N`, `02_out.txt`, `trace_auc_N.csv`
For each class, AUC(missed event rows against their matched controls) on each of the 190 vector columns (benchmark N tables) and on
31 label-free raw aggregates that have no column in the vector: log amount against the sender's median / max and the receiver's
median; sender's share of rows with the same txn_type, receiver_type, category, city, app; hour, login time and balance against
the sender's medians; sender's and receiver's lifetime, counts and new-payee share; receiver's own outgoing rows; +-48 h counts,
distinct counterparties and new-payee counts on sender, receiver and device; `new_payee` against the observed pair history.
- All classes pooled: every raw aggregate is inside .496-.506 in August and .497-.504 in September, while detected events differ
  (amount against sender's median .680, receiver lifetime .290, new-payee rows on the device in +-48 h .650).
- Per class (11 classes x 31 aggregates): largest |AUC - .5| is .039 in August and .032 in September (both loan_app_extortion, 435 events); no aggregate exceeds .03
  with the same sign in both months (0 of 372 cells, pooled group included). Vector columns: 4 of 1,800 cells, all sparse columns with few non-missing rows
  (`takeover__next_bal_ratio`, `flow__sin_r10_frac24`).
- Status: **no detected difference**. No case was found where a raw link exists that the module keys or gaps (2 h sender session,
  600 s burst, 12 h receiver segment, 6 h hop) drop: the missed events have no excess links on sender, receiver or device at 48 h.
- New_payee flags over all rows (label-free): `new_payee == 1` although the pair was seen before .0045 / .0108 / .0155 / .0187 in
  July / August / September / October; it grows with the length of visible history, as expected.

## 5. Raw CSV text representation (task 3): `03_rawtext.py`, `03_out.txt`
`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv` read as text; pattern = each digit -> 9, leading digit run -> N.
- One text format per column in train and test: timestamp `N-99-99 99:99:99` (100%), ids `AC999999` / `MR999999` / `DV9999999`,
  sim_age / device_age / app_version `N.9`, ip `N.999 / N.99 / N.9` with shares .900 / .091 / .009 for legit and .899 / .091 / .010
  for fraud labels, qr_vpa_mismatch empty or `N.9`. No integer-versus-float variant and no `nan`-versus-empty variant exists.
- amount `N.9` (at most one decimal) .328 legit / .538 fraud labels; balance `N.9` .101 / .131. These are the known value signals
  (integer amounts, balance = amount / .909), present in the vector as `cents`, `round100`, `ring__b11`.
- Status: **no detected difference** in representation; `prep.py:10-13` loses nothing.

## 6. Model-independent generator view (ADDENDUM): `04_mix.py`, `06_lost_cells.py`, `07_qr_tracefree.py`
No model output and no device zone is used as truth here. Labels = supplied labels. Traces come from fields the vector drops.

### C1 (rank 1). Fraud-path "finalize" traces in dropped fields count what the pipeline loses. Status: supported as a measurement
1. Generation step (hypothesis; `fresh_columns.md` section 2): every fraud-path row passes one finalize step that redraws
   `app_version` (3.x on about 60%) and `ip_risk_score` (high on about 47%) and, on non-QR rows, sets `qr_vpa_mismatch = 0`
   with p about .35. Camouflage rows skip it.
2. Trace (label-free): `J = (app_version < 4) & (ip_risk_score >= 0.441)`; stamp `S = (txn_type != "QR") & (qr_vpa_mismatch == 0)`.
   Measured on home-device rows (device number = sender number): legit-labelled `app < 4` .1504 / .1503 / .1503 and `ip >= .441`
   .0442 / .0442 / .0442 and S .0007 / .0007 / .0008 in July / August / September; among S rows `app < 4` .596 / .644 / .629 and
   `ip >= .441` .463 / .474 / .482. J among legit-labelled decoded-legit rows .00668 (Aug) / .00653 (Sep); among rows that are fraud by
   label and by decode .279 / .280.
3. Alternative explanation: the traces belong to the senders or devices chosen for fraud (old app on old phones), not to a per-row
   step. Then J would follow the sender onto that sender's legit-path rows.
4. Test against matched legit: (a) per class, the fraud-path share of home-device fraud labels estimated from `app < 4` and from
   `ip >= .441` (reference = S rows) agree in all 11 classes and 3 months (`04_out.txt`), e.g. July mule .310 / .292, layering
   .441 / .452, ATO .166 / .191, velocity .037 / .027. (b) Missed events show the legit rate of both traces, as do their matched
   controls (section 3). The sender alternative is not excluded by my tests; `fresh_columns.md` c1 (app_version lower than the sender's
   previous row: .457 legit, .63 core) supports the per-row step.
5. Threshold .441 frozen on July; legit and reference rates are recomputed inside each month (they are stable: item 2). Checked on
   August and September, benchmark N, home-device rows
   (85.5% of fraud labels, 92.1% of legit):

| class | Aug share by `app<4` / by ip | Aug share decoded fraud | Sep share by `app<4` / by ip | Sep share decoded fraud |
|---|---|---|---|---|
| mule_account | .299 / .295 | .301 | .319 / .286 | .308 |
| velocity_bot | .085 / .081 | .092 | .081 / .088 | .082 |
| layering_chain | .389 / .436 | .428 | .383 / .413 | .405 |
| sim_swap | .310 / .282 | .299 | .309 / .301 | .320 |
| account_takeover | .193 / .198 | .192 | .184 / .180 | .166 |
| phishing_collect | .393 / .388 | .379 | .357 / .349 | .366 |
| qr_swap | .359 / .386 | .368 | .372 / .362 | .359 |
| refund_abuse | .343 / .358 | .382 | .426 / .386 | .392 |
| fake_cashback | .251 / .300 | .316 | .386 / .332 | .364 |
| loan_app_extortion | .294 / .286 | .307 | .259 / .286 | .293 |
| merchant_collusion | .293 / .323 | .296 | .326 / .273 | .291 |

   The share of each class's labels that the decoder calls fraud equals the generator-trace share within noise, in both months.
   Joint trace over all decoded-legit rows (`06_out.txt` N, `06_out_D.txt` D): N August J rows 3,503 against 3,463 expected, excess 40,
   about 146 fraud-path rows decoded legit; N September 3,308 / 3,278, excess 30, about 111; D August excess 45, about 165;
   D September excess 42, about 155. By supplied class, N (J observed / expected): qr_swap Aug 20 / 10.2, Sep 26 / 9.5; phishing
   Aug 27 / 9.1, Sep 13 / 8.6; collusion Aug 7 / 4.2, Sep 11 / 3.8; every other class within +-5.5.
6. For agent C (executable): score a candidate decode by running `06_lost_cells.py` on it (it reads `held_<B>.parquet`: txn_id, q).
   Fraud-path rows lost = (J count among decoded-legit rows - rJ x n) / (rF - rJ), with rJ = J rate on legit-labelled decoded-legit
   rows and rF = J rate on rows fraud by label and decode, both taken from the SAME decode and month. Do not plug in a constant:
   the excess is about 1% of the expected count, so a fixed .0066 gives wrong numbers. The count covers fraud-labelled rows only. A candidate that only moves camouflage or label noise leaves this number
   unchanged. It needs August / September rows (app_version is 5.x on every October row), so it is a development metric, not a feature.
   Difference from earlier work: `fresh_columns.md` section 3 used the stamp S on h6 for September and test, non-QR rows only; this uses
   two independent dropped fields, covers QR rows, both benchmarks, both months, per class, and returns a count.

### C2 (rank 2). Cells that hold the lost fraud-path rows. Status: promising but unconfirmed (small)
1. Generation step: qr_swap fraud-path rows to a merchant keep `qr_vpa_mismatch = 1` with a non-integer amount; phishing fraud-path
   rows are integer new-payee COLLECTs. Single rows of such events fall below the decode threshold.
2. Trace (label-free cell of decoded-legit rows): `(txn_type == "QR") & (receiver_type == "MERCHANT") & (qr_vpa_mismatch == 1) & (amount != round(amount))`;
   (A COLLECT-P2P new-payee integer cell was also read; it is dropped: see item 4.)
3. Alternative: the J excess in these cells is chance (small counts), or legit rows of old-app senders.
4. Test: J observed against expected at the legit rate, benchmark N. QR cell: Sep 11 J rows in 567 rows against 3.7 expected (z 3.3 in the
   new_payee = 0 half, 1.8 in the other); Aug new_payee = 1 half 2 in 118 against 0.8. In the QR cell 7 of the 11 September J rows carry a fraud label; August has 2 J rows. COLLECT cell: only 4 of 24 (Aug) and 3 of 17 (Sep)
   J rows carry a fraud label, so its excess is mostly legit-labelled rows and the month-wide baseline is not valid per cell:
   **not evaluated adequately**. The phishing-topped fraud-labelled J rows sit in integer PAY-P2P cells (Aug 22 with new_payee = 0, 13 with new_payee = 1).
5. Cells were read on August and September together, so this is not a frozen test. The QR cell rests on September.
6. For agent C: purity is low (about 27 fraud-path rows among 567 decoded-legit rows, 5%), so a forcing rule would lose. A testable option
   is a lower qr_swap threshold inside the QR cell only; upper bound about 25 rows per month. Not evaluated adequately by me.

### C3. Trace-free fraud labels against trace-free legit, original fields only. Status: no detected difference
1. Step: camouflage rows are drawn by the legit path for the same sender; the question is whether the choice of the row that gets the
   fraud label depends on any original field or combination of fields.
2. Cohort (no model output): home device, no S, `app_version >= 4`, `ip < .441`, non-integer amount: about 7,000 fraud labels and
   281,000 legit labels per month.
3. Alternative: uniform relabelling of legit rows. Both produce identical rows (`camo.md`).
4. Test: one small LightGBM (15 leaves, 150 rounds) on 45 original and joined fields including the dropped ones (ages, receiver bank
   and os, cities, second, weekday, day of month, account numbers), fit on July only.
5. August AUC .5047, September .5100; fraud-label rate in the top 1% .075 / .063 against base .024. Per class .475-.531 except
   qr_swap .564 / .555. `07_qr_tracefree.py`: that is the QR-to-P2P cell (qr rate .68-.92 in the cell, no legit rows; in the six largest other
   cells the qr rate is .0016-.0034; QR-MERCHANT with mismatch 1 has .021 in July on 9 rows). It is the known signature, already decoded (QR-P2P miss rate .000).
6. Nothing to score.

## 7. Not reached
- Hand trace of individual events through each join (only aggregate checks were made).
- `adapt`, `mnorm`, `core2`, `ts_victim` columns in production matrices other than the benchmark tables (they are not among the 190 columns
  of `scratch_gap/{N,D}_f*.parquet`).
- `02_trace.py` and `04_mix.py` on benchmark D (only `01` and `06` cover D).
- The sender-identity alternative of C1 item 3; a threshold test for C2; any scored fit (agent C).
- The table of October rates per predicate is in `01_out.txt` lines 3-18 (wide); not copied here.

## 8. Follow-up: are trace-free fraud labels placed independently, or on chosen entities / moments? (added 08:13)
Scripts `09_placement.py` (`09_out.txt`), `10_recur.py` (`10_out.txt`), `11_recur_tracefree.py` (`11_out.txt`). Supplied labels of
July / August / September only; no model output in any test (one reading aid in 8.5 is marked).
- Cohort A = the C3 trace-free cohort: 855,614 rows, 20,599 fraud labels (Jul 6,870, Aug 7,023, Sep 6,706).
- Cohort B = all home-device rows: 1,436,654 rows, 49,188 fraud labels.
- Null: fraud flag shuffled among cohort rows inside month x txn_type x receiver_type, 200 shuffles, seed 2026. Entity activity,
  daily and hourly volume and sender row times are preserved exactly. Format below: observed | null mean | null 2.5%..97.5%.
- Device on home-device rows is the sender's own phone, so device results equal sender results (in `09_out.txt`, not repeated).

**Hypothesis:** the generator picks accounts, merchants or time windows and labels some of their ordinary rows.
**Alternative:** independent per-row placement at a class rate; any concentration only reflects activity.
**Result: cohort A agrees with the alternative on every test. Cohort B concentrates strongly, as fraud-path events must.**

### 8.1 Test 1, entities holding >= 2 and >= 3 fraud rows inside a month. Cohort A: no detected difference
| entity | Jul >= 2 | Aug >= 2 | Sep >= 2 | Jul >= 3 | Aug >= 3 | Sep >= 3 |
|---|---|---|---|---|---|---|
| sender | 200 / 178 / 149..206 | 181 / 186 / 164..210 | 161 / 171 / 149..195 | 13 / 9 / 4..15 | 14 / 10 / 5..17 | 11 / 9 / 3..15 |
| receiver | 500 / 478 / 453..506 | 482 / 489 / 459..515 | 459 / 468 / 444..493 | 250 / 250 / 233..266 | 260 / 256 / 236..272 | 248 / 246 / 229..262 |
| merchant | 499 / 475 / 451..503 | 481 / 487 / 457..512 | 458 / 467 / 442..492 | 250 / 250 / 233..266 | 260 / 256 / 236..272 | 248 / 246 / 229..262 |
| sender-receiver pair | 5 / 3 / 1..7 | 8 / 3 / 0..9 | 8 / 3 / 0..6 | 0 / 0 / 0..0 | 1 / 0 / 0..0 | 0 / 0 / 0..0 |

- Only the pair line leaves the null range (September 8 against 0..6; August one triple). `10_recur.py` (a): the 43 rows on repeated
  pairs are REFUND-to-merchant rows (5 / 11 / 10 rows in Jul / Aug / Sep, top class refund_abuse; two classes in Jul and Aug) plus QR-merchant rows with mixed classes
  (5 / 5 / 3 rows, 5 / 5 / 3 different classes) and 4 PAY-merchant rows. The refund rows are fraud-path refund episodes that carry none of the three traces
  (refund amounts are non-integer in 72%); the vector already holds them (`burst__rs_*`, `burst__pair_ref14d`). About 5 pairs per month.
- Cohort B for comparison: sender >= 3: 348 / 91, 369 / 95, 367 / 86 (z +27 to +36); pair >= 2: 243 / 20, 280 / 21, 248 / 23.

### 8.2 Test 2, across months. Cohort A: no detected difference
| entity (cohort A) | P(Aug \| Jul) | P(Sep \| Aug) | P(Sep \| Jul or Aug) | Sep predicate lift | Sep predicate coverage |
|---|---|---|---|---|---|
| sender | .0551 / .0520 / .0473..0575 | .0508 / .0503 / .0462..0541 | .0489 / .0480 / .0448..0509 | 1.018 / 1.007 / .939..1.068 | .1026 / .1011 / .094..108 |
| receiver | .2044 / .2111 / .2012..2208 | .2055 / .2040 / .1942..2130 | .1539 / .1553 / .1490..1620 | .990 / .991 / .979..1.003 | .6633 / .6623 / .654..672 |
| merchant | .3054 / .3175 / .3031..3325 | .3111 / .3094 / .2949..3240 | .2461 / .2487 / .2382..2596 | .990 / .990 / .978..1.003 | .6625 / .6609 / .653..669 |
| pair | .0009 / .0009 / .0004..0016 | .0007 / .0009 / .0003..0016 | .0005 / .0009 / .0004..0014 | .530 / .997 / .501..1.510 | .0010 / .0018 / .0009..0028 |

- Predicate here = "entity held a trace-free fraud label in July or August", scored on September cohort rows (label-informed).
  Frozen July to August, checked August to September: every value is inside the shuffle range. No lift.

### 8.3 Test 3, class agreement on one entity. Cohort A: no detected difference beyond the refund pairs
Share of fraud-row pairs on one entity with the same class, against shuffling the class among the month's cohort fraud rows.
- receiver / merchant (about 350,000 pairs per month): .1087 / .1132, .1142 / .1123, .1199 / .1152 (z -1.9, +0.9, +2.0).
- sender (229 / 215 / 183 pairs): .1266 / .1119, .1814 / .1149, .1639 / .1163 (z +0.7, +3.2, +2.1): about 3, 14 and 9 same-class
  pairs above chance per month.
- pair (5 / 10 / 8 pairs): .40, .60, .625 against .11 to .12 (z +1.9, +4.8, +4.6): the refund episodes of 8.1.
- Check (`12_sender_pairs.py`, `12_out.txt`): same-class sender pairs observed / expected under class shuffle: Jul 29 / 26.0,
  Aug 39 / 24.2, Sep 30 / 21.1. REFUND-to-merchant refund_abuse pairs: 4, 11, 8 (about half to the same merchant). Without them:
  25, 28, 22. So refund episodes explain 11 of the 15 excess pairs in August and 8 of 9 in September; the rest (about +4 and +1) is
  inside sampling noise (sd about 4.6). No QR-to-P2P pair is involved. Not a second mechanism.
- Cohort B: sender .63 to .64 against .115 (z > 90); pair .92 to .94.

### 8.4 Test 4, time. Cohort A: no detected difference
- Chi-square / df of fraud counts per day against the stratum-preserving expectation: Jul .81, Aug .74, Sep .76 (null mean .95 to .98,
  range about .56..1.55). Per hour of day: .88, .75, 1.23 (null .95 to 1.00, range about .47..1.62).
- Consecutive fraud rows of one sender: 1,591 pairs against 1,546 (1,472..1,615); gap < 1 h 3 against 2 (0..5); < 1 day 41 against 39
  (26..51); < 7 days 278 against 260 (225..292); median gap 24.09 days against 24.11 (22.81..25.64).
- Cohort B: per hour 70.5 / 67.6 / 58.3; per day 2.48 / 2.00 / 1.50; gap < 1 h 1,529 against 16.

### 8.5 The one concentration found (cohort B only): P2P receivers recur across months. Status: supported; no lever on undetected rows
- In cohort B, P(receiver has a fraud row in the next month | it had one) is .1612 / .1069 (Jul to Aug) and .1628 / .1054 (Aug to Sep),
  z +24 and +26. Merchants do not recur (.3231 / .3375 and .3321 / .3270); the effect is on P2P accounts.
- Merchants and pairs in cohort B go the other way: merchants flagged in July or August hold 6,207 September fraud rows against
  6,388..6,534 under the shuffle (lift .763 against .793, z -7); pair P(Sep | Jul or Aug) .0011 against .0014..0024. Plausible reason:
  fraud-path merchant rows sit on small merchants and on pairs that are not used again. Not tested further.
- Exact predicate for October (**label-informed: it reads train labels, so it is not a label-free feature**):
  `receiver_type == "P2P"` and the receiver account held a fraud label in July to September.
- Frozen July to August, all train rows: predicate rows 5,435, fraud-labelled 1,351 (rate .249; month base .0367, lift 6.8; P2P base
  .0493, lift 5.0); coverage 7.0% of August fraud labels, 14.3% of P2P fraud labels.
- Checked August to September: 5,190 rows, 1,250 fraud-labelled (.241; lift 6.6 / 5.0); coverage 6.7% / 13.9%.
- July + August to September: 9,712 rows, 2,251 fraud-labelled (.232; lift 6.3 / 4.8); coverage 12.1% / 25.0%. By class: mule 501,
  layering 449, phishing 332, qr 286, ATO 233, sim 218, fake_cashback 101, velocity 78, loan 24, collusion 21, refund 8.
  Control (P2P rows to receivers seen in the source months without a fraud label): 108,252 rows, rate .0373.
- Home-device rows only: 7,807 rows, 1,624 fraud-labelled (.208, lift 6.1 / 4.5), coverage 10.2% / 20.5%.
- Model-free check that it is fraud-path structure and not label placement (`11_recur_tracefree.py`): inside the trace-free cohort,
  non-QR P2P rows, the predicate has lift .75 (16 fraud rows against 21.3 expected, Jul to Aug), 1.00 (20 / 20.0, Aug to Sep),
  .87 (33 / 38.0, Jul + Aug to Sep). With QR rows kept the lift is 1.5 to 2.1, all from the known QR-to-P2P signature.
- Reading aid, uses the benchmark-N final decode (not part of the test): of the 2,251 fraud-labelled September predicate rows 2,079 are
  already decoded fraud. Among decoded-legit predicate rows the fraud-label rate is .0234 (172 of 7,353) against .0231 for the control and
  .0231 for all decoded-legit P2P rows; August: .0245 against .0233. So the predicate adds nothing on the rows the pipeline calls legit.
  280 legit-labelled predicate rows are decoded fraud (11.9% of decoded-fraud predicate rows, the known label-noise rate).
- Not reached: the number of October rows that match the predicate.
- For agent C: nothing to score for detection. Possible use for the type decision only (the receiver's earlier class as a
  label-informed input); I did not measure it.

### 8.6 Status per test
| test | cohort A (trace-free) | cohort B (all home-device fraud labels) |
|---|---|---|
| 1 entity concentration within month | no detected difference (pair excess about 5 per month = refund episodes) | supported (fraud-path events) |
| 2 recurrence across months | no detected difference | supported for P2P receivers; senders weak (Sep predicate lift 1.15 against .97..1.04); merchants and pairs recur LESS than the shuffle (see 8.5) |
| 3 class agreement | no detected difference once the refund episodes are removed (sender z +3.2 / +2.1 before removal; 1 of about 100 statistics read) | supported |
| 4 day / hour / sender gaps | no detected difference | supported |

Conclusion of the test: trace-free fraud labels behave as independent per-row placements inside month x txn_type x receiver_type.
No entity, pair, month-to-month or time concentration was found, so no October predicate follows from them.

## 9. Follow-up: the receiver's earlier fraud class as a label-informed input to the type decision (added 08:20)
Script `14_hist_class.py`, output `14_out.txt`. Measurement only, no fits. **Label-informed: the rule reads train labels of earlier months.**
- History: supplied labels of earlier months only (August rows: July; September rows: July + August), P2P rows, keyed on the receiver account.
  `hist_class` = majority class of the receiver's earlier fraud-labelled rows (ties: none), `hist_n` = their count, `hist_share` = majority share.
- Predicate row: `receiver_type == "P2P"` and `hist_n >= 1`. Decodes: `held_N.parquet` (`C_q`), `held_D.parquet` (`h6_q`).
- Signature (never overridden): takeover archetype, zone device (static edges and pool), bait rule 1 or 2, or final label != forced decode.

**Result: no setting of the rule is above the unchanged decode on August, on either benchmark. Stopped there; September was not scored
for the rule. Status: measurably worse.**

### 9.1 Decoded-fraud predicate rows
| benchmark, month | predicate rows | decoded fraud | fraud-labelled | decoder == label | hist_class == label | decoder == hist_class | rows where they differ | decoder right, hist wrong | decoder wrong, hist right | both wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| N Aug | 5,435 | 1,428 | .877 | .745 | .169 | .212 | 980 | 736 | 20 | 224 |
| N Sep | 9,712 | 2,359 | .881 | .769 | .172 | .211 | 1,430 | 1,103 | 21 | 306 |
| D Aug | 5,435 | 1,393 | .896 | .765 | .172 | .210 | 956 | 738 | 18 | 200 |
| D Sep | 9,712 | 2,307 | .896 | .782 | .174 | .209 | 1,397 | 1,094 | 22 | 281 |

- If `hist_class` were independent of the label, agreement would be .115 / .102 (N Aug / Sep) and .117 / .103 (D). Observed .169 to .174:
  the earlier class carries a small amount of information (about +.055 to +.07), far below the decoder (.75 to .78).
  A tie (`hist_class` = none) never counts as agreement.
- With and without a signature (N Aug; no signature 1,123 rows, signature 305): decoder == label .745 / .744; hist == label .157 / .216;
  where they differ, decoder right 581 / 155, hist right 12 / 8. N Sep (1,814 / 545 rows): decoder .772 / .760; hist .168 / .183; 838 / 265 against 12 / 9.
- By `hist_n` (N Aug: 1,135 rows with 1; 175 with 2; 118 with 3+): hist == label .188 / .046 / .178; decoder right where they differ 657 / 20 / 59,
  hist right 18 / 0 / 2. N Sep (1,599 / 460 / 300): .212 / .046 / .150; 934 / 56 / 113 against 19 / 0 / 2. Most `hist_n = 2` rows are ties (two different classes).
- By `hist_share` (N Aug: 1.0 on 1,250 rows; .6 to .9 on 33; < .6 on 145): hist == label .181 / .485 / .000; decoder right where they differ
  724 / 12 / 0, hist right 18 / 2 / 0. N Sep (1,770 / 142 / 447): .207 / .218 / .016; 1,018 / 81 / 4 against 20 / 1 / 0.
- Best cell seen: no signature, `hist_n >= 3`, share .6 to .9, N Aug: 21 rows, hist == label .714, decoder .857; where they differ decoder
  right 4, hist right 1. In N Sep the same cell has 99 rows, hist .273, decoder .788. It does not repeat.
- D is within +-.04 of N on every share in lines with about 90 or more rows; smaller cells differ more.

### 9.2 Rule R(n0, s0), chosen on August only (whole-month macro-F1 after the decoder, all 12 classes)
| setting | N Aug macro-F1 (unchanged .4980) | rows changed / corrected / newly wrong | D Aug macro-F1 (unchanged .5004) | rows changed / corrected / newly wrong |
|---|---|---|---|---|
| R(1, .6) | .4688 (-.0292) | 772 / 12 / 581 | .4704 (-.0300) | 757 / 10 / 585 |
| R(1, .8) | .4690 (-.0290) | 766 / 12 / 577 | .4706 (-.0298) | 751 / 10 / 581 |
| R(1, 1.0) | .4690 (-.0291) | 765 / 11 / 577 | .4706 (-.0299) | 750 / 9 / 581 |
| R(2, .6) | .4959 (-.0022) | 63 / 1 / 49 | .4983 (-.0021) | 62 / 1 / 48 |
| R(2, .8) | .4960 (-.0020) | 57 / 1 / 45 | .4985 (-.0019) | 56 / 1 / 44 |
| R(2, 1.0) | .4960 (-.0020) | 56 / 0 / 45 | .4984 (-.0020) | 55 / 0 / 44 |
| R(3, .6) | .4965 (-.0015) | 46 / 1 / 36 | .4989 (-.0015) | 45 / 1 / 36 |
| R(3, .8) | .4967 (-.0013) | 40 / 1 / 32 | .4991 (-.0013) | 39 / 1 / 32 |
| R(3, 1.0) | .4967 (-.0014) | 39 / 0 / 32 | .4990 (-.0014) | 38 / 0 / 32 |

- Every setting loses on August. Per the instruction the rule was not applied to September, so event counts and per-class changes
  for a chosen setting do not exist. The September rows of table 9.1 show the same picture (1,103 against 21).

### 9.3 Which explanation the numbers support
- Hypothesis (the earlier class of the receiver decides the type): **not supported**. Where it differs from the decoder it is right in
  2% of rows (20 of 980, N Aug; 21 of 1,430, N Sep).
- Alternative (the model already encodes receiver behaviour, so `hist_class` agrees with the decoder and changes nothing): **not
  supported either**. They agree on only 21% of rows.
- What the numbers show instead: P2P receiver accounts recur across months (section 8.5) but not in the same class. The same account
  receives fraud rows of different classes in different months, so its earlier class is close to uninformative (.17 against
  .10 to .12 under independence). Camouflage labels in the history cannot explain this: of the 1,624 September home-device P2P
  predicate fraud rows about 500 are expected under the shuffle (section 8.5), so about two thirds are real recurrence, and a
  persistent class would put the agreement well above .4. This fits a shared pool of receiver accounts used by several fraud classes; that reading is mine and
  was not tested further. Not reached: the `hist_class` x label crosstab, and whether the earlier label was a fraud-path row.

### 9.4 October count (count only, no label inferred)
History = July to September labels: 7,000 of 260,582 test rows match the predicate (95,042 test rows are P2P); 1,559 with `hist_n >= 2`,
537 with `hist_n >= 3`; 5,407 distinct receiver accounts.

## (superseded stub below, written 07:46 by the clock; its header said 07:48)

## Running / done
- `01_missed.py` (running, second start 07:47): missed-fraud groups (N then D, August and September), one row per independent
  event (`fraud_events`, 48 h), 5 matched legit controls per event (month, txn_type, receiver_type, amount band, sender
  activity band; seed 2026), 23 label-free raw-field predicates and 20 numeric raw fields scanned. Output `01_out.txt`,
  `predicate_rates.csv`, `sample_{N,D}_m{8,9}_events.parquet`, `sample_*_controls.parquet`, `missed_events_*.parquet`.
- `field_coverage.md`: in progress.
- `02_trace.py`: planned: missed events whose raw links (sender / receiver key / device within 48 h) exist while the size
  features of the 190-column vector show a singleton (module key or gap loses the link).

## Examined so far by code reading (no numbers yet)
- Raw fields with no path into the 190 columns: `ip_risk_score`, `app_version` (dropped on purpose, quick_eval.py:30,
  notes/drift.md), `sender_account_age_days`, `snd_opened_on`, `rcv_opened_on`, `rcv_bank_code`, `rcv_os`,
  `mer_onboarded_on`, timestamp date / weekday / second / microsecond, raw `device_city` / `mer_city` / `rcv_home_city`
  values (only equality flags survive), `device_id` (aggregates plus decoder zones only).
- Nine category columns enter as ordered Enum codes, null -> "NA" (quick_eval.py:45). Already tested as categorical:
  notes/assumption_tests.md section B, no detected difference. Not repeated.
