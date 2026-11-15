# Forensic round on missed fraud: original data, events, generator view (15 Nov 2026)

Question: which observable information does the current representation or pipeline lose, especially on fraud-labelled
rows that the final pipeline predicts as legit? Neither a dataset ceiling nor the existence of recoverable signal was
assumed. This note states what was examined and what the tests found. It makes no ceiling claim.

Three agents, separate directories under `scratch_atomic/` (shared brief `CONTEXT.md`, generator addendum `ADDENDUM.md`):
`representation/` (A, original fields against model features), `relationships/` (B, roles, devices, ordering),
`verification/` (C, audit and whole-month scoring). Every number has a script in its directory; full tables are in each
`FINDINGS.md` / `AUDIT.md` / `RESULTS.md`. Benchmarks N (labels hidden, full density) and D (density-matched,
label-informed stress test) are reported separately. September is reused development evidence.
The two public scores (cmb80 .5484, sq35 .5494) were recorded as aggregates and used for nothing here.

## Who is missed (A, `01_missed.py`)
| benchmark, month | fraud rows | missed rows | fraud events (48 h) | events with a missed row |
|---|---|---|---|---|
| N Aug | 19,345 | 12,014 | 15,016 | 11,912 |
| N Sep | 18,661 | 11,596 | 14,400 | 11,476 |
| D Aug | 19,345 | 12,026 | 15,016 | 11,929 |
| D Sep | 18,661 | 11,624 | 14,400 | 11,506 |

- 95.5% of missed rows are confident-legit after the bias (margin below -3); 71 rows are borderline (N Aug).
- Missed events are single rows (12,014 rows in 11,912 events), on the sender's own device, inside ordinary history.
- Miss rate with any signature (archetype, bait rule, `qr_vpa_mismatch == 1`, zone device): .019; without: .737.
- By context (N Aug): QR-MERCHANT .905, AUTOPAY-MERCHANT .967, PAY-MERCHANT .669, PAY-P2P .510, COLLECT-P2P .362,
  REFUND .215, QR-P2P .000. Per class .555 to .690.

## Original fields against the feature vector (A)
- Coverage table for every transaction, account and merchant field (preserved / transformed / aggregated / dropped,
  with file:line): `scratch_atomic/representation/field_coverage.md`.
- 23 label-free predicates on original fields, one row per independent event, 5 matched legit controls each (month,
  transaction type, receiver type, amount band, sender activity): missed = matched controls = all legit, while detected
  fraud differs strongly. Example N Aug, missed / control / all legit / detected: `app_version < 4` .152 / .150 / .151 /
  .564; `ip_risk_score >= .441` .047 / .045 / .044 / .430. Same picture in September and on D (within .006).
- 20 raw numeric fields: AUC missed against controls .49 to .51. 31 raw-history aggregates (sender and receiver
  profiles, +-48 h link counts, pair history): pooled AUC .496 to .506; 0 of 372 class cells above .03 with the same
  sign in both months.
- Raw CSV text: one text format per column in train and test, whole-second timestamps, no missing-versus-zero or
  int-versus-float variant lost by `prep.py`.
- Model-independent check on home-device rows (`04_mix.py`): per class, the share of fraud labels that carry a
  fraud-path trace in the DROPPED fields (`app_version < 4`, high `ip_risk_score`) equals the share the decoder calls
  fraud, in all 11 classes, both months (August: mule .299 / .295 by trace against .301 decoded; layering .389 / .436
  against .428; qr_swap .359 / .386 against .368). Status: supported as a measurement.
- Measured loss on trace-carrying fraud: about 110 to 165 fraud-path rows per month are still decoded legit, of about
  12,000 missed rows (2 SE about 80). Excess in qr_swap (N: 20 against 10.2 expected in August, 26 against 9.5 in
  September), phishing_collect, merchant_collusion. `app_version` is 5.x on every October row, so the trace itself
  cannot be a feature.
- Trace-free cohort (home device, no stamp, `app_version >= 4`, low ip, non-integer amount; about 7,000 fraud labels
  against 281,000 legit per month): a model on 45 original and joined fields, dropped ones included, fit on July, gives
  AUC .505 (August) and .510 (September). Status: no detected difference.
- Small candidate cell (A's C2): QR-to-merchant rows with `qr_vpa_mismatch == 1` and a non-integer amount, decoded
  legit: 11 fraud-path rows in 567 against 3.7 expected (N September); upper bound about 25 rows per month. Scored by C.

## Relationships, roles, devices, ordering (B)
Cohorts were defined by labels only (single-row fraud events, all fraud events, own-device events, events with 3+ rows)
as well as by the September misses. Controls: legit rows in the same month x type x receiver type x amount band x
sender-activity cell; nulls: timestamps shuffled within day and within sender.
- a. Ordered role transitions (previous event of the sender was a receive within 24 h, role strings of length 3):
  all fraud events +.9 points (z 6); after conditioning on `flow__sin_n24h`, `flow__rin_n12h`, `scam__rcv_in_15d`:
  +.09 / -.05 / -.08. On 3+ row events most of the effect survives the day shuffle (+14.7 real, +10.2 shuffled), so it is
  same-day co-occurrence, not order. Status: redundant.
- b. Secondary device with directional counterparty: owner edges at most .12%; co-user edge +1.8 points on multi-row
  events, -.05 / +.08 after `burst__devrcv_c300`, `burst__rs_n_device`, `burst__ds_n_sender`; identical under both
  shuffles. Status: redundant / empty.
- c. Activity change against the account's own earlier activity (7 days against the 21 days before): +22 points on
  multi-row events, +9 after the exact recent count, -.2 / +.8 after the recent count and the session / burst / ring
  columns together. The own-history normalisation adds nothing; what remains is a plain 7-day sender count, which is
  not in the matrix (`burst__snd_c*` stops at 30 minutes). It halves in October (density). Scored by C.
- Missed September events: all 17 motifs within .5 point and |z| <= 1.3. Univariate AUC of all 26 columns .485 to .519.
  Positive control: on detected September events the same motifs give +5.3 (z 10) and +11.0 (z 13).

## Generator view: are trace-free fraud labels placed on chosen entities or moments? (A, `09_placement.py` to `12_`)
Supplied labels only, no model output. Null: the fraud flag shuffled inside month x transaction type x receiver type,
200 shuffles. Cohort: the trace-free cohort (855,614 rows, 20,599 fraud labels).
- Entities with >= 2 labels in a month, observed against null mean: senders 200 / 178, 181 / 186, 161 / 171; receivers
  500 / 478, 482 / 489, 459 / 468; all inside the null range. Only sender-receiver pairs leave it (5 to 8 against 3):
  about 5 refund episodes per month, covered by `burst__rs_*`.
- Recurrence across months: P(sender labelled in August | labelled in July) .0551 against .0520; receivers .2044
  against .2111. Predicate "entity held a trace-free label in July or August" on September: lift 1.018 (senders),
  .990 (receivers), .990 (merchants).
- Class agreement of two labels on one entity: at the shuffle level once the refund pairs are removed.
- Time: per-day and per-hour dispersion inside the null; sender gaps under 1 hour 3 against 2 expected.
- Status: no detected difference from independent per-row placement, in every test.
- Contrast, all home-device fraud labels: sender gaps under 1 hour 1,529 against 16; same-class share on a pair .92
  against .115. That structure belongs to the fraud the pipeline already detects.
- One recurring entity type among all fraud labels: P2P receivers with an earlier fraud label (September fraud rate
  .232, lift 6.3). 2,079 of the 2,251 such fraud rows are already decoded fraud; among decoded-legit rows the rate is
  .0234 against .0231 for controls. As a class hint (label-informed rule "use the receiver's earlier class"), chosen on
  August: every setting lowers macro-F1 (best -.0013 on N and D); where it differs from the decoder it is right in
  about 2% of rows. Status: measurably worse; not scored on September. Receivers recur, but not in the same class.

## Whole-month scoring of the surviving mechanisms (C)
Harness (`scratch_atomic/verification/`): matched baseline equal to the stored references bit for bit (N `PN_C`, D
`PD_A_C`); the decode equals the stored decode files; every new column passes a label-permutation check (labels
permuted, columns must stay bit-identical; a deliberately leaky column is rejected). Whole held month scored, August
first, September once. Mechanisms frozen at 08:09 IST. Disclosure by agent C: its audit runs had printed September
diagnostic lines for both items before the freeze; no column, grid or factor was set from them.

Change in macro-F1 after the decoder (September: 5%..95% event bootstrap):

| arm | N Aug | N Sep | D Aug | D Sep |
|---|---|---|---|---|
| control: one hash-noise column, same seeds | +.0005 | +.0007 (-.0002..+.0016) | -.0012 | +.0006 (-.0004..+.0016) |
| control: other seed pair (2,3 against 0,1) | +.0014 | +.0002 (-.0015..+.0025) | | |
| `n7` (three backward 7-day sender counts), seeds 0,1; D with density-matched columns | +.0000 | +.0012 (+.0002..+.0023) | -.0005 | -.0002 (-.0012..+.0007) |
| `n7`, seeds 2,3, against the seed-pair control | +.0004 | +.0005 (-.0017..+.0021) | | |
| `n7` against the noise column | -.0005 | +.0006 (-.0005..+.0018) | +.0007 | -.0008 (-.0018..+.0003) |
| `c2` (QR-merchant cell, P(qr_swap) x 3, factor chosen on August N from {3, 10, 30}) | +.0002 | -.0003 (-.0004..-.0001) | +.0000 | -.0001 (-.0004..+.0002) |

- `n7`: one positive reading (N September, seeds 0,1). A noise column moves the same reading by +.0007, the second seed
  pair covers zero, D is negative. Status: no detected difference (promising but unconfirmed by the letter of the frozen
  rule on that one reading). N September rows: 18 missed fraud rows recovered in 14 events, 13 new false positives, 24
  correct fraud rows damaged; the noise column gives 16 / 14 / 21.
- `c2`: measurably worse on N (by .0003), no detected difference on D. September: N recovers 6 missed fraud rows for 37
  new false positives, D 10 for 30. The cell holds about 18 qr_swap-labelled rows among about 525 decoded-legit rows.
- Re-measured by C and found empty: A's `sd_first` (the deficit is the known legit fresh-phone block), A's COLLECT cell
  (lift 1.20 / 1.13), the label-informed receiver-history predicate (fraud share inside decoded-legit rows .0245 / .0234
  against base .0232 / .0231), B's 26 pattern columns (AUC inside decoded-legit rows .485 to .516).
- C confirmed that the dropped-field trace (`app_version < 4`, high `ip_risk_score`) vanishes in October by a label-free
  check: `app_version < 4` on stamp rows is .60 to .63 in July to September and .000 in October.
- Not fitted (two-slot limit): B's own-history ratio and the 26-column block. D has one seed pair. Per-class tables,
  event counts before and after the decoder and reproduction commands: `scratch_atomic/verification/RESULTS.md`.

## Outcome
- No mechanism from this round stayed competitive, so no candidate file was built. `subs/dd_final.csv` is unchanged.
- What the round measured, model-independently: (1) per class, the fraud labels that carry a generation trace in the
  original data are the ones the pipeline calls fraud; about 110 to 165 trace-carrying rows per month remain missed;
  (2) the other fraud labels are indistinguishable from legit rows on every original field, raw-history aggregate,
  relationship, ordering, entity and time statistic tested here, and their placement is statistically the same as
  independent per-row placement inside month x transaction type x receiver type.
- This is a statement about the tests listed in this note. It is not a claim that no signal exists. Not tested here:
  amount-conditioned transitions (receive X then send about X), role strings longer than 3, merchant-side histories,
  other event-gap definitions, a hand trace of single events through each join, A's aggregate tests on benchmark D.
