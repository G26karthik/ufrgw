# Agent B (relationships / ordering) FINDINGS — running file

Started 07:38 IST 15 Nov. Stub written 07:46 IST. Numbers below this line are final only where a script is named.

## Population (script `s0_overview.py`)
- September, benchmark N, `scratch_gap/dec_N_wt.npz` (`y`, `C_q`). Missed = `y > 0` and `C_q == 0`.
- Events: `clean_labels.fraud_events(tx, g=48*3600)`.
- Missed rows per class (rows / missed rows / events / events with a missed row): mule 3367/2276/2430/2271, velocity 2659/1561/1661/1560,
  layering 2182/1261/1556/1260, sim_swap 1267/778/989/778, ATO 1280/784/1007/784, qr_swap 2352/1462/2299/1462,
  refund 1075/633/867/633, cashback 831/517/709/517, loan 645/431/551/431, collusion 841/575/676/572.
- Structure of missed events: ~97% are single-row events (mule: 2214 of 2271 missed events have exactly one fraud-labelled row).
- Consequence for freezing: July / August have no decoded predictions, so the freeze population there is the label-only stand-in
  "single-row fraud event" (g = 48 h). September is reported on the real missed set.

## Patterns (in progress)
a. ordered role transitions; b. secondary device + directional counterparty; c. in/out activity change vs own history. Results follow.

## Task 1: sample + timelines (script `s2_sample_timelines.py`, written 07:49 IST)
- `sample.parquet`: 40 independent missed September events (seed 0, one row per event) + 40 legit controls matched exactly on
  month | txn_type | receiver_type | amount band (100/300/1000/3000/10000) | sender rows in month band (1/2/4/8).
- `timelines.txt`: for each of the 80 rows, every event of the sender and of the P2P receiver in +-14 d in BOTH roles, device owner
  (`DV0<n>` belongs to `AC<n>`), device users.
- Reading the 40 missed timelines: the labelled row sits inside an ordinary account history (own device, old payee, no burst, no
  other fraud-labelled row on either account). Examples: `TX80bdc478` labelled qr_swap = P2P PAY 390.00 to an old payee, no QR;
  `TX233b936d` labelled sim_swap = merchant QR on the sender's own long-used device; `TX260b9627` labelled mule_account = merchant PAY 113.00.

## Row-type mix of missed events vs legit (script `s3_typemix.py`, `typemix.csv`; status: supported, September N)
Total variation distance over (txn_type, receiver_type, amount band, new_payee), one row per event:
| class | missed ev | caught ev | TV missed vs legit | TV same-size legit sample vs legit (noise floor) | TV caught vs legit | P2P share missed / caught (legit .361) |
|---|---|---|---|---|---|---|
| mule | 2271 | 168 | .052 | .050 | .595 | .361 / .738 |
| velocity | 1560 | 101 | .063 | .057 | .432 | .352 / .386 |
| layering | 1260 | 298 | .065 | .089 | .849 | .356 / .919 |
| sim_swap | 778 | 213 | .098 | .112 | .816 | .389 / .915 |
| ATO | 784 | 223 | .083 | .083 | .844 | .380 / .901 |
| phishing | 1314 | 762 | .070 | .075 | .881 | .355 / .963 |
| qr_swap | 1462 | 839 | .073 | .057 | .747 | .341 / .694 |
| refund | 633 | 235 | .090 | .100 | .745 | .348 / .064 |
| cashback | 517 | 192 | .112 | .104 | .841 | .377 / .922 |
| loan | 431 | 120 | .147 | .148 | .781 | .316 / .267 |
| collusion | 572 | 106 | .098 | .111 | .609 | .390 / .330 |
Reading: for all 11 classes the missed events' row-type mix is at the legit sampling-noise floor and far from the class's caught events.
This is a measurement about row type only; the three relational patterns below test the remaining (ordering / device / role) channels.
Foreign-device share (device not owned by sender): missed .027-.060 by class, legit .079 (legit includes the known fresh-phone region), caught .13-.80.

---
# RESULTS (written 07:57 IST; UPDATED 08:02 IST: read the UPDATE section at the end first, it supersedes the status of pattern c)

## How every number below is computed
- Feature code: `s1_patterns.py::feat(ts)`; 26 label-free columns, each uses only events strictly earlier than the row. Output for all
  1,824,642 rows (train + test): **`rel_feats.parquet`** (`txn_id` + `rel__*`), asserted unique. Agent C can join it on `txn_id`.
- Motif = boolean threshold on those columns (`s1_patterns.py::motifs`).
- Control = legit-labelled rows of the same month in the same cell: month x txn_type x receiver_type x amount band
  (100/300/1000/3000/10000) x sender rows in the month (1, 2, 3-4, 5-8, 9+). `diff` = event rate minus the cell-weighted control
  rate, in percentage points; `z` = mean / standard error of the per-event (motif - cell control rate). One row per event.
- **Warning on z:** when the event-side rate is exactly 0 the z is meaningless (values like -23, -30 in the csv files for motifs with
  rate < 0.3%). Read `fraud_pct` / `ctrl_pct` there, not z.
- Populations. Model-defined (September only): `missed_m9` (11,476 events), `caught_m9` (3,005). Model-independent (addendum):
  `single_m7/8/9` = single-row fraud events (13,626 / 13,713 / 13,195), `allfraud_m7/8/9` = every fraud-labelled event
  (14,892 / 15,016 / 14,400), `owndev_m*` = fraud events on the sender's own device (13,657 / 13,777 / 13,193),
  `multi3_m*` = events with >= 3 fraud rows (696 / 704 / 695). Event ids come from `fraud_events(g = 48 h)`, itself a hypothesis:
  "single-row" means no other fraud-labelled row on the same sender / receiver key / device within 48 h.
- Time controls: `shuf_day` = timestamps permuted among all rows of the same calendar day (daily degree kept, order destroyed);
  `shuf_snd` = timestamps permuted among the rows of the same sender (sender's timestamp set kept, in/out order destroyed).
- Conditioning = the control cell is additionally split by bins of the closest existing columns (read from `scratch_gap/N_f1.parquet`).
- Files: `results.csv`, `results_class.csv` (s1), `auc.csv`, `info_cells.csv` (s4), `cond_allfraud.csv` (s5), `class_degree.csv` (s6); logs `s1.out`, `s4.out`, `s5.out`, `s6.out`.
- Pattern c uses a fixed 28 d lookback and is NaN before 29 July, so its freeze month is August (July has only 3 valid days).

## Headline
1. **Single-row / missed events: no detected difference for any of the 17 motifs or 26 columns.** `missed_m9` (n = 11,474 matched events):
   every |diff| <= 0.5 point, |z| <= 1.3, except `B1_sec` (see pattern b, a model-selection artefact). Univariate AUC vs same-month legit:
   0.489-0.519 for all 26 columns on `single_m7`, `single_m8`, 0.485-0.505 on `missed_m9` (`auc.csv`).
   With ~11-13k events per month the 2-SE resolution is ~0.5-0.7 point for motifs at 5-20% base rate, so an
   effect larger than that is not there for these predicates.
2. On fraud events as a whole the motifs do separate (positive control works: `caught_m9`, `multi3_m*`), and the separation is carried
   by multi-row events.
3. After conditioning on the closest existing columns: pattern a is **redundant**, pattern b is **redundant**, pattern c
   (`rel__c_snd_act`, `rel__c_snd_out_ratio`) keeps a residual on multi-row events in August and September under each conditioning separately, and none under the joint conditioning (UPDATE section).

## Ranking
| rank | pattern | status | key number |
|---|---|---|---|
| 1 | c. sender activity change vs own prior window (`rel__c_snd_act`, `rel__c_snd_out_ratio`) | SUPERSEDED by the UPDATE section (joint test: no detected difference; the residual is a plain 7 d sender count). Earlier text: promising but unconfirmed (multi-row events only); no detected difference on single-row / missed | multi-row events, after conditioning on `takeover__sess_n`, `burst__snd_c1800`, `ring__pair_n`, `ring__s4_np_int`: Aug +13.4 pts (z 8.1, n 790), Sep +11.9 pts (z 7.1, n 742); single-row events Aug +0.36 (z 1.2), Sep +0.06 (z 0.2) |
| 2 | a. ordered role transition (`rel__a_snd_prev_role`, `a_snd_in_run`, `a_rcv_*`) | redundant with `flow__sin_n24h` | all-fraud `A1d`: base Aug +0.89 (z 6.2), Sep +0.91 (z 6.3); after flow conditioning Aug -0.05 (z -0.5), Sep -0.08 (z -0.8) |
| 3 | b. secondary device + directional counterparty (`rel__b_*`) | redundant with `burst__devrcv_c300` / `burst__rs_n_device` / `burst__ds_n_sender`; owner edges empty | multi-row `B4`: base Aug +1.80 (z 4.8), Sep +1.67 (z 4.4); after burst-device conditioning Aug -0.05 (z -0.6), Sep +0.08 (z 0.9). Owner-receiver / owner-sender edges: 0.00-0.12% everywhere |

## Pattern c: receiver-versus-sender activity change relative to the account's own earlier activity
1. Hypothesized generation step: a fraud event injects several rows on an account that the background (legit) generator keeps at a
   steady low rate, so the account's event rate in the last 7 d jumps relative to its own previous 21 d; a legit heavy user has high
   counts in both windows.
2. Trace (label-free, `feat`): for account A at row time t, `i7,o7` = received / sent counts in [t-7d, t), `iP,oP` in [t-28d, t-7d).
   `c_snd_act = log((i7+o7+1) / ((iP+oP)/3 + 1))`, `c_snd_out_ratio = log((o7+1)/(oP/3+1))`, `c_snd_in_ratio`, `c_snd_dshare =
   i7/(i7+o7) - iP/(iP+oP)`, `c_snd_first_out`; same five for the P2P receiver (`c_rcv_*`, `c_rcv_first_in`). NaN when t-28d is before the data start.
   Motif `C5 = c_snd_act > 0.5`, `C7 = c_snd_out_ratio > 0.5`.
3. Alternative explanation: it is only absolute recent degree (already in `burst__snd_c*`, `takeover__sess_n`, `ring__s4_*`), or the
   event's own earlier rows inside the 7 d window (position in a burst the model already sees).
4. Tests (`s5_cond_allfraud.py`, `s6_class_degree.py`):
   - vs matched legit, base cells, all-fraud: `C5` Aug +2.47 (z 8.1, n 14,691), Sep +2.06 (z 6.7, n 14,033).
     Split: multi-row events Aug +22.2 (z 16.4, n 1,285), Sep +21.4 (z 15.4, n 1,195); single-row events Aug +0.58 (z 1.9), Sep +0.27 (z 0.9).
   - beyond absolute degree (cell additionally split by the exact recent count n7 = i7+o7, capped at 8): multi-row Aug +9.29 (z 8.7),
     Sep +9.33 (z 8.1); single-row Aug +0.20 (z 0.9), Sep +0.06 (z 0.3). So at equal recent degree the prior-window count still differs.
   - after the closest existing columns (`takeover__sess_n`, `burst__snd_c1800`, `ring__pair_n`, `ring__s4_np_int` bins): multi-row
     Aug +13.35 (z 8.1, n 790 of 1,285 with a control cell), Sep +11.89 (z 7.1, n 742); all-fraud Aug +1.10 (z 3.6), Sep +0.72 (z 2.4).
   - after flow + burst columns (`flow__sin_n24h`, `flow__rin_n12h`, `scam__rcv_in_15d`, `takeover__sess_n`, `burst__snd_c1800`):
     `multi3` Aug +20.7 (z 7.6, n 323), Sep +17.4 (z 6.4); `C7` Aug +18.8 (z 7.9), Sep +21.3 (z 8.8).
   - per class after ring / burst conditioning (Aug / Sep, all events of the class): layering +2.19 (z 2.3) / +2.28 (z 2.4);
     fake_cashback +2.44 (1.7) / +3.24 (2.2); refund +4.82 (3.5) / +1.06 (0.8); collusion +3.49 (2.4) / +0.38 (0.3);
     mule, velocity, phishing, qr_swap, sim_swap, ATO, loan: |z| < 1.3 both months.
   - shuffled time: `multi3_m8` `C5` real +24.3 (z 13.1), `shuf_day` +23.3 (z 12.7), `shuf_snd` +17.8 (z 9.3). Both shuffles keep the
     sender's 7 d / 21 d counts almost unchanged by construction, so they are NOT a discriminating null for this pattern (it is a
     window statistic, not an order statistic). The discriminating tests are the n7 and existing-column conditionings above.
   - receiver side (`C6 = c_rcv_in_ratio > 0.5`, `C2`, `C3`): base multi3 Aug +12.1 (z 8.2); after flow columns -0.62 (z -0.6):
     **redundant** with `flow__rin_n12h` / `scam__rcv_in_15d`. `C8` (no sender event in the prior 21 d) after ring / burst
     conditioning: multi-row Aug +2.26 (z 1.7), Sep +0.25 (z 0.2): no detected difference.
   - model-defined cohorts: `caught_m9` `C5` +10.96 (z 13.1; conditioned z 10.1); `missed_m9` +0.06 (z 0.2).
5. Freeze / check: thresholds (0.5) and windows (7 d / 21 d) were fixed before looking at any month; August is the freeze month,
   September agrees in sign and size for every multi-row number above. July is not usable (lookback).
6. Feature for agent C: `rel__c_snd_act`, `rel__c_snd_out_ratio`, `rel__c_snd_in_ratio`, `rel__c_snd_dshare`, `rel__c_snd_first_out`
   in `rel_feats.parquet`. Expected reach: multi-row events (about 8% of events, most already caught: `caught_m9` z 13), so any
   macro-F1 gain would come from class assignment / borderline multi-row rows, not from single-row missed events. Not scored by me.
   Caveats: (i) built at full density, while the production matrix is density-matched (benchmark D): the 7 d / 21 d counts include
   fraud rows, so check the column on D before use; (ii) October is a half month but its lookback reaches into September, so the
   column is defined there; (iii) cells with < 5 controls are dropped (n shown); (iv) the conditioning uses coarse bins of 4-5
   existing columns, not the 190-column model, so "residual" here is an upper bound on what a fit will see.

## Pattern a: directed, time-ordered role transitions
1. Generation step: a mule / layering account receives and then forwards; the ordered string "receive, then send" (and the receiver
   that was a sender just before) is the trace of a pass-through step, as opposed to an account that happens to do both in a day.
2. Trace: `a_snd_prev_role` (1 = the sender's immediately previous event in either role was a receive), `a_snd_prev_gap_log`,
   `a_snd_in_run` (consecutive receives immediately before this send, cap 5), `a_snd_str3` (ordered role string of the last 3
   events, base 3), and the mirror for the P2P receiver (`a_rcv_prev_role` 0 = previous event was a send, `a_rcv_out_run`, `a_rcv_str3`).
   Motifs: `A1d` = previous event is a receive within 24 h; `A3` = in-run >= 2; `A2`, `A4` receiver side.
3. Alternative: unordered co-occurrence of an inbound and an outbound in the same day, which `flow__sin_n24h` (count of P2P
   inbound to the sender in the previous 24 h, `feats/flow.py` section B) already encodes.
4. Tests:
   - all-fraud base: `A1d` Jul +0.92 (z 6.6), Aug +0.89 (z 6.2), Sep +0.91 (z 6.3); `multi3` Jul +12.2 (9.6), Aug +15.5 (11.3), Sep +12.9 (9.8).
   - conditioned on `flow__sin_n24h`, `flow__rin_n12h`, `scam__rcv_in_15d`: all-fraud Jul +0.09 (z 0.9), Aug -0.05 (z -0.5),
     Sep -0.08 (z -0.8); `multi3` Aug -1.37 (z -3.8), Sep -1.04 (z -2.4). `A3`: all-fraud Aug +0.36 (z 2.1), Sep +0.08 (z 0.4). `A2`: |z| <= 1.5.
   - shuffled time, `multi3_m8` `A1d`: real +14.65 (z 10.9), `shuf_day` +10.16 (z 7.8), `shuf_snd` +2.80 (z 3.2). Two thirds of the
     effect survives destroying intra-day order: it is same-day in/out co-occurrence, not order.
   - new information exists but carries no fraud signal: inside `flow__sin_n24h == 0` the new column splits rows 135k / 34k
     (previous role send / receive) with single-row fraud share 2.99% vs 3.03% (Aug) and missed share 2.31% vs 2.14% (Sep) (`info_cells.csv`).
   - single-row / missed: `A1d` single_m8 +0.02 (z 0.1), missed_m9 -0.01 (z -0.1); own-device single-row Aug +0.01, Sep +0.01.
5. Freeze / check: same sign and size in all three months before conditioning; zero in all three after.
6. Verdict: **redundant** with `flow__sin_n24h` (residual -0.05 / -0.08 point on 14-15k events per month). Columns are in
   `rel_feats.parquet` if agent C wants to confirm with a fit; I do not expect a gain.

## Pattern b: shared secondary device combined with directional counterparty relationships
1. Generation step: a fraud operator device (or a borrowed phone: `DV0<n>` is the phone of `AC<n>`) is used by several accounts that
   pay the same receiver, or the phone's owner has a prior payment relation with the receiver / sender (collusion, mule herding).
2. Trace: `b_sec` (device not the sender's own), `b_own_to_rcv`, `b_rcv_to_own`, `b_snd_to_own`, `b_own_to_snd` (a payment edge in
   that direction exists strictly earlier; owner = account with the device's number), `b_dev_nco_before` (other accounts that used the
   device earlier), `b_co_to_rcv` / `b_rcv_to_co` (number of earlier co-users of the device that had already paid this row's
   receiver / been paid by it). Motifs `B2` owner-receiver edge, `B3` sender-owner edge, `B4` co-user-receiver edge, `B5` co-user exists.
3. Alternative: plain device sharing inside a burst, already in `burst__devrcv_c300`, `burst__rs_n_device`, `burst__ds_n_sender`,
   `takeover__n_other_dev`; and for the missed cohort, selection by the model.
4. Tests:
   - owner edges are empty: `B3` 0.00% in every population and month; `B2` <= 0.12% (missed_m9 0.12% vs control 0.10%; caught 0.00%
     vs 0.04%). A borrowed phone's owner has no payment relation to the parties. Status: empty.
   - `B4`: multi-row events Aug +1.80 (z 4.8), Sep +1.67 (z 4.4); velocity_bot class Aug +1.34 (z 4.5), Sep +1.43 (z 4.5); every
     other class |diff| <= 0.21 (refund Sep -0.21, event rate 0.00%). After conditioning on `burst__devrcv_c300`, `burst__rs_n_device`, `burst__ds_n_sender`: multi-row
     Aug -0.05 (z -0.6), Sep +0.08 (z 0.9); velocity Aug -0.08 (z -0.9), Sep +0.34 (z 1.9). **Redundant.**
   - shuffled time, `multi3_m8` `B4`: real +3.37 (z 4.9), `shuf_day` +3.23 (z 4.8), `shuf_snd` +3.09 (z 4.7): the "earlier" direction
     adds nothing; it is co-occurrence.
   - `B1_sec` on `missed_m9`: 4.99% vs 7.92% control (-2.93, z -14.4), but on the model-independent single-row cohorts it is
     Jul -0.12, Aug -0.25 (z -1.1), Sep +0.09 (z 0.4). The September gap is created by the model (it catches foreign-device fraud and
     the legit fresh-phone region stays in the control), not by the generator. Not a usable signal.
   - single-row / missed: `B4` missed_m9 0.27% vs 0.23% (z 0.9); `B5` +0.21 (z 0.9).
5. Freeze / check: July / August / September agree (multi3 `B4` base +3.1 / +3.4 / +2.8 points).
6. Verdict: **redundant** (co-user-receiver) and **empty** (owner edges). No feature recommended.

## Observation that constrains all three (s3, s4)
For single-row fraud events the account neighbourhood matches the legit neighbourhood for every predicate tested here (role order,
role-share change, device co-use, owner / co-user edges) in July, August and September, and the row-type mix of missed events
equals the legit mix to within sampling noise for all 11 classes. This is a statement about these predicates only.

## Difference from earlier tests
- `notes/hop.md`, `notes/embed.md`: undirected adjacency / embeddings / hot-row linkage from model outputs on a model-defined cohort
  (p_legit > 0.9). Here: directed and time-ordered predicates, device ownership, own-history normalisation, label-only cohorts
  (single-row events, all fraud, own-device fraud) in three months, shuffled-time nulls, and conditioning on the closest column.
- `notes/art_acct.md`: static account attributes and other-base device shares by class. Here: owner / co-user payment edges in time order.

---
# UPDATE (written 08:02 IST) — supersedes the status of pattern c above; adds the items the addendum asked for

## Final ranking
| rank | pattern | final status |
|---|---|---|
| 1 | c. activity change relative to own history | **no detected difference** as defined (own-history part). A plain backward 7 d sender count is what carried the residual: **promising but unconfirmed**, and it drifts on October |
| 2 | a. ordered role transitions | **redundant** with `flow__sin_n24h` |
| 3 | b. secondary device + directional counterparty | **redundant** (co-user edge) / **empty** (owner edges) |
Single-row and missed events: no detected difference for any predicate of a, b, c.

## Pattern c, joint test (`s7_joint.py`, `joint.csv`)
Control cell = base cell x bins of `takeover__sess_n`, `burst__snd_c1800`, `ring__pair_n`, `ring__s4_np_int` x exact recent 7 d count
n7 (cap 8). All fraud-labelled events, one row per event:
| motif | group | Aug | Sep |
|---|---|---|---|
| `C5` (`c_snd_act > 0.5`) | all | +0.17 (z 0.7, n 13,247) | -0.03 (z -0.1, n 12,763) |
| `C5` | multi-row events | -0.18 (z -0.1, n 563) | +0.78 (z 0.5, n 543) |
| `C5` | single-row events | +0.19 (z 0.8, n 12,684) | -0.07 (z -0.3, n 12,220) |
| `C8` (no sender event in prior 21 d) | all | -0.30 (z -1.0) | -0.37 (z -1.2) |
| `C8` | multi-row events | +0.49 (z 0.3) | +1.23 (z 0.7) |
Per class: every |z| <= 2.0 for `C5` (largest sim_swap Aug -1.85, z -2.0, Sep -0.31).
Reading: the earlier separate survivals (+9.3 points under n7 alone; +13.4 / +11.9 under ring / burst bins alone) do not survive
together. The own-history normalisation (prior 21 d) adds nothing; what the ring / burst bins left over was the absolute count of the
sender's events in the previous 7 d. Limits: only 563 of 1,285 (Aug) and 543 of 1,195 (Sep) multi-row events have a control cell with
>= 5 legit rows, so the test speaks for the events that have legit look-alikes, which are the ones that matter for missed fraud.
A multi-row event is a burst by its label definition (>= 2 fraud rows within 48 h on a shared key), so a raised 7 d count on it is
partly definitional.

## Candidate handed to agent C (low prior): plain backward 7 d sender counts
- File `rel_n7.parquet` (`s8_n7.py`): `rel__snd_n7` (sender's events in [t-7d, t), both roles), `rel__snd_out7`, `rel__snd_in7`;
  all 1,824,642 rows, NaN before 8 July. Label-free, backward only.
- Evidence: multi-row events, `C5` after ring / burst bins without n7: Aug +13.35 (z 8.1, n 790), Sep +11.89 (z 7.1, n 742); gone when
  n7 is added. Closest existing columns: `burst__snd_c60/300/1800` (`feats/burst.py:72`, windows up to 30 min), `takeover__sess_n`
  (`feats/takeover.py:32`, 2 h session), `ring__s4_np_int` (`feats/ring.py:42-43`, symmetric 4 d window but only new-payee
  whole-rupee rows >= 1000), `ring__pair_n` (`feats/ring.py:80`, per pair). None is a general multi-day sender count.
- **Drift warning (measured):** month means of `rel__snd_n7`: Aug 1.136, Sep 1.144, **Oct 0.637**. This is the half-density of
  October legit rows described in `notes/drift.md` item 1. Use only through `density.run_feature_matched` (benchmark D), never at
  full density. Reach is multi-row events, which the model mostly catches already (`caught_m9` `C5` z 13.1), and single-row events
  show nothing (`single_m8` `C5` +0.58, z 1.9; `missed_m9` +0.06, z 0.2), so I expect little or no macro-F1 gain. Status: promising
  but unconfirmed; not scored.
- Not tested against `burst__ss_n` (sender sessions) or `ring__s2_*` / `ring__s10_*`.

## Shuffled-time control on the model-independent populations (`results.csv` from the 07:58 rerun of `s1_patterns.py`)
Numbers differ slightly from s5 / s6 because the one-row-per-event draw differs. diff in points (z).
| motif | population | real | conditioned (sin_n24h, rcv_in_15d, n_other_dev) | shuf_day | shuf_snd |
|---|---|---|---|---|---|
| `A1d` | allfraud Jul | +1.06 (7.4) | +0.08 (0.8) | +0.71 (5.1) | +0.38 (3.0) |
| `A1d` | allfraud Aug | +0.99 (6.9) | -0.05 (-0.5) | +0.76 (5.4) | -0.02 (-0.2) |
| `A1d` | allfraud Sep | +0.88 (6.1) | -0.07 (-0.7) | +0.66 (4.7) | +0.14 (1.1) |
| `A1d` | own-device fraud Aug / Sep | +0.99 (6.6) / +1.20 (7.6) | -0.04 (-0.4) / -0.05 (-0.4) | +0.70 (4.8) / +0.89 (5.8) | 0.00 (0.0) / +0.19 (1.4) |
| `A3` | allfraud Aug / Sep | +0.73 (4.3) / +0.46 (2.4) | +0.37 (2.2) / +0.06 (0.3) | +0.55 (3.3) / +0.41 (2.1) | +0.42 (2.7) / +0.61 (3.3) |
| `B4` | allfraud Jul / Aug / Sep | +0.15 (3.8) / +0.05 (1.3) / +0.17 (3.3) | +0.16 / +0.06 / +0.18 | +0.15 / +0.06 / +0.16 | +0.12 / +0.07 / +0.15 |
| `C5` | allfraud Aug / Sep | +2.46 (8.1) / +1.83 (6.0) | +1.69 (5.5) / +1.18 (3.8) | +2.41 (7.9) / +1.77 (5.8) | +1.45 (4.4) / +1.23 (3.6) |
| `C5` | own-device fraud Aug / Sep | +2.26 (7.1) / +1.97 (6.2) | +1.52 (4.8) / +1.20 (3.7) | +2.22 (7.0) / +1.80 (5.6) | +1.15 (3.3) / +0.98 (2.8) |
Reading:
- a: within-day shuffle keeps 70-75% of the `A1d` effect in all three months (order inside the day is not the carrier); within-sender
  shuffle removes it in August and September (z -0.2, 1.1). With conditioning on `flow__sin_n24h` it is zero in all three months.
- b: `B4` is identical under both shuffles: no order information.
- c: unchanged by the within-day shuffle, as expected for a 7 d / 21 d window statistic.

## Own-device (home-device) fraud, as the addendum asked (`cond_allfraud.csv`)
Own-device single-row fraud events vs matched legit (Aug n 12,673 / Sep n 12,160), base cells: `A1d` +0.01 (z 0.1) / +0.01 (z 0.1);
`A3` +0.13 (0.7) / +0.24 (1.1); `A2` -0.05 (-0.2) / -0.19 (-0.7); `C5` +0.65 (2.1) / +0.30 (1.0); `C7` +0.44 (1.4) / +0.11 (0.4);
`C4` +0.44 (2.0) / +0.10 (0.4); `C3` +0.42 (1.6) / -0.43 (-1.6); `C6` -0.41 (-2.7) / 0.00 (0.0); `C2` -0.11 / +0.12.
No motif keeps its sign with |z| >= 2 in both months: no detected difference.
Artefact warning: `B5` (-2.5, z -16) and `B4` (-0.12, z -8) on the own-device populations are produced by the population definition
(events restricted to own devices, controls not restricted). They are not findings.

## Reuse of earlier notes
- `notes/camo.md` item 3 already reports for a model-defined cohort: sender activity `snd_n` AUC 0.499, sender personal amount
  profile identical (.787 vs .791), fraud-label neighbourhood of sender / device / receiver at the legit rate. My single-row nulls
  agree with it and extend it to: label-only cohorts in three months, ordered role strings, own-history ratios, device-owner and
  co-user payment edges, shuffled-time nulls.
- `notes/drift.md` item 1 (October = 50% thinning of legit, fraud events complete) is the reason for the drift warning above.
- `notes/generator_experiment.md`, `notes/monthsep.md`: grepped for order / sequence / sender-activity tests, none equivalent found;
  not read in full (time).

## Code locations of the closest existing columns
`flow__sin_n24h` `feats/flow.py:72-75`; `flow__rin_n12h` `feats/flow.py:56-59`; `scam__rcv_in_15d` `feats/scam.py:119`;
`takeover__sess_n`, `takeover__n_other_dev` `feats/takeover.py:32, 78`; `burst__snd_c*` `feats/burst.py:72`; `burst__devrcv_c300`
`feats/burst.py:100`; `burst__ds_*` (device sessions, 600 s gap) `feats/burst.py:106`; `burst__rs_*` `feats/burst.py:111`;
`ring__s4_np_int` `feats/ring.py:42-43`; `ring__pair_n` `feats/ring.py:80`; `scam__rev_pair` `feats/scam.py:98-100`.

## Not done
- Benchmark D (`dec_D_sq6.npz`): every missed-cohort number is benchmark N only.
- July for pattern c (28 d lookback leaves 3 days); August is the only freeze month for c.
- No model fit of any kind; residuals are from coarse-bin conditioning on 3-5 columns, not from the 190-column model.
- Matched controls are cell-standardised rates over all legit rows of the cell, not 1:1 draws (the 1:1 draw exists only for the
  40-event timeline sample).
- Receiver-side ordered strings beyond the last 3 events, amount-conditioned transitions (for example receive X then send ~X),
  and merchant-side role histories were not tested.
- `fraud_events` gap sensitivity (48 h only).
