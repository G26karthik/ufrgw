# Agent C audit of the proposals from agents A and B (15 Nov 2026)

Clock times are from `date` (IST). File started 07:48.

## Bar a proposal must clear (from the brief)
(a) exact label-free feature or predicate (code); (b) support across multiple independent events (`fraud_events`, 48 h), not
rows; (c) measured rate among legit controls; (d) month-specific results; (e) how it differs from the earlier tests
(`notes/final_comparison.md`, `fresh_columns.md`, `fresh_errors.md`, `assumption_tests.md`, module notes).
Examples alone do not qualify. Every decisive number is re-run with `va_pred.py` on the whole held month: all legit rows that
the reference decodes legit are the controls (no sampling), shown both unmatched and matched on the reference P(legit).

## Addition to the bar (07:50, `scratch_atomic/ADDENDUM.md`, received from the coordinator while the controls were fitting)
Each mechanism needs all six items, and the verdict table below has one column per item:
1. hypothesized generation step; 2. exact observable trace (label-free predicate or statistic on original fields);
3. an alternative explanation for the same trace; 4. a test of both explanations against matched legit examples;
5. frozen on July / August, with its September prediction checked; 6. a general feature or executable rule with its
incremental whole-month macro-F1 (this harness).
Rejected regardless of effect size: a mechanism that only matches a distribution or explains selected rows (it must predict
beyond the examples that suggested it); any argument that uses the model's predictions or the inferred device zones as ground
truth (supplied July / August / September labels are the only labels); a cohort defined only by model confidence without the
same test on all fraud-labelled rows against legit rows. Earlier negative results are reused where equivalent
(`notes/generator_experiment.md`, `fresh_columns.md`, `fresh_errors.md`, `camo.md`, `drift.md`, `monthsep.md`, `relabel.md`).
No seed brute-forcing and no generic ID scans.
Consequence for `va_pred.py`: it reports the July comparison on ALL fraud-labelled rows against ALL legit rows (no model
involved) next to the August / September comparisons inside the decoded-legit rows.

## Decision rule for the scored fits (declared 07:48, before any mechanism was fitted or scored)
- Freeze on July / August only. Fold 0 (held August) is fitted and scored first for every mechanism; the frozen definition and
  the clock time are written below before any fold-1 (September) fit is started.
- Reference: N `PN_C`, D `PD_A_C` (seeds 0,1). Noise floor: the null mechanism `vanull` (one column of hash noise, same seeds)
  and on N the stored second seed pair `A_Cb2`.
- supported: September after the decoder above zero with the 5 % bound of the paired event bootstrap above zero on N AND on D,
  August after the decoder not negative on either, and the September gain larger than the null mechanism's absolute change.
- promising but unconfirmed: the September interval is above zero on one benchmark only, or D cannot stress the new column
  (a window / count column joined to D at full density, see `va_feat.py`), or August is negative.
- no detected difference: intervals cover zero. measurably worse: the upper (95 %) end of the 5..95 % interval below zero on N or D.
- not evaluated adequately: the fit or the invariance check could not be completed in time.
- September is reused development evidence (read more than 40 times before this round), not a fresh holdout.

## FREEZE (written 08:09 IST, before any September score of a mechanism was read; fold-1 fits were still in fold 0)
Two mechanisms, both named by A / B as their only scorable items (coordinator message 08:04). Definitions are the code files; nothing
in them is changed after this line.
Disclosure: before this freeze my audit runs had already printed September lines for both items (`va_pred.py` on the n7 columns and
on the C2 cells, `va_c2cell.py` top-k counts), because the instrument prints July, August and September together. No column, grid
or value of F was set from them: the n7 columns are B's file unchanged and the F grid is fixed here before any decode of the rule.
So "frozen on July / August" holds for the definitions, not for my having been blind to September.
Added 08:11, before any n7 score existed: contrast arm `van7full` on D (the same columns at full density), not a status carrier.
1. `n7` (agent B, `relationships/s8_n7.py` -> `verification/mech_n7.py`, output equal to B's `rel_n7.parquet` on all 1,824,642 rows):
   three columns, backward 7-day counts of the sender: `n7__snd_n7` (events in [t-7 d, t), both roles), `n7__snd_out7`, `n7__snd_in7`;
   NaN before data start + 7 d. Added to the 190 columns, same LightGBM recipe. N: built on the fold world. D: density-matched build
   (`va_feat.py` frame DM, halves of `density.halves(mlink=86400)`), as B requires; file tag `van7dm`.
   Free parameters chosen on August: none.
2. `c2` (agent A, C2 QR cell, `verification/va_c2rule.py`): reference P(qr_swap) x F on rows with QR to merchant, `qr_vpa_mismatch == 1`,
   non-integer amount; unchanged decoder. F is the one free parameter: grid {3, 10, 30}, chosen by August macro-F1 after the decoder
   on N (mean of the two out-of-half scores; ties -> smaller F). September is scored for the chosen F only.
Noise floors scored with the same code: `vanull` (one hash-noise column, seeds 0,1, N and D), `A_Cb2` (N, seeds 2,3 control).
Confirmation if time allows (declared now): `n7` with seeds 2,3 on N against `A_Cb2`.
Not given a slot: B's patterns a / b (B: redundant / empty; my whole-month AUCs below agree), B's pattern c ratios (B: superseded by
n7 in its joint test; my extension `mech_relcx.py` and the 26-column block `mech_rel.py` were built and pass the permutation check
but are not fitted), A's C1 (measurement, fields dead in October, verified below), A's `sd_first` (known fresh-phone block, below).

F chosen 08:21 from the cached August decodes only (`va_c2pick.py`, `log/c2pick.log`): N August after decoder, mean of halves:
reference .49779, F=3 .49799, F=10 .49794, F=30 .49686 -> **F = 3**. (D August, not used for the choice: .50036 / .50040 / .50021 /
.49970.) September is decoded for F = 3 only; the F = 10 and F = 30 September files exist on disk and were never decoded.

## Proposals: what A and B delivered, what I re-measured, verdict
Instrument for every re-measurement: `va_pred.py` = whole held month, ALL legit rows that the reference decodes legit as controls
(N August 506,485; September 490,326), unmatched and matched on the reference P(legit) (20 quantile bins of the missed rows), plus
July = all fraud-labelled rows against all legit rows with no model involved. Matching on baseline confidence moved two predicates: the QR cell
(lift 2.56 -> 1.98 Aug, 2.36 -> 1.98 Sep) and the COLLECT cell (1.20 -> 1.13, 1.13 -> 1.07); every other lift moved by at most .02
and every AUC by at most .003. So matching did not hide a signal; it removed part of two small ones. Logs: `log/pred_*.log`.

| # | proposal (agent) | six items delivered? | my re-measurement (N; D within noise unless stated) | verdict |
|---|---|---|---|---|
| A1 | `sd_first` = sender's first row on a device, not the sender's first row (A sec. 3) | 2, 3, 4 yes; 1 by reference to earlier notes; 5 yes; 6 "not a lever" | reproduced: missed fraud 68 / 12,014 (.0057) vs legit 15,442 / 506,485 (.0305) Aug, lift .19 (matched .19); Sep 51 rows, lift .14; D Aug .18, Sep .13. Split by device class (`mech_sd3.py`): the whole deficit is the legit fresh-phone block (device number >= 428455): 15,071 legit rows vs 10 fraud-labelled (Aug), 14,849 vs 11 (Sep), July 15,224 vs 11. On base devices `sd_first` has lift 1.03 (Aug) / 1.16 (Sep, 42 rows). The reference calls 0 (N) / 1 (D) of the 15,081 August fresh-phone rows fraud (`va_block.py`). | true trace of a legit-only generator step, already documented (`art_acct.md:25`, `fresh_errors.md` sec. 1, `typesplit_victim.md:28`). Marks rows that are legit and already decoded legit: nothing to recover and no false positive to remove. Not scored: a fit can only move the 10-11 fraud-labelled rows of the block. |
| A2 | C1: dropped-field traces J = `app_version < 4` & `ip_risk_score >= .441` count the fraud-path rows the pipeline loses (A sec. 6) | all six, item 6 as a development metric | reproduced: J among missed fraud 120 / 12,014 (.0100) vs legit .0067, lift 1.50 (matched 1.49), excess 40 rows Aug; Sep 106 rows, lift 1.40, excess 30. July all fraud vs legit: J .1137 vs .0071 (lift 16). October check without labels or model (`va_ipoct.py`): stamp rows (non-QR, mismatch 0) have mean ip month-rank .76 / .77 / .78 in Jul / Aug / Sep and .53 in the October test rows; share with rank >= .95 is .46-.48 vs .084; `app_version < 4` is .60-.63 vs .000. | supported as a measurement: about 110-165 fraud-path rows per month are decoded legit (0.6-0.9 % of fraud-labelled rows), which bounds what any fraud-path feature can recover. That range is A's conversion (excess / .273); I re-derived only the excess J rows among missed fraud: N 40 / 30, D 43 / 43 (A: 40 / 30 / 45 / 42). Not a feature: both fields lose the trace in October (same conclusion as `drift.md` item 2, here without model output). A fit would measure a signal that the test month does not carry, so none was run. |
| A3 | C2: QR-to-merchant, `qr_vpa_mismatch == 1`, non-integer amount; lower qr_swap threshold in the cell (A sec. 6) | 1-4 yes; 5 no (A: cells read on Aug + Sep together); 6 left to C | cell inside decoded-legit rows: Aug 30 fraud-labelled (18 qr_swap) vs 495 legit, fraud share .057 (base .023), lift 2.56, matched 1.98; Sep 30 (15 qr_swap) vs 537, lift 2.36, matched 1.98. July all rows: 31 fraud-labelled vs 545 legit, lift 1.48. Ranking the cell by the reference's own P(qr_swap) (`va_c2cell.py`): top 20 hold 6 (Aug) / 4 (Sep) qr_swap labels on N, 6 / 7 on D. COLLECT cell: lift 1.20 / 1.13, fraud share .028 / .026: empty. | below the bar on (b) / (c): about 18 qr_swap rows per month at 3-6 % purity; the best case of the threshold form is about 6 true rows per 20 flagged. Scored anyway as mechanism `c2` (no fit needed) so that a measured number exists. |
| A4 | C3: trace-free fraud labels vs trace-free legit, LightGBM on 45 original fields (A sec. 6) | n/a (negative result) | not re-run (no decision depends on it; equivalent earlier result `camo.md` item 2, AUC .5025). | accepted as reported: no detected difference. |
| A5 | 23 predicates and 20 numeric raw fields, missed events vs 5 matched controls (A sec. 3-5) | n/a (negative results) | only `sd_first` and J re-run (the two that differ). | accepted as reported: no detected difference. A's controls are matched on row type and sender activity but not on model confidence; my `va_pred.py` runs of A1-A3 use all legit rows with and without confidence matching and agree. |
| B1 | pattern a, ordered role transitions, 8 columns (B) | all six; B's verdict: redundant with `flow__sin_n24h` | all 26 `rel__*` columns rebuilt from B's code (`mech_rel.py`): equal to B's `rel_feats.parquet` on every row; fold-world build equal to the all-rows build on every scored row (4 Oct-dated train rows differ), so the columns respect the fold's history. Whole-month AUC missed fraud vs legit: Aug .494-.503, Sep .495-.502 (matched within .002). July all fraud vs legit: `a_rcv_prev_gap_log` .385 (mule .215), `a_rcv_prev_role` .588 (mule .709). | agree with B: signal exists on fraud as a whole and none inside decoded-legit rows. B's conditional test (zero after `flow__sin_n24h`) was not re-run; no slot. |
| B2 | pattern b, secondary device + directional counterparty edges, 8 columns (B) | all six; B's verdict: redundant / empty | AUC Aug .485-.501, Sep .486-.501; July velocity_bot .68-.69 on three columns; `b_sec` ATO .613, sim .542; the rest .50. `b_sec` .485 is the fresh-phone block again (A1). | agree with B. No slot. |
| B3 | pattern c, sender / receiver activity change against own earlier window, 10 columns (B) | all six; B's final verdict: own-history part no detected difference, superseded by n7 | AUC Aug .491-.516, Sep .495-.503; July .47-.57. B's columns are NaN before 29 July, so a July fit cannot use them; `mech_relcx.py` (my extension, equal to B's columns from 29 July on) was built and not fitted. | B's joint test was not re-run. No slot (superseded by B itself). |
| B4 | `n7`: backward 7-day sender counts, 3 columns (B, final candidate) | 1-5 for pattern c; for n7 itself only the joint-test residual on multi-row events; B: "low prior", drift in October (mean 1.14 -> .64) | `mech_n7.py` equals B's file on every row; density-matched build mean .577. AUC missed fraud vs legit: Aug .506 (matched .505), Sep .500 (.4995), per class .483-.520; July all fraud vs legit .537 (ATO .584, refund .583, qr_swap .489). | below the bar on (b): no effect on missed events; the residual is on multi-row events that are caught already. Not new in kind: sender multi-day windows exist in `ring` (+-2 / 4 / 10 d) and `scam` (+-3 / 15 d) for restricted row types (`representation/field_coverage.md`), and `camo.md` item 3 found sender activity identical on generic rows. Scored anyway as mechanism `n7`. |
| A6 | section 8 (added by A 08:13): are trace-free fraud labels placed on chosen entities / moments? plus the label-informed predicate "P2P receiver held a fraud label in an earlier month" (8.5) | hypothesis, alternative, shuffle null, freeze Jul -> Aug, check Aug -> Sep: yes; predicate flagged label-informed by A | `va_recur.py` (whole held month, labels of earlier months only; `log/recur.log`): predicate rows 5,435 (Aug) / 9,712 (Sep), fraud-labelled 1,351 (.249) / 2,251 (.232) = A's numbers. Inside decoded-legit rows: 98 fraud-labelled of 4,007 = .0245 (base .0232) Aug; 172 of 7,353 = .0234 (base .0231) Sep; D .0255 / .0251. Inside decoded-fraud rows where the receiver's earlier majority class differs from the decoded class (N: 1,082 Aug, 1,733 Sep): decoded class right 812 / 1,323, earlier class right 23 / 30. | A's own reading confirmed: no concentration in the trace-free cohort; the receiver recurrence is fraud-path structure that the decode already catches, and the earlier class does not help typing. Equivalent earlier result: `art_acct.md:19-23`. Third item, no slot, not scored: the two measured upper bounds (no lift in decoded-legit rows, 23-30 rows where the earlier class is right against 812-1,323 where the decode is right) leave nothing a fit could gain. |
| B5 | B's joint test (`relationships/s7_joint.py`), the test that moved B's candidate from the pattern-c ratios to n7 | n/a | re-run from B's source with the csv redirected (`B_joint_rerun.py`): all 56 rows equal B's `joint.csv` (max abs difference of `diff` and `z` 0.0, same n). | reproduced exactly. |

Clock note: A's stub header said 07:48 when the clock read 07:46 (A corrected it); B's stub said "07:46" at 07:41 file time. No
result depends on it.

Addendum checks: no proposal uses device zones or model predictions as truth. Cohorts: A1-A3 and B1-B4 each carry a model-free
comparison (July, all fraud-labelled rows vs all legit rows) next to the model-defined one. A's C1 arithmetic uses supplied labels
and the label-free stamp only. The one place where the model defines the effect is `b_sec` / `sd_first` (the model catches
foreign-device fraud, the legit fresh-phone block stays in the controls); both agents say so themselves.

## Outcome of the audit (08:26)
- No proposal from A or B met the full bar. Nothing they found separates missed fraud from legit rows on more than a few dozen
  rows per month: every predicate and column is at lift .9-1.1 / AUC .485-.52 inside the decoded-legit rows, with or without matching
  on the reference confidence, except three: the fresh-phone block (legit-only, already decoded legit), the dropped-field traces
  (about 40 excess rows per month, dead in October) and the QR cell (about 18 qr_swap rows per month at 3 % purity).
- Scored anyway, as instructed, so that measured whole-month numbers exist: `n7` (B) and `c2` (A). Results and status words:
  `RESULTS.md`. `n7`: by the letter of the rule frozen at 07:48, "promising but unconfirmed" (one reading, N seeds 0,1 September, +.0012 with the interval above zero); after the controls declared at the freeze (noise column +.0007 on the same reading, second seed pair +.0005 covering zero, D density-matched -.0002), "no detected difference". Both readings are reported; no candidate should be built from it. `c2`: measurably worse on N by .0003, no detected difference on D.
- Not verified by me: A's C3 classifier (AUC .505 / .510), A's 20 numeric fields and 21 of the 23 predicates, A's 31 raw
  aggregates and 190-column per-class AUC scan (`02_trace.py`), A's raw-text format scan, B's shuffled-time controls, B's
  conditional tests for patterns a and b, B's row-type-mix table. None of them decided a slot.
- Verified by re-running or re-deriving: A `sd_first` (and its device-class split, new here), A J-trace excess, A C2 cells,
  A 8.5 predicate counts, the October fate of the ip / app traces (label-free, new here), B's 26 columns (rebuilt, equal),
  B's n7 file (rebuilt, equal), B's joint test (equal), whole-month AUCs of all 29 B columns.
- Logs of the audit runs: `log/pred_A_sd.log`, `log/pred_A_c2.log`, `log/pred_B_rel_N.log`, `log/pred_B_n7.log`, `log/block.log`,
  `log/c2cell.log`, `log/ipoct.log`, `log/recur.log`, `log/B_joint_rerun.log`, `log/n7_world_vs_all.log`. The `sd`, block, c2cell
  and ipoct runs were first read on the terminal and re-run into these files at 08:32 with unchanged code and data.
  `log/pred_B_n7.log` was re-run at 08:16 after a fix in `va_pred.py` (NaN no longer counts as satisfying `>=`; it affected only
  the July predicate lines of n7, which have NaN in the first 7 days; no number quoted in this file came from those lines).
