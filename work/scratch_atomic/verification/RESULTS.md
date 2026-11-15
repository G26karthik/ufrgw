# Agent C results: incremental whole-month scoring (15 Nov 2026)

All numbers: whole held month (August 526,852 rows, September 509,996 rows), supplied labels, 12-class macro-F1.
`pre` = after the per-class bias only; `q` = after takeover forcing, device zones and the bait rule.
August = bias tuned on one event-grouped half, the other half decoded, halves swapped (same split as `scratch_gap/aug_dec.py`);
September = bias tuned on all of August (same as `scratch_gap/bench_eval.py`). September is reused development evidence.
Interval = paired event bootstrap, 200 draws, 5 %..95 % (fraud events whole, legit rows singly). It covers event sampling only,
not seed noise, not bias-tuning noise, not the reuse of September.
N and D are separate benchmarks: N = held labels hidden, full density; D = production density-matched matrix (label-informed
density stress test, not independent truth).

## Harness and its checks
| file | what it does |
|---|---|
| `va_fit.py` | benchmark table + new columns, recipe of `scratch_wt/wt.py` arm `wC` (same rows, keep mask, seeds 0,1, `T.LGB`, 150 rounds, lr .05, weights 5 / 1, 4 threads) |
| `va_feat.py` | builds a mechanism's columns on every frame and runs the label-permutation check; frames `N_f0`, `N_f1`, `ALL`, `DM` |
| `va_score.py` | decode + all counts (`report`, `summary`, `check`); decodes cached in `dec/` |
| `va_pred.py` | audit instrument: predicate / column inside decoded-legit rows, all legit rows as controls, with and without matching on P(legit); July = all fraud vs all legit |
| `mech_*.py` | one `build(tx)` per mechanism or audit column set |

Checks (logs in `log/`):
- Control `vaC` equals the stored references bit for bit: N `PN_C` August and September, D `PD_A_C` August and September
  (`np.array_equal` True, max abs diff 0; `log/check2.log`).
  Those four files were fitted before `va_fit.py` gained the `SEED0` / `DFEAT` / `TAG` options; the patched script was re-checked at
  08:33 on N fold 0 (`PN_vaCchk_aug.npy` equals `PN_C_aug.npy` bit for bit, `log/check3.log`). D and fold 1 were not re-fitted with the patched script.
- My September decode of the references equals `scratch_gap/dec_N_wt.npz` (`C_pre`, `C_q`) and `dec_D_wt.npz` (`A_C_pre`, `A_C_q`)
  exactly; my August half scores equal `scratch_gap/aug_dec_*.log` to 4 decimals (N .4696 / .4978, D .4716 / .5004; `log/check1.log`).
- Label-permutation invariance (`va_feat.py`): the true labels are restored on every row of the frame and permuted across ALL its
  rows, the columns are rebuilt and must be bit-identical (polars `equals` and numpy buffers with NaN == NaN). Passed for `n7`
  (N_f0, N_f1, ALL, DM), `null`, `rel` (26 columns) and `relcx` (N_f0, N_f1, ALL), and the audit column sets `c2`, `sd`, `sd2`, `sd3`
  (ALL; `sd` also N_f0). Limit of the check: a column that read `is_test` directly would pass it; none of the mechanism files does. A deliberately label-dependent column
  (`mech_leak.py`, sender's fraud-label share) is rejected by the same check.
- Information convention. N follows `scratch_gap/nat_data.py` / `bench_nat.world`: the frame holds only the rows of the fit months
  and the held month (no later month, no October test row), held rows are visible but unlabelled, one build at full density.
  D follows `scratch_gap/d_data.py` = production: one build on all train + test rows; for window / count columns the production
  density split (`density.halves(mlink=86400)`, label-informed split, label-free values) is used (frame `DM`).
  `n7` counts only rows strictly earlier than the row, so its fold-world values equal its all-rows values on every scored row.
- Events: fraud rows by `clean_labels.fraud_events(g = 48 h)`; legit rows (new false positives) by the same linking (sender,
  receiver key, device, 48 h) applied to the row set itself.

## Status per mechanism (September after the decoder; August is the freeze month)

| mechanism | N August | N September (5..95 %) | D August | D September (5..95 %) | status |
|---|---|---|---|---|---|
| noise-column control `vanull` (one hash-noise column, seeds 0,1) | +.0005 | +.0007 (-.0002..+.0016) | -.0012 | +.0006 (-.0004..+.0016) | noise floor of "add a column, same seeds" |
| seed-pair control `A_Cb2` (N, seeds 2,3 against seeds 0,1, no new column) | +.0014 | +.0002 (-.0015..+.0025) | - | - | noise floor of the seed pair |
| `n7` = B's backward 7-day sender counts (3 columns), seeds 0,1; D = density-matched columns | +.0000 | +.0012 (+.0002..+.0023) | -.0005 | -.0002 (-.0012..+.0007) | letter of the frozen rule: **promising but unconfirmed**; after the declared controls: **no detected difference**; not competitive, build nothing from it (see reading) |
| `n7`, seeds 2,3, N, against `A_Cb2` | +.0004 | +.0005 (-.0017..+.0021) | - | - | confirmation: interval covers zero |
| `n7` against the noise-column control (same seeds) | -.0005 | +.0006 (-.0005..+.0018) | +.0007 | -.0008 (-.0018..+.0003) | intervals cover zero |
| `n7` at full density on D (contrast arm, not a status carrier) | - | - | -.0005 | -.0008 (-.0017..+.0001) | leaning worse |
| `c2` = A's QR cell, P(qr_swap) x 3 (F chosen on August N from 3 / 10 / 30) | +.0002 | -.0003 (-.0004..-.0001) | +.0000 | -.0001 (-.0004..+.0002) | **measurably worse** on N by .0003, no detected difference on D |

Reading of `n7`:
- By the letter of the rule frozen at 07:48 the single N seeds-0,1 September reading (interval above zero on one benchmark)
  would be "promising but unconfirmed". The controls declared at the freeze remove it: the noise column moves the same reading by
  +.0007, the paired comparison n7 against the noise column covers zero on N and is negative on D, the second seed pair on N
  covers zero, and D (density-matched, the only build that matches October) is -.0002. August is +.0000 / -.0005.
  The project's earlier convention (`notes/assumption_tests.md`: both N seed pairs above zero for "promising") gives the same
  answer. Status used: no detected difference. Not competitive; no candidate should be built from it.
- Row counts are at the noise-column level. N September after the decoder: n7 changes 91 rows (45 corrected, 37 newly wrong;
  18 missed fraud rows recovered in 14 events, 13 new false positives, 24 correct fraud rows damaged in 23 events); the noise
  column changes 78 rows (16 recovered, 14 new false positives, 21 damaged); the seed-pair control 124 rows (18 / 19 / 36).
- Gain share of the new columns in the fits: `n7__snd_n7` .0025-.0044 (rank 56-81 of 193), `n7__snd_out7` .0019-.0042,
  `n7__snd_in7` .0015-.0020. The noise column takes .0089-.0138 (rank 19-26 of 191). The two are not comparable in size (a
  continuous noise column offers far more split points than a small count), but it shows that gain share alone proves nothing here.
- What this fit does not test: B's own-history ratio (pattern c as first defined) and the 26-column block; B's joint test, which
  replaced them by n7, reproduces exactly (`B_joint_rerun.py`). `mech_relcx.py` / `mech_rel.py` are built and unfitted.

Reading of `c2`:
- The rule moves 13-17 rows in August and 42-44 in September. September after the decoder: N recovers 6 missed fraud rows (6
  events) for 37 new false positives; D recovers 10 for 30. August: N 0 recovered / 3 new false positives (+.0002 comes from 7
  corrected against 5 newly wrong rows after the bias re-tune), D 4 / 10.
- F = 10 and F = 30 in August: N +.0002 and -.0009, D -.0001 and -.0007 (whole-month numbers; the choice used the mean of halves).
- The cell holds about 18 qr_swap-labelled rows among about 525 decoded-legit rows per month (3 %), so no threshold form of this
  cell can pay; the measured result agrees with A's own expectation.

## Summary tables (all counts; `pre` = bias only, `q` = after the decoder rules)
The `reference` column was added to this table after the run (the run printed the reference's macro-F1 only; the script checks that each named reference has one macro-F1 per benchmark, month and stage). `va_score.py summary` now prints the column itself; `log/summary_check.md` is one row re-run with the current code.
Columns: rows changed / corrected / newly wrong against the reference; missed fraud recovered = reference legit, truth fraud,
variant equals truth; new false positives = reference legit, truth legit, variant fraud; false positives removed; correct fraud
damaged = reference equals truth (fraud), variant differs (in brackets: sent to legit). Events as defined above.
The reference changes by block: `C` (N, seeds 0,1), `A_Cb2` (N, seeds 2,3), `vanull` (noise-column control), `A_C` (D, seeds 0,1).
`vac2f10`, `vac2f30` appear for August only (never decoded on September).

| benchmark | month | stage | variant | reference | macro-F1 ref | macro-F1 variant | change (event bootstrap 5..95 %) | rows changed / corrected / newly wrong | missed fraud recovered rows / events | new false positives rows / events | false positives removed | correct fraud damaged rows / events (to legit) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | aug | pre | vanull | C | 0.4698 | 0.4707 | +0.0008 (+0.0000..+0.0017) | 101 / 57 / 32 | 15 / 13 | 7 / 7 | 22 / 21 | 25 / 25 (11 / 11) |
| N | aug | q | vanull | C | 0.4980 | 0.4985 | +0.0005 (-0.0002..+0.0012) | 92 / 53 / 27 | 14 / 12 | 4 / 4 | 22 / 21 | 23 / 23 (10 / 10) |
| N | aug | pre | A_Cb2 | C | 0.4698 | 0.4715 | +0.0016 (+0.0002..+0.0031) | 162 / 71 / 66 | 25 / 23 | 22 / 22 | 17 / 17 | 44 / 33 (23 / 12) |
| N | aug | q | A_Cb2 | C | 0.4980 | 0.4995 | +0.0014 (+0.0004..+0.0026) | 134 / 63 / 48 | 24 / 21 | 22 / 22 | 16 / 16 | 26 / 26 (9 / 9) |
| N | aug | pre | van7 | C | 0.4698 | 0.4706 | +0.0007 (-0.0004..+0.0019) | 123 / 60 / 41 | 7 / 7 | 0 / 0 | 27 / 26 | 41 / 38 (22 / 22) |
| N | aug | q | van7 | C | 0.4980 | 0.4980 | +0.0000 (-0.0010..+0.0012) | 106 / 48 / 37 | 5 / 5 | 0 / 0 | 23 / 22 | 37 / 34 (19 / 19) |
| N | aug | pre | vac2f3 | C | 0.4698 | 0.4700 | +0.0002 (-0.0001..+0.0005) | 13 / 7 / 5 | 0 / 0 | 3 / 3 | 2 / 2 | 2 / 2 (1 / 1) |
| N | aug | q | vac2f3 | C | 0.4980 | 0.4982 | +0.0002 (-0.0001..+0.0005) | 13 / 7 / 5 | 0 / 0 | 3 / 3 | 2 / 2 | 2 / 2 (1 / 1) |
| N | sep | pre | vanull | C | 0.4815 | 0.4823 | +0.0007 (-0.0002..+0.0017) | 81 / 33 / 37 | 17 / 17 | 16 / 16 | 2 / 2 | 21 / 20 (8 / 8) |
| N | sep | q | vanull | C | 0.5070 | 0.5077 | +0.0007 (-0.0002..+0.0016) | 78 / 32 / 35 | 16 / 16 | 14 / 14 | 2 / 2 | 21 / 20 (7 / 7) |
| N | sep | pre | A_Cb2 | C | 0.4815 | 0.4821 | +0.0006 (-0.0012..+0.0030) | 127 / 58 / 54 | 19 / 18 | 20 / 20 | 5 / 5 | 34 / 29 (8 / 6) |
| N | sep | q | A_Cb2 | C | 0.5070 | 0.5072 | +0.0002 (-0.0015..+0.0025) | 124 / 55 / 55 | 18 / 17 | 19 / 19 | 5 / 5 | 36 / 31 (8 / 6) |
| N | sep | pre | van7 | C | 0.4815 | 0.4830 | +0.0014 (+0.0002..+0.0026) | 99 / 49 / 40 | 18 / 14 | 14 / 14 | 16 / 16 | 26 / 25 (18 / 18) |
| N | sep | q | van7 | C | 0.5070 | 0.5082 | +0.0012 (+0.0002..+0.0023) | 91 / 45 / 37 | 18 / 14 | 13 / 13 | 16 / 16 | 24 / 23 (17 / 17) |
| N | sep | pre | vac2f3 | C | 0.4815 | 0.4813 | -0.0002 (-0.0004..-0.0000) | 44 / 6 / 37 | 6 / 6 | 37 / 37 | 0 / 0 | 0 / 0 (0 / 0) |
| N | sep | q | vac2f3 | C | 0.5070 | 0.5067 | -0.0003 (-0.0004..-0.0001) | 44 / 6 / 37 | 6 / 6 | 37 / 37 | 0 / 0 | 0 / 0 (0 / 0) |
| N | aug | pre | vac2f10 | C | 0.4698 | 0.4700 | +0.0002 (-0.0003..+0.0007) | 56 / 22 / 32 | 10 / 10 | 20 / 20 | 4 / 4 | 12 / 11 (7 / 7) |
| N | aug | q | vac2f10 | C | 0.4980 | 0.4982 | +0.0002 (-0.0003..+0.0007) | 54 / 21 / 31 | 9 / 9 | 20 / 20 | 4 / 4 | 11 / 10 (5 / 5) |
| N | aug | pre | vac2f30 | C | 0.4698 | 0.4689 | -0.0010 (-0.0016..-0.0003) | 146 / 36 / 101 | 16 / 16 | 61 / 61 | 12 / 11 | 40 / 40 (38 / 38) |
| N | aug | q | vac2f30 | C | 0.4980 | 0.4971 | -0.0009 (-0.0015..-0.0003) | 142 / 35 / 99 | 15 / 15 | 61 / 61 | 12 / 11 | 38 / 38 (35 / 35) |
| N | aug | pre | van7b2 | A_Cb2 | 0.4715 | 0.4709 | -0.0006 (-0.0020..+0.0011) | 126 / 62 / 44 | 19 / 15 | 4 / 4 | 12 / 12 | 40 / 35 (15 / 15) |
| N | aug | q | van7b2 | A_Cb2 | 0.4995 | 0.4999 | +0.0004 (-0.0010..+0.0017) | 103 / 52 / 33 | 12 / 12 | 4 / 4 | 11 / 11 | 29 / 27 (13 / 13) |
| N | sep | pre | van7b2 | A_Cb2 | 0.4821 | 0.4829 | +0.0007 (-0.0017..+0.0026) | 132 / 47 / 68 | 17 / 12 | 14 / 14 | 7 / 7 | 54 / 25 (8 / 7) |
| N | sep | q | van7b2 | A_Cb2 | 0.5072 | 0.5077 | +0.0005 (-0.0017..+0.0021) | 128 / 45 / 69 | 19 / 14 | 15 / 15 | 5 / 5 | 54 / 24 (8 / 7) |
| N | aug | pre | van7 | vanull | 0.4707 | 0.4706 | -0.0001 (-0.0013..+0.0010) | 97 / 38 / 44 | 4 / 4 | 1 / 1 | 13 / 13 | 43 / 39 (25 / 24) |
| N | aug | q | van7 | vanull | 0.4985 | 0.4980 | -0.0005 (-0.0015..+0.0005) | 83 / 27 / 42 | 3 / 3 | 1 / 1 | 6 / 6 | 41 / 37 (23 / 22) |
| N | sep | pre | van7 | vanull | 0.4823 | 0.4830 | +0.0007 (-0.0006..+0.0019) | 124 / 62 / 49 | 18 / 15 | 12 / 12 | 28 / 28 | 37 / 37 (26 / 26) |
| N | sep | q | van7 | vanull | 0.5077 | 0.5082 | +0.0006 (-0.0005..+0.0018) | 115 / 57 / 46 | 17 / 14 | 12 / 12 | 27 / 27 | 34 / 34 (24 / 24) |
| D | aug | pre | vanull | A_C | 0.4719 | 0.4707 | -0.0012 (-0.0028..+0.0002) | 136 / 46 / 71 | 5 / 5 | 19 / 19 | 16 / 16 | 52 / 41 (33 / 28) |
| D | aug | q | vanull | A_C | 0.5006 | 0.4994 | -0.0012 (-0.0025..+0.0001) | 127 / 41 / 70 | 5 / 5 | 18 / 18 | 14 / 14 | 52 / 42 (32 / 27) |
| D | aug | pre | van7dm | A_C | 0.4719 | 0.4709 | -0.0009 (-0.0024..+0.0008) | 129 / 58 / 57 | 3 / 3 | 6 / 6 | 25 / 25 | 51 / 45 (33 / 30) |
| D | aug | q | van7dm | A_C | 0.5006 | 0.5001 | -0.0005 (-0.0019..+0.0012) | 117 / 52 / 52 | 3 / 3 | 6 / 6 | 21 / 21 | 46 / 42 (27 / 25) |
| D | aug | pre | van7full | A_C | 0.4719 | 0.4714 | -0.0004 (-0.0018..+0.0010) | 109 / 46 / 49 | 3 / 3 | 12 / 12 | 16 / 16 | 37 / 35 (22 / 21) |
| D | aug | q | van7full | A_C | 0.5006 | 0.5001 | -0.0005 (-0.0017..+0.0008) | 100 / 40 / 48 | 3 / 3 | 11 / 11 | 13 / 13 | 37 / 36 (21 / 20) |
| D | aug | pre | vac2f3 | A_C | 0.4719 | 0.4719 | +0.0000 (-0.0002..+0.0003) | 17 / 4 / 11 | 4 / 4 | 10 / 10 | 0 / 0 | 1 / 1 (0 / 0) |
| D | aug | q | vac2f3 | A_C | 0.5006 | 0.5006 | +0.0000 (-0.0002..+0.0003) | 17 / 4 / 11 | 4 / 4 | 10 / 10 | 0 / 0 | 1 / 1 (0 / 0) |
| D | sep | pre | vanull | A_C | 0.4824 | 0.4830 | +0.0005 (-0.0006..+0.0016) | 105 / 48 / 43 | 10 / 9 | 10 / 10 | 19 / 19 | 33 / 31 (20 / 20) |
| D | sep | q | vanull | A_C | 0.5078 | 0.5084 | +0.0006 (-0.0004..+0.0016) | 96 / 46 / 38 | 10 / 9 | 7 / 7 | 18 / 18 | 31 / 29 (18 / 18) |
| D | sep | pre | van7dm | A_C | 0.4824 | 0.4822 | -0.0002 (-0.0013..+0.0009) | 71 / 29 / 31 | 5 / 5 | 5 / 5 | 10 / 10 | 26 / 21 (8 / 7) |
| D | sep | q | van7dm | A_C | 0.5078 | 0.5076 | -0.0002 (-0.0012..+0.0007) | 61 / 25 / 28 | 4 / 4 | 5 / 5 | 9 / 9 | 23 / 19 (7 / 6) |
| D | sep | pre | van7full | A_C | 0.4824 | 0.4812 | -0.0013 (-0.0025..-0.0002) | 84 / 31 / 42 | 4 / 4 | 7 / 7 | 11 / 11 | 35 / 31 (12 / 12) |
| D | sep | q | van7full | A_C | 0.5078 | 0.5070 | -0.0008 (-0.0017..+0.0001) | 70 / 27 / 35 | 3 / 3 | 6 / 6 | 10 / 10 | 29 / 26 (11 / 11) |
| D | sep | pre | vac2f3 | A_C | 0.4824 | 0.4824 | -0.0001 (-0.0004..+0.0002) | 42 / 10 / 31 | 10 / 10 | 30 / 30 | 0 / 0 | 1 / 1 (0 / 0) |
| D | sep | q | vac2f3 | A_C | 0.5078 | 0.5078 | -0.0001 (-0.0004..+0.0002) | 42 / 10 / 31 | 10 / 10 | 30 / 30 | 0 / 0 | 1 / 1 (0 / 0) |
| D | aug | pre | vac2f10 | A_C | 0.4719 | 0.4717 | -0.0001 (-0.0003..+0.0001) | 26 / 7 / 18 | 5 / 5 | 15 / 15 | 2 / 2 | 3 / 3 (2 / 2) |
| D | aug | q | vac2f10 | A_C | 0.5006 | 0.5005 | -0.0001 (-0.0003..+0.0000) | 25 / 6 / 18 | 4 / 4 | 15 / 15 | 2 / 2 | 3 / 3 (2 / 2) |
| D | aug | pre | vac2f30 | A_C | 0.4719 | 0.4712 | -0.0006 (-0.0012..-0.0001) | 96 / 28 / 65 | 11 / 11 | 38 / 38 | 13 / 13 | 27 / 27 (21 / 21) |
| D | aug | q | vac2f30 | A_C | 0.5006 | 0.4999 | -0.0007 (-0.0013..-0.0002) | 93 / 27 / 64 | 10 / 10 | 38 / 38 | 13 / 13 | 26 / 26 (20 / 20) |
| D | aug | pre | van7dm | vanull | 0.4707 | 0.4709 | +0.0003 (-0.0008..+0.0015) | 145 / 77 / 51 | 19 / 17 | 12 / 12 | 34 / 33 | 39 / 35 (21 / 21) |
| D | aug | q | van7dm | vanull | 0.4994 | 0.5001 | +0.0007 (-0.0003..+0.0019) | 137 / 76 / 47 | 22 / 20 | 12 / 12 | 31 / 30 | 35 / 31 (19 / 19) |
| D | sep | pre | van7dm | vanull | 0.4830 | 0.4822 | -0.0007 (-0.0019..+0.0004) | 134 / 55 / 62 | 22 / 20 | 18 / 18 | 14 / 14 | 44 / 36 (14 / 13) |
| D | sep | q | van7dm | vanull | 0.5084 | 0.5076 | -0.0008 (-0.0018..+0.0003) | 115 / 46 / 57 | 19 / 18 | 17 / 17 | 10 / 10 | 40 / 33 (13 / 12) |

## Per-class precision / recall / F1 (reference against variant; `pre` and `q`; August then September; N then D)
Full reports with every block: `log/score_*.log`. Blocks below: n7 (status carriers), c2 (F = 3), and the noise-column control after the decoder.

```
N aug van7 vs C [pre] macro-F1 0.4706 vs 0.4698 = +0.0007 (event bootstrap 5-95 %: -0.0004, +0.0019) | mean of halves 0.4703 vs 0.4696
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.976  0.998  0.987  518912 |  0.976  0.998  0.987  518953 | -0.000 +0.000 +0.0000
mule_account            3448 |  0.734  0.256  0.380    1203 |  0.734  0.257  0.381    1207 | +0.000 +0.001 +0.0010
velocity_bot            2612 |  0.841  0.312  0.455     968 |  0.841  0.312  0.455     968 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.753  0.415  0.535    1275 |  0.753  0.416  0.536    1277 | -0.000 +0.000 +0.0003
sim_swap                1393 |  0.630  0.261  0.369     578 |  0.627  0.257  0.365     571 | -0.003 -0.004 -0.0048
account_takeover        1333 |  0.603  0.245  0.349     542 |  0.607  0.246  0.350     540 | +0.004 +0.001 +0.0014
phishing_collect        2254 |  0.742  0.331  0.457    1004 |  0.757  0.326  0.455     970 | +0.015 -0.005 -0.0020
qr_swap                 2489 |  0.813  0.354  0.494    1085 |  0.831  0.351  0.494    1052 | +0.018 -0.003 +0.0001
refund_abuse            1115 |  0.843  0.370  0.514     489 |  0.841  0.370  0.513     490 | -0.002 +0.000 -0.0003
fake_cashback            811 |  0.603  0.224  0.327     302 |  0.596  0.242  0.344     329 | -0.007 +0.017 +0.0168
loan_app_extortion       660 |  0.803  0.247  0.378     203 |  0.807  0.247  0.378     202 | +0.004 +0.000 +0.0004
merchant_collusion       919 |  0.818  0.259  0.393     291 |  0.805  0.257  0.389     293 | -0.012 -0.002 -0.0039
```
```
N aug van7 vs C [q] macro-F1 0.4980 vs 0.4980 = +0.0000 (event bootstrap 5-95 %: -0.0010, +0.0012) | mean of halves 0.4978 vs 0.4978
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518499 |  0.977  0.998  0.987  518535 | -0.000 +0.000 +0.0000
mule_account            3448 |  0.737  0.274  0.399    1283 |  0.737  0.275  0.400    1287 | +0.000 +0.001 +0.0009
velocity_bot            2612 |  0.843  0.332  0.476    1029 |  0.843  0.332  0.476    1029 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.762  0.434  0.553    1317 |  0.761  0.434  0.553    1318 | -0.001 +0.000 -0.0002
sim_swap                1393 |  0.684  0.377  0.486     767 |  0.684  0.375  0.484     763 | -0.000 -0.002 -0.0019
account_takeover        1333 |  0.825  0.259  0.394     418 |  0.826  0.260  0.395     419 | +0.000 +0.001 +0.0009
phishing_collect        2254 |  0.751  0.348  0.476    1045 |  0.760  0.343  0.473    1018 | +0.009 -0.005 -0.0028
qr_swap                 2489 |  0.814  0.376  0.514    1149 |  0.831  0.373  0.515    1117 | +0.017 -0.003 +0.0007
refund_abuse            1115 |  0.841  0.385  0.528     510 |  0.840  0.385  0.528     511 | -0.002 +0.000 -0.0003
fake_cashback            811 |  0.632  0.248  0.356     318 |  0.617  0.256  0.362     337 | -0.015 +0.009 +0.0063
loan_app_extortion       660 |  0.811  0.261  0.394     212 |  0.815  0.261  0.395     211 | +0.004 +0.000 +0.0005
merchant_collusion       919 |  0.826  0.274  0.412     305 |  0.814  0.272  0.408     307 | -0.012 -0.002 -0.0039
```
```
N sep van7 vs C [pre] macro-F1 0.4830 vs 0.4815 = +0.0014 (event bootstrap 5-95 %: +0.0002, +0.0026)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.976  0.998  0.987  502285 |  0.976  0.998  0.987  502286 | +0.000 +0.000 +0.0000
mule_account            3367 |  0.752  0.271  0.399    1215 |  0.751  0.271  0.398    1215 | -0.002 -0.001 -0.0009
velocity_bot            2659 |  0.846  0.339  0.484    1066 |  0.846  0.339  0.484    1066 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.773  0.401  0.529    1133 |  0.769  0.403  0.529    1143 | -0.004 +0.001 +0.0002
sim_swap                1267 |  0.664  0.281  0.395     536 |  0.676  0.281  0.397     527 | +0.011 +0.000 +0.0020
account_takeover        1280 |  0.650  0.270  0.381     531 |  0.649  0.273  0.384     538 | -0.001 +0.003 +0.0029
phishing_collect        2162 |  0.735  0.329  0.454     967 |  0.742  0.325  0.452     946 | +0.007 -0.004 -0.0027
qr_swap                 2352 |  0.830  0.348  0.490     986 |  0.833  0.346  0.489     977 | +0.004 -0.002 -0.0011
refund_abuse            1075 |  0.804  0.373  0.510     499 |  0.804  0.378  0.514     505 | +0.000 +0.005 +0.0044
fake_cashback            831 |  0.639  0.254  0.363     330 |  0.650  0.255  0.366     326 | +0.011 +0.001 +0.0030
loan_app_extortion       645 |  0.867  0.262  0.402     195 |  0.862  0.262  0.402     196 | -0.004 +0.000 -0.0005
merchant_collusion       841 |  0.830  0.250  0.384     253 |  0.808  0.260  0.394     271 | -0.022 +0.011 +0.0100
```
```
N sep van7 vs C [q] macro-F1 0.5082 vs 0.5070 = +0.0012 (event bootstrap 5-95 %: +0.0002, +0.0023)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  501922 |  0.977  0.998  0.987  501923 | +0.000 +0.000 +0.0000
mule_account            3367 |  0.760  0.290  0.419    1283 |  0.758  0.289  0.418    1283 | -0.002 -0.001 -0.0009
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.358  0.504    1125 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.777  0.415  0.541    1166 |  0.773  0.417  0.541    1176 | -0.004 +0.001 +0.0002
sim_swap                1267 |  0.697  0.367  0.481     667 |  0.701  0.367  0.482     663 | +0.004 +0.000 +0.0010
account_takeover        1280 |  0.839  0.290  0.431     442 |  0.837  0.290  0.431     443 | -0.002 +0.000 -0.0003
phishing_collect        2162 |  0.748  0.348  0.475    1007 |  0.756  0.345  0.473     986 | +0.008 -0.004 -0.0019
qr_swap                 2352 |  0.830  0.365  0.507    1034 |  0.833  0.363  0.506    1025 | +0.003 -0.002 -0.0010
refund_abuse            1075 |  0.809  0.389  0.525     517 |  0.809  0.393  0.529     523 | +0.000 +0.005 +0.0043
fake_cashback            831 |  0.647  0.273  0.384     351 |  0.658  0.276  0.388     348 | +0.011 +0.002 +0.0044
loan_app_extortion       645 |  0.870  0.279  0.423     207 |  0.865  0.279  0.422     208 | -0.004 +0.000 -0.0005
merchant_collusion       841 |  0.825  0.270  0.407     275 |  0.805  0.281  0.416     293 | -0.020 +0.011 +0.0094
```
```
D aug van7dm vs A_C [pre] macro-F1 0.4709 vs 0.4719 = -0.0009 (event bootstrap 5-95 %: -0.0024, +0.0008) | mean of halves 0.4707 vs 0.4716
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.976  0.998  0.987  519081 |  0.976  0.998  0.987  519133 | -0.000 +0.000 -0.0000
mule_account            3448 |  0.749  0.256  0.381    1178 |  0.747  0.257  0.382    1185 | -0.002 +0.001 +0.0007
velocity_bot            2612 |  0.839  0.312  0.454     970 |  0.842  0.312  0.455     967 | +0.003 +0.000 +0.0004
layering_chain          2311 |  0.779  0.411  0.538    1218 |  0.782  0.410  0.538    1212 | +0.003 -0.000 +0.0003
sim_swap                1393 |  0.660  0.261  0.374     550 |  0.672  0.257  0.372     533 | +0.012 -0.004 -0.0019
account_takeover        1333 |  0.624  0.239  0.346     511 |  0.637  0.246  0.355     515 | +0.013 +0.007 +0.0090
phishing_collect        2254 |  0.733  0.335  0.459    1028 |  0.734  0.331  0.456    1016 | +0.001 -0.004 -0.0032
qr_swap                 2489 |  0.830  0.354  0.496    1060 |  0.828  0.353  0.495    1062 | -0.003 -0.000 -0.0008
refund_abuse            1115 |  0.857  0.360  0.507     468 |  0.861  0.354  0.502     459 | +0.004 -0.005 -0.0047
fake_cashback            811 |  0.614  0.236  0.340     311 |  0.625  0.224  0.330     291 | +0.011 -0.011 -0.0102
loan_app_extortion       660 |  0.813  0.244  0.375     198 |  0.817  0.244  0.376     197 | +0.004 +0.000 +0.0004
merchant_collusion       919 |  0.867  0.263  0.404     279 |  0.858  0.263  0.403     282 | -0.009 +0.000 -0.0010
```
```
D aug van7dm vs A_C [q] macro-F1 0.5001 vs 0.5006 = -0.0005 (event bootstrap 5-95 %: -0.0019, +0.0012) | mean of halves 0.4999 vs 0.5004
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518642 |  0.977  0.998  0.987  518682 | -0.000 +0.000 -0.0000
mule_account            3448 |  0.752  0.274  0.401    1255 |  0.749  0.275  0.402    1265 | -0.004 +0.001 +0.0004
velocity_bot            2612 |  0.837  0.333  0.476    1040 |  0.839  0.332  0.476    1033 | +0.003 -0.001 -0.0007
layering_chain          2311 |  0.787  0.429  0.555    1259 |  0.789  0.428  0.555    1254 | +0.002 -0.000 +0.0002
sim_swap                1393 |  0.711  0.377  0.493     738 |  0.722  0.376  0.495     726 | +0.010 -0.001 +0.0018
account_takeover        1333 |  0.834  0.256  0.392     409 |  0.840  0.263  0.401     418 | +0.006 +0.008 +0.0094
phishing_collect        2254 |  0.749  0.353  0.480    1063 |  0.750  0.349  0.477    1049 | +0.001 -0.004 -0.0034
qr_swap                 2489 |  0.829  0.375  0.516    1125 |  0.827  0.374  0.515    1127 | -0.002 -0.000 -0.0008
refund_abuse            1115 |  0.855  0.376  0.522     490 |  0.859  0.370  0.518     481 | +0.004 -0.005 -0.0046
fake_cashback            811 |  0.639  0.260  0.370     330 |  0.650  0.252  0.363     314 | +0.010 -0.009 -0.0072
loan_app_extortion       660 |  0.821  0.258  0.392     207 |  0.825  0.258  0.393     206 | +0.004 +0.000 +0.0005
merchant_collusion       919 |  0.871  0.279  0.422     294 |  0.862  0.279  0.421     297 | -0.009 +0.000 -0.0010
```
```
D sep van7dm vs A_C [pre] macro-F1 0.4822 vs 0.4824 = -0.0002 (event bootstrap 5-95 %: -0.0013, +0.0009)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.976  0.998  0.987  502486 |  0.976  0.998  0.987  502491 | +0.000 +0.000 +0.0000
mule_account            3367 |  0.763  0.272  0.401    1202 |  0.757  0.273  0.401    1213 | -0.006 +0.000 -0.0005
velocity_bot            2659 |  0.847  0.339  0.484    1065 |  0.847  0.340  0.486    1069 | -0.000 +0.001 +0.0011
layering_chain          2182 |  0.809  0.394  0.530    1063 |  0.811  0.390  0.527    1050 | +0.002 -0.004 -0.0028
sim_swap                1267 |  0.696  0.277  0.396     504 |  0.700  0.276  0.396     500 | +0.004 -0.001 -0.0002
account_takeover        1280 |  0.703  0.264  0.384     481 |  0.702  0.266  0.385     484 | -0.000 +0.002 +0.0016
phishing_collect        2162 |  0.720  0.325  0.448     975 |  0.722  0.323  0.446     967 | +0.002 -0.002 -0.0014
qr_swap                 2352 |  0.820  0.348  0.488     998 |  0.820  0.348  0.489     997 | +0.001 +0.000 +0.0001
refund_abuse            1075 |  0.842  0.367  0.512     469 |  0.842  0.366  0.510     467 | -0.001 -0.002 -0.0019
fake_cashback            831 |  0.660  0.250  0.363     315 |  0.656  0.255  0.367     323 | -0.004 +0.005 +0.0044
loan_app_extortion       645 |  0.868  0.254  0.393     189 |  0.868  0.254  0.393     189 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.880  0.260  0.402     249 |  0.882  0.258  0.399     246 | +0.003 -0.002 -0.0026
```
```
D sep van7dm vs A_C [q] macro-F1 0.5076 vs 0.5078 = -0.0002 (event bootstrap 5-95 %: -0.0012, +0.0007)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  502077 |  0.977  0.998  0.987  502083 | -0.000 +0.000 +0.0000
mule_account            3367 |  0.770  0.290  0.422    1270 |  0.765  0.291  0.421    1280 | -0.005 +0.000 -0.0005
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.360  0.505    1129 | -0.000 +0.001 +0.0011
layering_chain          2182 |  0.811  0.408  0.543    1097 |  0.814  0.405  0.541    1085 | +0.003 -0.003 -0.0023
sim_swap                1267 |  0.722  0.365  0.485     640 |  0.721  0.364  0.484     639 | -0.000 -0.001 -0.0008
account_takeover        1280 |  0.852  0.288  0.431     433 |  0.850  0.288  0.431     434 | -0.002 +0.000 -0.0003
phishing_collect        2162 |  0.740  0.344  0.469    1004 |  0.745  0.342  0.469     992 | +0.005 -0.002 -0.0008
qr_swap                 2352 |  0.820  0.365  0.505    1046 |  0.821  0.365  0.505    1045 | +0.001 +0.000 +0.0001
refund_abuse            1075 |  0.846  0.383  0.528     487 |  0.845  0.381  0.526     485 | -0.001 -0.002 -0.0019
fake_cashback            831 |  0.652  0.271  0.383     345 |  0.648  0.277  0.388     355 | -0.004 +0.006 +0.0052
loan_app_extortion       645 |  0.871  0.271  0.414     201 |  0.871  0.271  0.414     201 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.871  0.281  0.424     271 |  0.873  0.278  0.422     268 | +0.002 -0.002 -0.0025
```
```
N aug vac2f3 vs C [pre] macro-F1 0.4700 vs 0.4698 = +0.0002 (event bootstrap 5-95 %: -0.0001, +0.0005) | mean of halves 0.4698 vs 0.4696
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.976  0.998  0.987  518912 |  0.976  0.998  0.987  518912 | -0.000 -0.000 -0.0000
mule_account            3448 |  0.734  0.256  0.380    1203 |  0.734  0.256  0.380    1203 | +0.000 +0.000 +0.0000
velocity_bot            2612 |  0.841  0.312  0.455     968 |  0.841  0.312  0.455     968 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.753  0.415  0.535    1275 |  0.753  0.415  0.535    1275 | +0.000 +0.000 +0.0000
sim_swap                1393 |  0.630  0.261  0.369     578 |  0.630  0.261  0.369     578 | +0.000 +0.000 +0.0000
account_takeover        1333 |  0.603  0.245  0.349     542 |  0.603  0.245  0.349     542 | +0.000 +0.000 +0.0000
phishing_collect        2254 |  0.742  0.331  0.457    1004 |  0.743  0.332  0.459    1008 | +0.001 +0.002 +0.0019
qr_swap                 2489 |  0.813  0.354  0.494    1085 |  0.816  0.354  0.493    1078 | +0.003 -0.001 -0.0002
refund_abuse            1115 |  0.843  0.370  0.514     489 |  0.841  0.370  0.513     490 | -0.002 +0.000 -0.0003
fake_cashback            811 |  0.603  0.224  0.327     302 |  0.603  0.224  0.327     302 | +0.000 +0.000 +0.0000
loan_app_extortion       660 |  0.803  0.247  0.378     203 |  0.803  0.247  0.378     203 | +0.000 +0.000 +0.0000
merchant_collusion       919 |  0.818  0.259  0.393     291 |  0.816  0.260  0.394     293 | -0.002 +0.001 +0.0010
```
```
N aug vac2f3 vs C [q] macro-F1 0.4982 vs 0.4980 = +0.0002 (event bootstrap 5-95 %: -0.0001, +0.0005) | mean of halves 0.4980 vs 0.4978
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518499 |  0.977  0.998  0.987  518499 | -0.000 -0.000 -0.0000
mule_account            3448 |  0.737  0.274  0.399    1283 |  0.737  0.274  0.399    1283 | +0.000 +0.000 +0.0000
velocity_bot            2612 |  0.843  0.332  0.476    1029 |  0.843  0.332  0.476    1029 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.762  0.434  0.553    1317 |  0.762  0.434  0.553    1317 | +0.000 +0.000 +0.0000
sim_swap                1393 |  0.684  0.377  0.486     767 |  0.684  0.377  0.486     767 | +0.000 +0.000 +0.0000
account_takeover        1333 |  0.825  0.259  0.394     418 |  0.825  0.259  0.394     418 | +0.000 +0.000 +0.0000
phishing_collect        2254 |  0.751  0.348  0.476    1045 |  0.752  0.350  0.478    1049 | +0.001 +0.002 +0.0018
qr_swap                 2489 |  0.814  0.376  0.514    1149 |  0.817  0.375  0.514    1142 | +0.003 -0.001 -0.0001
refund_abuse            1115 |  0.841  0.385  0.528     510 |  0.840  0.385  0.528     511 | -0.002 +0.000 -0.0003
fake_cashback            811 |  0.632  0.248  0.356     318 |  0.632  0.248  0.356     318 | +0.000 +0.000 +0.0000
loan_app_extortion       660 |  0.811  0.261  0.394     212 |  0.811  0.261  0.394     212 | +0.000 +0.000 +0.0000
merchant_collusion       919 |  0.826  0.274  0.412     305 |  0.824  0.275  0.413     307 | -0.002 +0.001 +0.0010
```
```
N sep vac2f3 vs C [pre] macro-F1 0.4813 vs 0.4815 = -0.0002 (event bootstrap 5-95 %: -0.0004, -0.0000)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.976  0.998  0.987  502285 |  0.976  0.998  0.987  502241 | +0.000 -0.000 -0.0000
mule_account            3367 |  0.752  0.271  0.399    1215 |  0.752  0.271  0.399    1215 | +0.000 +0.000 +0.0000
velocity_bot            2659 |  0.846  0.339  0.484    1066 |  0.846  0.339  0.484    1066 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.773  0.401  0.529    1133 |  0.773  0.401  0.529    1133 | +0.000 +0.000 +0.0000
sim_swap                1267 |  0.664  0.281  0.395     536 |  0.664  0.281  0.395     536 | +0.000 +0.000 +0.0000
account_takeover        1280 |  0.650  0.270  0.381     531 |  0.650  0.270  0.381     531 | +0.000 +0.000 +0.0000
phishing_collect        2162 |  0.735  0.329  0.454     967 |  0.735  0.329  0.454     967 | +0.000 +0.000 +0.0000
qr_swap                 2352 |  0.830  0.348  0.490     986 |  0.800  0.350  0.487    1030 | -0.030 +0.003 -0.0028
refund_abuse            1075 |  0.804  0.373  0.510     499 |  0.804  0.373  0.510     499 | +0.000 +0.000 +0.0000
fake_cashback            831 |  0.639  0.254  0.363     330 |  0.639  0.254  0.363     330 | +0.000 +0.000 +0.0000
loan_app_extortion       645 |  0.867  0.262  0.402     195 |  0.867  0.262  0.402     195 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.830  0.250  0.384     253 |  0.830  0.250  0.384     253 | +0.000 +0.000 +0.0000
```
```
N sep vac2f3 vs C [q] macro-F1 0.5067 vs 0.5070 = -0.0003 (event bootstrap 5-95 %: -0.0004, -0.0001)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  501922 |  0.977  0.998  0.987  501878 | +0.000 -0.000 -0.0000
mule_account            3367 |  0.760  0.290  0.419    1283 |  0.760  0.290  0.419    1283 | +0.000 +0.000 +0.0000
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.358  0.504    1125 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.777  0.415  0.541    1166 |  0.777  0.415  0.541    1166 | +0.000 +0.000 +0.0000
sim_swap                1267 |  0.697  0.367  0.481     667 |  0.697  0.367  0.481     667 | +0.000 +0.000 +0.0000
account_takeover        1280 |  0.839  0.290  0.431     442 |  0.839  0.290  0.431     442 | +0.000 +0.000 +0.0000
phishing_collect        2162 |  0.748  0.348  0.475    1007 |  0.748  0.348  0.475    1007 | +0.000 +0.000 +0.0000
qr_swap                 2352 |  0.830  0.365  0.507    1034 |  0.801  0.367  0.504    1078 | -0.028 +0.003 -0.0030
refund_abuse            1075 |  0.809  0.389  0.525     517 |  0.809  0.389  0.525     517 | +0.000 +0.000 +0.0000
fake_cashback            831 |  0.647  0.273  0.384     351 |  0.647  0.273  0.384     351 | +0.000 +0.000 +0.0000
loan_app_extortion       645 |  0.870  0.279  0.423     207 |  0.870  0.279  0.423     207 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.825  0.270  0.407     275 |  0.825  0.270  0.407     275 | +0.000 +0.000 +0.0000
```
```
D aug vac2f3 vs A_C [pre] macro-F1 0.4719 vs 0.4719 = +0.0000 (event bootstrap 5-95 %: -0.0002, +0.0003) | mean of halves 0.4716 vs 0.4716
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.976  0.998  0.987  519081 |  0.976  0.998  0.987  519067 | +0.000 -0.000 -0.0000
mule_account            3448 |  0.749  0.256  0.381    1178 |  0.749  0.256  0.381    1178 | +0.000 +0.000 +0.0000
velocity_bot            2612 |  0.839  0.312  0.454     970 |  0.839  0.312  0.454     970 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.779  0.411  0.538    1218 |  0.779  0.411  0.538    1218 | +0.000 +0.000 +0.0000
sim_swap                1393 |  0.660  0.261  0.374     550 |  0.660  0.261  0.374     550 | +0.000 +0.000 +0.0000
account_takeover        1333 |  0.624  0.239  0.346     511 |  0.624  0.239  0.346     511 | +0.000 +0.000 +0.0000
phishing_collect        2254 |  0.733  0.335  0.459    1028 |  0.733  0.335  0.459    1028 | +0.000 +0.000 +0.0000
qr_swap                 2489 |  0.830  0.354  0.496    1060 |  0.821  0.355  0.495    1076 | -0.010 +0.001 -0.0005
refund_abuse            1115 |  0.857  0.360  0.507     468 |  0.860  0.359  0.506     465 | +0.003 -0.001 -0.0003
fake_cashback            811 |  0.614  0.236  0.340     311 |  0.614  0.236  0.340     311 | +0.000 +0.000 +0.0000
loan_app_extortion       660 |  0.813  0.244  0.375     198 |  0.813  0.244  0.375     198 | +0.000 +0.000 +0.0000
merchant_collusion       919 |  0.867  0.263  0.404     279 |  0.868  0.264  0.405     280 | +0.000 +0.001 +0.0013
```
```
D aug vac2f3 vs A_C [q] macro-F1 0.5006 vs 0.5006 = +0.0000 (event bootstrap 5-95 %: -0.0002, +0.0003) | mean of halves 0.5004 vs 0.5004
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518642 |  0.977  0.998  0.987  518628 | +0.000 -0.000 -0.0000
mule_account            3448 |  0.752  0.274  0.401    1255 |  0.752  0.274  0.401    1255 | +0.000 +0.000 +0.0000
velocity_bot            2612 |  0.837  0.333  0.476    1040 |  0.837  0.333  0.476    1040 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.787  0.429  0.555    1259 |  0.787  0.429  0.555    1259 | +0.000 +0.000 +0.0000
sim_swap                1393 |  0.711  0.377  0.493     738 |  0.711  0.377  0.493     738 | +0.000 +0.000 +0.0000
account_takeover        1333 |  0.834  0.256  0.392     409 |  0.834  0.256  0.392     409 | +0.000 +0.000 +0.0000
phishing_collect        2254 |  0.749  0.353  0.480    1063 |  0.749  0.353  0.480    1063 | +0.000 +0.000 +0.0000
qr_swap                 2489 |  0.829  0.375  0.516    1125 |  0.820  0.376  0.516    1141 | -0.009 +0.001 -0.0006
refund_abuse            1115 |  0.855  0.376  0.522     490 |  0.858  0.375  0.522     487 | +0.003 -0.001 -0.0003
fake_cashback            811 |  0.639  0.260  0.370     330 |  0.639  0.260  0.370     330 | +0.000 +0.000 +0.0000
loan_app_extortion       660 |  0.821  0.258  0.392     207 |  0.821  0.258  0.392     207 | +0.000 +0.000 +0.0000
merchant_collusion       919 |  0.871  0.279  0.422     294 |  0.871  0.280  0.423     295 | +0.000 +0.001 +0.0013
```
```
D sep vac2f3 vs A_C [pre] macro-F1 0.4824 vs 0.4824 = -0.0001 (event bootstrap 5-95 %: -0.0004, +0.0002)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.976  0.998  0.987  502486 |  0.976  0.998  0.987  502445 | +0.000 -0.000 -0.0000
mule_account            3367 |  0.763  0.272  0.401    1202 |  0.763  0.272  0.401    1202 | +0.000 +0.000 +0.0000
velocity_bot            2659 |  0.847  0.339  0.484    1065 |  0.847  0.339  0.484    1065 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.809  0.394  0.530    1063 |  0.809  0.394  0.530    1063 | +0.000 +0.000 +0.0000
sim_swap                1267 |  0.696  0.277  0.396     504 |  0.696  0.277  0.396     504 | +0.000 +0.000 +0.0000
account_takeover        1280 |  0.703  0.264  0.384     481 |  0.703  0.264  0.384     481 | +0.000 +0.000 +0.0000
phishing_collect        2162 |  0.720  0.325  0.448     975 |  0.720  0.325  0.448     975 | +0.000 +0.000 +0.0000
qr_swap                 2352 |  0.820  0.348  0.488     998 |  0.796  0.352  0.488    1039 | -0.024 +0.004 -0.0006
refund_abuse            1075 |  0.842  0.367  0.512     469 |  0.842  0.367  0.512     469 | +0.000 +0.000 +0.0000
fake_cashback            831 |  0.660  0.250  0.363     315 |  0.660  0.250  0.363     315 | +0.000 +0.000 +0.0000
loan_app_extortion       645 |  0.868  0.254  0.393     189 |  0.868  0.254  0.393     189 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.880  0.260  0.402     249 |  0.880  0.260  0.402     249 | +0.000 +0.000 +0.0000
```
```
D sep vac2f3 vs A_C [q] macro-F1 0.5078 vs 0.5078 = -0.0001 (event bootstrap 5-95 %: -0.0004, +0.0002)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  502077 |  0.977  0.998  0.987  502036 | +0.000 -0.000 -0.0000
mule_account            3367 |  0.770  0.290  0.422    1270 |  0.770  0.290  0.422    1270 | +0.000 +0.000 +0.0000
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.358  0.504    1125 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.811  0.408  0.543    1097 |  0.811  0.408  0.543    1097 | +0.000 +0.000 +0.0000
sim_swap                1267 |  0.722  0.365  0.485     640 |  0.722  0.365  0.485     640 | +0.000 +0.000 +0.0000
account_takeover        1280 |  0.852  0.288  0.431     433 |  0.852  0.288  0.431     433 | +0.000 +0.000 +0.0000
phishing_collect        2162 |  0.740  0.344  0.469    1004 |  0.740  0.344  0.469    1004 | +0.000 +0.000 +0.0000
qr_swap                 2352 |  0.820  0.365  0.505    1046 |  0.798  0.369  0.504    1087 | -0.023 +0.004 -0.0008
refund_abuse            1075 |  0.846  0.383  0.528     487 |  0.846  0.383  0.528     487 | +0.000 +0.000 +0.0000
fake_cashback            831 |  0.652  0.271  0.383     345 |  0.652  0.271  0.383     345 | +0.000 +0.000 +0.0000
loan_app_extortion       645 |  0.871  0.271  0.414     201 |  0.871  0.271  0.414     201 | +0.000 +0.000 +0.0000
merchant_collusion       841 |  0.871  0.281  0.424     271 |  0.871  0.281  0.424     271 | +0.000 +0.000 +0.0000
```
```
N aug vanull vs C [q] macro-F1 0.4985 vs 0.4980 = +0.0005 (event bootstrap 5-95 %: -0.0002, +0.0012) | mean of halves 0.4983 vs 0.4978
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518499 |  0.977  0.998  0.987  518510 | +0.000 +0.000 +0.0000
mule_account            3448 |  0.737  0.274  0.399    1283 |  0.735  0.276  0.401    1293 | -0.001 +0.002 +0.0017
velocity_bot            2612 |  0.843  0.332  0.476    1029 |  0.843  0.332  0.476    1029 | +0.000 +0.000 +0.0000
layering_chain          2311 |  0.762  0.434  0.553    1317 |  0.763  0.434  0.554    1315 | +0.002 +0.000 +0.0009
sim_swap                1393 |  0.684  0.377  0.486     767 |  0.686  0.380  0.489     771 | +0.002 +0.003 +0.0028
account_takeover        1333 |  0.825  0.259  0.394     418 |  0.823  0.259  0.394     419 | -0.002 +0.000 -0.0002
phishing_collect        2254 |  0.751  0.348  0.476    1045 |  0.754  0.348  0.476    1040 | +0.003 -0.000 +0.0001
qr_swap                 2489 |  0.814  0.376  0.514    1149 |  0.824  0.376  0.517    1137 | +0.010 +0.001 +0.0028
refund_abuse            1115 |  0.841  0.385  0.528     510 |  0.857  0.381  0.528     496 | +0.016 -0.004 -0.0004
fake_cashback            811 |  0.632  0.248  0.356     318 |  0.618  0.249  0.355     327 | -0.014 +0.001 -0.0011
loan_app_extortion       660 |  0.811  0.261  0.394     212 |  0.819  0.261  0.395     210 | +0.008 +0.000 +0.0009
merchant_collusion       919 |  0.826  0.274  0.412     305 |  0.823  0.273  0.410     305 | -0.003 -0.001 -0.0016
```
```
N sep vanull vs C [q] macro-F1 0.5077 vs 0.5070 = +0.0007 (event bootstrap 5-95 %: -0.0002, +0.0016)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  501922 |  0.977  0.998  0.987  501899 | +0.000 -0.000 -0.0000
mule_account            3367 |  0.760  0.290  0.419    1283 |  0.759  0.290  0.419    1284 | -0.001 +0.000 -0.0001
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.358  0.504    1125 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.777  0.415  0.541    1166 |  0.774  0.416  0.541    1173 | -0.003 +0.001 +0.0001
sim_swap                1267 |  0.697  0.367  0.481     667 |  0.700  0.367  0.482     664 | +0.003 +0.000 +0.0007
account_takeover        1280 |  0.839  0.290  0.431     442 |  0.839  0.289  0.430     441 | -0.000 -0.001 -0.0009
phishing_collect        2162 |  0.748  0.348  0.475    1007 |  0.751  0.346  0.474     996 | +0.003 -0.002 -0.0015
qr_swap                 2352 |  0.830  0.365  0.507    1034 |  0.829  0.366  0.507    1038 | -0.001 +0.001 +0.0006
refund_abuse            1075 |  0.809  0.389  0.525     517 |  0.809  0.390  0.526     518 | +0.000 +0.001 +0.0009
fake_cashback            831 |  0.647  0.273  0.384     351 |  0.630  0.285  0.393     376 | -0.016 +0.012 +0.0086
loan_app_extortion       645 |  0.870  0.279  0.423     207 |  0.869  0.278  0.421     206 | -0.001 -0.002 -0.0019
merchant_collusion       841 |  0.825  0.270  0.407     275 |  0.826  0.271  0.408     276 | +0.001 +0.001 +0.0014
```
```
D aug vanull vs A_C [q] macro-F1 0.4994 vs 0.5006 = -0.0012 (event bootstrap 5-95 %: -0.0025, +0.0001) | mean of halves 0.4992 vs 0.5004
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 507507 |  0.977  0.998  0.987  518642 |  0.977  0.998  0.987  518665 | -0.000 -0.000 -0.0000
mule_account            3448 |  0.752  0.274  0.401    1255 |  0.745  0.275  0.402    1273 | -0.007 +0.001 +0.0006
velocity_bot            2612 |  0.837  0.333  0.476    1040 |  0.843  0.332  0.476    1029 | +0.006 -0.001 -0.0002
layering_chain          2311 |  0.787  0.429  0.555    1259 |  0.789  0.427  0.554    1251 | +0.002 -0.002 -0.0010
sim_swap                1393 |  0.711  0.377  0.493     738 |  0.715  0.373  0.490     726 | +0.003 -0.004 -0.0029
account_takeover        1333 |  0.834  0.256  0.392     409 |  0.835  0.258  0.394     412 | +0.001 +0.002 +0.0028
phishing_collect        2254 |  0.749  0.353  0.480    1063 |  0.741  0.350  0.476    1066 | -0.008 -0.003 -0.0040
qr_swap                 2489 |  0.829  0.375  0.516    1125 |  0.830  0.372  0.514    1115 | +0.001 -0.003 -0.0025
refund_abuse            1115 |  0.855  0.376  0.522     490 |  0.852  0.370  0.516     485 | -0.004 -0.005 -0.0059
fake_cashback            811 |  0.639  0.260  0.370     330 |  0.648  0.256  0.367     321 | +0.009 -0.004 -0.0024
loan_app_extortion       660 |  0.821  0.258  0.392     207 |  0.815  0.261  0.395     211 | -0.006 +0.003 +0.0028
merchant_collusion       919 |  0.871  0.279  0.422     294 |  0.859  0.279  0.421     298 | -0.012 +0.000 -0.0014
```
```
D sep vanull vs A_C [q] macro-F1 0.5084 vs 0.5078 = +0.0006 (event bootstrap 5-95 %: -0.0004, +0.0016)
class                      n |  ref P      R     F1   npred |  var P      R     F1   npred |     dP     dR     dF1
legit                 491335 |  0.977  0.998  0.987  502077 |  0.977  0.998  0.987  502096 | -0.000 +0.000 +0.0000
mule_account            3367 |  0.770  0.290  0.422    1270 |  0.770  0.291  0.422    1273 | -0.000 +0.001 +0.0006
velocity_bot            2659 |  0.847  0.358  0.504    1125 |  0.847  0.358  0.504    1125 | +0.000 +0.000 +0.0000
layering_chain          2182 |  0.811  0.408  0.543    1097 |  0.808  0.411  0.545    1109 | -0.003 +0.003 +0.0017
sim_swap                1267 |  0.722  0.365  0.485     640 |  0.722  0.363  0.483     637 | +0.000 -0.002 -0.0013
account_takeover        1280 |  0.852  0.288  0.431     433 |  0.854  0.288  0.431     432 | +0.002 +0.000 +0.0003
phishing_collect        2162 |  0.740  0.344  0.469    1004 |  0.752  0.337  0.466     969 | +0.012 -0.006 -0.0037
qr_swap                 2352 |  0.820  0.365  0.505    1046 |  0.834  0.361  0.503    1017 | +0.014 -0.004 -0.0016
refund_abuse            1075 |  0.846  0.383  0.528     487 |  0.834  0.388  0.530     500 | -0.012 +0.005 +0.0020
fake_cashback            831 |  0.652  0.271  0.383     345 |  0.645  0.284  0.394     366 | -0.007 +0.013 +0.0117
loan_app_extortion       645 |  0.871  0.271  0.414     201 |  0.870  0.270  0.412     200 | -0.001 -0.002 -0.0019
merchant_collusion       841 |  0.871  0.281  0.424     271 |  0.868  0.281  0.424     272 | -0.003 +0.000 -0.0004
```

## Columns for all rows, including October test rows
No mechanism stayed competitive, so no candidate columns are recommended. The files exist anyway (txn_id + columns, 1,824,642
rows = 1,564,060 train + 260,582 test), written by `va_feat.py` after the permutation check:
- `feat/n7_DM.parquet`: `n7__snd_n7`, `n7__snd_out7`, `n7__snd_in7`, density-matched build (the form scored on D; the only form
  that is comparable between train rows and October rows). `feat/n7_ALL.parquet`: the same at full density (equal to B's
  `relationships/rel_n7.parquet`; month means 1.14 Aug / Sep against .64 in October, do not use).
- `feat/c2_ALL.parquet`: `va_c2__qr` (the C2 cell flag); rule = P(qr_swap) x 3 on flagged rows, then the unchanged decoder.
No submission file was built.

## Reproduction (cwd = `work`; prefix every line with
`PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4 ../.venv/Scripts/python.exe`)
```
# harness control and checks
BM=N  scratch_atomic/verification/va_fit.py C            # env BM=N|D goes before the interpreter
BM=D  scratch_atomic/verification/va_fit.py C
scratch_atomic/verification/va_score.py check
# noise-column control
scratch_atomic/verification/va_feat.py scratch_atomic/verification/mech_null.py null
BM=N  scratch_atomic/verification/va_fit.py null ;  BM=D  scratch_atomic/verification/va_fit.py null
# mechanism n7
scratch_atomic/verification/va_feat.py scratch_atomic/verification/mech_n7.py n7 N_f0,N_f1,ALL,DM
BM=N                    scratch_atomic/verification/va_fit.py n7      # -> PN_van7_{aug,sep}.npy
BM=D DFEAT=DM TAG=dm    scratch_atomic/verification/va_fit.py n7      # -> PD_van7dm_{aug,sep}.npy   (status carrier on D)
BM=D DFEAT=ALL TAG=full scratch_atomic/verification/va_fit.py n7      # -> PD_van7full_{aug,sep}.npy (contrast)
BM=N SEED0=2            scratch_atomic/verification/va_fit.py n7      # -> PN_van7b2_{aug,sep}.npy   (reference scratch_gap/PN_A_Cb2)
# mechanism c2 (no fit)
scratch_atomic/verification/va_feat.py scratch_atomic/verification/mech_c2.py c2 ALL
scratch_atomic/verification/va_c2rule.py 3 10 30
scratch_atomic/verification/va_score.py N C vac2f3,vac2f10,vac2f30 aug ;  scratch_atomic/verification/va_c2pick.py     # August only -> F
scratch_atomic/verification/va_score.py N C vac2f3 sep ;  scratch_atomic/verification/va_score.py D A_C vac2f3 sep     # chosen F only
# scoring (full report with per-class tables; one process per variant and month, they share the decode cache)
scratch_atomic/verification/va_score.py N C vanull,A_Cb2,van7 aug        # then the same with sep
scratch_atomic/verification/va_score.py D A_C vanull,van7dm,van7full aug   # then sep
scratch_atomic/verification/va_score.py N A_Cb2 van7b2 aug,sep
scratch_atomic/verification/va_score.py N vanull van7 aug,sep ;  scratch_atomic/verification/va_score.py D vanull van7dm aug,sep
scratch_atomic/verification/va_score.py summary N C vanull,A_Cb2,van7,vac2f3 aug,sep     # the tables above (log/summary.md)
# audit instrument and audit runs
scratch_atomic/verification/va_pred.py N scratch_atomic/verification/feat/n7_N_f0.parquet,scratch_atomic/verification/feat/n7_N_f1.parquet "n7__snd_n7" "n7__snd_n7 >= 3"
scratch_atomic/verification/va_pred.py N scratch_atomic/verification/feat/c2_ALL.parquet "va_c2__qr = 1" "va_c2__collect = 1" "va_c2__J = 1"
scratch_atomic/verification/va_pred.py N scratch_atomic/verification/feat/sdj_ALL.parquet "va_sd__first = 1" "va_sd__first = 1 AND va_sd3__dn >= 428455" "va_sd__first = 1 AND va_sd3__dn < 420000"
scratch_atomic/verification/va_pred.py N scratch_atomic/relationships/rel_feats.parquet rel__a_snd_prev_role ...   (all 26 columns: log/pred_B_rel_N.log)
scratch_atomic/verification/va_block.py ; scratch_atomic/verification/va_c2cell.py ; scratch_atomic/verification/va_ipoct.py ; scratch_atomic/verification/va_recur.py
scratch_atomic/verification/B_joint_rerun.py
```
`feat/sdj_ALL.parquet` is the txn_id join of `feat/sd_ALL.parquet` (`mech_sd.py`) and `feat/sd3_ALL.parquet` (`mech_sd3.py`).
Fit time on this machine during the round: 125-300 s (fold 0) and 176-510 s (fold 1) per arm with two fits in parallel; one
decode (bias tuning) about 2 minutes.
