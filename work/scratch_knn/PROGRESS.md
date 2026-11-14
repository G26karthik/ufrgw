# Experiment B: class-conditional neighbour features (benchmark N)

Started 14 Nov 2026 21:40 IST, hard stop 23:30 IST. CPU only. No production file is touched.

## Design, declared 21:46 IST before any run

**Distance space (45 dimensions), built per world, scaling fitted on fit rows only** (`knn_feats.py::space`)
- 17 numeric columns, quantile-transformed to a normal (1000 quantiles, fitted on the world's fit rows), clipped to +-3,
  missing set to 0: `log_amount, cents, sim_age_hours, device_age_days, login_to_txn_sec, balance_before, amt_share_bal,
  burst__dev_c300, burst__snd_c300, burst__snd_ref7d, flow__rin_n12h, flow__sin_n24h, flow__rout_n6h,
  takeover__dev_frac, takeover__n_other_dev, scam__rcv_in_15d, ring__rcv_n`.
- `hour` as sin and cos (2 dimensions). `minute_of_day` is excluded (stored wrapped, -128..127, not a metric).
- 6 binary or small ordinal columns kept raw: `round100, round1000, new_payee, pin_retries, away_from_home, is_p2p`.
- 5 categorical columns one-hot, divided by sqrt(2) so a category mismatch costs 1 unit, null as its own level:
  `txn_type, receiver_type, app, qr_vpa_mismatch, mer_same_city` (20 dimensions). Never used as numbers.
- Euclidean distance, exact brute-force search (sklearn), float32.

**Reference banks, from `y_fit` of fit rows only**
- Each of the 11 fraud classes: all fit rows of the class.
- Legit: random sample of 20% of the fit legit rows (fold0 about 101.5k, fold1 about 203k), so the fraud:legit bank
  ratio is the same in both worlds. Exact sizes of every class bank are logged per fold (`feats_f*.log`).

**Cross-fitting**
- Fit rows are split in 10 folds. Fraud events (`clean_labels.fraud_events`, 48 h, computed on the fit months only, so
  held labels do not shape them) stay whole; all other rows get a random fold.
- A fit row in fold j queries banks built from the other 9 folds (legit bank resampled from those folds at the same
  absolute size as the held bank). It cannot meet itself or a row of its own event.
- Held rows query the full fit bank. Known gap: fraud banks are 90% size for fit rows and 100% for held rows; the log
  prints the median distance of fit legit rows against held rows per class as the density check.
- Fit legit rows that neither training seed keeps (64% of them) are not queried.

**New columns (50)**
- 36: distance to the 1st, 5th and 15th neighbour in each of the 12 class banks (`kn_c<class>_k<k>`).
- 3: nearest fraud-class distance over legit distance, for k = 1, 5, 15 (`kn_rmin_k<k>`).
- 11: each fraud class's 5th-neighbour distance over the legit 5th-neighbour distance (`kn_r5_c<class>`).

**Candidates (fixed now)**
- `kA`: baseline recipe (`numrep.py::run`: `train.LGB`, 150 rounds, lr 0.05, legit 20% with weight 5, seeds 0 and 1,
  new `lgb.Dataset` per fit) on the 190 columns + 50 new columns.
- `kBL`: 0.65 x C + 0.35 x kA probabilities.
- Harness check: the same script with zero new columns, fold 0, compared with `PN_C_aug.npy`.

**Verdict rule (fixed now)**
- supported improvement: August after-decoder gain over C > 0 and September after-decoder paired event-bootstrap
  interval (5%..95%) entirely above 0.
- measurably worse: that interval entirely below 0.
- promising but unconfirmed: August gain > 0 and September point gain > +0.001 with an interval that includes 0.
- no detected difference: everything else (RAW and AC moved +-0.001 against C from target changes alone).
- not evaluated adequately: a step did not finish.

## Status
- 21:46 design declared; neighbour search timed at 4.5 s per 20k queries x 100k bank (estimate: fold0 4 min, fold1 9 min).
- 21:49 fold0 neighbour columns done (174 s), `KF_f0.parquet`. Space is 43 dimensions, not 45 (two one-hot levels do not occur).
  Bank sizes fold0, legit then fraud classes in `train.LABELS` order: held (full) 101510, 3422, 2834, 2325, 1409, 1437, 2280,
  2421, 1041, 888, 678, 778; cross-fit minimum 101510, 3008, 2417, 2021, 1242, 1277, 2027, 2151, 916, 782, 592, 683.
  Density check: median 5th-neighbour distance of fit legit rows is about 1% larger than for held rows on every fraud bank
  (for example mule 2.467 vs 2.445), as expected from the 90% bank; legit bank 1.645 vs 1.655.
- 21:51 harness check: zero new columns, fold 0, reproduces `PN_C_aug.npy` exactly (max probability difference 0.00000,
  0 argmax changes). Noise floor of the harness is zero.
- 21:52 kA fold 0 training started; fold1 neighbour search running.
- 21:56 fold1 neighbour columns done (423 s), `KF_f1.parquet`. Bank sizes fold1: held (full) 203012, 6870, 5446, 4636, 2803,
  2769, 4534, 4910, 2156, 1699, 1338, 1697; cross-fit minimum 203012, 6041, 4634, 4091, 2486, 2434, 4039, 4383, 1904, 1512,
  1193, 1501. Density check as in fold0 (about 1% larger for fit rows on fraud banks). No fit row has a zero own-class distance.
- 21:57 August diagnostic (`recov_aug.log`): among the 518,920 August rows C calls legit, 12,371 are fraud-labelled. No
  neighbour column separates them from legit rows: AUC 0.503 / 0.505 / 0.505 for nearest-fraud over legit ratio at k 1 / 5 / 15,
  0.496..0.516 for the per-class ratios, against 0.506 for C's own fraud probability.
- 21:58 kA fold 0 trained (343 s). New columns take 0.50 / 0.47 of the gain (seeds 0 / 1). Argmax differs from C on 272 rows.
- 22:03 kA fold 1 trained (331 s). New columns take 0.52 / 0.49 of the gain. Argmax differs from C on 259 rows.
- 22:03 August selection (`aug_dec.log`), mean of the two swapped halves:
  C before 0.4696, after 0.4978 | kA before 0.4703, after 0.4977 | kBL before 0.4708, after 0.4987.
- **22:04 FREEZE.** Candidates kA and kBL as declared; nothing was tuned on August. September is scored once from here.
- 22:09 September scored once (`eval_sep.log`, `diff_sep.log`, `recov_sep.log`, `scratch_gap/dec_N_knn.npz`).
  `subs/dd_final.csv` hash re-checked: `8e04353d...29cd`, unchanged.

## Result (September, benchmark N, supplied labels; intervals = paired event bootstrap 5%..95% against C)

| variant | August after decoder | Sep before decoder | Sep after decoder | change vs C before | change vs C after | rows changed (after) | corrected | newly wrong |
|---|---|---|---|---|---|---|---|---|
| C | 0.4978 | 0.4815 | 0.5070 | | | | | |
| kA | 0.4977 | 0.4824 | 0.5073 | +0.0008 (-0.0009..+0.0022) | +0.0003 (-0.0013..+0.0016) | 198 | 72 (fraud 54, legit 18) | 112 (fraud 64, legit 48) |
| kBL | 0.4987 | 0.4828 | 0.5078 | +0.0012 (+0.0001..+0.0024) | +0.0008 (-0.0004..+0.0019) | 86 | 39 (fraud 32, legit 7) | 42 (fraud 29, legit 13) |

- Missed fraud (after decoder; C calls 11,596 of 18,661 fraud-labelled rows legit): kA types 26 of them right and flags 4
  more with a wrong type; kBL types 15 right. Against that, fraud rows C had right and the variant gets wrong: kA 64
  (30 of them now legit), kBL 29 (5 now legit). New legit rows called fraud: kA 48, kBL 13; removed: kA 18, kBL 7.
- Largest per-class F1 moves after the decoder: kA phishing_collect -0.015, merchant_collusion +0.012, fake_cashback
  +0.008, qr_swap -0.004; kBL phishing_collect -0.010, fake_cashback +0.010. Per-class P/R/F1 in `eval_sep.log`.
- Two observations: the model gives the new columns about half of the gain, and the held-out decisions barely move.
  Why was not tested. Either the columns stand in for existing columns, or they carry a fit-only signal that does not
  transfer (the space holds sender and receiver window counts, so the same entity in another event can sit close by).
- August diagnostic: missed fraud rows are not nearer to any fraud bank than legit rows (AUC 0.50).

**Verdict (declared rule): no detected difference**, for kA and for kBL. kBL's September point gain (+0.0008) is below the
+0.001 threshold and its after-decoder interval includes 0.

## Limits
- One declared 43-dimension space and one bank design. Other spaces (all 190 columns, learned embeddings, exact-match
  restricted banks, larger k) were not tried; this result does not cover them.
- Fraud banks are 90% size for fit rows and 100% for held rows (median distances about 1% apart).
- Distances are not normalised by bank size. `kn_rmin_*` takes a minimum over fraud banks of 1.3k to 6.9k rows (fold1),
  so it partly reflects bank size.
- Events come from fit-month labels only; legit rows are cross-fitted at random, not by entity.
- Two seeds. The blend has no seed-averaging control (C seeds 0,1 + two more C seeds was not run), so its +0.0008 is not
  separated from plain averaging of more models.
- September is a reused development month; N does not reproduce October's lower density.
