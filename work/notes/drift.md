# Drift and label-noise audit (drift agent, 2026-11-13)

Scripts: `work/scratch_drift/d*.py`, outputs `work/scratch_drift/*.txt`. Code deliverables: `work/density.py`, `work/clean_labels.py`.

## TL;DR (ranked by measured impact)

1. **Oct has half the density, but fraud events are complete.** Legit rows per day are 8.4k in Oct and 17.0k in Jul-Sep. Every legit-background statistic matches a random 50 % thinning of Sep exactly: senders, tx per sender, merchant volume quantiles, sender gap. Fraud events are not thinned, but there are fewer of them. Velocity device bursts in Oct keep full size (38 rows, 9 s gaps), while a thinned Sep would give 19 rows and 18 s gaps.
   Window and graph features built on full-density train therefore shift on test. Adversarial AUC Sep vs Oct: burst 0.70, ring 0.87, scam 0.89. Flow (0.51) and takeover (0.54) are fine.
   **Measured on Sep** with base+burst+ring+scam, tuned macro-F1 (`scratch_drift/eval_octlike.py`):
   - Full-density features: 0.4559.
   - Oct-density Sep features scored by the current full-density fit: **0.4363 (-0.020)**. merchant_collusion drops .36 to .21, refund .54 to .47, loan .38 to .35.
   - **Density-matched fit: 0.4527**. This recovers about 80 % of the loss: argmax 0.4234 goes to 0.4458, mule .31 to .35, ATO .26 to .31.
   - flow+takeover (low drift): REF 0.4127, FULL 0.4142, MATCH **0.4200**. Matching does not hurt modules that do not drift, and it adds +0.006.
   - Fix: `density.run_feature_matched(name)` (cached as `feats_cache/<name>__dm.parquet`, same contract as `run_feature`). Use it for every window/graph module in train AND test. Validate on its output, because Sep rows then sit at Oct density.
2. **ip_risk_score: keep dropped. A within-month rank does not fix it.**
   - The vendor score was recalibrated in test.csv only. The 147 Oct-dated train rows keep the old scale and app 3.x/4.x, so the change follows the file, not the date.
   - Sep legit looks like Beta(2,8) (mean .20, sd .12). Fraud of every type looks like Beta(1.3,3.3) (mean .28, sd .19). All of Oct looks like Beta(4.2,15) (mean .22, sd .09).
   - Fraud signal is mostly gone. Score Oct with a model that has no ip: rows with P(fraud)>0.95 sit at a mean ip rank of 0.54 in Oct, against 0.78 in Sep. corr(P(fraud), ip) falls from 0.19 to 0.025. ip has no entity persistence (per-sender split-half corr 0.10, Sep-to-Oct corr 0.01).
   - Note: the "legit mean 0.44 in Oct" claim in the brief came from the 18 legit-labelled Oct-dated train rows. Those rows are mislabelled fraud on the old scale, not Oct legit.
3. **app_version: keep dropped.** In train, fraud has 32 % app 3.x against 15 % for legit, a real train signal. Test is 100 % 5.0/5.1/5.2, spread uniformly over apps, so the signal does not exist there.
4. **QR evasion ("rings adapt").** In Oct, 140 P2P QR rows have `qr_vpa_mismatch` = null. Train has 0 such rows out of about 2,000 P2P QR rows. In train, P2P QR with mismatch=1 is 1084/1300 qr_swap. Oct P2P QR mm=1 drops to 112, where ~200 would be expected at half density.
   So about half of Oct qr_swap P2P-QR rows probably now show null in place of 1. Trees send this unseen null pattern down an arbitrary branch.
   Suggested fix (not testable on Sep): for `txn_type=QR & receiver_type=P2P`, treat null as 1 (`qr_mm_p2p = QR & P2P & (mm==1 | mm is null)`). Merchant-QR null exists in train (mc/refund/qr_swap) and keeps its rate.
5. **Festival amounts: do NOT rescale.** All legit amounts in Oct are x1.17 (every txn_type and category gives 1.15-1.20; integer share unchanged at 26.9 %). Fraud signature amounts are not scaled: COLLECT 1999/4999/9999 and merchant 1000/2000/5000/10000 still appear at the exact values, about 0.5x as often, in line with the density.
   Simulation on Sep (`d13_amount.py`):
   - legit x1.17 alone costs only -0.0035.
   - Deflating all amounts by 1.17 costs -0.06: layering .35 goes to .06 and merchant_collusion to .02.
   - Keep raw amounts. Amount signatures stay exact.
6. **Ages drift by design.** No account is opened after 1 Jul, so every age grows about 30 days per month.
   - Sep-vs-Oct adversarial AUC is 0.80 on base features, driven by mer_age_days x merchant_category (merchant identity), sender_account_age_days and rcv_acct_age_days. Without ip and ages the AUC is 0.56: weekday (calendar) and amount.
   - sender_account_age / rcv_acct_age / mer_age carry no fraud signal: label medians match legit. They are candidates to drop. quick_eval with `--drop sender_account_age_days,rcv_acct_age_days,mer_age_days,weekday` gives 0.3566 / 0.3756, against 0.3577 / 0.3773 baseline. That is no real loss, and it removes the time-trend and calendar features.
   - device_age_days and sim_age_hours are per-row draws. They are stable and informative: ATO and velocity have low device_age.
7. **Stable:** hour mix, txn_type mix, receiver_type, app shares, merchant_category mix, new_payee, pin_retries, login_to_txn_sec, `cents` (integer share 0.2685 vs 0.269), balance (+1.6 %).

## Other findings

### Split rule: 147 Oct train rows, 46 Nov test rows

- Rows are assigned by **event start month**. A fraud event that starts in Sep goes to train whole, even its Oct rows (147 rows up to 10 Oct). An event that starts in Oct goes to test whole, including Nov rows (46 rows up to 9 Nov, mostly REFUND and layering).
- The legit background is generated per month, so:
  - Every Nov-dated test row is fraud. These are free positives: predict a fraud class for them, never legit.
  - Every Oct-dated train row is fraud. 18 of the 147 are labelled legit, which is a direct read of the noise rate.
  - The Oct-dated train rows have old ip scale and app 3.x/4.x. The ip/app transform was applied to test.csv, not by date.
- Early-Oct parts of Sep events are in train with labels. Graph features see them at train density.

### Test class prior (`d5_prior.py`)

Base model (no ip/app) fit on Jul-Aug, mean predicted probability per 1k rows, Sep vs Oct:

| class | Sep | Oct |
|---|---|---|
| mule | 6.96 | 7.35 |
| velocity | 5.23 | **4.37** |
| layering | 4.54 | 4.64 |
| sim | 2.59 | 2.65 |
| ATO | 2.46 | 2.52 |
| phishing | 4.47 | 4.54 |
| qr | 4.94 | **4.42** |
| refund | 2.12 | 2.34 |
| fake_cashback | 1.39 | 1.42 |
| loan | 0.97 | 0.99 |
| merchant_collusion | 1.28 | 1.29 |

- The fraud share per row is about the same in Oct. The true clean share is about 4.1 %, against 3.66 % noisy-labelled.
- Argmax counts changed more: velocity 2.06 to 1.05 per 1k, qr 2.32 to 1.78, merchant_collusion .22 to .16, refund .76 to .98.
- **velocity_bot changed.**
  - Device bursts of 4+ rows within 120 s: 0.23 per day in Oct, against 0.87 in Sep and 0.93 in a 50 %-thinned Sep.
  - Sender bursts: 0.35 against 1.07.
  - The Oct bursts that exist look like train velocity (38 rows, 9 s gaps, 100 % gaming/ecommerce).
  - Expect about 1/3 as many velocity events, or bots that spread out. Do not tune the velocity bias up on Sep expecting Sep rates.
- **qr_swap changed** through the mismatch-null evasion (item 4).

### Label noise (`d3_octrows.py`, `d6_events.py`)

Noise is **per row and independent**, not per event. Inside one mule event of 30 rows, 2 rows are labelled legit, 1 velocity_bot and 1 account_takeover.

- Fraud labelled legit: about **11-12 %**.
  - Oct-dated train rows (all fraud): 18/147 = 12.2 %.
  - Legit rows inside tight events on the event key:
    - mule receiver: 273 in-span against 43 in same-length control windows, so ~8 % of mule rows.
    - velocity device: 428 against 34, so ~11 %.
    - merchant_collusion pair: 51 against 9.5.
- Fraud with the wrong type: about **4 %**. In events of 3 or more rows (6 h, keys sender / P2P receiver or sender|merchant / device), 3.8 % of rows carry a different fraud type. The wrong type is uniform over the other 10 types (0.3-0.4 % each), with no confusion structure.
- Tight events of 3 or more rows exist mostly for mule, velocity, layering, sim_swap and ATO. phishing, qr_swap, refund, loan and fake_cashback are mostly single rows on these keys, so their noise cannot be fixed by event voting.

**Cleaning recipe**, `clean_labels.clean(tx)`:
- Majority relabel inside events of 3+ rows (majority at least 60 %).
- Legit rows inside the event span on the receiver key (mule / velocity / merchant_collusion) or on the device (velocity) are relabelled to the event type.
- Changes 1,165 train rows.

**Result** (`scratch_drift/eval_clean.py`, quick_eval protocol, base features, 1 seed):

| train labels | Sep raw tuned | Sep cleaned tuned |
|---|---|---|
| raw y | 0.3761 | 0.3844 |
| cleaned y | **0.3781** | **0.3877** |

- The gain is +0.002 to +0.003, at noise level: layering +0.03, merchant_collusion +0.02, mule and velocity flat.
- Optional; it is cheap. The bigger value is the **cleaned Sep** as the validation target: 396 Sep rows change, and the model agrees with the relabels (+0.008 macro).

**Decode biases under noise** (noise-on-noise on `oof/base.npy`): biases tuned on labels with an extra 11 %/4.5 % noise lose ≤0.0004 F1 on the original labels, and the bias shifts are ≤0.13. So bias tuning on noisy Sep transfers to clean test without correction. The extra noise costs 0.027 F1 at fixed predictions, so **clean-test F1 should read about +0.03 above noisy-val F1**, before drift.

## Gotcha found on the way

`quick_eval.base_features` encodes categoricals with `cast(pl.Categorical).to_physical()`. In polars 2.1 the codes depend on the frame: computing it on a subset gives different codes (txn_type 97 % differ, merchant_category 100 %). It is safe only when computed once on the full `load_all()` frame, which is what quick_eval and train.py do. Anyone who calls it on a subset (Oct-only inference, simulations) gets garbage. A fixed mapping would remove the trap: sorted uniques, or `pl.Enum`.

## Recommendations for the trainer

1. Build window/graph modules with `density.run_feature_matched`, for all of train and test.
2. Keep `DRIFT_DROP`. Also drop `sender_account_age_days,rcv_acct_age_days,mer_age_days,weekday`: it costs -0.001 on Sep and removes features that drift by design.
3. Add the QR-P2P null fix (item 4) inside a module or in base_features.
4. Hard-assign Nov-dated test rows to a fraud class (argmax over the non-legit classes).
5. Optional: train on `clean_labels.clean(tx)` and tune biases on cleaned Sep.
6. Expect velocity_bot and qr_swap to be the classes that slip on the leaderboard.
