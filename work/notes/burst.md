# burst module: velocity_bot and refund_abuse

Module: `work/feats/burst.py`. Cache: `feats_cache/burst.parquet` (73 cols). Build takes about 40 s.
Scratch scripts: `work/scratch_burst/`.

## Generator signatures (from train labels)

### velocity_bot (8106 train rows)
- About 35% of the rows sit in device bursts. There are about 28 bursts per month in train and 7 in Oct.
  - Each burst is one `device_id` with 20-60 rows (median 40) in 2-14 min (median 8 min).
  - All amounts are integers, median 245 and mostly at or below 500.
  - The burst pays 1-3 receivers. 99% are merchants: ecommerce in 74/85 bursts, gaming in 10/85.
- There are two bot styles:
  - (a) A new sender account for every txn: n_sender = n. `login_to_txn_sec` is about 3-4 s.
  - (b) Exactly 3 sender accounts rotate on the device. Some of these bursts have `login_to_txn_sec` of 40-60 s (a "humanised" bot).
- Label noise inside bursts: 70-97% of each burst is labelled velocity_bot. About 13% is labelled legit, and the rest is spread over other fraud types.
  - These are the "fraud labelled legit" and "wrong type" noise. Test labels are clean, so all burst rows should be velocity_bot.
- The other 65% of the rows are singletons that look like legit:
  - No shared sender, device or receiver with the bursts, and no closeness in time to them.
  - Entity overlap with other fraud is the same as for legit rows.
  - They differ only slightly: integer amounts 31% vs 26% for legit, night share 4.9% vs 3.2%, slightly larger amounts.
  - The features tested here do not find them. If they stay unfound, class F1 on noisy Sep stays near 0.5 (a scenario, not a bound).

### refund_abuse (3305 train rows)
- Only 950 rows are REFUND rows. In each event, one sender files 3-8 REFUNDs (a few have up to 15) over about 4-9 days (median span 7 d).
  - The refunds go to 1-2 merchants, with mean nrcv 1.5.
  - About 70% are ecommerce and 25% electronics.
  - Amounts have paise and sit around 1500-4500 (median about 2250). Usually one device.
  - There are about 60-65 events per month in train and about 32 in Oct, with the same shape (checked on test: n 3-8, span 2-10 d, 1-2 merchants, ecommerce/electronics, amounts 1.5k-6k).
- Legit refunders mostly have 1 refund. Legit senders with 3 or more refunds spread them over 10-110 days, to 3 different merchants, with small amounts.
- The other 2355 rows (70%) are non-REFUND singletons that look like legit. Their sender overlaps with abuser senders in only 0.7% of rows, and they have no link in time.
  - The one oddity is amount % 100 == 0: 16.5% vs 2.4% for legit. That is a row feature and the base `cents` already uses it.
  - The features tested here do not find these rows.
- In Oct, abusers spread refunds over more merchants: the ratio of pair_ref14d≥5 to pair_ref14d≥2 is 0.52 in Oct vs 0.67 in Sep. So counts per sender are more robust than counts per pair.

### General finding, likely true for all types
Each fraud label seems to be about 1/3 detectable "event" rows and about 2/3 camouflage rows drawn like legit.
The ratio is 1757:902 for VB in Sep and 745:330 for RA.
So per-class F1 on noisy validation stays near 0.5 while these rows are unfound, which matches the organiser's note that "0.50 is not expected".

## DRIFT (important for everyone)
- Oct (test) has about 8.4k rows per day vs about 17k per day in Jul-Sep. Volume is half, distinct senders are 148k vs 215k per month, and fraud event counts are also about half.
  - Any raw count of activity per entity over long windows drifts. Example: sender ±7d count has mean 2.88 in Sep vs 1.96 in Oct.
  - Raw merchant in-counts drift. Example: merchant integer-payment count ±1d, clipped at 10, has mean 2.67 in Sep vs 1.94 in Oct, and the share at ≥10 is 17% vs 10%.
  - Use counts that come from the event itself, ratios, or P2P-only counts. P2P in-counts are stable: mean 0.79 in both Sep and Oct.
- Oct amounts are larger (festival). For example, the mean of the session amount median went from 1234 to 1434. Avoid copies of raw amount.
- A continuous statistic per event (such as a session's median amount) works as an event ID. LightGBM memorises it and overfits, so I replaced it with coarse fractions.
- Train has 147 labelled rows dated Oct 1-10 and test has rows into Nov 1-9. These are events that started in the previous month, which supports "events are assigned to a month by their start time".

## Features (all symmetric windows)
- `dev_c{60,300,1800}`, `snd_c{60,300,1800}`: count of the same device or sender within ±w seconds.
- `ds_*` / `ss_*`: device or sender sessions, where a gap over 600 s starts a new session.
  - Columns: n, n_sender / n_device, nrcv, dur_min, int_frac, small_frac (amount ≤ 500), cat_frac (ecommerce/gaming), min_gap (clipped at 3600 s).
  - The stats are null when the session has fewer than 3 rows.
- `devrcv_c300`, `devrcv_share300`: same (device, receiver) within ±5 min. This separates bot bursts (one device) from mule in-flows (many devices).
- `snd_ref{3,7,14}d`, `dev_ref7d`, `snd_ref_share7d`, `pair_ref14d`: REFUND counts per sender, device or pair.
- `rs_*`: the sender's refund episode, where a gap over 4 days starts a new episode. Only on REFUND rows.
  - Columns: n, n_device, nrcv, dur_min, int_frac, small_frac, cat_frac (ecommerce/electronics).
- `rcv_int_share{300,1d}`, `rcv_ref_share1d`: ratios per receiver, safe against the volume change.
- `p2p_int_in1d`: integer-amount payments into a P2P receiver within ±1 day, clipped at 10. This is a strong mule and layering signal.
- `mer_int_burst`: a merchant's integer in-count over ±1d divided by its count over ±15d.

## Eval (quick_eval.py burst --threads 2; Jul-Aug fit, Sep score)
Baseline: macro argmax 0.3577 and tuned 0.3773. VB 0.457, RA 0.327.

| version | macro argmax / tuned | VB | RA | notes |
|---|---|---|---|---|
| v1 raw windows + sessions | 0.3912 / 0.3993 | 0.453 | 0.420 | merchant raw counts helped MC (0.19), but they drift |
| v2/v3 drift-safe, refund episodes | 0.3837 / 0.3979 | 0.482 | 0.443 | |
| v4 session stats nulled for singletons | 0.3818 / 0.3982 | 0.477 | 0.434 | VB burst recall 98.2%, RA REFUND recall 97.9% |
| v7 + merchant int count clip10 + devrcv | 0.4243 / 0.4320 | 0.454 | 0.473 | mule 0.37, layering 0.48, MC 0.31; merchant count drifts |
| **v9 FINAL (P2P-only in-count + mer_int_burst)** | **0.4004 / 0.4122** | 0.459 | 0.438 | mule 0.377, layering 0.478, MC 0.084 |

v9 per-class argmax F1:
legit 0.987, mule 0.377, velocity 0.459, layering 0.478, sim_swap 0.332, ATO 0.304, phishing 0.404, qr_swap 0.439, refund 0.438, fake_cashback 0.158, loan_app 0.343, merchant_collusion 0.084.

The per-class numbers move by about ±0.03 between runs for small changes, so this eval is noisy. Merchant_collusion is the least stable.

Where the remaining Sep loss for VB and RA comes from:
- The camouflage singletons, which no tested feature finds.
- The 13% of burst rows and the 15% of abuser REFUND rows labelled legit. The model correctly predicts these as fraud, but they count as false positives on noisy Sep. On clean test they should count as true positives.

## Ideas not done
- Merchant_collusion owner: the raw count of integer-amount payments into a merchant per day, clipped at 10, raised MC to 0.31 on Sep. It drifts with Oct volume, so a version normalised by volume is needed.
  - My ratio (`mer_int_burst`) did not capture it. Try distinct integer-paying senders per merchant per day, or a rank within the month.
- Final model: post-process at session level. If a device session has n ≥ 20, int_frac = 1 and cat_frac ≥ 0.8, force velocity_bot for every row. Likewise force refund_abuse for every REFUND row in an episode with n ≥ 3 and span ≤ 10 d.
  - This fixes the trees' instability on burst rows: VB burst recall swung between 82% and 98% across versions.

## v10 (adversarial review, scratch_burstrev/)
- No label leak, no absolute time, symmetric windows. Edge truncation is negligible: test runs to Nov 9, late-Oct vs early-Oct means match.
- Fix: the 6 ratio features (`snd_ref_share7d`, `rcv_int_share300`, `rcv_ref_share1d`, `rcv_int_share1d`, `mer_int_burst`, `devrcv_share300`) are now leave-self-out, NaN when the row is alone in its window.
  - Reason: with self included, half density pulls each share toward the row's own flag. On 50%-thinned Sep, legit REFUND rows had `snd_ref_share7d` +19% and `rcv_ref_share1d` +18%.
- quick_eval: v9 0.4004 / 0.4122, v10 **0.4083 / 0.4246**.
- Oct-density eval (`scratch_drift/eval_octlike.py burst`, tuned): v9 REF .4145, FULL .4146, MATCH .4101. v10 REF .4255, FULL .4259, MATCH .4261.
  - Most of the gain is merchant_collusion (.05 to .10-.14), which is noisy. Velocity gains about +.01.
- Still density-sensitive but harmless for burst alone (FULL is about REF): raw `snd_ref*d` / `dev_ref7d` (legit mean -30% at half density) and the composition of `rs_*` / `ds_cat_frac`.
