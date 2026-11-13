# takeover module: sim_swap + account_takeover

Module: `work/feats/takeover.py` (34 cols, prefix `takeover__`). Cache: `work/feats_cache/takeover.parquet`.
Scratch scripts: `work/scratch_takeover/` (eda*.py, qe_dump.py, qe_clean.py, rule.py, drift.py).

## Generator signatures (train Jul-Sep)

Sender session = rows of the same sender with gaps <= 2h.

1. **About 61% of sim_swap/ATO labels are single rows that look exactly like legit.** sim_swap has 2486/4072 such rows; ATO has 2478/4049.
   - Every stat matches legit: sim<48h 0.7%, share>0.5 4%, new_payee 0.24, P2P 0.37, txn_type mix like legit.
   - Their sender has a burst elsewhere at the base rate (1.2%, same as legit).
   - Their receiver was drained by another sender at about the base rate (1.6-2.4% vs 1.2% for legit).
   - These rows are label noise or undetectable decoys, at about 830 per month per class. The same singleton check gives high values for other classes too (layering 0.98, mule 0.95, qr 0.95 of rows in a 1-row sender session). Those classes are not sender bursts, so their owners need their own check.
2. **Real events are drain bursts.** Each is 2-4 rows (n=2: 428 rows, n=3: 570, n=4: 588 for sim) from the victim's account.
   - P2P PAY to a NEW payee, with integer amounts (99.5%).
   - Gaps between rows: median 10-14 min, p95 37-52 min. Session span p90 is 64-79 min.
   - Each row pays about 0.7x the current balance: median share is 0.70 at pos1, 0.74 at pos2, 1.0 at pos3.
   - The last rows are often exactly Rs 500 even when the balance is lower (share > 1).
   - Balance falls between rows but is not exactly consistent.
   - Receivers are normal accounts. Some are reused across events: 457 receivers got drains from 2 victims, 66 from 3. That is far above chance, so a ring is reusing mules.
3. **There are two archetypes, and the type labels are ~30% swapped between them.**
   - **SIM archetype:** same device as usual, sim_age_hours < 48 (min ~7h), home city (away 7%), night 72% (hour <= 5), login 14s, pin 0.13. Labels: 350 sim_swap / 152 ATO events.
   - **DEVICE archetype:** device_age_days < 3 (min 0.3), a device the sender never uses outside the burst, away from home 90%, pin_retries 0.83, daytime (night 5%), login 48s. Labels: 323 ATO / 143 sim_swap events.
   - Within one archetype, sim-labelled and ATO-labelled events are identical on every other stat (night, away, pin, login, amounts, span, size). So the label is noise: the clean answer is SIM -> sim_swap and DEVICE -> account_takeover.
   - Small leftovers: "D" (device only used in the burst but old device_age) has 18 ATO / 19 sim events. "none" has 88 / 66.
   - For legit rows, sim_age_hours and device_age_days are redrawn at random on every row, even on the same device.
4. **Oct (test):**
   - Oct has half the daily rows: 8.4k vs 17k per day. It has 148k senders vs 215k, and 1.76 vs 2.37 txns per sender.
   - The test set is NOT row-thinned: drain sessions of size 2/3/4 all shrink by the same ratio (~0.55).
   - Fraud burst profile in Oct is unchanged from Sep: simf 0.45, dev_young 0.44, away 0.48, night 0.39, share 0.82, span 1600s.
   - Archetype mix is unchanged: Oct S=91 / DAW=84 vs Sep 161 / 155. No sign that these rings adapted.

## Drift check of module features (Sep vs Oct)

- Stable: dev_excl, city_excl, sim/dev flags, away, night, pin, login, rcv_drain_snd. Fraction nonzero matches within 0.002.
- Shifted (prior shift only, kept): fewer legit multi-row sessions in Oct.
  - sess_span nonzero: 0.033 -> 0.018.
  - sess_bal_down: 0.019 -> 0.011.
  - next_bal_ratio: p99 48 -> 81.
- Dropping these three cost 0.006 macro (mostly layering), so they stay.
- share (amount/balance) rises for legit in Oct: mean 0.144 -> 0.160, share > 0.5 rate 0.043 -> 0.052. This is festival spending. The drain flag also needs int amount + new P2P payee + 2 drains in the session, so its nonzero rate only moves 0.0071 -> 0.0079.

## Eval (quick_eval, fit Jul-Aug -> Sep, noisy Sep labels)

| run | macro argmax | macro tuned | sim | ATO |
|---|---|---|---|---|
| baseline | 0.3577 | 0.3773 | 0.320 | 0.290 |
| v1 session/device/drain feats | 0.3881 | 0.4028 | 0.334 | 0.323 |
| v2 + archetype flags (FINAL) | 0.3864 | 0.4007 | 0.351 | 0.307 |
| v3 v2 minus drifty feats | 0.3817 | 0.3996 | 0.326 | 0.319 |

- Other classes gain from rcv_drain_snd and sess_drain: mule .22 -> .31, layering .35 -> .42.
- Burst detection is nearly perfect on Sep: 409 of 417 sim burst rows are caught as sim or ATO.
- What is left is singleton noise (can't fix) and sim/ATO type noise.

## Label cleaning: biggest remaining gain, a trainer-level change

Cleaning rule for train labels with y in {sim_swap, ATO}:
- arch_sim=1 -> sim_swap
- arch_ato=1 -> account_takeover
- sess_n==1 -> legit

Results (`scratch_takeover/qe_clean.py`):

| train labels | val noisy Sep | val clean-proxy Sep (same cleaning) |
|---|---|---|
| raw | macro .3864, sim .351, ATO .307 | sim .692, ATO .647 |
| cleaned | macro .3944, sim .389, ATO .382 | sim .905, ATO .882 |

- The hard archetype override on top of the raw model gives sim .375 / ATO .379 on noisy Sep.
- Recommendation: clean labels before training the final model.
- Bias tuning on noisy Sep under-calls these classes in clean test, because 61% of Sep positives are undetectable. Tune on cleaned Sep, or keep biases near 0 for sim/ATO.
