# typesplit (victim angle): sim_swap vs account_takeover, and a device-id allocation leak

Agent key `victim`, 13 Nov. Scratch: `work/scratch_ts_victim/` (ev.py / rows.py build the event tables; a1-a23 analyses; post.log).
New code: `work/feats/ts_victim.py` (label-free features), `work/ts_victim_post.py` (post-rule + validation, writes the subs).

## 1. Victim-side history: null
Events are sender sessions (gap 2 h) with >= 2 drain rows; event label = the sim/ATO label of its rows.
- Labels are event-pure: of 1137 labelled events, only 6 mix sim and ATO rows. The ~30% swap is per event.
- Inside each archetype (S = sim_age < 48 h on the own phone, D = phone younger than 3 days), sim- and ATO-labelled events
  match on every victim stat. Checked:
  - usual device and pre/post activity on the own phone
  - gap to the last and the next own activity
  - number of devices, device shared with other accounts
  - home, event city and usual city, fraud-hub cities
  - app, version, OS, KYC, bank, account age
  - sim/device-age increments (redrawn per row, never consistent)
  - receiver ring reuse: label agreement is .63 vs .565 expected; cross-archetype pairs agree less than chance
  - receiver account numbers
- An LGB on 46 event features (Jul+Aug -> Sep) scored .677 against .727 for the archetype rule.
- The only weak signal is event size. Swapped events are smaller: S-archetype n=2 gives .58 sim, n=4 gives .85. It never
  flips the archetype's majority.
- N archetype (neither flag): night at home -> sim .81; away -> ATO 1.00 (28/28); D at home -> ATO .97.

## 2. The real rule: device numbers are allocated per fraud class
- An account's own phone is `DV0<account n>`. Fresh phones get n >= 420000 from counters. In train the counters run in
  class order, in contiguous blocks:
  `mule 423000 | velocity 423657 | layering 424154 | sim_swap 424627 | ATO 425036 | phishing 426801 | qr 427166 | refund 427847 | fake_cashback 428061 | loan 428201 | collusion 428305-428454`
- After the fraud blocks come legit fresh phones (428455+), where every number is used. Multi-sender velocity bot phones
  sit below 423000.
- Sorted by device number, adjacent D-archetype events share a label 99.8% of the time; sorted by time, 58%.
- Per month, a block's class equals the row label 85% of the time. The misses are 12% legit and 3-4% another type, which
  is exactly the label-noise rate. So the block gives the true class, including camouflage rows and noise-flipped rows.
- Every fraud class puts ~3% of its rows on block phones (the legit fresh-phone rate). Sim and ATO put more:
  - sim: 12% of rows, the D-archetype sim events
  - ATO: 25% of rows
- Test (Oct) is numbered separately: fraud region 473632-474551, legit from ~474552. The class order is the same, and
  block widths are proportional to train (sim ends at .389 of the region vs .373; ATO at .712 vs .697).
- The sim block is dense (mean gap 1.40 train / 1.54 test). The ATO block is sparse (3.77 / 3.83). A two-rate gap
  changepoint finds the sim|ATO boundary label-free: train 425032 (true ~425035), test 473987.
- Limit: S-archetype events run on the victim's own phone, so the leak cannot split them. They stay at ~72%.

## 3. Out-of-time validation (`ts_victim_post.py`)
- Block cores are fit label-free: a monotone DP over the region's devices, scored by h6_ppq's own decoded preds, plus the
  gap changepoint for sim|ATO.
- They are fit on Aug OOF preds only and scored on Sep.

| Sep | noisy macro | cleaned macro | sim / ATO F1 (cleaned) | rows changed |
|---|---|---|---|---|
| h6_ppq | .4855 | .4979 | .409 / .396 | - |
| **ts_victim** (sim/ATO-decoded rows on a sim/ATO block phone take the block type) | .4932 | .5062 | .483 / .422 | 102 |
| **ts_victim_blk** (every row on a block-core phone takes the block class) | **.5051** | **.5180** | .492 / .440 | 483 |

- Sep takeover-event accuracy: archetype rule .727 -> block rule **.848**.
  - On block phones: 1.000 (154 events).
  - On own phones: .722 (187 events). There is no rule for these.
- blk per class, cleaned Sep delta: mule +.014, layering +.011, sim +.083, ATO +.044, phishing +.017, qr +.017,
  refund +.013, fc +.011, loan +.009, collusion +.022. Velocity: 0.
- In train the DP merges the velocity block into the mule core (velocity rows predicted mule). In test the cores leave the
  velocity zone 473783-473838 untouched.
- Oct consistency: D-archetype events split 31 sim-block / 59 ATO-block (34%; Sep 27%). Sim-block events are away 100%
  and small (n 2.35), as in Sep (100%, 2.71).
- Test cores:
  - mule 473642-473782, layering 473839-473915
  - sim 473920-473987, ATO 473990-474279
  - phishing 474285-474351, qr 474372-474458, refund 474462-474495, loan 474517-474519
  - fake_cashback and collusion: under 2 confident devices, so not set.

## 4. Candidates (decode == subs/h6_ppq.csv verified bit-equal before overrides)
- `subs/ts_victim.csv`: 75 test rows change vs h6_ppq, all sim<->ATO. Narrow, safe.
- `subs/ts_victim_blk.csv`: 233 test rows change, 154 of them from legit. Expected about +.02 macro (Sep).
  **Recommended upload.**
- `feats/ts_victim.py` (blkpos, gapden, ext) is label-free but not yet in any model. blkpos transfers because block
  proportions match, but boundary offsets of ~.015 make tree splits fuzzy. The explicit post-rule is more exact. Retraining
  with it is optional.

## Risks
- This exploits a generator artifact (device-id numbering). It is valid data in test.csv, but an organizer could see it as
  leak use. Decide before the final selection.
- Test block boundaries come from h6's own preds. Uncertain gaps between cores are left untouched.
- The bait-pair targets (phishing/fake_cashback, loan/phishing) are only fixed for rows on block phones (~3%). The general
  60/40 ambiguity is not addressed here.
