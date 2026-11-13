# bc_velocity: velocity_bot class sweep (13 Nov)

Scratch: `work/scratch_bc_velocity/` (b0 frame, b1 counts, b2/b3 zone row profiles, b4 p_vel tail, b5 receiver link,
b6/b8 fan-in hosts, b7 zone volume, b9 rule + bursts, b10 EV). Rule: `work/bc_rules_velocity.py`. Candidate: `subs/bc_velocity.csv`
(= sc_final + 4 rows layering -> mule).

**Verdict: velocity is saturated.** It cannot supply the +.004. The only velocity-adjacent lever is one disputed zone cut.

## (a) Test velocity decodes in sc_final: 298 rows
| source | rows | model (h6) decode |
|---|---|---|
| pool DV0420000-422999 (7 bursts, 31-58 rows each) | 269 | 269 velocity |
| velocity zone (fresh single-row phones, judge edges [473749, 473835)) | 28 | 26 legit, 1 mule, 1 layering (zone override) |
| own device, model | 1 | velocity |
- No pool row is decoded anything else. Pool devices never send P2P (train 0/3341, test 0/269).
- Device or sender sessions with >=5 rows (gap 600 s) on non-pool devices: 0 in train and in test. Bursts exist only on the pool.
- Train velocity labels per month: own device about 1,700 (camouflage), pool 800-1,100, velocity zone 41-60.

## (b) Oct structure
- **p_vel tail (b4): null.** 305 test rows have p_vel >= .05 and are not decoded velocity. All are decoded mule (fan-in rows).
  On Sep, the same bins hold 0/167 velocity for p_vel >= .1, and .098 for [.05, .1). Break-even precision is about .2.
- **Receiver link (b5, = art_acct / zprop R1): all 13 test hits come from 2 band devices, 473749 and 473782.**
  The pool contributes 0 (it never sends P2P), and the velocity core contributes 0 (0 hosts among 17 devices).
  So every Oct velocity lever beyond the 297 pool and zone rows reduces to the mule|vel cut.
  Train precision is high when the host is truly velocity: Aug 73/74, Sep 22/23 cleaned.
  The only upload that carried these rows (final1, .5435) scored below ts_judge_h6 (.5443), confounded by c2s6.
- **mule|vel band [473749, 473783): unresolved.**
  - Row-profile NB (b3; receiver type x int, amount band, gaming/ecom): Sep AUC .61. The band's summed log LR is about 0 (-0.2 to +0.8).
  - Zone volume (b7): mule share of mule+vel devices is .72/.64/.63 in Jul/Aug/Sep. Oct gives .74 with band=vel (z +1.6) and .84 with band=mule (z +3.9).
    vel/other is .20-.26 in train, against .21 (band=vel) or .13 (band=mule).
    This is NOT decisive. Velocity is the class with a known Oct volume break (bursts .28x), so band=mule (zone devices 17/60 = .28x) fits "velocity shrank everywhere".
    Moving the cut by one device changes the likelihood by only about 0.7x, so the count cannot place the edge devices 473749 or 473782.
  - Hosting: fan-in hosts per P2P row are mule .49, vel .12, lay .17. The band has 2/4, which favours mule about 3-5:1 per device. Train velocity hosts sit at both zone ends (rel pos .04-.11 and .88-.95).
  - Single cut: 473749's fan-in cannot be velocity unless 473782's is too.

### Conditional EV of the 13 band-hosted rows -> velocity (art_acct / R1), b10
| P(host velocity) | .2 | .4 | .5 | .6 | .8 | .9 |
|---|---|---|---|---|---|---|
| EV macro | -.0002 | +.0001 | +.0003 | +.0005 | +.0008 | +.0010 |
Break-even is P about .35. **Not shipped:** P cannot be estimated out-of-time. Edge refinements (dpocc, e22 emission DP, occupancy) have all failed OOT. The evidence is 2 test devices.

## (c) Shipped rule: LFM (layering fan-in -> mule), 4 test rows
Rule: an off-zone row decoded layering, whose P2P receiver gets a mule-zone or velocity-zone fresh-device row within +-48 h and no layering-zone row -> mule_account.
This does not depend on the mule|vel cut. Both zones are on the same side of the rule.
- Train, label-only, label-fit zones: the layering share of linked rows (y in mule/vel/lay) is .5% / .8% / .3% (Jul/Aug/Sep; n 418/392/323).
  This is asserted in `bc_rules_velocity.check_train`.
- Judge-decoded layering hits: Aug 8 (cleaned 6 mule, 1 vel, 1 legit) and Sep 4 (4 mule). 0 are layering.
  The window was chosen on Aug (48 h: 8 hits; 24 h: 2) and checked on Sep (4/4).
- Test: 4 rows layering -> mule, all from the 473749-hosted fan-in. Its mule-core-hosted rows are already decoded mule (0 layering hits).
- Target choice, by EV per class (b10; Oct per-row value from Sep cleaned prec/rec):

  | P(host mule) | .2 | .5 | .8 |
  |---|---|---|---|
  | -> mule | +.0001 | +.00017 | +.00023 |
  | -> velocity | +.00035 | +.00023 | +.00012 |

  Mule is positive under both band resolutions (no-regret). Fan-in hosts are mule 103 : velocity 8 in train. Velocity would be a bet on the cut.
- Expected gain: about +.0001 to +.0002 macro (mule/layering, not velocity). Probably 1-2 rows are public, so the LB will not show it.

Per-row macro values on test (b10): mule +TP .000043 / FP .000013; velocity +TP .000078 / FP .000015; layering +TP .000070 / FP .000027.
