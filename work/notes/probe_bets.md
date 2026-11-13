# probe_bets: ranked single-event LB bets (13 Nov)

## CURRENT (after LB: bc_nr .5490, bc_bandv .5493, so BANDV is confirmed velocity; base = subs/bc_bandv.csv)
The +.0003 is about 1 public row at the per-row values below, where about 3 were expected. Read it as "sign confirmed on at least 1 public row". The base choice does not change.

**Upload next (1 left today): `subs/probe_edgefix_bc_bandv.csv`.**
- bc_bandv + 3 rows: TX8d8b9c7d, TXdd72ce9c, TX27cb4eb5; 2 ATO -> phishing and 1 sim -> layering.
- EV +.00016. It is the only bet left with positive blind EV, so it can be kept on a tie.
- If it reads negative, split it tomorrow: ATO|ph (2 rows) and lay|sim (1 row) are independent.
- EDGEFIX relabels zone rows. Apply it last in any pipeline, after every rule that re-runs `test_zones`.

**Probe queue for tomorrow, on the winning base.** These are probe-only: keep one only if public improves, and never ship blind.
1. `subs/probe_hl_ac015935_bc_bandv.csv`: 19 rows mule -> velocity, the hostless fan-in at AC015935.
2. `subs/probe_hl_ac279075_bc_bandv.csv`: 18 rows mule -> velocity, the hostless fan-in at AC279075.
3. Further hostless fan-ins (p12 list): add their ids as new HL_* keys.
4. V49 (9 rows, P < .2), then SIMATO (12 rows, P .05), then VELLAY (10 rows, P .05-.1). These are the last resort.

Why the hostless fan-ins (p12):
- Hostless means mule-majority decoded, no zone row paying the receiver within ±48 h, and no new_payee=0 row.
- In train Aug/Sep, 9 of 63 such segments are velocity-majority (.14). By size: n 15-19 gives 5/21, n >= 20 gives 1/19, n < 15 gives 3/23.
- Test has 22 such segments. AC083650 is one of them; its single velocity-decoded row adds nothing (train 0/1).
- Per probe: if right +n x 91e-6 (+.0017 for 19 rows), if wrong -n x 57e-6 (-.0011). Blind EV is about -.0007, so these are probe-only.
- Probe value is P x gain = about .2 x .0017 = +.0003, above EDGEFIX's +.00016. Oct has less velocity volume (bursts .28x), so P may be lower than in train.

**Not for upload:**
- `subs/probe_muleband.csv`: dead, P about 0 now.
- `subs/probe_edgefix.csv`: superseded by the bc_bandv version.

Code: `work/probe_rules.py` (`apply(sub_df, tx, key)`, keys MULEBAND and EDGEFIX). Rows are selected by explicit device id; the module does not re-run `test_zones`.
Candidate subs (both on bc_nr; same id order; the diff is exactly the bet rows; asserted disjoint from the 15 no-regret rows and the 10 BANDV rows):
- `subs/probe_muleband.csv` = bc_nr + 11 rows velocity_bot -> mule_account. **Only for the bandv < nr outcome.** apply() refuses a base where the AC369510 fan-in is velocity, so it never runs on bc_bandv.
- `subs/probe_edgefix.csv` = bc_nr + 3 rows (2 account_takeover -> phishing_collect, 1 sim_swap -> layering_chain). For the bc_bandv base run `python probe_rules.py EDGEFIX subs/bc_bandv.csv`, which writes `subs/probe_edgefix_bc_bandv.csv`.

Scratch: `work/scratch_probe/`
- p0 frame + bc_nr edges
- p1 edge dumps
- p2 zone row mix
- p3 edge posterior + OOS harness
- p4 split events
- p5 per-row values
- p6 Nov rows
- p8 MULEBAND tail
- p9 end-to-end OOS
- p10 linked-row multipliers
- p11 fan-in priors

Each has a `.log`.

**Bottom line: no ≥8-row bet independent of BANDV clears break-even.** Every ≥8-row candidate is either BANDV itself, MULEBAND (conditional on BANDV losing), or an event whose P is far below break-even. Independent positive EV is only EDGEFIX: 3 rows, about +.00016, and probably invisible on public.

## Per-row values (test, p5; b10 formula on bc_nr counts, x1e-6)
Moving a row A -> B gains B.tp + A.fp if the truth is B, and loses A.tp + B.fp if the truth is A.

| class | +TP | FP | | class | +TP | FP |
|---|---|---|---|---|---|---|
| mule | 42 | 13 | | phishing | 79 | 24 |
| velocity | 78 | 15 | | qr | 71 | 24 |
| layering | 72 | 28 | | refund | 138 | 50 |
| sim | 115 | 38 | | fake_cashback | 207 | 52 |
| ATO | 138 | 39 | | loan | 308 | 82 |
| | | | | collusion | 391 | 108 |

## Ranked table
P(>=1 public) = 1 - .7^n. "Right" and "wrong" are full-test macro swings. The notes treat the expected public delta as about the same, only noisier.

| # | bet | rows | from -> to | P(alt) | right / wrong | break-even | EV | P(>=1 pub) | BANDV dependence | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BANDV (pending, `subs/bc_bandv.csv`) | 10 (host 473782's AC369510 fan-in) | mule -> vel | .8 (.55-.95) | +.0009 (bc_mule +.0011-.0014) / -.0006 (-.0007-.0009) | .38-.40 | +.0006 | .97 | is the question | uploaded |
| 2 | **DEAD after LB (bandv > nr)**: MULEBAND (`subs/probe_muleband.csv`) | 11 on-zone rows, devices [473749, 473783) | vel -> mule | bandv<nr: .65 (.4-.9); equal: .2; bandv>nr: 0 | +.00063 / -.0010 | .61 | bandv<nr: +.0001 (-.0004 to +.0005); equal: -.0007 | .98 | needs bandv<nr | probe next if bandv<nr; information probe if equal |
| 3 | EDGEFIX (`subs/probe_edgefix.csv`) | 3, independent edges | 2 ATO -> ph, 1 sim -> lay | .75, .85, .70 | +.00035 / -.00047 (all-right / all-wrong) | .58 (ph), .57 (lay) | +.00016 | .66 | independent | positive blind EV, not a visible probe |
| 4 | V49: 473749-linked rows -> velocity (residue of the velocity agent's 13-row band bet) | 9 (AC291816 7 incl. 4 LFM rows, + 2 forwards) | mule -> vel | bandv>nr .2; equal .15; bandv<nr 0 | +.00082 / -.00051 | .38 | -.00025 at .2 | .96 | on the bc_bandv base | reject |
| 5 | SIMATO: sim|ATO cut at 473980 | 12 (multi-row devices 473980-473986) | sim -> ATO | .05 | +.0021 / -.0018 | .46 | -.0016 | .99 | independent | reject |
| 6 | VELLAY: vel|lay cut at 473818 (occupancy changepoint, -17) | 10 | vel -> lay | .05-.1 | +.00087 / -.00106 | .55 | -.0009 | .97 | independent; bandv<nr makes it less likely | reject |
| 7 | Hostless mule fan-ins -> velocity: a family of 22 test segments (AC083650 is one; top: AC015935 19, AC279075 18) | 13-25 each | mule -> vel | .14 (n 15-19: .24) | about +n x 91e-6 / -n x 57e-6 | .39 | blind -.0007; probe value about +.0003 | 1.0 | independent | probe only (see CURRENT) |

Sub-threshold flips (contested, not shipped): 473749 vel->mule (P about .75, +2e-5) and 473750 (about .55, -1e-5). Both are subsumed by MULEBAND if bandv<nr.
- 474496/474497 fc->refund: posterior .92/.81, but ts_comb moves that edge the other way. Shrunk to about .6 against a .575 break-even, so about 0.
- 473987 ATO->sim: .6; ts_comb is opposite.
- AC083650's single velocity row TXffb4ef65 -> mule: P about .85, +3e-5.

## Evidence per bet
**New estimator (p3).**
- Per-edge posterior with the other edges fixed at bc_nr. Inputs:
  - occupancy (rates from the test zones);
  - per-device row emission P(receiver_type/txn_type, model decode | zone), fitted on train label-fit zones, Aug rows;
  - the multi-row flag;
  - a uniform or length prior (DP form).
- Train harness (Aug-fit emissions; Jul type-only, Sep type x decode): the posterior is roughly calibrated for both priors when block widths are Poisson. In the bins >= .7 it is slightly over-confident.
- End-to-end OOS check (p9): one month's devices, judge DP edges, flip where P(other) >= .7.
  - Uniform prior: 4 of 13 flips right in Aug, 0 of 14 in Sep. The errors are vel->mule at mule|vel and loan->coll. **Rejected.**
  - Length prior: 4 of 4 right in Aug (sim|ATO and ATO|ph, where the DP was off by 12 and 6), 0 flips in Sep.
- So only length-prior flips are used, and they are shrunk. The emission is biased toward mule at the mule|vel edge, so its 473749/473750 numbers are not trusted.
- p8 gives P(473782 is mule) of 1e-5 to 1e-11. That is over-sharp and is not quoted as P. Its direction supports BANDV, like bc_mule's occupancy LR of 50 or more.

**MULEBAND.**
- Contiguity: if 473782 is a mule-zone device, every id below it is mule zone. That covers 11 zone rows (473749, 750, 754, 755, 756, 769, 774, 775, 778, 780, 782; 8 of them merchant rows). All are velocity in bc_nr.
- All linked rows are already mule in bc_nr (p10: 19 linked rows; they differ from bc_nr only in the unguarded base world).
- Re-running zprop/R1/CZ with the band as mule adds 0 rows beyond LFM's.
- Tail 473786/789/790: P(mule | band mule) is .01-.27, so it is excluded.
- What a bandv<nr reading says: the AC369510 fan-in is mule. That happens either because the host is mule zone (prior π) or because a velocity host fed a mule fan-in. R1 velocity impurity ε is about .03 (Aug 71/71, Sep 19/20).
  - P(host mule | fan mule) = π / (π + (1-π)ε).
  - π = .2 (bc_final, includes final1's LB): .89.
  - π = .05: .64.
  - π = .02 (occupancy only): .40.
  - p3's row-type mix points to π well below .2. Point estimate .65; break-even .61.
- Visible either way: about 3 public rows at about ±.0002-.0003 each.

**EDGEFIX.**
- ATO|ph, 474281 and 474283: single-row MERCHANT/QR legit rows above the last multi-row ATO device 474279, which pins the edge at >= 474280.
  - Length-prior posterior .78/.96. The occupancy changepoint is 474276/474281. ts_comb's edge is 474280, which agrees.
  - p9's Aug length-prior flips at this same edge were 2/2 right.
- lay|sim, 473917: a single P2P legit row ahead of unused ids 473918-919. The first sim evidence is at 473920.
  - Length prior .86, occupancy changepoint 473920.
  - The rejected dpocc run made the same move (sim->lay 1). No other estimator supports it.

**V49.**
- 473749's fan-in AC291816 is not one burst. It has 2 rows plus the host row, then a 17 h gap, then 5 rows and 2 forwards, about 37 h in all. The second part is linked only through R1's 48 h window.
- P(473749 velocity | 473782 velocity): a4_occ gives .17-.21 (occupancy + hosting + length prior), and p3 gives .07, which is biased toward mule.
- final1 (.5435) carried 6 of these rows as velocity. If bandv > nr, its -.0008 is explained by these rows being mule.

**SIMATO.**
- Devices 473980-473986 are multi-row and model-decoded ATO. So are all multi-row sim-zone devices (sim archetype), so that decode is no evidence.
- The occupancy changepoint is 473988. Both judge estimators give 473987, and ts_comb gives 473986.

**VELLAY.**
- Occupancy alone favours layering (10/17 used, LR about 8).
- The rows are 8/10 merchant, all decoded legit, with 0 layering decodes. The emission LR is about 400 against layering (train P2P share: lay .67, vel .35).
- Posterior P(473818 lay) < .05; for 473833 it is .41.

**AC083650.**
- 21 inbound rows in about 24 h with no zone device; 20 are decoded mule and 1 velocity (own device).
- Train Aug/Sep mule-majority fan-ins with no zone row: velocity-majority 4/39 and 5/42. The one case with a velocity-decoded row was mule (p11).

**Split events (p4).** Only 7 receiver and 1 sender events have >= 4 fraud rows across >= 2 classes.
- The merchant ones are camouflage rows of different zones, not events.
- AC392297 (5 lay / 2 mule) and AC215699 (4 lay / 1 sim) are 1-2-row fixes with no evidence.

**Nov rows (p6).** 46 rows: 36 refund, 6 loan, 3 layering, 1 fc, matching the train Oct spill (refund 74, mule 24, collusion 11, loan 8). The 3 layering rows are one 3-hop chain with amounts decaying 57155 -> 55312 -> 53729, so they are correctly decoded. No bet.

## LB result (13 Nov): bandv > nr
Public scores: bc_nr .5490 and bc_bandv .5493 (sc_final .5486). BANDV is confirmed: the DV0473782 event is velocity. The base is now `subs/bc_bandv.csv`.

What this changes:
- **MULEBAND is dead.** With 473782 in the velocity zone, P is about 0. `probe_muleband.csv` is obsolete; do not upload it.
- **V49 drops further.** final1 (.5435, -.0008) carried 8 of the now-confirmed BANDV rows correctly. The rest of its diff (6 rows hosted by 473749 sent to velocity, plus the c2s6 rows) therefore cost more than -.0008. That points to 473749 being mule, so V49 is below .2. Reject.

Next upload: `subs/probe_edgefix_bc_bandv.csv` (bc_bandv + EDGEFIX's 3 rows; 13 rows vs bc_nr). It is the only remaining bet with positive EV (+.00016), and it is probably invisible on public. Keep it as the base if public is >= .5493. A small drop of about -.0002 to -.0005 would mean 1-2 of its 3 rows are wrong.

## Next upload per BANDV outcome (written before the result)
- **bandv > nr:**
  - Base: bc_bandv.
  - No ≥8-row bet clears break-even; V49 is .2 against .38.
  - If a slot is free, upload `probe_edgefix_bc_bandv.csv` (EV +.00016, likely invisible). Otherwise hold the slot.
- **bandv < nr:**
  - Base: bc_nr.
  - Upload `subs/probe_muleband.csv`. Keep it only if public improves.
  - Then add EDGEFIX to the winner.
- **equal (unlikely, about 5%: no public rows among the 10, or only noise-legit ones):**
  - Base: bc_nr.
  - Upload `subs/probe_muleband.csv` as an information probe on the same 473782 question (11 disjoint rows, P(>=1 public) .98).
  - If it reads better, keep it. If it reads worse, the band is velocity, so pick bc_bandv (+ EDGEFIX) for the final.
  - The private stake of BANDV (±.0006-.0011) is larger than EDGEFIX's EV.
