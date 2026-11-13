# bc_mule: mule_account dive (13 Nov)

Code: `work/bc_rules_mule.py` (`apply(sub_df, tx)`). Scratch: `work/scratch_bc_mule/` (`prep.py` frame, `a1`-`a8`, `chk_rules.py`; each has a `.log`).
Candidates on top of `subs/sc_final.csv`:
- `subs/bc_mule.csv`: 10 rows mule -> velocity. Recommended.
- `subs/bc_mule_b.csv`: the same 10 rows plus 4 rows layering -> mule. This is a bet on the guarded ids; see below.

## (a) Test rows decoded mule in sc_final: 794
| source | fan-in | forward | P2P lone | P2P multi | merchant |
|---|---|---|---|---|---|
| zone (mule zone device) | 17 | 2 | 25 | 0 | 35 |
| model | 627 | 75 | 2 | 7 | 0 |
| rule (v3 R3 lay->mule) | 0 | 0 | 4 | 0 | 0 |

- Fan-in means a P2P receiver segment with a 12 h gap, at least 5 senders and at least 80% round-10 amounts. Forward means a P2P send by a fan-in receiver within 6 h after the segment.
- Mule zone rows (79) by model decode: 56 legit, 19 mule, 4 layering.

## (b) Oct-specific structure: all null except the band
- **Burst completeness (null).**
  - Oct has 55 fan-in segments (721 rows). None of their rows is decoded legit.
  - In train, the legit-decoded fan-in rows are all labelled legit: Aug 29/29 and Sep 20/20. They are small coincidental segments (median 6-11 rows), not missed mule rows.
- **Forward rows (null).**
  - Oct has 78 forward rows; 1 is decoded legit.
  - Train legit-decoded forward rows: Aug 0 mule of 8 (6 legit), Sep 0 of 9 (9 legit).
- **Fan-in type.**
  - Label majority over 321 train segments: 194 mule, 19 velocity, 108 layering.
  - Decoded majority vs label majority (Aug/Sep): mule-decoded segments are mule 122, velocity 15; layering-decoded segments are layering 63, mule 3.
  - Within-segment smoothing of mixed segments is not consistently positive, because the minority decodes are often right (velocity zone hosts). Not proposed.
- **Velocity-labelled fan-ins vs mule-labelled ones** (`a5`, `a7`).
  - They match on size, amount, duration, login, device age, SIM age, hour and r100.
  - They never contain a new_payee=0 row, but 82-88% of mule fan-ins have none either, so P(velocity | all rows new_payee=1) is only .05-.15.
  - No label-free tell exists except a velocity-zone host row. Forward rows follow the fan-in's type: velocity fan-ins 35/37 velocity, mule fan-ins 330/388 mule.
- **Oct-only cells:** nothing new for mule. The zone, structural and decoded rates are all about 1.2x Sep (octcal).

## The disputed band [473749, 473783): resolved for 473757 and above by device-id occupancy
Zones have class-specific occupancy (used ids / ids). Train: mule .53, velocity .35. Test: mule zone .68, velocity rest .33, band 11/34 = .32.

DV0473782 can only be mule if the whole band is mule zone, because zones are contiguous.
- Train 34-id windows with at most 11 used ids: mule zone 0 of 625 overlapping windows (minimum 12; at most 12 occurs in 1%), velocity zone 41%. So the likelihood ratio is at least about 50 for velocity (`a8_boot`). This is conservative, because the test mule zone is denser than train.
- A band in the mule zone would also mean 90 mule zone devices (57 expected from volume) and only 17 velocity devices (about 30 for a normal half month).
- The victim-core claim "mule up to 473782" rests on the host's own decode. Velocity fan-ins decode as mule 15/16 times, so that evidence is circular.
- Hosting rate cannot localise the edge (v3_zprop a8). Train velocity hosts sit at both ends of the zone: deciles [3,1,0,0,0,1,0,0,1,2].
- The changepoint estimator itself is noisy at test scale. In the block bootstrap of test-size train blocks, it lands within 8 ids of the true edge only about 25% of the time, with a tail of -109 at the 5th percentile. The occupancy step at 473757 is therefore NOT used to place [473749, 473757). Those ids stay guarded.
- Posterior with a uniform prior, length prior and fixed rates: P(473749 velocity) .21, P(473750) .40, P(473754-473756) .85-.92. This is over-sharp given the overdispersion; it is quoted only for variant b.

## (c) Rules (bc_rules_mule.py)
Sources are zone rows in the mule, velocity or layering zones, except those on the guarded ids [473749, 473757). Candidates are off-zone P2P rows decoded mule, velocity or layering.
- **R1** is v3_zprop R1, unchanged: a zone row pays the candidate's receiver within ±48 h.
- **R1f** (new): the candidate's sender received a zone row in the previous 48 h. This catches forward rows.
- If the links give a single zone type that differs from the decode, the row takes that type.

Train evidence (`chk_rules.py`). Label-only purity uses label-fit zones; the "decode" is the noisy label.

| rule / source zone | Jul | Aug | Sep |
|---|---|---|---|
| R1 mule | 414/418 | 318/321 | 302/303 |
| R1 velocity (shipped direction) | - | 71/71 | 19/20 |
| R1 layering | 21/21 | 26/28 | 55/55 |
| R1f mule | 46/47 | 39/39 | 33/34 |
| R1f velocity (shipped direction) | - | 9/9 | 0/1 |
| R1f layering | 14/14 | 15/16 | 16/16 |

- OOF gate (judge decode, no guard):
  - Aug changes 74 rows: 63 mule -> velocity (57 by R1, 6 by R1f), 73 correct on cleaned labels. Macro noisy .5021 -> .5048, cleaned .5146 -> .5175.
  - Sep changes 5 rows, 4 correct. Macro noisy .5096 -> .5098, cleaned .5224 -> .5226.
  - Sep has no mule -> velocity hit, so the shipped direction is checked out-of-time only by label-only purity (Sep R1 velocity 19/20).
- Test: 10 rows change, all from one event. Host DV0473782 feeds receiver AC369510: 8 fan-in rows (R1) and 2 forward rows (R1f), all mule -> velocity.

### Expected gain (one event, so a two-outcome bet)
- Oct counts used:
  - mule: P 794, TP about 675, T 1720-2120.
  - velocity: P 298, TP about 270, T 700-1115. T is the non-burst Sep velocity volume at half month plus 269 pool rows; bursts are not scaled.
- If the band is velocity (truth = velocity): **+.0011 to +.0014** macro. Velocity F1 gains .011-.015 and mule .002.
- If it is mule: -.0007 to -.0009.
- Break-even is P(velocity) = .38-.40. At P = .95 (LR about 50, R1 purity .99, R1f .9) the EV is about **+.0010 to +.0013**. Roughly 3 of the 10 rows are public.

### Variant b (bc_mule_b.csv): guarded ids act as mule sources
- Host DV0473749 feeds receiver AC291816: 4 more rows go layering -> mule.
- Under each truth for the AC291816 cluster:
  - mule: +.0003;
  - velocity: +.00005 (the layering rows are wrong either way);
  - layering: -.00035. This is unlikely, since the host is a mule or velocity zone device.
- At P(mule) about .6, EV is about +.0002. It depends on placing DV0473749, which occupancy cannot do reliably. Ship only if bc_mule is accepted and a further bet is wanted. The 473749 row itself is left alone: break-even P(mule) is .62 and the estimate is .6-.8.

## Against the target
- bc_mule gives about +.001 expected; variant b adds about +.0002.
- Mule's own levers are saturated: no missed burst rows, no missed forward rows, and camouflage is unreachable. The +.004 target is not reachable from this class.
- Mule rows are cheap: one TP is worth about 5e-5 macro, because P+T is about 2900.
