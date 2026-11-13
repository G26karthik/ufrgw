# typesplit_event: sim_swap vs account_takeover (event-mechanics angle)

Scratch: `work/scratch_ts_event/` (ev.py event rows, a2.py ~110 event aggregates + per-archetype AUC, a3-a6 composition/linkage,
a7/a8/b4 device-id zones, b6 DP edges, b7 out-of-time validation, b8 txn_id check).
Code: `work/feats/ts_event.py` (label-free module), `work/ts_event_post.py` (post-rule, writes `subs/ts_event.csv` and `subs/ts_event_simato.csv`).

## 1. Event mechanics alone do not split sim/ATO inside an archetype
- Events are sender sessions (gap 2 h) with 2 or more drain rows. Train has 1,137 labelled events.
- Labels are homogeneous inside an event: only 0.4-1.2% of events mix the two labels. So the type is chosen per event by the generator; it is not row noise.
- Archetype x label counts (events):

  | archetype | ATO | sim |
  |---|---|---|
  | SIM | 165 | 352 |
  | DEV | 357 | 159 |
  | none | 68 | 36 |

- Within each archetype, ~110 event aggregates were scored by AUC for the label: exact sim_age and device_age values (min, max, median, first, n-unique), hour, minute, login, pin, gaps, span, amounts, shares, 500-floor rows, balance drain, app and version, OS, KYC, receiver reuse, receiver profile, and victim history before and after (same device, other device, gaps).
  - Best in SIM: n rows, AUC .63.
  - Best in DEV: .57.
- One real but weak cue: the minority-label events rarely have 4 drains (DEV-sim: 26 of 159; SIM-ATO: 21 of 165, against about 1/3 for majority events). Not enough to use.
- sim_age, device_age and app_version are redrawn per row, as for legit rows, so their exact values carry nothing.
- Receiver rings carry no type. Events that share a drain receiver agree on the label at .62 (same archetype) and .37 (cross archetype). That is what archetype alone predicts (.58 / .42). sim and ATO drain into one shared mule pool.
- Device linkage carries no type either: DEV-archetype devices are never shared across events.

## 2. The separator: generator device-id zones (a data-generation leak)
- Device ids are allocated sequentially by the generator:
  - DV0000000-0419999: one device per account (device id == account id for 96.8% of devices).
  - DV0420000-0422999: velocity-bot pool, multi-row and reused across months, including Oct. All rows are velocity_bot.
  - Then one contiguous zone per fraud type, in a fixed order, covering all of Jul-Sep: mule, velocity (single rows), layering, **sim_swap 424628-425038**, **account_takeover 425039-426768**, phishing, qr, refund, fake_cashback, loan, collusion (ends 428454).
  - Then 45k single-use legit devices (428455-473631).
  - The Oct test repeats this pattern: fraud zones 473632-474551 in the same order, then dense legit singles 474552-482013.
- Inside a zone, rows of a device are that type up to label noise. Train Sep: 85% label == zone, 11% legit (the known 12% fraud->legit noise), 4% other type. The ratio is flat over Jul/Aug/Sep.
- DEV-archetype events use a fresh device from the sim or the ATO zone, so the zone gives their type exactly. SIM-archetype events use the victim's own device (base range), so they get no signal. txn_id is not a counter hash: md5/sha1/sha256 hits are at chance.
- Zones also hold the "new-device" rows of every other fraud type. h6 decodes about half of those rows as legit.

## 3. Out-of-time validation
Zone edges were fit on Jul+Aug labels only; the scores below are on Sep.

| | events | archetype rule | zone rule |
|---|---|---|---|
| Sep takeover events, all | 341 | .707 | **.836** |
| DEV archetype | 157 | .720 | **.994** |
| SIM archetype | 157 | .694 | .701 |
| none | 27 | .704 | .704 |

- The 90% target is not reachable: SIM-archetype events (about 46%) stay at ~70% and look inseparable. 0.836 is the ceiling of this rule.
- h6 OOF (postproc qr_null + h6 bias), with zone rows set to the zone type:

  | variant | Sep noisy | Sep cleaned | Aug noisy |
  |---|---|---|---|
  | h6 baseline | .4855 | .4979 | .4730 |
  | all zone rows overridden | **.5099** (+.024) | **.5227** | .5021 |
  | sim/ATO-only override | .4938 | .5068 | - |

- Per class, Sep noisy: sim +.086, ATO +.043, and every other fraud class +.012 to +.023.
- The Aug column is in-sample for the edges.

## 4. Test (Oct) zones
- `test_block()` finds them label-free: lo = first device id above the train max (473632); hi = start of the final gap-free id run (474552).
- Edges are the train layout scaled to [lo, hi), then refined by a DP in `ts_event_post.py`. The DP keeps the zones contiguous and in order, uses a Poisson length prior, and takes h6 fraud-type predictions plus "multi-row device" as evidence.
  - DP edges: 473632, 473749, 473835, 473916, **473987 (sim->ATO)**, 474284, 474352, 474462, 474496, 474515, 474529, 474552.
  - The DP result does not depend on the evidence weight (0.5x-2x gives the same edges).
- Consistency checks:
  - Test/train zone row ratio is 0.14-0.24 for every type. Expected about 1/6: one month at half density.
  - Inside each estimated zone, h6's non-legit predictions are mostly that zone's type, for example 180/196 ATO in the ATO zone.
  - The last multi-row device is 474279, inside the ATO zone.
  - Test rows on velocity-pool devices: 269.

## 5. Candidates (from preds/h6_test.npy; decode reproduces subs/h6_ppq.csv exactly, which is asserted)

| file | rows changed vs h6_ppq | what changes |
|---|---|---|
| `subs/ts_event.csv` | **297** | 73 ATO->sim; about 200 legit->fraud type (56 mule, 27 qr, 26 velocity, 21 layering, ...); a few type fixes |
| `subs/ts_event_simato.csv` | 75 | sim/ATO choice only |

Recommend `ts_event.csv` (Sep +.024 noisy / +.025 cleaned, against +.008 for simato). In the clean test every zone row is fraud, so the clean gain should be at least the Sep gain.

## Risks / not done
- **This exploits a generator artefact (sequential device ids), not fraud behaviour.** The rules allow code rules ("rules and features you write in code are fine"). Still, the user should decide whether to use it. It applies equally to the private 70%.
- The sim/ATO edge inside the test block depends on the length prior only, because h6 cannot tell sim from ATO in DEV events. A wrong edge by ±8 ids moves about 5 devices (~10 rows).
- `feats/ts_event.py` uses the proportional test edges, not the DP edges. It has not been trained into a model; the post-rule is the deliverable.
- Not checked: the other base-range devices (id != owner account, 3%) as a fraud signal, and the phishing/fake_cashback pair beyond its zone rows.
