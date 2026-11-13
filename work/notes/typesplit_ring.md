# typesplit_ring: can fraud-ring linkage separate sim_swap from account_takeover? (13 Nov)

Verdict: **NO.** Ring linkage does not decide the type. Receivers, devices, time, cities and SIM-swap time were all tested.
No label-free rule beats the archetype rule out of time. Best Sep event accuracy is 0.727 (archetype plus small N cells), against 0.710 for the archetype alone. The target was 0.85.
No module and no submission were written. Test rows changed: 0.

Scripts: `work/scratch_ts_ring/e1.py` to `e13.py`. Event = sender session (gap <= 2 h) with >= 2 drain rows: 1369 events, of which 1137 are labelled (Jul-Sep).
Event label = majority of the sim/ATO-labelled rows. Labels inside an event are almost never mixed (< 2% of events), so the swap happens per event.

## Ring linkage (e2, e3, e10)
- **Shared drain receivers.** 919 receivers are hit by 2-5 events, giving 1207 event pairs.
  - Label agreement on shared-receiver pairs is .494; chance given the archetypes is .500.
  - Archetype agreement is .41, which is chance (.45² + .45² + .10²).
  - The median gap between two events on one receiver is 33 days, and only 27% fall in the same month. The receivers form one shared mule pool, not type-specific rings.
  - Connected components over receivers form one giant component (661 events). Leave-one-out label from the component neighbours gives .50 accuracy.
  - Same-archetype pairs agree .63 against .57 expected: about 1.6 sd, too weak for a rule.
- **Devices.** Only 2 devices appear in more than 1 event, out of 1451. There is no device reuse across victims.
- **Time proximity.** For same-archetype event pairs within 1 / 3 / 6 / 24 / 72 h, agreement is .54-.57, which is chance (.57).
- **Same device city or home city within 24-72 h.** Agreement is .60-.62 against .57.
  - Per-city majority rules fit on Jul+Aug lose on Sep: home city .657, device city .620 and bank .688, against .710 for the archetype.
- **SIM-swap time** (t0 - sim_age) within 1 h / 6 h: agreement .59 / .58 against .57.

## Receiver-side and victim-history features (e4, e5, e9)
- Within-archetype AUC (sim=1) is .45-.55 for every feature tested, after stratifying by size:
  - receiver attributes: age, bank and OS match, home city vs device city, KYC;
  - receiver onward sends within 1 h / 1 d, receiver inbound volume, receiver use of the event device;
  - device sender-count;
  - victim activity before and after the event, and on the same or another device;
  - victim device and city continuity, account age, app, app_version, bank, OS, KYC;
  - hour, sim_age, device_age, pin, login, amounts, share, span.
- An LGB per archetype, fit on Jul+Aug and scored on Sep, without size features: AUC .516 (S) and .522 (D). Accuracy is below the majority baseline.

## The only signal: event SIZE (number of drain rows). Label-free and stable, but it never flips the majority
Share of events whose label matches the archetype (S -> sim, D -> ATO), by number of drain rows:

| ndrain | Jul | Aug | Sep |
|---|---|---|---|
| 2 | .60 | .67 | .63 |
| 3 | .68 | .59 | .64 |
| 4 | .80 | .82 | .88 |

- The swapped events are mostly small: in the S archetype, ATO-labelled events have ndrain 2/3/4 = 81/62/21, against 121/119/112 for sim-labelled events.
- The LGB on size alone gives Sep AUC .64 (S) and .62 (D). Accuracy stays equal to the archetype majority, because no size flips the majority.
- Oct size mix (S+D events) for ndrain 2/3/4 is .475/.318/.207, against Sep .405/.302/.293. Oct has more 2-row events, which are less often archetype-consistent.
- Inside S events the swap does NOT follow night (.68 sim at night vs .71 by day). This is the strongest evidence that the archetype is the generated profile and the label is a separate draw.
- The swap rate is not uniform:
  - D events with no away rows and no night rows are 30/30 ATO, against an expected ~10 sim.
  - Sim-labelled D events are 100% away; ATO-labelled D events are 92% away.
  - So the label mechanism depends on size and on away. It is not uniform per-event noise.

## Small "N" events (neither archetype; 10% of events): already handled by the model
- Cell rule fit on Jul+Aug: night -> sim, away -> ATO, home and daytime -> ATO.
- On Sep: N_away 10/10, N_night 6/6, N_home_day 9/11.
- On the h6 OOF (Aug+Sep), the model's sim/ATO calls on N rows match cleaned labels .845, against .762 for the rule. The model already wins.
- On test, the rule would change only 4 rows. Not worth it.

## Implications
- The sim/ATO swap is not ring identity, receiver pool, device, place or time. Inside an archetype, the only label-free correlate found is burst size, plus the away/home asymmetry in D.
- Neither reaches the 85% accuracy target. If the leader's edge comes from sim/ATO, it is not from linkage.
- Not tried here: a size-weighted soft decode. It cannot change the argmax inside an archetype (P(arch type) >= .59 at every size), so its macro-F1 effect is nil.
