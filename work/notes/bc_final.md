# bc_final: judge of the bc_* class dives (13 Nov)

Code: `work/bc_final.py`. Output: `subs/bc_final.csv` = `subs/sc_final.csv` + 25 rows. Scratch and logs: `work/scratch_bc_final/` (occ, final1_diff, bc_final).

Each rule's `apply()` runs on sc_final, never on another rule's output, because `test_zones` re-runs the edge DP on the decode. Each output is asserted equal to that agent's own `subs/bc_<key>.csv`. The four diff sets are disjoint, which is also asserted.

| rule | source | rows | change | EV (macro, noisy labels) |
|---|---|---|---|---|
| CZ | bc_rules_layering | 10 | layering -> mule | +.0004 to +.0007, no-regret |
| LFM | bc_rules_velocity | 4 | layering -> mule | +.0001 to +.0002, no-regret |
| X999 | bc_rules_phishing | 1 | legit -> phishing | +.00005 |
| BANDV | bc_rules_mule | 10 | mule -> velocity | +.0010 if the band is velocity, -.0007 if mule; EV about +.0005 at P = .8 |

Total EV is about +.0012 to +.0016. The no-regret part is about +.0006 to +.0009. This is short of the +.004 target.

## Re-verification
- **CZ** (`scratch_bc_layering/a7.log`):
  - Off-zone rows of mule-zone chains are labelled mule 24/30 (Jul-Sep). The rest are 5 legit and 1 other-G1.
  - Legit misses still net positive, because the layering FP is removed.
  - Judge OOF increment over v3: 8/9 correct.
  - The 4 source devices sit deep in the mule zone (473637-473731, occupancy .67). They do not depend on the band.
- **LFM:**
  - `check_train` re-ran inside apply. Linked-row layering share: .5% / .8% / .3% (Jul/Aug/Sep).
  - Judge-decoded hits: 0/12 layering.
  - The 4 rows equal `bc_mule_b`'s extras. LFM targets mule whichever side DV0473749 falls on, so variant b is subsumed and rejected.
- **X999:**
  - The train cell is 74% phishing and 13% legit (noise). The rule never fires on train OOF, so its precision is the cell prior.
  - Break-even P(phishing) is about .23.
- **BANDV** (independent check, `scratch_bc_final/occ.log`):
  - Raw-tx used-id occupancy: test mule zone 79/117 (.675), band [473749, 473783) 11/34 (.32), velocity rest 17/52 (.33).
  - P(<= 11 of 34) is 1e-3 at mule rate .60, 1.2e-2 at the train rate .53, and 4e-5 at .67.
  - Train month-level mule tails show no sparse end.
  - Counter-evidence 1, LB (`final1_diff.log`): final1 (.5435, -.0008 vs ts_judge_h6) contained 8 of these 10 rows as mule -> velocity. It also contained 3 rows hosted by 473749 (mule -> velocity), 3 LFM rows as layering -> velocity, and 12 c2s6/zonemax rows.
    - With per-row public effects at a .3 public share, -.0008 fits "band = mule" better.
    - The likelihood ratio is about 1:5 against velocity, because "velocity" needs the 12 c2s6 rows to have cost about -.0014 public (z about -1.8).
  - Counter-evidence 2, hosting: about 3:1 for mule, but it does not localise the edge (v3_zprop a8).
  - Net: P(velocity) is about .8, with a range of .55-.95. Break-even is .38-.40, so EV is positive but this is a bet on one event.
  - To drop the bet, rebuild without the BANDV entry in `RULES`.

## Rejected
- Phishing MIX: no out-of-time support. Aug has 0 guarded events; the proxy is ph 5 / fc 4 against a .645 break-even.
- Mule variant b: subsumed by LFM, which reaches the same 4 rows without placing 473749.
- qr: identity; nothing was proposed.
- Velocity R1 on the 473749 host (-> velocity): the LFM mule target dominates.
