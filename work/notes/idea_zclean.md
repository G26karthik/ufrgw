# idea zclean: zone-truth label cleaning of the fit target (13 Nov) — FAILED the gate, closed

Idea: train rows on generator zone devices (feats/ts_event.py, TR_EDGES) get the zone's type as the fit target. This includes
legit-labelled rows. Optional extension: propagate the zone type to other rows of the same event. Val labels stay the original noisy ones.
Code: `relabel.py` letters Z / P / S (additive; used only when requested). Scratch: `work/scratch_idea_zclean/`
(s1-s4 stats, gate.py paired gate, gate_*.log). Logs: notes/train_zc3_val.log, notes/train_zcs3_val.log.

## Data facts (train Jul-Sep)
- 6573 rows sit on zone devices. In every month, 85% have label == zone and 12% are labelled legit. Only ~10% of fraud rows are on a zone device.
  - By class: velocity 37%, ATO 25%, sim 12%, other classes ~3% each.
  - Zone devices carry 1 row each, except velocity (13.7 rows/device), sim (2.0) and ATO (2.6).
- Relabel C fights the zone in the sim zone. It turns 371 of the 553 sim-zone rows into ATO, because they carry the ATO
  archetype (fresh device). The raw labels agree with the zone there (478 sim).
- Event propagation is weak:
  - Receiver bursts on zone receivers are mostly hubs or merchants (97% legit), so this link is useless.
  - Same sender within 1 h: 31% legit, which is above the 12% noise rate.
  - Usable links: receiver == zone sender, or sender == zone P2P receiver, within 6 h, for the mule/layering/phishing/fc zones.
    They agree with the zone 85% of the time with 13% legit, but only add ~50 changed rows.

## Gate (paired: same matrix_core2swap recipe and seeds 0-2 as `c2swap`, judge decode = postproc qr_null + pooled bias + zone/bait override)
| tag | relabel | rows changed vs raw y | Sep noisy | cleaned | OOS Aug->Sep | pre-zone Sep noisy |
|---|---|---|---|---|---|---|
| c2swap (base) | C | 750 | .5102 | .5229 | .5085 | .4861 |
| zc3 | CZP | 1416 | .5088 (-.0014) | .5214 (-.0016) | .5081 (-.0003, sd .0005) | .4844 |
| zcs3 | CZPS (keep C on sim/ATO zones) | 1786 | .5098 (-.0004) | .5224 (-.0005) | .5082 (-.0002, sd .0005) | .4856 |
The sd is a sender-cluster bootstrap of the paired diff (200 reps). Raw pooled biased: c2swap .4803, zc3 .4788, zcs3 .4798.
The train DP edges barely move (mule|velocity 423707 -> 423714).

## Verdict
Null-to-negative on all three metrics, for both variants, so there was no 6-seed or full run and no sub.
- Why: zone rows are overridden at decode anyway. The model cannot see device ids, so their cleaned targets only reach
  non-zone rows as feature-space spillover.
- Undoing C on sim-zone rows (zc3) costs the most: sim/ATO and phishing/loan go down. This repeats relabel A, whose
  legit -> velocity relabel was also neutral.
