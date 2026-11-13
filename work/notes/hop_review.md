# hop review (adversarial check of notes/hop.md, 2026-11-13)

**Verdict: the null is confirmed.** Multi-hop graph linkage carries no G-vs-L signal on camouflage rows, which makes it stronger than the claim in hop.md.

## Checks
- **Label leakage.** h1.py features use only tun_best OOF/test p_fraud. Labels (yc) are used only to tag G/L. Leakage could inflate a result but could not hide one, so it cannot explain a null.
- **Circularity.** G and L both need p_legit > 0.9, so neither group's own row is ever "hot" (pf < 0.1). The own-row term is 0 in both groups and adds no bias.
- **Out-of-time.** Univariate AUCs are on Sep. LightGBM fits on Aug and scores Sep with fixed rounds and no early stopping on Sep. The AUC is 0.4995/0.5018, with train AUC 0.72/0.82 (overfit).
- **Oct distribution / macro-F1 lift.** Skipped: there is no signal and no module, so neither has anything to measure.
- **Gap in hop.** Seeds were model-hot rows only. A ring made only of camo rows (all p_legit > 0.9) would leave no trace in hop's features, because none of its rows is ever hot.

## Extra test (scratch_hop_review/r1.py, output r1.txt)
This is a diagnostic, not a feature. It uses labels from the same month, so it is an oracle upper bound. Own-row contribution is removed exactly at hops 0 and 1. All AUCs are Sep G vs L.

| seed | graph | best AUC |
|---|---|---|
| all clean fraud (same month allowed) | all-time, hop 0/1, s/r/d | 0.4956-0.4999 |
| camo G rows only (camo-ring test) | all-time, hop 0/1 | 0.496-0.499 |
| all fraud / camo G | ±3 d window, hop 0/1 | 0.4998-0.5013 |
| out-of-time fraud labels (Jul+Aug for Sep) | all-time, hop 0/1/2, raw + normalised | 0.4984-0.5049 |

Even oracle label linkage, including camo-to-camo, finds nothing. For example, the share of rows whose receiver had another camo row within ±3 d is 0.407 for G and 0.407 for L. G rows are graph-independent of every fraud label.

## Recommendation
Close the graph line on camouflage, node2vec included, since it is a smoothed form of the same adjacency tested here.
Remaining headroom is on core rows: type confusion, the partial rows, and Oct drift.
