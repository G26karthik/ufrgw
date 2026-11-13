# Review of postproc (m5dm), 2026-11-13

Scripts: `work/scratch_postprocrev/r1.py` (reproduce, alignment, OOS, arch breakdown), `r2.py` (bias side-effects, QR null), `r3.py` (QR table).

**Verdict: the claim holds.** The archetype forcing gain is real and survives out-of-sample decoding in both directions.

## Reproduced (same numbers as the claim)

| | pooled noisy | pooled cleaned | Sep noisy | Sep cleaned |
|---|---|---|---|---|
| raw m5dm, bias re-tuned | .4574 | .4700 | .4650 | .4772 |
| post, bias re-tuned | .4641 | .4764 | .4721 | .4845 |
| post, raw bias kept | .4626 | .4751 | .4706 | .4831 |
| post, raw bias + post sim/ATO bias only | .4635 | .4757 | .4711 | .4835 |

Out-of-sample decoding (bias tuned on one fold, scored on the other):

| direction | raw noisy / cleaned | post noisy / cleaned | gain |
|---|---|---|---|
| Aug -> Sep | .4627 / .4749 | .4690 / .4814 | +.0063 / +.0065 |
| Sep -> Aug | .4494 / .4623 | .4543 / .4662 | +.0049 / +.0039 |

`subs/m5dm_pp.csv` and `subs/m5dm_ppq.csv` are reproduced exactly (100% label match).

## Checks that passed

- **Leakage:** none. `apply` uses only txn columns, the label-free `takeover__arch_*` flags and `month`. The only label use is bias tuning, and the OOS table above covers that.
- **Alignment:**
  - `oof/m5dm_ids.parquet` y matches tx y. Fold 0 is Aug and fold 1 is Sep.
  - `preds/m5dm_test.npy` decoded with the meta bias gives `subs/m5dm.csv` exactly, in sample_submission order.
- **OOF/test path:** both use the same `apply`. The arch flags from `takeover__dm.parquet` match `matrix_m5dm.parquet` on all rows (0 mismatches).

## Caveats (none of them refutes the gain)

1. **The test gain should be smaller than the OOF gain.**
   - On test, forcing flips 57 of 492 arch rows (39 sim->ATO fixed in arch_sim, 18 in arch_ato), which is 12%.
   - On Sep val it flips 136 of 864 rows (16%). The full-fit model is already more archetype-aligned.
   - Expect roughly +.004-.005 on LB, not +.007.
2. **Re-tuning moves biases of unrelated classes.**
   - loan goes .4 -> 2.2, phishing .9 -> 1.3, refund .5 -> .2, collusion 1.1 -> .8. This is coordinate-ascent drift.
   - The sim/ATO biases are unstable: ATO is 2.3 pooled, 3.2 on Aug, 0.3 on Sep. They sit on a flat plateau.
   - Re-tuning only sim/ATO gives the same OOS score (.4693 vs .4690 on Aug->Sep, .4542 vs .4543 on Sep->Aug). It changes 24 test rows against the full re-tune.
   - Harmless, but most of the pooled +.0067 is +.0052 from forcing and +.0015 from in-sample re-tuning.
3. **The Oct archetype-to-type majority is assumed, not validated.** clean() does not touch sim/ATO, so the cleaned proxy does not test it. Forcing the majority is optimal whether the 30% swap is train noise or the generator's design, unless Oct flips the majority.
4. **Cache-staleness hazard.**
   - `apply` calls `run_feature_matched("takeover")`. That call silently rebuilds and overwrites the shared `takeover__dm` cache if `feats/takeover.py`, `density.py` or `clean_labels.py` is newer than it.
   - The halves depend on clean(), so the rebuilt flags can differ from the ones the model was trained on.
   - It is consistent for m5dm today. For later tags it is safer to read the `takeover__arch_*` columns from `feats_cache/matrix_<tag>.parquet`.
5. **QR-null flag:** supported, but not validated.
   - Test QR->P2P with category has 57 rows with mm=0, 112 with mm=1 and 140 with NULL. Train has no P2P rows with NULL.
   - Train P2P-cat rows are 81-83% qr_swap on cleaned labels. They are 12-14% legit, which is about the 12% fraud->legit flip rate, so clean-label precision is likely above 85%.
   - Expected gain is +.001-.002. ppq is the better pick.
