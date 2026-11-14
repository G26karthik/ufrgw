# Gap reset: benchmark audit, candidate recheck, score budget (14 Nov 2026, 18:00 to 19:00 IST)

Outcome: no improvement found under these tests. `subs/dd_final.csv` (public 0.5505, SHA-256 `8e04353d...29cd`) is unchanged.
No upload, no publication, no change to the production pipeline. This is not a statement that a higher score is impossible.

## 1. Two benchmarks, reported side by side

**N, natural label-hidden benchmark** (`scratch_gap/bench_nat.py`). A world holds only the fit months and the held month.
Held rows have their labels removed and no later row exists. All 190 features are built once on that world at full
density; `density.halves` is not used, so no label reaches a feature. Fit targets are computed on fit rows only, from
the world's own features. Check run: rebuilding the features with the held labels permuted gives an identical frame
(190 columns, 1,564,060 rows). Fit July, calibrate on August; refit July + August, score September.

**D, density stress test** (`scratch_gap/bench_den.py`). The production matrix. Its neighbourhood features come from
`density.halves`, which reads the labels of every train month, the scored one included, to keep fraud events whole.
It is a label-informed simulation of October's half density, not an independent benchmark.

Limits of both: September has been used for development many times. N does not reproduce October's lower density.

**Decoder applied to every variant** (`scratch_gap/bench_eval.py`): bias tuned on August, takeover archetype forcing,
bias re-tuned on August, device-zone override with edges found on the September decode alone, bait rule
(steps 5 and 6 of `run_all.ps1`). The edges found without labels are within 2 ids of the label-fit edges.
Not applied: steps 12 to 17 (test-only rule layers, about 300 test rows, +0.006 public).

## 2. Did the old gate reject a materially different solution?

Candidates were fixed before scoring: the current fit target (C, takeover archetype type), supplied raw labels (RAW),
event-majority cleaning + C (AC), and C with fraud labels lacking a class signature dropped (CE, fit on structured rows only).
Two seeds each. September macro-F1, supplied labels; intervals are paired event-bootstrap 5%..95% against C.

| benchmark | variant | before decoder | after decoder | change vs C after decoder | rows differing from C (before / after) | rows the decoder overrides |
|---|---|---|---|---|---|---|
| N | C (current) | 0.4815 | 0.5070 | | | 504 |
| N | RAW | 0.4795 | 0.5071 | +0.0001 (-0.0008..+0.0009) | 132 / 56 | 570 |
| N | AC | 0.4809 | 0.5064 | -0.0006 (-0.0015..+0.0002) | 72 / 71 | 503 |
| N | CE | 0.4594 | 0.4861 | -0.0209 (-0.0241..-0.0180) | 706 / 679 | 529 |
| D | C (current) | 0.4824 | 0.5078 | | | 542 |
| D | RAW | 0.4787 | 0.5076 | -0.0003 (-0.0012..+0.0006) | 116 / 49 | 603 |
| D | AC | 0.4820 | 0.5074 | -0.0004 (-0.0013..+0.0005) | 76 / 71 | 544 |
| D | CE | 0.4577 | 0.4849 | -0.0229 (-0.0264..-0.0197) | 736 / 703 | 573 |

- No variant beats the current recipe on either benchmark. RAW is worse before the decoder (N: -0.0020, interval
  -0.0038..-0.0004) and equal after it: archetype forcing does what relabel C does.
- The two benchmarks rank the variants the same way and differ by 0.0008 for the current recipe. The label dependence
  of `density.halves` is real, but it did not change this decision.
- Noise sensitivity: on event-cleaned labels the after-decoder scores are C 0.5196, RAW 0.5197, CE 0.4983 (N). AC is not
  scored on cleaned labels because its target uses the same cleaning. Zone truth proxy (1,011 Sep rows on class-zone
  devices): 50.9% carry the zone class before the decoder, 99.8% after, for every variant.
- Per-class precision / recall / F1 after the decoder: `scratch_gap/eval_N.log`, `eval_D.log`.

## 3. Error compartments and the score budget

September, N benchmark, current recipe after the decoder (macro 0.5070; 18,661 fraud-labelled rows, 8,074 predicted fraud).
"Oracle" = that compartment alone set to its supplied label (`scratch_gap/compartments.py`, `mistyped.py`).

| compartment | rows | oracle gain |
|---|---|---|
| fraud label, predicted legit, no class signature | 10,851 | +0.4138 |
| fraud label, another fraud class predicted | 732 | +0.0570 |
| of which five look-alike pairs (velocity as mule 124, takeover as sim_swap 105, mule as layering 100, phishing as fake_cashback 71, fake_cashback as phishing 65) | 465 | +0.0320 |
| of which 86 other pairs, 3.1 rows each | 267 | +0.0251 |
| fraud label, predicted legit, with a class signature (584 are QR-type rows of qr_swap) | 745 | +0.0296 |
| legit label, fraud predicted | 1,009 | +0.0165 |

- The 267 scattered mistyped rows match the train label noise (about 4% random wrong types on 7,065 rows = 283). The test
  labels are stated to be clean, so those rows and most of the "legit label, fraud predicted" rows are not errors there.
- Largest compartment a typing rule could address: the five look-alike pairs, +0.032 on September.
- Largest compartment overall: fraud-labelled rows with no class signature.

Score budget (`scratch_gap/budget.py`). Known: predicted counts of `dd_final`, public 0.5505, public F1 of velocity_bot
(0.373) and merchant_collusion (0.337) from knockout uploads. Assumed: per-class type precision and the class supports it
implies. The gap is 0.0833 macro = 0.9996 summed class F1.

- Exact identity: rows added to class c raise its F1 only if their precision exceeds F1_c / 2. That is 0.17 to 0.33 by class
  (0.19 for velocity_bot, 0.17 for merchant_collusion).
- New correct rows needed, spread over classes like the missed rows (scenario with September type precision):

| target | precision 1.00 | precision 0.75 | precision 0.50 | precision 0.35 |
|---|---|---|---|---|
| +0.0833 (whole gap) | 898 correct | 1,044 correct / 1,392 flagged | 1,547 / 3,094 | 4,236 / 12,104 |
| +0.0417 | 434 | 496 / 661 | 691 / 1,382 | 1,422 / 4,062 |
| +0.0208 | 214 | 242 / 322 | 328 / 656 | 610 / 1,741 |

- "Every current fraud prediction typed right, no new row found" gives 0.6075 with September type precision, 0.6366 with
  the lower precision of the model decode before zones, 0.71 to 0.75 if precision were 0.70 to 0.83. The answer depends on
  an assumed precision, so none of these is a bound. With the 4% label noise removed, the look-alike part is about +0.03.

Ceiling statements reworded as scenarios: `README.md` (camouflage bullet), `HANDOFF.md`, `notes/burst.md`,
`scratch_lead/critic/k15_testceiling.py`, `validate_sub.py` (now uses the post-decoder type precision).
Earlier figures of 0.564 to 0.593 assumed class supports equal to predicted counts, a camouflage volume and
September-to-October transport. The reported 0.6338 is above them, so at least one assumption is wrong; which one is unknown.

## 4. New mechanism: none pursued

Steps 1 to 3 give a size for each compartment but no new observable input for any of them.
- Look-alike pairs: earlier tests stand (about 110 event aggregates per archetype in `notes/typesplit_event.md`; the
  grouped pair classifier `scratch_x/classwise/e24.py`, event AUC 0.52 to 0.55, phishing as fake_cashback 0.80 with
  precision 0.56). I found no input those tests did not already include.
- Rows with no class signature: same negative evidence as before, limited to the representations tested.

## 5. Remaining material uncertainty

Whether the events generated by one class path with another class's template (the five look-alike pairs) carry any
trace of their generating class besides the device-zone id. On September that is worth about +0.03. It does not cover
the whole gap; the rest needs rows that no tested representation finds.

## Commands (cwd = work, `PYTHONPATH=.`)

```
python scratch_gap/bench_nat.py
python scratch_gap/bench_den.py
python scratch_gap/bench_eval.py N
python scratch_gap/bench_eval.py D
python scratch_gap/compartments.py N
python scratch_gap/mistyped.py
python scratch_gap/budget.py
```
Seeds: model 0 and 1; bootstrap `default_rng(2026)`; label permutation `default_rng(1)`.
