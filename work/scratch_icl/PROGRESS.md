# TabICLv2, all 12 classes, benchmark N (14 Nov 2026, started 21:38 IST, hard stop 23:30)

Script: `scratch_icl/icl.py` (modes `sub` = August subset timing + quality, `full` = both months). Logs: `scratch_icl/*.log`.
Package: tabicl 2.2.0, checkpoint `tabicl-classifier-v2-20260212.ckpt`, RTX 4060 8 GB, AMP auto, `batch_size=1`.

## API facts (read from the installed source)
- The model's native limit is 10 classes. 12 classes use the package's many-class path (mixed-radix target embedding +
  hierarchical in-context classification). That path refuses `kv_cache`, so every `predict_proba` call re-encodes the context.
- NaN inputs are accepted. No categorical argument: the 9 categorical codes enter as ordinal floats.
- Column-embedding output is offloaded to pinned host RAM: about 97 kB per (context + query) row with 190 features.
  89k rows per call failed (8.3 GB host allocation); 69k rows per call worked.

## Timing (August subset: all 19,345 fraud rows + 30,000 legit rows reweighted; ensemble 1; 190 features)
| context (fraud cap / legit) | rows | chunk | rows/s | weighted log-loss | mean fraud-class AP |
|---|---|---|---|---|---|
| LightGBM C (reference) | | | | 0.17974 | 0.2666 |
| 250 / 2,250 | 5,000 | 20k | 2882 | 0.18364 | 0.2540 |
| 1,000 / 10,000 | 20,344 | 50k | 1896 | 0.18219 | 0.2613 |
| 4,000 / 20,000 (= all July fraud) | 39,513 | 20k | 659 | 0.18079 | 0.2673 |

Fixed cost per call grows with the context (about 10 s at 20k, about 22 s at 40k); per query row about 0.3 ms.
Chunking check (same rows in a 10k call and in a larger call): max abs diff 0.0000 at 5k and 40k context, 0.0455 max at 20k.

## Plan (declared before any full August score)
- Context recipe, same for both folds: 20,000 legit fit rows + every fraud class capped at 2,500 fit rows, seed 0.
  Fold 0 (fit July) and fold 1 (fit July + August) each draw from their own fit rows.
- Ensemble: `n_estimators` 1 unless the ensemble-4 subset run shows a clear gain that fits the time.
- Prior correction: p_k *= (fold fit share of class k) / (context share of class k), renormalised. Saved files are corrected.
- Blend weight fixed at 0.35 (0.65 C + 0.35 model), not tuned.
- "Adequate" = full August and September rows, context >= 38k rows, ensemble >= 1.
- Selection uses the August subset (log-loss / AP) and `aug_dec.py` only. September is scored once with `bench_eval.py`.

## Decision 21:52 (from the subset runs only)
| extra subset run | rows/s | weighted log-loss | mean fraud-class AP |
|---|---|---|---|
| 1,000 / 10,000 context, ensemble 4 | 417 | 0.18127 | 0.2643 |

- Context size helps more per second than the package ensemble (40k ensemble 1 beats 20k ensemble 4 at 1.6x the speed).
- Running (job started 21:54): `icl.py full 2500 20000 1 20000 ICLs0 0`, then the same with context seed 1 (`ICLs1`).
  Candidates, fixed now: `iICL2` = mean of the two context bags if the second finishes by 22:50, else `iICLs0`;
  and the blend 0.65 C + 0.35 of that. C August reference (aug_dec): before 0.4696, after decoder 0.4978.

## August, full rows (22:12) - bag 0 (context seed 0), `scratch_icl/aug_dec.log`
Bag 0 August inference: 526,852 rows in 857 s (615 rows/s), 38,257 context rows, VRAM peak 3.9 GB.
| variant | before decoder (halves) | after decoder (halves) |
|---|---|---|
| C (LightGBM) | 0.4696 (0.4616, 0.4775) | 0.4978 (0.4920, 0.5036) |
| iICLs0 (TabICL alone) | 0.4686 (0.4618, 0.4753) | 0.4966 (0.4907, 0.5025) |
| iICLs0b35 (0.65 C + 0.35 TabICL) | 0.4704 (0.4611, 0.4796) | 0.4986 (0.4913, 0.5059) |

## Freeze (22:13, before any September score)
Bag 1 cannot finish both months by 22:50 (bag 0 needs about 31 min for both), so by the rule above the frozen candidates are
`iICLs0` and `iICLs0b35`. September is scored once: `bench_eval.py N C,iICLs0,iICLs0b35 _icl`.
Bag 1 August only is left running as a development-only side check of a second context bag; its September fold is stopped.

## September, scored once (22:40) - `scratch_icl/bench_eval.log`, `scratch_icl/diff.log`, `scratch_gap/dec_N_icl.npz`
Bag 0 September inference: 509,996 rows in 1,456 s (350 rows/s; `aug_dec.py` shared the CPU), 44,390 context rows
(20,000 legit; fraud 2500 x 7 classes, then 2156, 1699, 1338, 1697).
Intervals: paired event bootstrap, 5%..95%, vs C.
| variant | before decoder | after decoder | change before | change after | rows changed (after) | corrected | newly wrong |
|---|---|---|---|---|---|---|---|
| C | 0.4815 | 0.5070 | | | | | |
| iICLs0 | 0.4775 | 0.5027 | -0.0041 (-0.0064..-0.0018) | -0.0042 (-0.0063..-0.0021) | 384 | 91 (fraud 75, legit 16) | 263 (fraud 131, legit 132) |
| iICLs0b35 | 0.4829 | 0.5077 | +0.0013 (+0.0001..+0.0026) | +0.0007 (-0.0007..+0.0020) | 141 | 58 (fraud 49, legit 9) | 73 (fraud 25, legit 48) |

Largest per-class F1 moves after the decoder: alone mule_account -0.017, phishing_collect -0.009, refund_abuse -0.008,
fake_cashback +0.008; blend fake_cashback +0.010, phishing_collect -0.003, others within 0.002.
