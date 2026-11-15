# Second exploratory candidate: raw blend .65 h6 + .35 wSQ (15 Nov 2026, 07:20 IST)

Status: promising but unconfirmed. Above the current model on every reading taken (August and September, N and D);
no single interval excludes zero on D. A control without the weighted model gives the same gain (see Controls). Not uploaded; the user chooses. `subs/dd_final.csv` is unchanged
(SHA-256 `8e04353d6f52fe86ab9e44b5dd33d7adec9ba26df36f566d0bcfd68d49d829cd`).

| | file | SHA-256 | rows | non-legit rows |
|---|---|---|---|---|
| current | `subs/dd_final.csv` | `8e04353d...29cd` | 260,582 | 3,984 |
| first exploratory | `subs/cmb80_dd_final.csv` | `cff9925e...c3e3` | 260,582 | 3,977 |
| this candidate | `subs/sq35_dd_final.csv` | `bacec5494db5d06f40f590187face753268521c50096a3f725f88a332664a9fe` | 260,582 | 3,993 |

Rules were written before any score of this recipe existed: `scratch_wt/DECLARED_sq35.md` (07:06 IST).

## Recipe
- `wSQ6`: the existing `scratch_wt/wt.py` arm `wSQ` (baseline weights x class count^-0.5 over all 12 classes; total
  training weight rescaled to the baseline total, asserted) on the production density-matched matrix. Seed numbers 0 to 5
  with the same keep-mask formula as `train.py`, 150 rounds, learning rate .05, `T.LGB`: the recipe of h6 with other weights.
- Candidate = .65 h6 + .35 wSQ6, raw probabilities, blended before the decoder. The weight was fixed in advance.
- Then the unmodified decoder and every accepted rule layer (steps 12 to 17, no step 11).

## Results (change against the current model, after the decoder)
| reading | August | September (5%..95%) |
|---|---|---|
| D, six seeds, reference h6: **the shipped recipe** | .5007 against .5002 (+.0005) | .5092 against .5090, +.0002 (-.0003..+.0009) |
| D, seeds 0,1, reference A_C (earlier) | +.0006 | +.0004 (-.0003..+.0009) |
| N, seeds 0,1, reference C (earlier, N-trained, not shipped) | +.0007 | +.0016 (+.0006..+.0024) |
| N, seeds 2,3 (earlier, N-trained, not shipped) | +.0001 | +.0013 (+.0001..+.0024) |

- Every reading is above its reference: the six earlier two-seed readings and the two six-seed D readings (the D seeds
  0,1 row is contained in the six-seed row). August D: .5007 against h6 .5002 and the first exploratory file .5003.
- Gate (declared): August D not more than .0010 below h6, September D interval not entirely below zero. Passed.
- N and D are separate benchmarks. D is a label-informed density stress test, not independent ground truth. September
  has been inspected more than 40 times; the intervals cover event sampling for one comparison and not that reuse.

## September trade-offs (D, six seeds, after the decoder)
- 47 rows change: 17 corrected, 23 newly wrong, 7 wrong before and after. 8 missed fraud rows recovered, 15 legit rows
  newly called fraud.
- 17 of 47 changed rows became right (36%). The `validate_sub.py` break-even for the 18 October rows is 38% (below).
  On that basis the expected public effect is about zero in either direction.
- Per-class F1: loan_app_extortion +.0019, phishing_collect +.0011, mule_account +.0005, qr_swap +.0003;
  fake_cashback -.0004, layering_chain -.0003, refund_abuse -.0003; the other four within .0001.
  fake_cashback precision +.017, recall -.004. Table: `scratch_wt/sq6_table_D.log`.

## Controls: does the weighted model add more than a decision-scale change?
The weighted model's mean fraud probability is .281 (August) against .037 for h6. Two controls without any new model:
- K1, scalar fraud-against-legit shift of h6 (fraud probabilities x e^d), d picked on August from {0.5, 1.0, 1.5}:
  August .4998, .4999, .5001 (h6 .5002), so d = 1.5. As expected in advance, about zero on August: the decoder already
  tunes that knob. September at d = 1.5: +.0004 (-.0001..+.0011).
- K2, scale-matched mixture `.65 h6 + .35 (h6 x per-class ratio)`, ratio fitted on August predictions so that the
  class means equal wSQ6's (ratios 11.7 to 23.0 for the fraud classes): August .5007, September +.0003 (-.0003..+.0010).
- Candidate minus K2, September, paired: -.0001 (-.0009..+.0008); 66 rows differ, 34 right only for the candidate,
  27 right only for K2.

Reading: no detected difference between the candidate and K2. On these two months the gain of the blend is reproduced
by changing the probability scale of h6 alone; the weighted model is not shown to add information beyond that. This
does not show that the weighted model cannot help: the comparison is one recipe on two months. K2 was a control and
was not built into a file.

## Build and checks (`bash scratch_wt/build_sq35.sh`, log `scratch_wt/build_sq35.log`, `check_sq35.log`)
- Test blend against h6 before bias: argmax differs on 115 of 260,582 rows (validation: 213 of 1,036,848).
- Production bias (`postproc.py`, the bias the file is built with): h6 `[0,.4,.7,.6,.9,.6,.8,.3,.3,1.5,1.5,.9]`,
  candidate `[0,.2,.2,.1,.6,-.1,.4,.1,0,1.0,.6,.1]`, inside the grid (-4..6). Non-legit rows before the later rules
  (`ts_judge` files): 3,962 against 3,950 for h6.
- Transfer diagnostic (`sq6.py counts`: bias tuned on the August D rows only, raw decode, no forcing or zones; not the
  production bias): h6 `[.4,-.7,.6,.7,.4,.9,.7,.5,1.6,1.2,1.1]`, candidate `[.2,-1.2,0,.6,-.1,.2,.3,0,.6,.3,-.1]`, inside the grid.
- With that diagnostic bias the August tuning transfers to October: predicted non-legit rows, candidate / h6 = 1.0003 (August), 1.0031 (September),
  1.0048 (October). Largest per-class October ratios: refund_abuse 1.031, phishing_collect 1.015; merchant_collusion .959.
- Zone edges on October identical to h6's (both edge sets).
- `validate_sub.py`: format OK (`logs/validate_sq35.log`). Ids, order and labels valid. Equals no existing file in `subs/`.

| rule layer | rows changed, h6 base | rows changed, this base | note |
|---|---|---|---|
| v3 | 28 | 28 | same rows and targets |
| RS | 3 | 0 | all three already refund_abuse before the rule; same final labels |
| FCB | 5 | 5 | same |
| CZ | 10 | 10 | same |
| LFM | 4 | 2 | `TX200b82bf`, `TXe6d162e4` already mule_account before the rule; same final labels |
| X999 | 1 | 1 | same |
| BANDV | 10 | 11 | `TXe9a0295a` decoded mule_account here; the rule returns it to layering_chain, its dd_final label |
| EDGEFIX | 3 | 3 | same |
| pool-bait + stamp | 13 | 10 | guard refused the base; declared fallback (below) |

Pool-bait guard: `dd_rules_both.py` refused the base because three expected rows (`TX21726530`, `TX81feab4f`,
`TXe1644fe3`) are not selected: the model already labels them phishing_collect, the rule's own target. The declared
fallback applied the layer's 13 accepted rows; 3 listed conflicts (those three rows), all already at the rule's target. `scratch_mix/guard_rows.py
ts_judge_sq35`: 10 rows selected, none outside the expected list. No assertion was relaxed and no extra row fired.
Every rule row ends with the same label as in dd_final. No rule layer touches any of the 18 changed rows.

## The 18 rows that differ from dd_final (`notes/sq35_dd_final_diff.csv`)
| change | rows |
|---|---|
| legit -> phishing_collect | 8 (TXda35eb3c, TXb2fc98a4, TX0fb93918, TXa34a7647, TX911b86b0, TX5d91db16, TX4cbb5675, TXb6807aba) |
| phishing_collect -> fake_cashback | 5 (TX7bacc53f, TX96e28c86, TXf0208897, TX809fd398, TX46f60532) |
| layering_chain -> mule_account | 3 (TX6f86a56f, TXcba91e13, TX261a1b93) |
| velocity_bot -> mule_account | 1 (TXffb4ef65) |
| legit -> qr_swap | 1 (TXa910b735) |

- 4 of the 18 are also changed, to the same label, in the first exploratory file (TX261a1b93, TX96e28c86, TXf0208897,
  TXffb4ef65). The other 14 are new.
- `validate_sub.py` scenario (a truth model, not test truth): +.0021 if all 18 new labels are right, -.0013 if none is,
  break-even at 38% right (first exploratory file: 46%).

Per-class predicted counts:

| class | dd_final | this candidate | first exploratory |
|---|---|---|---|
| legit | 256,598 | 256,589 | 256,605 |
| mule_account | 798 | 802 | 800 |
| velocity_bot | 308 | 307 | 307 |
| layering_chain | 557 | 554 | 555 |
| sim_swap | 364 | 364 | 364 |
| account_takeover | 237 | 237 | 234 |
| phishing_collect | 522 | 525 | 513 |
| qr_swap | 537 | 538 | 537 |
| refund_abuse | 276 | 276 | 276 |
| fake_cashback | 200 | 205 | 202 |
| loan_app_extortion | 103 | 103 | 107 |
| merchant_collusion | 82 | 82 | 82 |

## Reproduce
```
cd work ; export PYTHONPATH=. PYTHONUTF8=1
BM=D python scratch_wt/wt.py wSQ ; BM=D SEED0=2 python scratch_wt/wt.py wSQ ; BM=D SEED0=4 python scratch_wt/wt.py wSQ
python scratch_wt/wt_oct.py 0 1 2 ; python scratch_wt/wt_oct.py 3 4 5
python scratch_wt/sq6.py assess
bash scratch_wt/build_sq35.sh
```
Assessment: `python scratch_gap/aug_dec.py D h6,SQ6b35,K2,K1d05,K1d10,K1d15` ; `python scratch_gap/bench_eval.py D
h6,SQ6b35,K2,K1d05,K1d10,K1d15 _sq6` ; `python scratch_gap/bench_eval.py D K2,SQ6b35 _sq6k`. Logs: `scratch_wt/sq6_*.log`.

## Uncertainty
- The size of the gain (+.0005 August, +.0002 September on D) is inside sampling noise; its sign is the same in all readings.
- Ranking uses the best public upload. A file that scores above .5505 on the public part becomes the file ranked on the
  private part. About 30% of the 18 rows are public.
- Not done: N with six seeds; K2 on October; the weight .65 / .35 was not searched (as instructed).
