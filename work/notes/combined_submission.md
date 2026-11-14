# Combined exploratory submission (15 Nov 2026, 03:00 IST)

Status: one authorised exploratory upload. Not a validated improvement: the August selector shows no detected difference
from the current model. `subs/dd_final.csv` is unchanged (SHA-256 `8e04353d6f52fe86ab9e44b5dd33d7adec9ba26df36f566d0bcfd68d49d829cd`).

| | file | SHA-256 | rows | non-legit rows |
|---|---|---|---|---|
| current | `subs/dd_final.csv` | `8e04353d...29cd` | 260,582 | 3,984 |
| candidate | `subs/cmb80_dd_final.csv` | `cff9925e5503b644bd16701916bea693d4942bf3c43b853a913fced4ea82c3e3` | 260,582 | 3,977 |

## Deviation from the brief, stated first
The brief asks for each source's actual feature pipeline. Five sources were developed on benchmark N (full density).
October has about half the row density, so every alternative source was refitted on the production density-matched
matrix with the same recipe otherwise. Their N results therefore belong to versions that are not in this file. The
shipped versions were assessed on benchmark D, a label-informed density stress test. Rules were written down before any
mixture score existed: `scratch_mix/DECLARED.md` (14 Nov, 23:35 IST).

## Pool (all complete before the 03:15 cut-off; none dropped)
| source | recipe on the density-matched matrix | August after decoder, alone (D) |
|---|---|---|
| anchor h6 | production LightGBM, 6 seeds | .5002 |
| c2s6 | production LightGBM + core2, 6 seeds | .5008 |
| sTS4 | soft two-stage, 4 seeds | see `final_comparison.md` |
| TabM | ple, seeds 0 and 1, epoch 30 for both | .4976, .4983 (single seeds) |
| kA | 190 + 50 neighbour columns, 2 seeds | not scored alone |
| TabICLv2 | 20,000 legit + up to 2,500 per fraud class, 1 context | not scored alone |
| BSEL | 1023 bins on 9 row-level columns, 2 seeds | see `final_comparison.md` |

TabM checkpoint choice: August after the decoder rose with every checkpoint (seed 0: .4943, .4952, .4965, .4976;
seed 1: .4953, .4971, .4973, .4983 for epochs 6, 12, 20, 30). Epoch 30 is the edge of the declared grid.

## Selection (August, after the decoder, grouped halves)
| mixture | D (selector) | N side reading (N-trained versions, not shipped) |
|---|---|---|
| reference | h6 .5002 | C .4978 |
| anchor .50 | .5001 | .4985 |
| anchor .65 | .5002 | .4985 |
| anchor .80 | .5003 | .4985 |

Declared rule: highest wins; within .0003 the higher anchor weight wins. Both clauses give anchor .80.
Reading: +.0001 against h6 on D is no detected difference (the two August halves differ by about .010).

## Build
`bash scratch_mix/build_all.sh`: `mix_blend.py build W` -> `postproc.py cmbW --qr-null` -> `ts_judge_post.py cmbW` ->
`scratch_mix/chain_fb.py` (the `alt_c2.py` rule chain, steps 12 to 17, no step 11) -> `scratch_mix/check.py`.
- Every validation and test row has all six sources; rows sum to 1.
- Test blend against h6 before bias: argmax differs on 6 of 260,582 rows (validation: 62 of 1,036,848).
- Tuned bias moved in every class (h6 `[0,.4,.7,.6,.9,.6,.8,.3,.3,1.5,1.5,.9]`, mixture
  `[0,.2,.5,.3,.6,.2,.6,.5,.6,1.3,1.7,1.2]`). Most of the 15 changed rows are margin rows decided by the retuned bias.
- Zone edges on October are identical to h6's (both edge sets).
- The three mixtures give three different files (4, 6 and 8 rows apart). None equals any existing file in `subs/`.
- `validate_sub.py`: format OK (`logs/validate_cmb80.log`).

## Rule layers on this base
| layer | rows changed, h6 base | rows changed, this base | note |
|---|---|---|---|
| v3 | 28 | 28 | same rows and targets |
| RS | 3 | 2 | `TX7f5156c9` is already refund_abuse before the rule; same final label |
| FCB | 5 | 5 | same |
| CZ | 10 | 10 | same |
| LFM | 4 | 2 | `TX200b82bf`, `TXe6d162e4` are already mule_account before the rule; same final labels |
| X999 | 1 | 1 | same |
| BANDV | 10 | 10 | same |
| EDGEFIX | 3 | 3 | same |
| pool-bait + stamp | 13 | 13 | guard refused the base; declared fallback applied (below) |

Fallback: the expected-id guard of `dd_rules_both.py` refused the base (`AssertionError: ('P_only', ['TX1647c469'])`).
As declared, no assertion was relaxed and the layer's 13 accepted rows were applied; none conflicts (all 13 were legit
before the layer, as recorded). `scratch_mix/guard_rows.py` lists what the rule selects on this base: the same 13 rows
plus `TX1647c469` (sender AC114656, amount 10). h6 decodes that row phishing_collect, the mixture decodes it legit, and
the rule's own criteria would send it back to phishing_collect. It stays legit here because the guard held. So 1 of the
15 differences exists only through the guard. It was not hand-fired: that would be a row choice made after seeing the result.

## The 15 rows that differ from dd_final (`notes/cmb80_dd_final_diff.csv`)
| change | rows |
|---|---|
| phishing_collect -> legit | 6 (TX1d444be7, TX6a3ada77, TX9cba4eb3, TX10754001, TXf05cc0e1, TX1647c469) |
| account_takeover -> loan_app_extortion | 3 (TX4a8d7538, TXfe2920db, TX504fdcb4) |
| phishing_collect -> fake_cashback | 2 (TX96e28c86, TXf0208897) |
| phishing_collect -> loan_app_extortion | 1 (TX37a8ee92) |
| layering_chain -> mule_account | 1 (TX261a1b93) |
| layering_chain -> legit | 1 (TXe9a0295a) |
| velocity_bot -> mule_account | 1 (TXffb4ef65) |

- No rule layer produced any of the 15; all are model plus bias decisions.
- 7 of the 15 are also in the c2s6 list of 12 (`notes/alt_c2s6.md`): TX1d444be7, TX6a3ada77, TX9cba4eb3, TX10754001,
  TX96e28c86, TXf0208897, TXffb4ef65. The only public evidence on those is the confounded `final1` result.
- `validate_sub.py` scenario (a truth model, not test truth): +.0023 if all 15 new labels are right, -.0020 if none is,
  break-even at 46% right.

## September (descriptive, read after the selection)
See the end of this note once `scratch_mix/sep_D.log` and `sep_N.log` are in. September has been inspected many times;
intervals do not correct for that.

## Uncertainty
- August D gives no detected difference. The file is an exploratory probe of whether a broader model family mix moves
  the public score, on 15 rows of which about 30% are public.
- Ranking uses the best public upload. If this file scores above .5505 on the public part, it becomes the file ranked
  on the private part.
- The public score of this upload is one observation. No further uploads or row flips follow from it.
