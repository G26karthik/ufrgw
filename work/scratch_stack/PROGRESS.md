# Combination / two-stage / TabM-checkpoint pass, 14 Nov 2026 (started 19:56 IST, hard stop 22:30)

New code and logs: `work/scratch_stack/n14/`. The other files in `work/scratch_stack/` belong to the 13 Nov stack
experiment (`notes/stack.md`) and are untouched. New prediction files: `work/scratch_gap/PN_s*_{aug,sep}.npy`
(prefix `s`, no existing name reused). Decodes: `work/scratch_gap/dec_N_ss*.npz`. `subs/dd_final.csv` untouched.

Benchmark N: fit Jul -> Aug (development); fit Jul+Aug -> Sep. September has been reused many times; not a fresh holdout.
Baseline reproduced 20:01: LightGBM `C` Sep before decoder .4815, after .5070 (`n14/eval_base.log`).

## Prior coverage (checked before running)
- Cross-model stacker: none. `notes/stack.md` = one LightGBM + neighbour aggregates (related, not equivalent).
  `notes/gpu.md` = weighted XGB/CatBoost/LightGBM probability blends on the older 182-feature matrix, pooled-OOF bias,
  no decoder, no TabM (related: at most +.0007). `notes/tabm_phfc.md` = LightGBM + TabM blend on N (equivalent for that pair; reused).
- Two-stage fraud/type decomposition: none (`grep -i two.stage|hierarch|binary notes/*.md`: only camouflage binary tests
  in `camo.md` and one-vs-rest specialists in `idea_ovr.md`; both related, not equivalent). Rerun needed.
- TabM checkpoints: only log-loss early stopping (`tabm_phfc.md`). Rerun needed.

## Selection rule (declared before any September scoring)
August selection score = `n14/nb.py aug_score`: bias tuned on one event-grouped half of August, other half scored after
forcing + zones + bait rule, swapped, mean of the two halves.
Part 3 rule: pick the epoch from {6, 12, 20, 30} by TabM-alone August score; keep 6 unless another epoch beats it by more
than the half-to-half spread of the difference. If 6 is kept: no fold-1 run, report "same checkpoint".

## Status (20:28)
| step | state | output |
|---|---|---|
| baseline eval | done: Sep .4815 / .5070 | `n14/eval_base.log` |
| sources sXGB (CPU 2 seeds), sCAT (GPU, 1 seed; the CPU run was too slow and was stopped), sTS (two-stage), sSUB (no burst/ring modules) | done | `n14/gbdt_{xgb,cat,ts,sub}.log` |
| sixth source | RAW replaced by sSUB (RAW argmax differs from C on 149 Aug rows only; its Sep result is already known) | |
| August score of single sources | C .4963, sXGB .4958, sCAT .4957, TABM .4939, sTS .4982, sSUB .4944 | `n14/combine.log` |
| TabM checkpoints seed 0 fold 0 | done; Aug final e6 .4915, e12 .4929, e20 .4938, e30 .4947 (C .4963); epoch 30 chosen (better on both halves); new e6 == existing TABMple0 exactly | `n14/tabm_ck0_f0.log`, `n14/ck_select0.log` |
| TabM seed 0 fold 1 (Sep at 6/12/20/30), then seed 1 fold 0 | running (GPU) | `n14/tabm_ck0_f1.log`, `n14/tabm_ck1_f0.log` |
| combine (blends, LR, tree; August only) | done 20:34: best blend C+TABM@.35 .4979 (+.0017; the same file as the earlier PN_BLEND); greedy keeps sTS alone (.4982); LR C=.001 .4957 (-.0006); tree .4940 (-.0023) | `n14/combine.log`, `n14/combine.json` |
| September scoring of the frozen list | not started | `n14/eval_ss1.log` |

## Frozen list for September, part 1 + 2 (written 20:35, before any September score of these files was read)
Reference C. Candidates, all scored once by `bench_eval.py N C,sTS,sCTS,sXGB,sCAT,sSUB,sBSEL,sLR,sLR5,sTREE _ss1`:
- sTS: soft two-stage LightGBM alone (best single source on August, .4982 vs C .4963).
- sCTS: 0.65 C + 0.35 sTS (two-stage as an added source in a blend; the better of the two declared weights on August, .4971).
- sXGB, sCAT, sSUB: the other new sources alone (August .4958, .4957, .4944).
- sBSEL: best declared simple blend on August = 0.65 C + 0.35 TABM. Its September result is already known (-.0005, notes/tabm_phfc.md).
- sLR: logistic-regression stacker on six sources, C = .001 (August grouped OOF .4957). sLR5: the same without sTS (.4952).
- sTREE: shallow residual tree stacker (August grouped OOF .4940).
Already known on September and not rescored: TABM alone (-.0041), RAW (+.0001).
Part 3 frozen: seed 0 epoch 30 (chosen on August) vs epoch 6 (log-loss choice), alone and in the fixed 0.35 blend with C.

## September, part 1 + 2 (20:45, `n14/eval_ss1.log`, `n14/diff_ss1.log`): after decoder, change vs C .5070 (boot 5%..95%)
sTS -.0008 (-.0020..+.0002) | sCTS +.0008 (+.0001..+.0014) | sXGB -.0021 | sCAT -.0029 | sSUB -.0012 | sBSEL -.0005 | sLR +.0009 (-.0004..+.0022) | sLR5 +.0009 | sTREE -.0032

## Replication of sCTS, declared 20:46 before running (weight .35 frozen, after the first sCTS September result was read)
- benchmark D: two-stage on D matrix -> PD_sTS, PD_sCTS = .65 PD_C + .35 PD_sTS; `bench_eval.py D C,sTS,sCTS _ss1d`.
- benchmark N, fresh seeds 2,3: PN_sTSb2, PN_sCTSb = .65 C + .35 sTSb2; `bench_eval.py N C,sTSb2,sCTSb _ss1b`.

## 20:56 results so far (September after decoder vs C .5070; logs in n14/)
- Part 3 seed 0 (`eval_ss3.log`): TabM alone e6 -.0021, e30 -.0024 (before decoder -.0021 vs -.0150); blend .35: e6 +.0002 (-.0008..+.0015), e30 +.0011 (+.0002..+.0018).
- sCTS replication: benchmark D +.0003 (-.0007..+.0012) (`eval_ss1d.log`); N seeds 2,3 +.0016 (+.0007..+.0026) (`eval_ss1b.log`). sTS alone: D -.0042, N seeds 2,3 +.0004.
## Declared 20:56 before running
- Control for seed averaging: plain 12-class LightGBM seeds 2,3 (PN_sLGBb2), sCC = .65 C + .35 sLGBb2; `bench_eval.py N C,sCC,sCTSb _ss1c` and paired `bench_eval.py N sCC,sCTSb _ss1p`.
- Paired e30 vs e6 blend: `bench_eval.py N sTB0e6,sTB0e30 _ss3p`.
- TabM seed 1: fold 0 grid 6/12/20/30 (running), same August rule; fold 1 to 30 epochs; frozen September comparison:
  .65 C + .35 chosen-epoch seed 1 vs .65 C + .35 TABMple1 (log-loss epoch 7), and the two-seed versions.

## 21:03
- Paired e30 blend vs e6 blend (seed 0): +.0009 (-.0006..+.0022) (`eval_ss3p.log`).
- Seed-averaging control: sCC (.65 C + .35 plain LightGBM seeds 2,3) .5071 (+.0001); sCTSb vs sCC paired +.0015 (+.0006..+.0025) (`eval_ss1c.log`, `eval_ss1p.log`).
- Declared before running: benchmark D with two-stage seeds 2,3 (PD_sTSb2, PD_sCTSb; `eval_ss1db.log`); N four-seed version
  sCTS4 = .65 C + .35 mean(sTS, sTSb2) (`eval_ss1f.log`); D four-seed PD_sCTS4 in the same D run.
- Candidate CSV: NOT attempted. Production features are benchmark-D style, where the sCTS gain is +.0003 with an interval across zero.

## 21:10
- N four-seed sCTS4: +.0017 (+.0008..+.0027) (`eval_ss1f.log`). D seeds 2,3 + four-seed: running (`eval_ss1db.log`).
- TabM seed 1 August (`ck_select1.log`): e6 .4935, e12 .4952, e20 .4937, e30 .4943, log-loss epoch 7 (TABMple1) .4949 -> rule picks 12 (seed 0 picked 30).
  Fold 1 seed 1 running to 30 epochs (`tabm_ck1_f1.log`). Frozen September comparison: .65 C + .35 e12(seed 1) vs .65 C + .35 TABMple1;
  two-seed: .65 C + .35 mean(e30 s0, e12 s1) vs existing BLEND. Also declared (replication of the seed-0 e30 blend): .65 C + .35 e30(seed 1).
- 21:10 CSV rule declared: attempt a candidate CSV only if the D four-seed sCTS4 interval is above zero (production features are D-style); otherwise not attempted.

## 21:16 two-stage replication complete
| blend .65 C + .35 two-stage | N Aug (cross-half) | N Sep after decoder | D Sep after decoder |
|---|---|---|---|
| seeds 0,1 (sCTS) | .4971 (+.0008) | +.0008 (+.0001..+.0014) | +.0003 (-.0007..+.0012) |
| seeds 2,3 (sCTSb) | .4970 (+.0007) | +.0016 (+.0007..+.0026) | -.0006 (-.0017..+.0003) |
| four seeds (sCTS4) | .4971 (+.0008) | +.0017 (+.0008..+.0027) | +.0001 (-.0009..+.0010) |
| control: .65 C + .35 plain LightGBM seeds 2,3 (sCC) | .4968 (+.0005) | +.0001 (-.0009..+.0010) | not run |
Two-stage alone: N Sep -.0008 (seeds 0,1), +.0004 (seeds 2,3); D Sep -.0042, -.0029. Plain LightGBM seeds 2,3 alone: N Aug .4973, Sep +.0002 (-.0015..+.0025) = seed noise.
Logs: `eval_ss1.log`, `eval_ss1b.log`, `eval_ss1c.log`, `eval_ss1p.log`, `eval_ss1f.log`, `eval_ss1d.log`, `eval_ss1db.log`, `aug_one.log`.
Verdict: positive on N (three readings), not reproduced on D (production-style features) -> promising but unconfirmed. CSV rule not met: no candidate CSV.
Still running: TabM seed 1 fold 1 (`tabm_ck1_f1.log`), then `eval_ss3s1.log`.

## FINAL (21:37) - all runs finished, nothing running. No candidate CSV. dd_final.csv untouched.

Benchmark N, reference C (LightGBM recipe, seeds 0,1): August cross-half score .4963; September before decoder .4815, after .5070.
August = `nb.aug_score` (bias tuned on one event-grouped half, other half scored after the full decoder, swapped).
September change = after decoder vs C, 90% paired event bootstrap (5%..95%), from `bench_eval.py`. Corrected / new errors = rows after decoder.
Logs are in `n14/` (`eval_<name>.log` and `diff_<name>.log`; August scores in `combine.log`, `ck_select0.log`, `ck_select1.log`, `aug_one.log`, `aug_one2.log`).

| candidate | Aug | Sep before | Sep after | change after (5%..95%) | rows changed / corrected / new errors | largest per-class F1 moves (after) | log |
|---|---|---|---|---|---|---|---|
| sTS two-stage alone, seeds 0,1 | .4982 | .4810 | .5062 | -.0008 (-.0020..+.0002) | 141 / 45 / 81 | phish -.010, loan -.005, merch +.005 | eval_ss1 |
| sTS two-stage alone, seeds 2,3 | .4972 | .4822 | .5074 | +.0004 (-.0011..+.0019) | 148 / 60 / 72 | | eval_ss1b |
| sCTS = .65 C + .35 sTS (0,1) | .4971 | .4825 | .5078 | +.0008 (+.0001..+.0014) | 34 / 19 / 11 | all within .003 | eval_ss1 |
| sCTSb = same, two-stage seeds 2,3 | .4970 | .4833 | .5086 | +.0016 (+.0007..+.0026) | 61 / 32 / 21 | merch +.011, refund +.005, fake +.005 | eval_ss1b |
| sCTS4 = same, four two-stage seeds | .4971 | .4835 | .5087 | +.0017 (+.0008..+.0027) | 70 / 34 / 29 | merch +.011, refund +.005, fake +.005 | eval_ss1f |
| control sCC = .65 C + .35 plain LightGBM seeds 2,3 | .4968 | .4821 | .5071 | +.0001 (-.0009..+.0010) | 57 | | eval_ss1c |
| paired: sCTSb minus sCC | | | | +.0015 (+.0006..+.0025) | 70 | | eval_ss1p |
| seed floor: plain LightGBM seeds 2,3 alone | .4973 | .4821 | .5072 | +.0002 (-.0015..+.0025) | 124 | | eval_ss1c |
| sXGB XGBoost alone | .4958 | .4796 | .5049 | -.0021 (-.0042..-.0001) | 227 / 95 / 102 | phish +.017, fake -.024 | eval_ss1 |
| sCAT CatBoost alone (GPU, 1 seed) | .4957 | .4778 | .5041 | -.0029 (-.0047..-.0010) | 202 / 78 / 98 | fake -.014, merch -.013, phish +.009 | eval_ss1 |
| sSUB LightGBM without burst/ring | .4944 | .4802 | .5058 | -.0012 (-.0033..+.0012) | 209 / 62 / 123 | mule -.015, loan -.013, fake +.010 | eval_ss1 |
| TABM ple, log-loss epochs, 2 seeds | .4939 | .4777 | .5029 | -.0041 (-.0059..-.0022) | 222 / 72 / 116 | accou -.017, refund -.017 | eval_ss3s2 |
| sBSEL best simple blend on Aug = .65 C + .35 TABM | .4979 | .4810 | .5065 | -.0005 (-.0014..+.0003) | 63 / 34 / 23 | refund -.004, fake -.004 | eval_ss1 |
| sLR logistic stacker, 6 sources, C=.001 | .4957 | .4821 | .5078 | +.0009 (-.0004..+.0022) | 121 / 61 / 44 | fake +.010, sim -.005 | eval_ss1 |
| sLR5 same without sTS | .4952 | .4817 | .5079 | +.0009 (-.0006..+.0021) | 121 / 63 / 42 | fake +.010, accou -.004 | eval_ss1 |
| sTREE shallow residual tree stacker | .4940 | .4773 | .5038 | -.0032 (-.0043..-.0015) | 103 / 33 / 57 | sim -.025, accou -.011 | eval_ss1 |
| TabM seed 0, epoch 6 (log-loss choice) alone | .4915 | .4794 | .5049 | -.0021 (-.0042..+.0002) | 231 / 103 / 94 | phish +.019, fake -.022 | eval_ss3 |
| TabM seed 0, epoch 30 (Aug macro-F1 choice) alone | .4947 | .4666 | .5046 | -.0024 (-.0043..-.0007) | 204 / 69 / 114 | before decoder velocity -.161 | eval_ss3 |
| .65 C + .35 TabM s0 e6 | .4972 | .4817 | .5072 | +.0002 (-.0008..+.0015) | 91 / 50 / 25 | phish +.008, fake -.009 | eval_ss3 |
| .65 C + .35 TabM s0 e30 | .4977 | .4820 | .5081 | +.0011 (+.0002..+.0018) | 95 / 44 / 42 | refund +.004 | eval_ss3 |
| paired: s0 e30 blend minus s0 e6 blend | | | | +.0009 (-.0006..+.0022) | 138 | | eval_ss3p |
| TabM seed 1, epoch 7 (log-loss choice) alone | .4949 | .4737 | .4991 | -.0079 (-.0101..-.0057) | 287 / 73 / 170 | accou -.031, refund -.020 | eval_ss3s1 |
| TabM seed 1, epoch 12 (Aug choice) alone | .4952 | .4766 | .5024 | -.0046 (-.0069..-.0024) | 377 / 113 / 228 | merch -.022 | eval_ss3s1 |
| .65 C + .35 TabM s1 e7 | .4980 | .4822 | .5072 | +.0002 (-.0009..+.0013) | 92 / 42 / 39 | fake +.012 | eval_ss3s1 |
| .65 C + .35 TabM s1 e12 | .4980 | .4819 | .5076 | +.0006 (-.0009..+.0025) | 165 / 86 / 55 | phish +.017, fake -.020 | eval_ss3s1 |
| .65 C + .35 TabM s1 e30 | .4955 | .4819 | .5076 | +.0006 (-.0009..+.0026) | 166 / 86 / 60 | phish +.017, fake -.020 | eval_ss3s2 |
| .65 C + .35 mean(s0 e30, s1 e12) | .4980 | .4822 | .5077 | +.0007 (-.0006..+.0022) | 143 / 75 / 48 | phish +.016, fake -.015 | eval_ss3s2 |

Benchmark D (production-style features; reference PD_C: Sep .4824 / .5078), two-stage only:

| candidate | Sep before | Sep after | change after (5%..95%) | changed / corrected / new errors | per-class | log |
|---|---|---|---|---|---|---|
| two-stage alone, seeds 0,1 | .4782 | .5036 | -.0042 (-.0065..-.0021) | 250 | | eval_ss1d |
| two-stage alone, seeds 2,3 | .4796 | .5049 | -.0029 (-.0043..-.0016) | 186 / 51 / 117 | refund -.016, layer -.011, phish -.010 | eval_ss1db |
| .65 C + .35 two-stage (0,1) | .4827 | .5081 | +.0003 (-.0007..+.0012) | 90 / 32 / 46 | fake +.011, merch .000 | eval_ss1d |
| .65 C + .35 two-stage (2,3) | .4818 | .5072 | -.0006 (-.0017..+.0003) | 73 / 22 / 38 | refund -.007, fake +.008 | eval_ss1db |
| .65 C + .35 two-stage (four seeds) | .4826 | .5080 | +.0001 (-.0009..+.0010) | 93 / 29 / 51 | fake +.011, merch .000 | eval_ss1db |

Verdicts
- Part 1 (combination): no detected difference. Best August blend -.0005; LR stacker +.0009 with an interval across zero and a negative
  August score (-.0006); tree stacker measurably worse (-.0032); XGBoost and CatBoost alone measurably worse.
- Part 2 (soft two-stage): alone = no detected difference on N, measurably worse on D. As a 35% blend source: positive on N September
  (two seed pairs and their average; same rows, same C; the replication was chosen after reading the first result), not supported by August
  (seed control +.0005 vs +.0008), not reproduced on D. More than half of the N gain is merchant_collusion (+.011), which does not move on D.
  -> promising but unconfirmed. No CSV (rule declared 21:10: the D four-seed interval must be above zero).
- Part 3 (TabM checkpoint by August macro-F1): no detected difference. The two seeds pick different epochs (30 and 12). Alone, the chosen
  checkpoints stay below LightGBM; epoch 30 loses .0150 before the decoder (velocity_bot -.161; an epoch count chosen on the July fit is
  twice the optimiser steps on the July+August refit). Blend changes are +.0006 to +.0011 vs C and +.0009 (-.0006..+.0022) vs the log-loss checkpoint.

Limitations
- September has been reused many times; about 30 variants were read on it in this pass. Intervals are 90% and cover September event
  sampling only. Measured seed floor: plain LightGBM seeds 2,3 vs seeds 0,1 = +.0002 (-.0015..+.0025).
- Decoder = bias + forcing + zones + bait rule (run_all steps 5-6). The generic rule layers (+.0016) and the test-only rules were not applied to any candidate.
- Stackers: August inputs come from July-only models, September inputs from July+August models; the August file is grouped out-of-fold,
  the September file is the full-August fit.
- PN_TABM August predictions come from checkpoints stopped on August log-loss, so August slightly favours every TABM blend and stacker.
- CatBoost: GPU, 1 seed, 1000 iterations (the CPU run was stopped after 18 minutes without finishing fold 0).
- The log-loss printed on the blend rows of `ck_select*.log` is the TabM-alone value (print bug); do not cite it.
- RAW was dropped from the pool for sSUB; the pool has one feature variant only (benchmark N has a single feature build).
- `combine_fail1.log` = first combine launch (worker pickling error, fixed); `gbdt_cat_cpu_killed.log` = the stopped CPU CatBoost run.
