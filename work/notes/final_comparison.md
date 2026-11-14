# Final comparison of the 14 Nov evening tests (19:56 to 23:00 IST)

Outcome: no supported candidate. `subs/dd_final.csv` is unchanged (SHA-256 `8e04353d...29cd`). Nothing was uploaded.

## How every test was scored
- Benchmark N: features built with held-month labels hidden, at full density. Benchmark D: the production feature matrix.
- Fit July, develop on August, freeze, then score September once (fit July + August).
- "After decoder" = bias, takeover forcing, device zones, bait rule (run_all steps 5 and 6).
- August = bias tuned on one event-grouped half, other half scored after the decoder, halves swapped. The two halves
  differ by about .010 for every source.
- September change = candidate minus the reference trained with the same seeds, paired event bootstrap 5%..95%.
- Seed floor: plain LightGBM seeds 2,3 against seeds 0,1 on N = +.0002 (-.0015..+.0025).
- September is a reused development month, not a fresh holdout. More than 40 variants were read on it this evening.

## Results (September after decoder; reference C: N .5070, D .5078)

| # | test | benchmark | August change | September change (5%..95%) | rows changed / corrected / newly wrong | status |
|---|---|---|---|---|---|---|
| 1 | Logistic stacker, 6 sources | N | -.0006 | +.0009 (-.0004..+.0022) | 121 / 61 / 44 | no detected difference |
| 1 | Shallow tree stacker | N | -.0023 | -.0032 (-.0043..-.0015) | 103 / 33 / 57 | measurably worse |
| 1 | XGBoost alone | N | -.0005 | -.0021 (-.0042..-.0001) | 227 / 95 / 102 | measurably worse |
| 1 | CatBoost alone (1 seed) | N | -.0006 | -.0029 (-.0047..-.0010) | 202 / 78 / 98 | measurably worse |
| 2 | Soft two-stage alone, seeds 0,1 / 2,3 | N | +.0019 / +.0009 | -.0008 / +.0004 (both across zero) | 141 / 45 / 81 | no detected difference |
| 2 | .65 C + .35 two-stage, four seeds | N | +.0008 | +.0017 (+.0008..+.0027) | 70 / 34 / 29 | positive on N |
| 2 | same, against a seed-averaging control | N | | +.0015 (+.0006..+.0025) | 70 | positive on N |
| 2 | .65 C + .35 two-stage, four seeds | D | | +.0001 (-.0009..+.0010) | 93 / 29 / 51 | not reproduced -> promising but unconfirmed |
| 3 | TabM, checkpoint by August macro-F1, alone | N | below C | -.0024 and -.0046 (two seeds) | 204 to 377 | measurably worse |
| 3 | .65 C + .35 TabM, that checkpoint | N | +.0014 to +.0017 | +.0011 (seed 0), +.0006 (seed 1), +.0007 (mean, -.0006..+.0022) | 143 / 75 / 48 | no detected difference |
| 4 | c2s6 instead of h6 (model change only) | D | +.0006 | +.0008 (+.0000..+.0017) | 100 / 41 / 51 | promising but unconfirmed |
| 5 | aug3btta instead of h6 | D | -.0006 | -.0017 (-.0038..+.0002) | | no detected difference, leaning worse |
| A | 1023 bins on row-level numeric columns | N seeds 0,1 | +.0005 | +.0011 (+.0001..+.0020) | 51 / 25 / 18 | first reading |
| A | same | N seeds 2,3 | -.0003 | +.0012 (+.0004..+.0022) | 62 / 30 / 26 | |
| A | same | D | +.0003 | -.0006 (-.0014..+.0003) | 51 / 16 / 26 | not reproduced -> no detected difference |
| A | 1023 bins on every feature | N seeds 0,1 | +.0006 | +.0009 (+.0001..+.0017) | 54 / 30 / 18 | first reading |
| A | same | N seeds 2,3 | +.0007 | +.0016 (+.0008..+.0024) | 50 / 27 / 16 | |
| A | same | D | -.0007 | -.0011 (-.0019..-.0003) | 58 / 19 / 30 | measurably worse on D (one seed pair) |
| A | exact-value features (5 raw columns) | N seeds 0,1 | +.0002 | +.0020 (+.0006..+.0031) | 145 / 63 / 63 | first reading |
| A | same | N seeds 2,3 | -.0015 | -.0015 (-.0035..+.0001) | 108 / 30 / 61 | |
| A | same | D | +.0003 | -.0005 (-.0027..+.0015) | 162 / 66 / 81 | not reproduced -> no detected difference |
| B | 50 class-conditional neighbour columns | N | -.0001 | +.0003 (-.0013..+.0016) | 198 / 72 / 112 | no detected difference |
| B | .65 C + .35 of that model | N | +.0009 | +.0008 (-.0004..+.0019) | 86 / 39 / 42 | no detected difference |
| C | TabICLv2, 12 classes, alone | N | -.0012 | -.0042 (-.0063..-.0021) | 384 / 91 / 263 | measurably worse |
| C | .65 C + .35 TabICLv2 | N | +.0008 | +.0007 (-.0007..+.0020) | 141 / 58 / 73 | no detected difference |

Tests 1 to 3: `scratch_stack/PROGRESS.md`. Tests 4 and 5: `notes/ledger.md` section 6. A: `scratch_gap/numrep.py`,
`numrep_rule.txt`, `eval_N_numrep.log`, `eval_N_numrep_b2.log`, `eval_D_numrep.log`, `aug_dec_*numrep*.log`,
`numrep_audit.log`. B: `scratch_knn/PROGRESS.md`. C: `scratch_icl/PROGRESS.md`, `eval_N_icl.log`, `diff.log`.

## Experiment A in detail (numeric representation and binning)
- Audit: the Float32 matrix loses nothing material. Collapsed distinct values: amount 0, balance_before 3, log_amount 27 of
  350,514. No finer raw precision exists than the printed one.
- 112 amount values have 500 or more fit rows; none is majority fraud. 15 values with 50 to 499 rows are majority fraud
  (3,192 rows).
- Rule declared at 22:13 before the confirmations: supported only if September is positive with the interval above zero
  in both confirmations (N seeds 2,3 and D seeds 0,1) and August is not negative in either.
- 1023 bins on row-level columns: September is positive on N with both seed pairs. It misses the rule on N only through
  an August change of -.0003, far inside the .010 spread between August halves. D is what fails. Zero of two.
- 1023 bins on every feature: N seeds 2,3 passes both months; D is below zero with the whole interval. One of two. By the
  letter of the rule this is "promising but unconfirmed", but the production pipeline uses the D feature matrix, where
  the arm is measurably worse. D was run with one seed pair (0,1); the interval covers September event sampling, not
  seeds. It is not a candidate.
- Exact-value features: the first +.0020 reverses to -.0015 with seeds 2,3. Zero of two.
- Missed fraud recovered (after decoder): N seeds 0,1: 10 / 10 / 33; N seeds 2,3: 15 / 9 / 4; D: 2 / 3 / 11.

## Experiment C in detail (TabICLv2)
- tabicl 2.2.0, checkpoint `tabicl-classifier-v2-20260212`, local RTX 4060. 12 classes use the package's many-class path.
- Context: 20,000 legit fit rows + up to 2,500 rows per fraud class (38,257 rows for July, 44,390 for July + August), one
  context bag, ensemble size 1, prior-corrected. Full August and September rows were scored (615 rows per second).
- Alone it reaches .4966 on August after the decoder (C .4978) and loses .0042 on September: 263 rows newly wrong against
  91 corrected, 132 new legit rows called fraud. mule_account -.017.
- Second context bag (seed 1), August only, development side check: alone .4958, two-bag mean .4965, two-bag blend
  .4986. No gain over one bag (.4966 and .4986). It has no September score.
- Limits: one context bag, ensemble 1, categorical codes entered as ordinal numbers, no fine-tuning. Context size
  was set by time (about 31 minutes for both months at 40k context rows). This result does not cover a fine-tuned or larger-context run.

## What the evening shows
- No source beats the current LightGBM alone. Blends at 35% weight move September by -.0005 to +.0017 on N.
- Every N gain that was re-tested on D failed to reproduce there (two-stage blend, both binning arms). The neighbour
  and TabICLv2 blends were not built on D; their N intervals already include zero.
- The N gains sit mostly in fake_cashback and merchant_collusion and lose phishing_collect. Different perturbations
  move the same two classes, so these readings are not independent evidence.
- A blend of several of these sources at once was not run: no component is supported on D, and choosing components
  after reading their September scores would bias the result.
- c2s6 stays the only D-side positive (+.0008, lower bound +.0000, one class). It changes 12 October rows
  (`scratch_gap/c2x_diff.csv`). It has never been measured alone on the public board.

## Not covered
- Benchmark D versions of B and C. A multi-source blend. A second TabICLv2 context bag on September.
- The binning and exact-value models were scored alone only, never as 35% blend sources.
- Benchmark D has one seed pair for experiment A.
- The generic rule layers (+.0016 on September) and the test-only rules were not re-applied on top of each variant.
- Benchmark N does not reproduce October's lower density; D is the closer match to production.
