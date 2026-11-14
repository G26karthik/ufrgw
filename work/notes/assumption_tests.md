# Tests of three shared assumptions (14 to 15 Nov 2026)

Read first:
- Benchmark N: labels hidden, features at full density. Benchmark D: the production density-matched matrix. D is a
  label-informed density stress test, not independent ground truth. "Fails D" does not mean "cannot generalise".
- Every variant was chosen or frozen on August, then September was read. September had been inspected more than 40 times
  before this round. The intervals (5%..95%, paired event bootstrap) cover September event sampling for one comparison.
  They do not correct for that reuse, for seed noise (about .002 between seed pairs) or for decoder-tuning noise.
- Status words: supported / promising but unconfirmed / no detected difference / measurably worse / not evaluated adequately.
- No candidate file was built from any of these tests. `subs/dd_final.csv` is unchanged.

## A. Does the model depend on the non-QR stamp? (`scratch_stamp/`)
Stamp = non-QR row with `qr_vpa_mismatch == 0` (ordinary non-QR rows carry null). Train: 7,454 stamped non-QR rows.

Test: on held months, set the stamp to null on every stamped non-QR row, rebuild all features, predict with the same
boosters (recipe C, seeds 0 and 1, benchmark N). Only the base column changes; the other 189 columns are identical.

| masked rows | n | decoded non-legit | mean P(fraud) | decoded = label |
|---|---|---|---|---|
| Sep fraud-labelled | 2080 | .995 -> .991 | .905 -> .893 | .857 -> .854 |
| Sep legit-labelled | 311 | .990 -> .990 | .908 -> .895 | .010 -> .010 |
| Aug fraud-labelled | 2135 | .994 -> .990 | .906 -> .892 | .844 -> .839 |
| Aug legit-labelled | 280 | .989 -> .986 | .900 -> .884 | .011 -> .014 |

- Largest class effect: phishing_collect, Sep decoded non-legit .984 -> .957 (n 258). Other classes move by .011 at most.
- Declared trigger for a training variant (detection falls by more than 20 points): not met (0.4 points). The variant
  (random masking of half the stamped fit rows) was written and not run.
- Context: 15% of fraud-labelled non-QR rows carry the stamp. Of the unstamped ones, .328 (Aug) / .333 (Sep) are decoded
  non-legit.

Status:
- Supported (N, both folds, LightGBM recipe C): stamped rows stay detected without the stamp column. These rows carry
  other marks; the model does not lean on the stamp alone.
- Not evaluated adequately: the claim in `fresh_columns.md` that detection of fraud-path rows is saturated. This test
  says nothing about the two thirds of unstamped fraud-labelled non-QR rows that are decoded legit.
- Not tested: the merchant-QR null stamp, fit-time masking, benchmark D, other model families, other seeds.
- Earlier work (`ls_stamp.md`, `adapt.md`, `--qr-null`, `dd_rules_stamp.py`) changed labels, QR rows or decoder rules.
  None masked the non-QR stamp at prediction time.

## C. Training influence of legit against fraud (`scratch_wt/`)
Baseline: legit rows downsampled to 20% with weight 5, fraud rows weight 1. Arms (same rows, seeds, labels; total weight
rescaled to the baseline total; biases retuned on August):
- `wL1`: every sampled row weight 1 (legit share of total weight .963 -> .839).
- `wSQ`: baseline weight x (class count)^-0.5 over all 12 classes (legit share .614; fraud rows 7.8 to 17.4 against legit 3.19).

Control with baseline weights is bit-identical to the stored references on N and D.

August after the decoder, change against the reference:

| variant | N seeds 0,1 | N seeds 2,3 | D seeds 0,1 |
|---|---|---|---|
| wL1 | +.0001 | -.0008 | +.0000 |
| wSQ | -.0005 | -.0021 | -.0015 |
| .65 ref + .35 wL1 | +.0002 | -.0005 | -.0001 |
| .65 ref + .35 wSQ | +.0007 | +.0001 | +.0006 |
| prior-matched blend wL1 | +.0005 | +.0001 | -.0003 |
| prior-matched blend wSQ | +.0005 | -.0002 | -.0002 |

September after the decoder, change against the reference (5%..95%):

| variant | N seeds 0,1 | N seeds 2,3 | D seeds 0,1 |
|---|---|---|---|
| wL1 | +.0022 (+.0010..+.0033) | +.0009 (-.0013..+.0029) | -.0010 (-.0024..+.0004) |
| wSQ | +.0013 (+.0002..+.0023) | +.0012 (-.0010..+.0029) | -.0004 (-.0015..+.0007) |
| .65 ref + .35 wL1 | +.0023 (+.0012..+.0034) | +.0007 (-.0003..+.0017) | +.0005 (-.0003..+.0012) |
| .65 ref + .35 wSQ | +.0016 (+.0006..+.0024) | +.0013 (+.0001..+.0024) | +.0004 (-.0003..+.0009) |
| prior-matched blend wL1 | +.0004 (-.0002..+.0011) | -.0004 (-.0014..+.0005) | +.0001 (-.0008..+.0008) |
| prior-matched blend wSQ | +.0005 (-.0004..+.0013) | -.0006 (-.0023..+.0006) | -.0003 (-.0010..+.0004) |

- Error trade of the single arms, September: more new legit rows called fraud than missed fraud recovered in every
  reading (N seeds 0,1: wL1 41 against 32, wSQ 56 against 28; D: wL1 57 against 24, wSQ 160 against 42).
- The N gain with seeds 0,1 sits in fake_cashback (+.012 F1) and merchant_collusion (+.009). With seeds 2,3 fake_cashback
  is -.008 for wL1; on D merchant_collusion is -.012. Other perturbations move the same two classes on N
  (`final_comparison.md`), so this is not independent evidence.
- The raw blend is not a 35% source: mean fraud probability on August is .037 (reference), .119 (wL1), .279 (wSQ). With
  the arm's prior scaled back to the baseline (prior-matched blend), no reading has an interval above zero.

Status (rule frozen before September: "supported" needs N seeds 0,1 and both confirmations above zero, August not negative):
- wL1 alone, wSQ alone, raw blend of wL1, both prior-matched blends: no detected difference.
- Raw blend .65 ref + .35 wSQ: promising but unconfirmed (both N seed pairs above zero, D not). Its gain is absent when
  the prior scale is matched.
- Equivalent history: `tuning.md` (d) tested `cw_pow .5` (fraud classes only, no rescale, single seed, before the
  decoder): neutral. The legit-weight-1 arm and the 12-class rescaled arm had not been run before.
- Not tested: other exponents, legit weight between 1 and 5, rounds or `lambda_l2` retuned under the new weights, the
  test-only rule layers on top, a production retrain.
