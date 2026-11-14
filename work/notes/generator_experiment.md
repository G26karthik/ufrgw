# Generator reconstruction experiment (14 Nov 2026, evening)

Result: no candidate. `subs/dd_final.csv` (public 0.5505) stays the submission.
Two mechanisms were built and measured. Both failed the pass rules that were written into the scripts before the
September comparison was read. No upload slot was used.

## Protocol (same for every arm)

- Features: the 190 columns of `feats_cache/matrix_m5dm2_pairsfd.parquet` (the matrix behind `h6`). Feature code never reads labels.
- Model: the `h6` recipe of `train.py` (LightGBM, 150 rounds, legit rows downsampled to 20%, fit target `relabel C`), 2 seeds.
- Development: fit July, predict August. Per-class decode bias tuned on August only.
- Evaluation: fit July + August, predict September, decode with the August bias. Scores use the supplied (noisy) labels;
  `sep_macro_clean` uses event-cleaned labels and is a proxy only. September has been looked at many times before; it is not a pristine holdout.
- Uncertainty: 200 fixed random 30% / 70% row splits of September (a simulation; the organiser's split is unknown), and
  200 event-bootstrap draws (fraud events resampled whole, legit rows singly).

## Mechanism 1: bait-event generator (fake_cashback and phishing bait events)

Hypothesis. fake_cashback is the class most sensitive to training volume, and its main typed error is the swap with
phishing (OOF Aug + Sep: 115 fake_cashback rows decoded phishing, 169 phishing rows decoded fake_cashback).
More complete bait events should sharpen that boundary. Falsified if the generator arm does not beat both the plain
recipe and a weighting control on September.

Recipe read from July to September labels (`scratch_gen/x1.py`, `x2.py`):
- 1 or 2 tiny P2P PAYs (1, 2, 5 or 10 rupees, uniform) from a fraudster to a victim, seconds to 10 minutes apart.
- One integer payback victim -> fraudster, PAY or COLLECT, median 10.6k rupees, balance = amount / 0.909.
- Delay bait -> payback: phishing bait events under 3 h; fake_cashback median 2.1 h with a tail to 47 h.
- Bait rows are legit-recipe rows apart from amount and receiver. Sim age, device age, pin retries and login time are
  drawn per row. `app` is fixed per account. Fraudster accounts are not reused (277 of 284 bait senders have one victim).

Generator (`gen_events.py`, version `bait-v2`): one generated event per source event. Both accounts are remapped to
accounts drawn like legit senders, all rows are shifted by the same whole number of days inside the source month,
per-row nuisance fields are redrawn from legit P2P PAY rows of that month, and shape, amounts, balances, types and the
observed noisy row labels are kept. Features of generated rows come from the same module code, built on the train
half plus the generated events. 845 events, 1,574 rows. A fold only uses events whose source month is a fit month.

Fidelity (`scratch_gen/fid.py`): a classifier separating generated rows from their real source rows on the 190
features scores AUC 0.69 with uniform account sampling (v1) and 0.59 with activity-weighted sampling and kept balances (v2).
Not modelled: fresh and secondary devices, role reuse across classes.

| arm | what changes | Sep macro-F1 | change vs A | event bootstrap 5%..95% | private side gains (share of 200 splits) | fake_cashback F1 (P / R) | phishing F1 | legit rows called fraud | Sep macro, cleaned labels |
|---|---|---|---|---|---|---|---|---|---|
| A | nothing (h6 recipe) | 0.4824 | | | | 0.363 (0.66 / 0.25) | 0.448 | 812 | 0.4948 |
| B05 | source-event rows weighted x1.5 | 0.4823 | -0.0001 | -0.0009..+0.0006 | 36% | 0.364 (0.69 / 0.25) | 0.450 | 826 | 0.4946 |
| B1 | source-event rows weighted x2 | 0.4830 | +0.0006 | -0.0011..+0.0027 | 98% | 0.361 (0.69 / 0.24) | 0.453 | 825 | 0.4953 |
| BR | `min_data_in_leaf` 50, `lambda_l2` 2.5 | 0.4789 | -0.0035 | -0.0055..-0.0014 | 0% | 0.341 (0.76 / 0.22) | 0.457 | 806 | 0.4911 |
| C05 | + 0.5x generated events | 0.4805 | -0.0019 | -0.0037..-0.0002 | 0% | 0.339 (0.80 / 0.22) | 0.465 | 816 | 0.4929 |
| C1 | + 1x generated events | 0.4818 | -0.0006 | -0.0014..+0.0004 | 10% | 0.362 (0.70 / 0.24) | 0.454 | 810 | 0.4942 |

August (development) macro-F1: A 0.4732, B05 0.4735, B1 0.4735, BR 0.4732, C05 0.4727, C1 0.4724.

What it establishes:
- The generator arms do not beat the plain recipe or the weighting control. Pass rule (+0.002 over A and over the best B) not met.
- Failure type: no gain over weighting. Generated events move rows between fake_cashback and phishing
  (phishing decoded fake_cashback 78 -> 62, fake_cashback decoded phishing 70 -> 76 for C1) with no net gain.
- Volume is not the limit at full data size: fake_cashback F1 is 0.343 with one fit month and 0.375 with two (`h6`, pooled bias).
  The swap with phishing is decided by the generator's own label choice for look-alike events, which no extra copies can resolve.
- For any fixed pair of submissions, the public and private changes share the same mean; a public gain is not extra evidence beyond it.

## Mechanism 2: October legit amounts are x1.17

Measured on the unlabeled October rows: median amount 541 vs 464 in September (x1.167); median balance x1.017, so only
amounts moved. Legit P2P integer amounts are multiples of 10 in 99.6% of rows in October and 99.3% in September
(`scratch_gen/amt_check.py`), so the scaling happens before rounding. Fraud signature amounts are not scaled (`notes/drift.md`).

Hypothesis. The model is fit on unscaled months, so it may lose on October; a fit on amount-scaled train months, or a
decode bias tuned on an amount-scaled month, should recover the loss. Falsified if neither beats the plain model on an
October-like September.

World (`scratch_gen/amt_world.py`): legit-labelled train rows that the model does not flag get amount x1.17, keeping
multiples of 10 and integers, capped at the observed maximum of 200,000; all 190 features rebuilt.

| run on October-like September | macro-F1 |
|---|---|
| A, real September (reference) | 0.4824 |
| A, bias from real August | 0.4821 |
| A, bias retuned on October-like August | 0.4821 (change -0.0000, bootstrap -0.0002..+0.0001) |
| M, fit on scaled months, bias from October-like August | 0.4815 (change -0.0006, bootstrap -0.0015..+0.0003, private side gains in 4% of splits) |

What it establishes:
- The amount shift costs the model 0.0003. There is nothing to recover; the matched fit is slightly worse. Mechanism falsified.
- A first version scaled integers without keeping multiples of 10 and showed a loss of 0.0043 with a bias-retune gain of
  +0.0023 (`scratch_gen/amt_arms_v1_unrounded.log`). That loss came from the emulation (amounts such as 1989 look like
  phishing limit-minus amounts), not from October. The -0.0035 in `notes/drift.md` item 5 has the same cause.

## Reproduction from raw files

Clean export of commit `330ad4c` (`git archive`) into an empty directory, raw CSVs only, no caches:
`.\run_all.ps1 -Tag repro -Threads 8`. All 17 steps ran. Every checked output is byte-identical to its reference,
including `repro_dd_final.csv` == `dd_final.csv`
(SHA-256 `8e04353d6f52fe86ab9e44b5dd33d7adec9ba26df36f566d0bcfd68d49d829cd`). Log: `work/logs/run_all_clean_14nov.log`.
Python 3.11.14, polars 2.1.0, numpy 2.4.6, lightgbm 4.7.0, scipy 1.17.1 (`requirements.txt`).

## Commands (cwd = work, `PYTHONPATH=.`)

```
python gen_events.py scratch_gen/gen_v2.parquet --mult 1 --seed 0
python scratch_gen/fid.py
python scratch_gen/arms.py A C1 B1 BR C05 B05
python scratch_gen/eval_arms.py
python scratch_gen/amt_world.py
python scratch_gen/amt_arms.py
python scratch_gen/amt_check.py
```
Seeds: generator 0; model seeds 0 and 1; split and bootstrap draws `default_rng(2026)`; 0.5x subset `default_rng(123)`.
Logs: `scratch_gen/arms.log`, `eval_arms.log`, `amt_arms.log`.
