# TabM test and audit of the phishing / fake_cashback signal (14 Nov 2026, 18:45 to 19:35 IST)

Outcome: no candidate. `subs/dd_final.csv` is unchanged (SHA-256 `8e04353d...29cd`). Nothing uploaded or published.
All numbers are on benchmark N (`notes/gap_reset.md`): fit July, develop on August, refit July + August, score September
with supplied labels. September has been used for development many times; it is not a fresh holdout.

## 1. Full 12-class TabM

No 12-class TabM run existed before (the earlier neural tests were MLP / TabPFN on camouflage rows only).
Implementation: official package `tabm` 0.0.3 (https://github.com/yandex-research/tabm) with `rtdl-num-embeddings` 0.0.12,
torch 2.15.0 + CUDA on the local RTX 4060. Installed in `.venv`; not needed by `run_all.ps1`, so not in `requirements.txt`.
Script: `scratch_gap/tabm_run.py`.

- Inputs: the 190 benchmark-N features. Same fit rows and target as LightGBM (relabel-C, legit rows at 20% with weight 5).
- Preprocessing fitted on fit rows only: 181 numeric columns through a quantile transform to a normal; missing values set
  to 0 after the transform plus a 0/1 indicator for each of the 60 columns with more than 0.1% missing; 9 categorical
  columns as integer codes of label-free vocabularies, one-hot inside TabM.
- Two configurations: `plain` (TabM, 3 blocks x 512, dropout 0.1, k = 32) and `ple` (same backbone with piecewise-linear
  embeddings, 48 bins, d = 16). AdamW, lr 2e-3, weight decay 3e-4, batch 1024, mixed precision.
- Stopping: August weighted log-loss, patience 5. Best epochs: plain 6 and 11, ple 6 and 7 (seeds 0, 1). The July + August
  refit runs for the epoch count found on August. September labels are never read by the training script.
- Selection on August only (`scratch_gap/tabm_select.py`): configuration by August macro-F1 after an August-tuned bias,
  then the blend weight w in {0.2, 0.35, 0.5, 0.65, 0.8} for w * TabM + (1 - w) * LightGBM.

August (development), TabM seeds 0 and 1 averaged:

| model | macro-F1 | log-loss |
|---|---|---|
| LightGBM (2 seeds) | 0.4725 | 0.18007 |
| TabM plain | 0.4664 | 0.17892 |
| TabM ple (chosen) | 0.4691 | 0.17868 |
| blend w = 0.35 (chosen) | 0.4731 | 0.17883 |

September. "After decoder" = bias, archetype forcing, device zones, bait rule. Intervals: paired event bootstrap 5%..95% vs LightGBM.

| model | TabM seeds | before decoder | after decoder | change after decoder | rows decoded differently (after) |
|---|---|---|---|---|---|
| LightGBM | | 0.4815 | 0.5070 | | |
| TabM plain | 0 | 0.4766 | 0.5027 | -0.0043 (-0.0063..-0.0022) | 269 |
| TabM ple | 0 | 0.4794 | 0.5049 | -0.0021 (-0.0042..+0.0002) | 231 |
| blend w = 0.35 | 0 | 0.4817 | 0.5072 | +0.0002 (-0.0008..+0.0015) | 91 |
| TabM ple | 0 + 1 | 0.4777 | 0.5029 | -0.0041 (-0.0059..-0.0022) | 222 |
| blend w = 0.35 | 0 + 1 | 0.4810 | 0.5065 | -0.0005 (-0.0014..+0.0003) | 63 |

Different errors (`scratch_gap/tabm_diff.py`, after decoder):

| model | fraud rows right only in it | of those, LightGBM said legit | fraud rows right only in LightGBM | oracle "either model right" |
|---|---|---|---|---|
| TabM ple, seed 0 | 88 | 30 | 75 | 0.5125 |
| TabM ple, seeds 0 + 1 | 55 | 20 | 105 | 0.5109 |
| blend, seeds 0 + 1 | 21 | 15 | 22 | 0.5080 |

- TabM has slightly better log-loss than LightGBM on August and lower macro-F1 on both months.
- The blend picked on August (+0.0006 there) gives +0.0002 with one TabM seed and -0.0005 with two on September;
  both intervals include zero. No gain.
- TabM does not open a new error group. Its disagreement with LightGBM is mostly the phishing / fake_cashback boundary
  (seed 0: phishing +0.019 F1, fake_cashback -0.022). It recovers 20 to 30 rows LightGBM calls legit and loses more elsewhere.
- What failed: TabM alone is worse; the blend is not better; the errors it fixes are offset by the ones it adds.
- Not tested: TabM tuned beyond these two configurations, other fit-row sampling, the density benchmark D, the test month.

## 2. Audit of "phishing label, fake_cashback look: event AUC .80, precision .56"

Source of the cited numbers: `scratch_x/classwise/e24.py`. It pooled August and September, split events with
GroupKFold(5) across both months (not in time), and defined the look by the model's argmax before any rule.
233 events, 73 of them phishing; at threshold 0.5, 68 events flagged with event precision 0.56.

Audit (`scratch_gap/phfc_audit.py`): same 47 features and classifier, but on the rows the final decode (after zones and the
bait rule) calls fake_cashback, events without a zone device, discovery on August only, check on September.

| | August | September |
|---|---|---|
| rows the final decode calls fake_cashback | 342 (210 fc, 81 phishing, 40 legit-labelled) | 351 (227 fc, 71 phishing, 48 legit-labelled) |
| independent events (connected accounts within the month, no zone device) | 134, of them 52 phishing | 141, of them 45 phishing |
| event AUC | 0.836 (grouped 5-fold inside August) | 0.784 (model fit on August) |

- Threshold search on August (flip whole flagged events to phishing): every threshold from 0.3 to 0.8 lowers August macro-F1
  (-0.0001 to -0.0008). The least bad is 0.8.
- September at the cited threshold 0.5: 60 of 141 events flagged, 34 of them phishing (event precision 0.57); 81 rows
  flipped = 48 phishing, 33 fake_cashback. Coverage: 48 of the 71 phishing rows that sit in fake_cashback.
  - fake_cashback precision / recall / F1: 0.647 / 0.273 / 0.384 -> 0.719 / 0.233 / 0.352.
  - phishing: 0.748 / 0.348 / 0.475 -> 0.736 / 0.370 / 0.493.
  - macro-F1 0.5070 -> 0.5058 (-0.0012, bootstrap -0.0031..+0.0005).
- September at the August-picked threshold 0.8: 32 events, 15 phishing (0.47); macro -0.0016 (-0.0030..-0.0002).
- The signal is event size: `e_n` carries 0.21 of the permutation importance, the next feature 0.03. Flagged events have
  1.5 rows and last 4 minutes; all look events average 2.2 rows and 1.6 hours.

Verdict: the AUC is real and holds in time, but it does not add a usable correction. At precision 0.57 each flipped
row costs fake_cashback (the smaller, lower-F1 class) more than it gives phishing, so macro-F1 falls. Not promoted.

## Commands (cwd = work, `PYTHONPATH=.`)

```
python scratch_gap/nat_data.py
python scratch_gap/tabm_run.py plain 0 ; python scratch_gap/tabm_run.py ple 0
python scratch_gap/tabm_run.py plain 1 ; python scratch_gap/tabm_run.py ple 1
python scratch_gap/tabm_select.py 0,1
python scratch_gap/bench_eval.py N C,TABM,BLEND _tabm
python scratch_gap/tabm_diff.py
python scratch_gap/phfc_audit.py
```
Logs: `scratch_gap/tabm_*.log`, `tabm_select_s0.log`, `tabm_select_s01.log`, `eval_N_plain0.log`, `eval_N_tabm_s0.log`,
`eval_N_tabm_s01.log`, `tabm_diff_s0.log`, `tabm_diff_s01.log`, `phfc_audit.log`.
