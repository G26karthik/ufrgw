# Hybrid rules: LLM overrides on top of the baseline label C_q (pilot rows, September)

Rows: natural 300 (17 fraud), pooled 525 (182 fraud) reweighted per true class to the September distribution (weight = population count / pooled count, legit 1432). Counts are pilot rows, not reweighted. corrected = baseline wrong, hybrid right; damaged = baseline right, hybrid wrong; unchanged-wrong = overridden, wrong before and after.

## Decisions lean pilot (`gpt-6-luna`, the setup of the full run)

Confidence = logged probability of the chosen label. Distribution over the 525 rows: >= 0.5: 493, >= 0.7: 411, >= 0.9: 258, >= 0.95: 181, >= 0.99: 77; minimum 0.250. Logged `confidence` field equals it on 181 rows (mean 0.812 against 0.827).
Disagreements with the baseline: 48 rows; of them D 20, T 16, V 12.

Baseline macro-F1: natural 0.6645 (classes present, unweighted), pooled reweighted 0.5063.

| family | t | natural: overridden | corrected | damaged | unchanged-wrong | macro-F1 hybrid (change) | pooled: overridden | corrected | damaged | unchanged-wrong | reweighted macro-F1 hybrid (change) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 0 | 13 | 0 | 12 | 1 | 0.5592 (-0.1053) | 20 | 0 | 14 | 6 | 0.3675 (-0.1387) |
| D | 0.5 | 11 | 0 | 10 | 1 | 0.5597 (-0.1048) | 17 | 0 | 12 | 5 | 0.3871 (-0.1192) |
| D | 0.7 | 7 | 0 | 6 | 1 | 0.5800 (-0.0845) | 9 | 0 | 7 | 2 | 0.4196 (-0.0867) |
| D | 0.9 | 3 | 0 | 2 | 1 | 0.6107 (-0.0538) | 3 | 0 | 2 | 1 | 0.4736 (-0.0326) |
| D | 0.95 | 2 | 0 | 1 | 1 | 0.6228 (-0.0417) | 2 | 0 | 1 | 1 | 0.4812 (-0.0251) |
| D | 0.99 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| T | 0 | 2 | 0 | 2 | 0 | 0.4978 (-0.1667) | 16 | 1 | 13 | 2 | 0.4273 (-0.0790) |
| T | 0.5 | 2 | 0 | 2 | 0 | 0.4978 (-0.1667) | 11 | 1 | 9 | 1 | 0.4567 (-0.0495) |
| T | 0.7 | 2 | 0 | 2 | 0 | 0.4978 (-0.1667) | 5 | 0 | 4 | 1 | 0.4819 (-0.0244) |
| T | 0.9 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 1 | 0 | 1 | 0 | 0.4989 (-0.0074) |
| T | 0.95 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| T | 0.99 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| V | 0 | 2 | 0 | 2 | 0 | 0.5391 (-0.1254) | 12 | 0 | 12 | 0 | 0.4375 (-0.0688) |
| V | 0.5 | 1 | 0 | 1 | 0 | 0.5809 (-0.0835) | 9 | 0 | 9 | 0 | 0.4539 (-0.0524) |
| V | 0.7 | 1 | 0 | 1 | 0 | 0.5809 (-0.0835) | 8 | 0 | 8 | 0 | 0.4620 (-0.0443) |
| V | 0.9 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 3 | 0 | 3 | 0 | 0.4915 (-0.0147) |
| V | 0.95 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| V | 0.99 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| ALL | 0 | 17 | 0 | 16 | 1 | 0.2949 (-0.3696) | 48 | 1 | 39 | 8 | 0.2504 (-0.2559) |
| ALL | 0.5 | 14 | 0 | 13 | 1 | 0.3268 (-0.3377) | 37 | 1 | 30 | 6 | 0.3016 (-0.2047) |
| ALL | 0.7 | 10 | 0 | 9 | 1 | 0.3447 (-0.3198) | 22 | 0 | 19 | 3 | 0.3557 (-0.1506) |
| ALL | 0.9 | 3 | 0 | 2 | 1 | 0.6107 (-0.0538) | 7 | 0 | 6 | 1 | 0.4516 (-0.0547) |
| ALL | 0.95 | 2 | 0 | 1 | 1 | 0.6228 (-0.0417) | 2 | 0 | 1 | 1 | 0.4812 (-0.0251) |
| ALL | 0.99 | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |

Rows where the LLM is right and the baseline wrong: 1
- TX6cc3cd5c (enriched): true account_takeover, baseline sim_swap, LLM confidence 0.580
Rows where the baseline is right and the LLM wrong: 39; their LLM confidence: >= 0.9 on 6, >= 0.99 on 0, median 0.620

## astra full-context pilot (`gpt-6-astra`), label only (no probabilities exist for it)

Disagreements with the baseline: 14 rows; of them D 0, T 10, V 4.

Baseline macro-F1: natural 0.6645 (classes present, unweighted), pooled reweighted 0.5063.

| family | t | natural: overridden | corrected | damaged | unchanged-wrong | macro-F1 hybrid (change) | pooled: overridden | corrected | damaged | unchanged-wrong | reweighted macro-F1 hybrid (change) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | label only | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 0 | 0 | 0 | 0 | 0.5063 (+0.0000) |
| T | label only | 3 | 0 | 3 | 0 | 0.4145 (-0.2500) | 10 | 0 | 9 | 1 | 0.4479 (-0.0583) |
| V | label only | 0 | 0 | 0 | 0 | 0.6645 (+0.0000) | 4 | 0 | 4 | 0 | 0.4869 (-0.0194) |
| ALL | label only | 3 | 0 | 3 | 0 | 0.4145 (-0.2500) | 14 | 0 | 13 | 1 | 0.4261 (-0.0802) |

Rows where the LLM is right and the baseline wrong: 0
Rows where the baseline is right and the LLM wrong: 13

## Verdict

- decisions: family / threshold with corrected > damaged on BOTH the natural and the pooled view and at least 5 corrected rows: NONE
- astra: family / threshold with corrected > damaged on BOTH the natural and the pooled view and at least 5 corrected rows: NONE

No hybrid rule is supported by the pilot.

Limits: 525 rows (17 fraud rows in the natural sample, 15-20 per fraud class pooled); thresholds are chosen and scored on the same rows, which favours the rules, and they still do not pass; pilot batches were 33% fraud against about 4% in the full run, so LLM confidences in the full run may be shifted; the baseline C_q here is the September label-hidden decode, not the October submission `dd_final.csv`.