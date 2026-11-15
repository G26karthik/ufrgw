# Pilot evaluation

predictions sha256 `454dbbbe772cd8755c5295943898f3cf3e401812f6e9d0f6f97cd5becb56bb6f` (matches seal). Samples: natural 300, enriched 225; distinct ids 525; ids in more than one sample 0

Coverage: PASS (every sample id exactly once, no extras, all labels in vocabulary)

Truth = raw September train label (509,996 rows, 18661 fraud = 3.66%). Baseline C_q on ALL of September: macro-F1 0.5070, fraud-vs-legit P 0.875 R 0.379. Compare with its score on each sample below to see the sampling noise.

## natural sample: 300 rows, 17 fraud, 283 legit, 0 without a valid LLM label

### natural: per class, LLM against baseline C_q, same rows (unweighted)

| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 | LLM recall 95% CI |
|---|---|---|---|---|---|---|---|---|
| legit | 283 | 0.964 | 0.954 | 0.959 | 0.969 | 0.996 | 0.983 | 0.92-0.97 |
| mule_account | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00-0.43 |
| velocity_bot | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00-0.56 |
| layering_chain | 1 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00-0.79 |
| sim_swap | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| phishing_collect | 3 | 0.286 | 0.667 | 0.400 | 0.667 | 0.667 | 0.667 | 0.21-0.94 |
| qr_swap | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 0.500 | 0.667 | 0.00-0.66 |
| refund_abuse | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.00-0.66 |
| fake_cashback | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.21-1.00 |
| loan_app_extortion | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| merchant_collusion | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |

- macro-F1 over classes with support > 0: LLM 0.2949, baseline 0.6645
- macro-F1 over all 12 classes (absent class = 0): LLM 0.1966, baseline 0.4430
- accuracy: LLM 0.9100, baseline 0.9633
- fraud versus legit (any fraud class counts): LLM P 0.350 R 0.412 F1 0.378; baseline P 0.889 R 0.471 F1 0.615

### natural: LLM confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 270 | . | 1 | . | 1 | . | 5 | . | 4 | . | 1 | 1 | . |
| mule_account | 4 | . | . | . | . | . | . | . | 1 | . | . | . | . |
| velocity_bot | 2 | 1 | . | . | . | . | . | . | . | . | . | . | . |
| layering_chain | . | 1 | . | . | . | . | . | . | . | . | . | . | . |
| phishing_collect | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . |
| qr_swap | 2 | . | . | . | . | . | . | . | . | . | . | . | . |
| refund_abuse | 1 | . | . | . | . | . | . | . | . | . | . | 1 | . |
| fake_cashback | . | . | . | . | . | . | . | . | . | 1 | . | . | . |

### natural: baseline confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 282 | . | . | . | . | . | 1 | . | . | . | . | . | . |
| mule_account | 5 | . | . | . | . | . | . | . | . | . | . | . | . |
| velocity_bot | 2 | 1 | . | . | . | . | . | . | . | . | . | . | . |
| layering_chain | . | . | . | 1 | . | . | . | . | . | . | . | . | . |
| phishing_collect | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . |
| qr_swap | 1 | . | . | . | . | . | . | 1 | . | . | . | . | . |
| refund_abuse | . | . | . | . | . | . | . | . | 2 | . | . | . | . |
| fake_cashback | . | . | . | . | . | . | . | . | . | 1 | . | . | . |

- rows where LLM and baseline differ: 17; LLM right and baseline wrong: 0; baseline right and LLM wrong: 16; both wrong, different labels: 1
- legit rows called fraud: LLM 13/283 (95% CI 0.0270-0.0770); baseline 1/283 (95% CI 0.0006-0.0197)
- fraud rows with the exact class: LLM 3/17 (95% CI 0.06-0.41); baseline 7/17 (95% CI 0.22-0.64)

Limits of the natural sample: 300 rows hold 17 fraud rows across 11 fraud classes, so per-class fraud F1 here is 0-or-1 noise and macro-F1 moves by about 0.03-0.06 per single row. It can show: whether the LLM over-flags legit rows (false-positive rate, interval above) and gross failure. It cannot rank LLM against baseline on fraud classes or predict leaderboard macro-F1.

## enriched sample: 225 rows, 165 fraud, 60 legit, 0 without a valid LLM label

### enriched: per class, LLM against baseline C_q, same rows (unweighted)

| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 | LLM recall 95% CI |
|---|---|---|---|---|---|---|---|---|
| legit | 60 | 0.356 | 0.967 | 0.520 | 0.375 | 1.000 | 0.545 | 0.89-0.99 |
| mule_account | 15 | 0.364 | 0.267 | 0.308 | 0.857 | 0.400 | 0.545 | 0.11-0.52 |
| velocity_bot | 15 | 0.444 | 0.267 | 0.333 | 0.875 | 0.467 | 0.609 | 0.11-0.52 |
| layering_chain | 15 | 0.625 | 0.333 | 0.435 | 0.778 | 0.467 | 0.583 | 0.15-0.58 |
| sim_swap | 15 | 0.500 | 0.067 | 0.118 | 0.625 | 0.333 | 0.435 | 0.01-0.30 |
| account_takeover | 15 | 0.750 | 0.200 | 0.316 | 1.000 | 0.267 | 0.421 | 0.07-0.45 |
| phishing_collect | 15 | 0.286 | 0.133 | 0.182 | 0.714 | 0.333 | 0.455 | 0.04-0.38 |
| qr_swap | 15 | 0.500 | 0.067 | 0.118 | 0.800 | 0.267 | 0.400 | 0.01-0.30 |
| refund_abuse | 15 | 0.833 | 0.333 | 0.476 | 1.000 | 0.333 | 0.500 | 0.15-0.58 |
| fake_cashback | 15 | 0.500 | 0.133 | 0.211 | 1.000 | 0.200 | 0.333 | 0.04-0.38 |
| loan_app_extortion | 15 | 0.875 | 0.467 | 0.609 | 1.000 | 0.533 | 0.696 | 0.25-0.70 |
| merchant_collusion | 15 | 1.000 | 0.067 | 0.125 | 1.000 | 0.067 | 0.125 | 0.01-0.30 |

- macro-F1 over classes with support > 0: LLM 0.3124, baseline 0.4706
- macro-F1 over all 12 classes (absent class = 0): LLM 0.3124, baseline 0.4706
- accuracy: LLM 0.4133, baseline 0.5111
- fraud versus legit (any fraud class counts): LLM P 0.968 R 0.364 F1 0.529; baseline P 1.000 R 0.394 F1 0.565

### enriched: LLM confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 58 | . | . | . | . | . | 1 | . | . | . | 1 | . | . |
| mule_account | 9 | 4 | . | 2 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 11 | . | 4 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 9 | 1 | . | 5 | . | . | . | . | . | . | . | . | . |
| sim_swap | 9 | 3 | 1 | . | 1 | 1 | . | . | . | . | . | . | . |
| account_takeover | 8 | 1 | 2 | . | 1 | 3 | . | . | . | . | . | . | . |
| phishing_collect | 11 | . | . | . | . | . | 2 | . | . | 2 | . | . | . |
| qr_swap | 13 | . | . | . | . | . | 1 | 1 | . | . | . | . | . |
| refund_abuse | 9 | . | . | 1 | . | . | . | . | 5 | . | . | . | . |
| fake_cashback | 9 | 1 | . | . | . | . | 3 | . | . | 2 | . | . | . |
| loan_app_extortion | 5 | 1 | 1 | . | . | . | . | . | 1 | . | 7 | . | . |
| merchant_collusion | 12 | . | 1 | . | . | . | . | 1 | . | . | . | 1 | . |

### enriched: baseline confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 60 | . | . | . | . | . | . | . | . | . | . | . | . |
| mule_account | 8 | 6 | . | 1 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 8 | . | 7 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 8 | . | . | 7 | . | . | . | . | . | . | . | . | . |
| sim_swap | 9 | . | . | 1 | 5 | . | . | . | . | . | . | . | . |
| account_takeover | 8 | . | . | . | 3 | 4 | . | . | . | . | . | . | . |
| phishing_collect | 10 | . | . | . | . | . | 5 | . | . | . | . | . | . |
| qr_swap | 11 | . | . | . | . | . | . | 4 | . | . | . | . | . |
| refund_abuse | 10 | . | . | . | . | . | . | . | 5 | . | . | . | . |
| fake_cashback | 9 | 1 | . | . | . | . | 2 | . | . | 3 | . | . | . |
| loan_app_extortion | 7 | . | . | . | . | . | . | . | . | . | 8 | . | . |
| merchant_collusion | 12 | . | 1 | . | . | . | . | 1 | . | . | . | 1 | . |

- rows where LLM and baseline differ: 31; LLM right and baseline wrong: 1; baseline right and LLM wrong: 23; both wrong, different labels: 7
- legit rows called fraud: LLM 2/60 (95% CI 0.0092-0.1136); baseline 0/60 (95% CI 0.0000-0.0602)
- fraud rows with the exact class: LLM 35/165 (95% CI 0.16-0.28); baseline 55/165 (95% CI 0.27-0.41)

### enriched: reweighted to the September class distribution (weight = population count / sample count per true class)

Weights: legit 8189, mule_account 224, velocity_bot 177, layering_chain 145, sim_swap 84, account_takeover 85, phishing_collect 144, qr_swap 157, refund_abuse 72, fake_cashback 55, loan_app_extortion 43, merchant_collusion 56

### enriched: reweighted per class (n = September population)

| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 |
|---|---|---|---|---|---|---|---|
| legit | 491335 | 0.975 | 0.967 | 0.971 | 0.978 | 1.000 | 0.989 |
| mule_account | 3367 | 0.606 | 0.267 | 0.370 | 0.960 | 0.400 | 0.565 |
| velocity_bot | 2659 | 0.667 | 0.267 | 0.381 | 0.957 | 0.467 | 0.627 |
| layering_chain | 2182 | 0.583 | 0.333 | 0.424 | 0.767 | 0.467 | 0.580 |
| sim_swap | 1267 | 0.497 | 0.067 | 0.118 | 0.623 | 0.333 | 0.434 |
| account_takeover | 1280 | 0.752 | 0.200 | 0.316 | 1.000 | 0.267 | 0.421 |
| phishing_collect | 2162 | 0.033 | 0.133 | 0.053 | 0.867 | 0.333 | 0.481 |
| qr_swap | 2352 | 0.737 | 0.067 | 0.122 | 0.918 | 0.267 | 0.413 |
| refund_abuse | 1075 | 0.893 | 0.333 | 0.485 | 1.000 | 0.333 | 0.500 |
| fake_cashback | 831 | 0.278 | 0.133 | 0.180 | 1.000 | 0.200 | 0.333 |
| loan_app_extortion | 645 | 0.035 | 0.467 | 0.066 | 1.000 | 0.533 | 0.696 |
| merchant_collusion | 841 | 1.000 | 0.067 | 0.125 | 1.000 | 0.067 | 0.125 |

- macro-F1 over classes with support > 0: LLM 0.3009, baseline 0.5138
- macro-F1 over all 12 classes (absent class = 0): LLM 0.3009, baseline 0.5138
- accuracy: LLM 0.9390, baseline 0.9764
- fraud versus legit (any fraud class counts): LLM P 0.277 R 0.337 F1 0.304; baseline P 1.000 R 0.401 F1 0.573

### enriched: LLM confusion, reweighted counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 474957 | . | . | . | . | . | 8189 | . | . | . | 8189 | . | . |
| mule_account | 2020 | 898 | . | 449 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 1950 | . | 709 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 1309 | 145 | . | 727 | . | . | . | . | . | . | . | . | . |
| sim_swap | 760 | 253 | 84 | . | 84 | 84 | . | . | . | . | . | . | . |
| account_takeover | 683 | 85 | 171 | . | 85 | 256 | . | . | . | . | . | . | . |
| phishing_collect | 1585 | . | . | . | . | . | 288 | . | . | 288 | . | . | . |
| qr_swap | 2038 | . | . | . | . | . | 157 | 157 | . | . | . | . | . |
| refund_abuse | 645 | . | . | 72 | . | . | . | . | 358 | . | . | . | . |
| fake_cashback | 499 | 55 | . | . | . | . | 166 | . | . | 111 | . | . | . |
| loan_app_extortion | 215 | 43 | 43 | . | . | . | . | . | 43 | . | 301 | . | . |
| merchant_collusion | 673 | . | 56 | . | . | . | . | 56 | . | . | . | 56 | . |

### enriched: baseline confusion, reweighted counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 491335 | . | . | . | . | . | . | . | . | . | . | . | . |
| mule_account | 1796 | 1347 | . | 224 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 1418 | . | 1241 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 1164 | . | . | 1018 | . | . | . | . | . | . | . | . | . |
| sim_swap | 760 | . | . | 84 | 422 | . | . | . | . | . | . | . | . |
| account_takeover | 683 | . | . | . | 256 | 341 | . | . | . | . | . | . | . |
| phishing_collect | 1441 | . | . | . | . | . | 721 | . | . | . | . | . | . |
| qr_swap | 1725 | . | . | . | . | . | . | 627 | . | . | . | . | . |
| refund_abuse | 717 | . | . | . | . | . | . | . | 358 | . | . | . | . |
| fake_cashback | 499 | 55 | . | . | . | . | 111 | . | . | 166 | . | . | . |
| loan_app_extortion | 301 | . | . | . | . | . | . | . | . | . | 344 | . | . |
| merchant_collusion | 673 | . | 56 | . | . | . | . | 56 | . | . | . | 56 | . |

Limits of this sample. The unweighted macro-F1 above is NOT representative: fraud is 165/225 of the sample but 3.66% of September. Recall per class rests on 15-15 rows (CI column above). Precision after reweighting rests on 60 legit rows, each standing for 8189 rows: one legit row called fraud adds 8189 false positives, more than any fraud class has rows (3367 at most). With 2/60 legit errors the 95% interval for the false-positive rate is 0.0092-0.1136, that is 4515-55835 false positives against 18661 fraud rows. So this sample estimates recall only; its reweighted precision and macro-F1 hang on 2 legit error(s).

## pooled (natural + enriched, post-stratified by true class) sample: 525 rows, 182 fraud, 343 legit, 0 without a valid LLM label

### pooled (natural + enriched, post-stratified by true class): per class, LLM against baseline C_q, same rows (unweighted)

| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 | LLM recall 95% CI |
|---|---|---|---|---|---|---|---|---|
| legit | 343 | 0.740 | 0.956 | 0.835 | 0.758 | 0.997 | 0.861 | 0.93-0.97 |
| mule_account | 20 | 0.308 | 0.200 | 0.242 | 0.750 | 0.300 | 0.429 | 0.08-0.42 |
| velocity_bot | 18 | 0.400 | 0.222 | 0.286 | 0.875 | 0.389 | 0.538 | 0.09-0.45 |
| layering_chain | 16 | 0.625 | 0.312 | 0.417 | 0.800 | 0.500 | 0.615 | 0.14-0.56 |
| sim_swap | 15 | 0.333 | 0.067 | 0.111 | 0.625 | 0.333 | 0.435 | 0.01-0.30 |
| account_takeover | 15 | 0.750 | 0.200 | 0.316 | 1.000 | 0.267 | 0.421 | 0.07-0.45 |
| phishing_collect | 18 | 0.286 | 0.222 | 0.250 | 0.700 | 0.389 | 0.500 | 0.09-0.45 |
| qr_swap | 17 | 0.500 | 0.059 | 0.105 | 0.833 | 0.294 | 0.435 | 0.01-0.27 |
| refund_abuse | 17 | 0.455 | 0.294 | 0.357 | 1.000 | 0.412 | 0.583 | 0.13-0.53 |
| fake_cashback | 16 | 0.600 | 0.188 | 0.286 | 1.000 | 0.250 | 0.400 | 0.07-0.43 |
| loan_app_extortion | 15 | 0.778 | 0.467 | 0.583 | 1.000 | 0.533 | 0.696 | 0.25-0.70 |
| merchant_collusion | 15 | 0.333 | 0.067 | 0.111 | 1.000 | 0.067 | 0.125 | 0.01-0.30 |

- macro-F1 over classes with support > 0: LLM 0.3249, baseline 0.5032
- macro-F1 over all 12 classes (absent class = 0): LLM 0.3249, baseline 0.5032
- accuracy: LLM 0.6971, baseline 0.7695
- fraud versus legit (any fraud class counts): LLM P 0.817 R 0.368 F1 0.508; baseline P 0.986 R 0.401 F1 0.570

### pooled (natural + enriched, post-stratified by true class): LLM confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 328 | . | 1 | . | 1 | . | 6 | . | 4 | . | 2 | 1 | . |
| mule_account | 13 | 4 | . | 2 | . | . | . | . | 1 | . | . | . | . |
| velocity_bot | 13 | 1 | 4 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 9 | 2 | . | 5 | . | . | . | . | . | . | . | . | . |
| sim_swap | 9 | 3 | 1 | . | 1 | 1 | . | . | . | . | . | . | . |
| account_takeover | 8 | 1 | 2 | . | 1 | 3 | . | . | . | . | . | . | . |
| phishing_collect | 12 | . | . | . | . | . | 4 | . | . | 2 | . | . | . |
| qr_swap | 15 | . | . | . | . | . | 1 | 1 | . | . | . | . | . |
| refund_abuse | 10 | . | . | 1 | . | . | . | . | 5 | . | . | 1 | . |
| fake_cashback | 9 | 1 | . | . | . | . | 3 | . | . | 3 | . | . | . |
| loan_app_extortion | 5 | 1 | 1 | . | . | . | . | . | 1 | . | 7 | . | . |
| merchant_collusion | 12 | . | 1 | . | . | . | . | 1 | . | . | . | 1 | . |

### pooled (natural + enriched, post-stratified by true class): baseline confusion counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 342 | . | . | . | . | . | 1 | . | . | . | . | . | . |
| mule_account | 13 | 6 | . | 1 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 10 | 1 | 7 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 8 | . | . | 8 | . | . | . | . | . | . | . | . | . |
| sim_swap | 9 | . | . | 1 | 5 | . | . | . | . | . | . | . | . |
| account_takeover | 8 | . | . | . | 3 | 4 | . | . | . | . | . | . | . |
| phishing_collect | 11 | . | . | . | . | . | 7 | . | . | . | . | . | . |
| qr_swap | 12 | . | . | . | . | . | . | 5 | . | . | . | . | . |
| refund_abuse | 10 | . | . | . | . | . | . | . | 7 | . | . | . | . |
| fake_cashback | 9 | 1 | . | . | . | . | 2 | . | . | 4 | . | . | . |
| loan_app_extortion | 7 | . | . | . | . | . | . | . | . | . | 8 | . | . |
| merchant_collusion | 12 | . | 1 | . | . | . | . | 1 | . | . | . | 1 | . |

- rows where LLM and baseline differ: 48; LLM right and baseline wrong: 1; baseline right and LLM wrong: 39; both wrong, different labels: 8
- legit rows called fraud: LLM 15/343 (95% CI 0.0267-0.0709); baseline 1/343 (95% CI 0.0005-0.0163)
- fraud rows with the exact class: LLM 38/182 (95% CI 0.16-0.27); baseline 62/182 (95% CI 0.28-0.41)

### pooled (natural + enriched, post-stratified by true class): reweighted to the September class distribution (weight = population count / sample count per true class)

Weights: legit 1432, mule_account 168, velocity_bot 148, layering_chain 136, sim_swap 84, account_takeover 85, phishing_collect 120, qr_swap 138, refund_abuse 63, fake_cashback 52, loan_app_extortion 43, merchant_collusion 56

### pooled (natural + enriched, post-stratified by true class): reweighted per class (n = September population)

| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 |
|---|---|---|---|---|---|---|---|
| legit | 491335 | 0.975 | 0.956 | 0.965 | 0.978 | 0.997 | 0.987 |
| mule_account | 3367 | 0.441 | 0.200 | 0.275 | 0.835 | 0.300 | 0.441 |
| velocity_bot | 2659 | 0.249 | 0.222 | 0.235 | 0.949 | 0.389 | 0.552 |
| layering_chain | 2182 | 0.630 | 0.312 | 0.418 | 0.812 | 0.500 | 0.619 |
| sim_swap | 1267 | 0.053 | 0.067 | 0.059 | 0.623 | 0.333 | 0.434 |
| account_takeover | 1280 | 0.752 | 0.200 | 0.316 | 1.000 | 0.267 | 0.421 |
| phishing_collect | 2162 | 0.051 | 0.222 | 0.083 | 0.354 | 0.389 | 0.370 |
| qr_swap | 2352 | 0.712 | 0.059 | 0.109 | 0.925 | 0.294 | 0.446 |
| refund_abuse | 1075 | 0.051 | 0.294 | 0.086 | 1.000 | 0.412 | 0.583 |
| fake_cashback | 831 | 0.393 | 0.188 | 0.254 | 1.000 | 0.250 | 0.400 |
| loan_app_extortion | 645 | 0.095 | 0.467 | 0.158 | 1.000 | 0.533 | 0.696 |
| merchant_collusion | 841 | 0.036 | 0.067 | 0.047 | 1.000 | 0.067 | 0.125 |

- macro-F1 over classes with support > 0: LLM 0.2504, baseline 0.5063
- macro-F1 over all 12 classes (absent class = 0): LLM 0.2504, baseline 0.5063
- accuracy: LLM 0.9286, baseline 0.9733
- fraud versus legit (any fraud class counts): LLM P 0.229 R 0.342 F1 0.274; baseline P 0.838 R 0.397 F1 0.539

### pooled (natural + enriched, post-stratified by true class): LLM confusion, reweighted counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 469848 | . | 1432 | . | 1432 | . | 8595 | . | 5730 | . | 2865 | 1432 | . |
| mule_account | 2189 | 673 | . | 337 | . | . | . | . | 168 | . | . | . | . |
| velocity_bot | 1920 | 148 | 591 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 1227 | 273 | . | 682 | . | . | . | . | . | . | . | . | . |
| sim_swap | 760 | 253 | 84 | . | 84 | 84 | . | . | . | . | . | . | . |
| account_takeover | 683 | 85 | 171 | . | 85 | 256 | . | . | . | . | . | . | . |
| phishing_collect | 1441 | . | . | . | . | . | 480 | . | . | 240 | . | . | . |
| qr_swap | 2075 | . | . | . | . | . | 138 | 138 | . | . | . | . | . |
| refund_abuse | 632 | . | . | 63 | . | . | . | . | 316 | . | . | 63 | . |
| fake_cashback | 467 | 52 | . | . | . | . | 156 | . | . | 156 | . | . | . |
| loan_app_extortion | 215 | 43 | 43 | . | . | . | . | . | 43 | . | 301 | . | . |
| merchant_collusion | 673 | . | 56 | . | . | . | . | 56 | . | . | . | 56 | . |

### pooled (natural + enriched, post-stratified by true class): baseline confusion, reweighted counts (rows = true, columns = predicted)

| true \ pred | legit | mule_a | veloci | layeri | sim_sw | accoun | phishi | qr_swa | refund | fake_c | loan_a | mercha | INVAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 489903 | . | . | . | . | . | 1432 | . | . | . | . | . | . |
| mule_account | 2189 | 1010 | . | 168 | . | . | . | . | . | . | . | . | . |
| velocity_bot | 1477 | 148 | 1034 | . | . | . | . | . | . | . | . | . | . |
| layering_chain | 1091 | . | . | 1091 | . | . | . | . | . | . | . | . | . |
| sim_swap | 760 | . | . | 84 | 422 | . | . | . | . | . | . | . | . |
| account_takeover | 683 | . | . | . | 256 | 341 | . | . | . | . | . | . | . |
| phishing_collect | 1321 | . | . | . | . | . | 841 | . | . | . | . | . | . |
| qr_swap | 1660 | . | . | . | . | . | . | 692 | . | . | . | . | . |
| refund_abuse | 632 | . | . | . | . | . | . | . | 443 | . | . | . | . |
| fake_cashback | 467 | 52 | . | . | . | . | 104 | . | . | 208 | . | . | . |
| loan_app_extortion | 301 | . | . | . | . | . | . | . | . | . | 344 | . | . |
| merchant_collusion | 673 | . | 56 | . | . | . | . | 56 | . | . | . | 56 | . |

Limits of this sample. The unweighted macro-F1 above is NOT representative: fraud is 182/525 of the sample but 3.66% of September. Recall per class rests on 15-20 rows (CI column above). Precision after reweighting rests on 343 legit rows, each standing for 1432 rows: one legit row called fraud adds 1432 false positives. With 15/343 legit errors the 95% interval for the false-positive rate is 0.0267-0.0709, that is 13108-34832 false positives against 18661 fraud rows. The reweighted macro-F1 is an estimate with wide error: per-class recall from 15-20 rows, precision from this false-positive interval.

Rows where LLM and baseline differ: `check/pilot_predictions_decisions_diff_rows.csv` (48 rows; true / llm / baseline: legit/phishing_collect/legit x5, legit/refund_abuse/legit x4, qr_swap/legit/qr_swap x4, velocity_bot/legit/velocity_bot x3, layering_chain/mule_account/layering_chain x2, legit/loan_app_extortion/legit x2, phishing_collect/fake_cashback/phishing_collect x2, sim_swap/mule_account/sim_swap x2, legit/velocity_bot/legit x1, legit/merchant_collusion/legit x1, refund_abuse/legit/refund_abuse x1, refund_abuse/merchant_collusion/refund_abuse x1, mule_account/refund_abuse/legit x1, legit/sim_swap/legit x1, fake_cashback/phishing_collect/fake_cashback x1)