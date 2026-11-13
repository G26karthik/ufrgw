# Research notes: techniques for UPI Fraud Rings (scout, 2026-11-13)

Scope: web + reasoning, light compute only (one drift check in `work/scratch_research/cents.py`).
Gains are estimates from the literature and from the structure of this task. They are not measured.
Baseline (quick_eval): macro argmax 0.3577 / bias-tuned 0.3773.

## Public material on this competition

- Searches for "UPI Fraud Rings", "ML Gradient Wars" and hack4hack + kaggle found no public kernels, discussions or dataset copies. Nothing to report or avoid.
- Closest public analogues: IBM AMLworld synthetic AML generator (arXiv 2306.16424): an agent-based generator with fan-in, fan-out, scatter-gather, gather-scatter, cycle, stack and bipartite patterns. The generator here is probably built the same way, with typed event templates injected into a legit background.

## Quick facts checked

- `cents` (amount has a non-zero paisa part) in train: legit 0.74; most fraud types 0.43-0.51; qr_swap 0.65; refund_abuse 0.72. Round-500 amounts: ATO, sim_swap, qr_swap and refund_abuse are each about 0.11, against legit 0.003.
- Oct overall cents fraction = 0.732, against a train mix of about 0.73. The marginal is stable, and the implied fraud share is about 3-4%, as in train. The artefact probably still holds in Oct. Verify per predicted class after the first submission.
- `train.parquet` holds 147 rows with month==10 (boundary rows; 18 of them are fraud). quick_eval's `month` filter ignores them. They are harmless, but do not treat month==10 as test.

## Ranked techniques

### 1. Event IDs from linked rows, plus per-event aggregate features (expected +0.05 to +0.12 macro)

Evidence: the IBM Graph Feature Preprocessor (Blanusa et al., ICAIF'24, arXiv 2402.08593) adds fan-in/fan-out degrees, scatter-gather and temporal-cycle counts, and per-vertex amount/time statistics (sum, mean, min, max, median, var, skew, kurt) in 1-day windows; scatter-gather uses 6 h. That lifts LightGBM minority F1 from 21 to 63 (HI-Small) and from 2 to 21 (LI-Small) on AMLworld. On a generator-made dataset this is the dominant lever.
The IEEE-CIS 1st-place "UID magic" (Deotte et al.) is the same idea: rebuild the hidden entity, then give the GBDT aggregates over that entity, never the raw ID.

How:
- Build edges among rows (the txns), using one key at a time. Sort by (key, ts). Link consecutive rows when the gap is under W. Do this per key type: sender (W = 30 min), device (30 min), P2P receiver (60 min), merchant receiver (10 min, and only when the merchant's own rate in that hour is low).
- Add a money-forward edge: incoming row r into account A at time t, linked to A's outgoing row s with t < ts_s <= t + 30 min and amount_s/amount_r in [0.7, 1.0]. This is the layering and mule onward-transfer signature. Do it with polars `join_asof` (`by` = account, `strategy="forward"`, `tolerance=1800`).
- Run connected components with `scipy.sparse.csgraph.connected_components` on a coo matrix of (row_i, row_j). Hubs are the danger: big merchants and busy legit accounts chain everything together. Two guards:
  - Keep one burst ID per key type (`sender_burst`, `device_burst`, `rcv_burst`, `flow_cc`) and do not merge them all.
  - Cap component size: if a component is larger than about 200 rows, fall back to the row's own burst.
- Per-component aggregates, attached to each row:
  - size; duration (max_ts - min_ts); rows per minute
  - n distinct senders, receivers and devices; senders per receiver
  - amount mean, CV and n_unique; fraction round-100/500; fraction just under a round limit (e.g. `amount % 1000 >= 900`)
  - fraction COLLECT, QR, REFUND and P2P; fraction new_payee
  - min sim_age_hours, min device_age_days; fraction away_from_home
  - fraction of rows at night (00-05) and in the evening (20-24)
  - max chain depth: longest path in the flow graph, approximated by iterating the as-of join 5 times
  - for flow chains, monotone-decreasing amount ratio (layering "a little less each hop")
  - fraction of rows in the component that are tiny (amount <= 10), for fake_cashback ("send Rs 1-10, then ask big")
  - reverse-direction pair exists (B paid A within 7 days): fake_cashback and merchant_collusion
- Use symmetric windows (past and future), as the rules allow. Normalise counts by a rate (per minute), not by absolute volume.

### 2. Event-level smoothing of predictions (expected +0.02 to +0.05)

The statement says every row in a mule burst, including the onward transfers, carries the same label. The generator labels events, not rows.
- After the model, average log-probs over the rows in the same tight event component (the component from item 1, size <= 50): `P_row' = a*P_row + (1-a)*mean_component(P)`. Tune `a` on Sep.
- This is the "Correct and Smooth" idea (Huang et al., ICLR 2021, arXiv 2010.13993): a simple base model plus label propagation over the graph matches or beats GNNs at a fraction of the cost. The "correct" step needs known labels on the graph. Train-month labels exist only for train rows, and Oct components are almost all-test, so use only the "smooth" step.
- Implementation: `df.with_columns(pl.col(p).mean().over("evt_id"))` for each of the 12 prob columns. It is cheap.

### 3. Event-consistent label cleaning for training (expected +0.02 to +0.04, also a cleaner validation)

Train labels are noisy: unreported fraud is labelled legit, and some fraud carries the wrong type. Test labels are clean. Each event has one true type.
- Majority-vote relabel: inside each tight event component (train rows), if at least 60% of the non-legit rows share type X and there are at least 3 of them, relabel the component's other non-legit rows to X. Relabel its legit rows to X only when they also pass the event signature: same key, inside the burst duration, and not a hub. This uses labels in the training step only and never inside `build()`, so it does not break the feature contract.
- Confident learning (Northcutt et al., JAIR 2021, arXiv 1911.00068; package `cleanlab`): fit 3-fold OOF LightGBM on Jul-Sep. Flag rows where p(given label) is below that class's mean self-confidence and another class has higher p. Then either:
  - drop them (`prune_by_noise_rate`), or
  - for legit rows with p(fraud type) > 0.5 and component support, relabel them.
  On GBDT, dropping the flagged rows usually gains more than relabelling. cleanlab is not needed: the `find_label_issues` logic is about 20 lines of numpy on the OOF matrix.
- Cheaper alternatives:
  - Weight legit rows down when they sit inside a component that has fraud rows.
  - Soft labels: LightGBM has no soft multiclass target, so instead duplicate the row with weights (0.7 given, 0.3 component-majority).
- Validation caveat: Sep labels are noisy too, so quick_eval under-reports. Score a second time on "cleaned Sep" (majority-vote relabelled). That is a closer proxy for the clean test.

### 4. Macro-F1 decision rule and prior shift (expected +0.01 to +0.03 over argmax; tuned bias already gives about +0.02)

- Per-class additive log-bias (already in quick_eval) is the right plug-in rule. For F1, the optimal threshold depends on base rate and achievable F1: with calibrated probabilities, the optimal binary threshold is F1*/2 (Lipton et al. 2014, arXiv 1402.1892). For rare, hard classes (merchant_collusion, fake_cashback) that pushes toward predicting them more often. Post-hoc logit adjustment (Menon et al., ICLR 2021) is the same per-class bias family.
- The bias is fit on noisy Sep labels while test is clean and in a different month. Make it robust:
  - Fit biases on OOF predictions over all three months, not on Sep only.
  - Shrink them toward 0 by about 30%.
  - Use the coarser grid already in place, and avoid a fine grid that overfits.
- Prior shift to Oct: run EM prior re-estimation (Saerens, Latinne & Decaestecker 2002) on the Oct predicted probabilities. Loop `pi_new = mean(P * pi_new/pi_train normalised)` about 20 times, then apply `log(pi_new/pi_train)` as an extra bias. It matters if "fraud rings adapt" or festival legit volume changes class shares. Run it at the event level, after smoothing.
- Class weighting: the current setup (legit downsampled 5x, weight 5) is fine. With bias tuning, extra inverse-frequency weights add little (usually +/-0.005). Skip it.

### 5. Temporal shift checks (protects the private LB; expected +0.00 to +0.03 robustness)

- Adversarial validation (as used by RecSys'23 GBDT solutions, arXiv 2310.03778): train LightGBM on Sep-train vs Oct-test rows using the final feature set (base plus graph modules). Expect AUC near 0.5. Take the top drifting features, then rank-normalise them inside the month or drop them; check that quick_eval does not lose. About 2 minutes of compute.
  - Likely offenders: amount and balance_before (festival spend), login_to_txn_sec and pin_retries (new app version), and anything counting per-key volume.
  - Remedies:
    - Use the within-month percentile rank of amount by merchant_category, e.g. `pl.col("amount").rank().over(["month", "merchant_category"]) / pl.len().over(...)`.
    - Use rates instead of counts.
- ip_risk_score: within-month rank might work, but legit and fraud moved differently, so the ranking itself may change. Keep it dropped unless adversarial validation plus a Sep-only check proves the rank is stable.
- Final fit: train on Jul+Aug+Sep, with the same number of rounds found on Jul-Aug to Sep times about 1.15.
- Pseudo-labelling on Oct, one round only and at the event level:
  - Take Oct events whose mean max-prob is above 0.9 and that have at least 3 rows.
  - Add them with weight 0.5 and refit.
  - Gain is usually small (+0.005 to +0.015) and helps mostly for "adapted" rings. Weak filtering amplifies errors, so keep the threshold strict.
- Seed or model bagging: 3 LightGBM seeds, plus CatBoost if time allows. Average the probabilities before bias tuning. Expected +0.005 to +0.01, with lower variance on the private 70%.

### 6. GNNs: skip

Multi-GNN (Egressy et al., AAAI'24) and GFP report that GNNs gain little over GBDT plus good subgraph features on AMLworld. GFP + LightGBM is competitive or better and far cheaper. With no GPU, 1.5 GB per agent and 48 h, a GNN does not pay. Engineered event features (item 1) plus smoothing (item 2) get most of the graph signal.

### 7. "e-means" / Smartigan (arXiv 2604.21798): not relevant here

- Verified: the paper exists. Authors: Francois Clement and Stefan Steinerberger, submitted 2026-04-23, cs.LG. The authors call the method "Smartigan". It is Hartigan's point-move k-means with a relaxed acceptance factor `3/2 - (1/2)*(iter/N_iter)`: a point moves when its best-cluster cost is at most that factor times its removal cost, so the factor anneals from 1.5 down to 1. They report 2-5% lower SSE than Hartigan, more at high d and k. No official code; it is a 3-line change to the Lederman-Group Hartigan code. I found no "e-means" package. The user's name for it is unconfirmed, and Smartigan is the closest match.
- Why it does not fit this problem:
  - Events here are defined by shared entities plus time adjacency, not by Euclidean proximity.
  - k is unknown and huge (thousands of events), and legit background swamps any SSE objective.
  - Connected components (item 1), or DBSCAN on time within a key, recover events exactly and deterministically.
- A k-means-style algorithm could only make an unsupervised "account behaviour profile" cluster ID. For example, cluster accounts on (txn rate, P2P share, night share, mean amount) and use the ID as a categorical. If someone tries that, use Smartigan rather than Lloyd's, per the user's preference. Expected gain is about 0; GBDT already sees those aggregates directly. Not recommended for the 48 h budget.

## Suggested order for the team

1. Event or burst components plus aggregates (item 1). Split by fraud-type signature: mule (fan-in to P2P, then onward), layering (flow chains), velocity (device burst to a merchant), collusion (merchant <-> few accounts, round amounts, reverse edges).
2. Majority-vote label cleaning on train plus a cleaned-Sep score (item 3).
3. Smoothing of event predictions (item 2), then bias tuning on all-month OOF with EM prior shift (item 4).
4. Adversarial-validation check of the final feature set, then the final fit on Jul-Sep, seed bagging, and optional one-round event pseudo-labels (item 5).

## Sources

- Graph Feature Preprocessor, IBM, ICAIF'24: https://arxiv.org/abs/2402.08593
- AMLworld synthetic AML generator: https://arxiv.org/abs/2306.16424
- Correct & Smooth: https://arxiv.org/abs/2010.13993
- Confident Learning: https://arxiv.org/abs/1911.00068 ; cleanlab tabular: https://cleanlab.ai/blog/label-errors-tabular-datasets
- Thresholding for F1: https://arxiv.org/abs/1402.1892
- Adversarial validation + GBDT: https://arxiv.org/abs/2310.03778
- Smartigan / Hartigan variant: https://arxiv.org/abs/2604.21798
- IEEE-CIS UID approach summary: https://towardsdatascience.com/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f/
