# flow module (mule_account, layering_chain)

Module: `work/feats/flow.py` -> `feats_cache/flow.parquet` (20 cols, ~35 s build, labels never read).

## Generator signatures (measured on train)

### mule_account
- Event = one P2P "mule" receiver gets a burst of inbound `PAY` rows: 5-24 rows (median 13, ~184 events Jul-Sep, ~61/month),
  each from a distinct sender, `new_payee=1` (89%), amount multiple of 10 (99%), Rs 500-5000 (median ~2600),
  spread over ~3-18 h. Sum inflow median Rs 38k.
- Then the mule forwards 1-4 integer (not round-10) amounts to other P2P accounts, 30-155 min after the last inflow,
  each 10-80% of the inflow sum.
- Normal P2P receivers get ~2 inbound rows in the whole 4 months -> `rin_n12h >= 6` is almost pure fraud.
- Lookalike: velocity_bot sometimes targets a P2P receiver (17 bursts) - same sender repeats; `pair_n12h` separates.
- Label noise: ~13% of burst rows labelled `legit` (fraud->legit). ~60% of mule labels are NOT structural:
  ~1500/3400 Sep mule labels are on merchant rows (popular merchants, look exactly like legit) and ~950 lone P2P rows.
  These look like random legit rows relabelled; nothing structural can catch them -> Sep mule F1 is capped (~0.40).

### layering_chain
- Event = chain A->B->C->... of P2P `PAY`, 3-6 rows (2-5 hops), integer amounts (100%, only ~10% multiple of 10),
  each hop = 0.966-0.994 x previous hop (median 0.980), gap 2 min - 5.6 h (median 17 min, no gap > 6 h anywhere),
  amounts mostly Rs 40k-170k; sender's balance_before only slightly above amount.
- Ratio-in-[0.95,1) pairs within 6 h are ~all layering (legit P2P forwarding is very rare).
- ~45% of P2P layering labels sit in such chains; the rest (and all ~2550 merchant layering labels) look like random
  legit rows (noise, unreachable).
- EVENT-LEVEL TYPE NOISE: ~15% of chains (85 of ~580 in train) are labelled mostly `mule_account`; rows are
  statistically identical to layering chains (amount, hop ratio, gaps, hour, sim, device...). Model memorises these and
  in Sep predicts `mule` with p~0.99 on 117 of 630 layering chain rows. Fix belongs in the trainer: relabel (see below).

## Drift (Sep vs Oct)
- Oct (test) has ~half the rows per day (3.1k vs 6.2k P2P/day; 72k vs 119k active senders) but fraud events are
  full-size (mule bursts mean 11 rows/day in both; chain hop ratio median 0.979 vs 0.980; gap median 18.5 vs 17.3 min).
- Hence legit-side density features drop a bit in Oct (rin_n12h>1: 2.7% -> 1.9% of P2P rows; sin_n24h nonzero
  2.9% -> 1.7%). Direction is safe (legit looks less bursty). chain_cspan / hop gaps have a slightly longer tail in
  Oct (mean 254 vs 189 min, median equal). No feature flagged as unsafe.

## Features
rin_n3h, rin_n12h (P2P inbound to receiver, symmetric windows), rin_np_frac12h, rin_r10_frac12h, pair_n12h,
rout_n6h, rout_ratio_in12 (receiver forwards within 6 h / inflow), sin_n24h, sin_amt_ratio24, sin_r10_frac24
(inflow to sender in prior 24 h), hop_nr/hop_ng/hop_pr/hop_pg (best next/prev hop with ratio [0.9,1), <=6 h,
middle account not a burst receiver), chain_size/cpos/cspan/csin/crin (connected components of hop edges).
All NaN on merchant rows.

## Eval (quick_eval.py flow --threads 2, fit Jul-Aug -> Sep, noisy Sep labels)
| run | macro argmax | macro tuned | mule | layering |
|---|---|---|---|---|
| baseline (no module) | 0.3577 | 0.3773 | 0.222 | 0.353 |
| flow v1 (no edge filter, no pair_n12h) | 0.3955 | 0.4140 | 0.370 | 0.443 |
| flow v2 = final module | 0.3955 | 0.4137 | 0.374 | 0.477 |
| v2, drop chain_cspan/cpos/crin | 0.3957 | 0.4166 | 0.369 | 0.471 (not kept, no clear gain) |
| v2 + train relabel A (chain mule->layering) | 0.3978 | 0.4128 | 0.367 | 0.475 |
| v2 + train relabel C (A + chain legit->layering + burst legit->mule) | 0.3990 | 0.4187 | 0.372 | 0.501 |
"clean-view" (Sep labels cleaned by the same structural rules, closer to clean test labels): none mule .411 / lay .545,
C mule .418 / lay .584. Script: scratch_flow/qe_relabel.py MODE.

## Recommendation for the final trainer
Relabel TRAIN rows (mode C):
- `flow__chain_size >= 3` and label in {mule_account, legit} -> layering_chain
- `rin_n12h >= 6 & rin_np_frac12h >= 0.7 & rin_r10_frac12h >= 0.9 & pair_n12h == 1` and label legit -> mule_account
Optionally post-process test: chain_size>=3 -> layering, burst rule -> mule (structural, high precision).
Merchant-row and lone-P2P mule/layering labels are unreachable noise; do not chase them.
