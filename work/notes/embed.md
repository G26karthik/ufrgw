# embed: graph embeddings and sender-sequence features on camouflage rows (embed agent, 2026-11-13)

Scripts are in `work/scratch_embed/`:
- `e0_gl.py` builds `gl.parquet`, the G vs L target frame.
- `e1_seq.py` builds `seq.parquet` (sequence features).
- `e2_emb.py` builds `emb.parquet` (embeddings).
- `ev.py` and `ev_cls.py` are the evaluators.

**Verdict: null. No feature module was written.** The best out-of-sample G vs L AUC is **0.5012**, against a bar of 0.56.

## Setup
- **G**: tun_best OOF rows with p_legit > 0.9 and a cleaned label (clean_labels.clean) that is fraud.
- **L**: the same p_legit > 0.9 band with a cleaned label of legit.
- Row counts: Aug G 12,274 / L 505,808; Sep G 11,858 / L 489,842.
- **Model**: XGBoost on cuda, fit on Aug. Early stopping uses an inner 20 % Aug split, so Sep is never touched before scoring. AUC is reported on Sep.
- **Positive control (`ctrl/`)**: the same 160 features and pipeline, on core fraud (p_legit < 0.5) vs L, give Sep AUC **0.9896**. The top-1 % G rate is 0.97. The features and the pipeline work, so the null on G is real.

## Results (fit Aug, AUC on Sep)
| feature set | n feats | XGB AUC | best univariate abs(AUC - 0.5) |
|---|---|---|---|
| (c) sender sequence / continuity | 40 | 0.4980 | 0.0053 (bal_err_next) |
| (a)+(b) embeddings | 120 | 0.4976 | 0.0053 (s_sd13) |
| all | 160 | **0.5012** | - |

- Top-1 % G rate: 0.021-0.025, against a base rate of 0.0236.
- Per-class models (`ev_cls.py`, G_k vs a 10 % sample of L, depth 3): Sep AUC lies between 0.4865 and 0.5079 for all 11 classes. The inner-Aug early stopping picks 0-13 rounds for most classes, which means nothing to learn.

## What was built (all rows of train+test, no labels)
- **(a) PPMI + TruncatedSVD, k=16.**
  - Inputs: the sender x receiver and sender x device count matrices, with cds 0.75 smoothing. This is the node2vec window-1 analogue.
  - Per row: sender, receiver and device vectors, plus edge plausibility (the low-rank PPMI reconstruction u_s·v_r and u_s·v_d).
- **(b) Multi-hop spectral (NetMF-style).**
  - Graph: an undirected log-weighted graph over 802k nodes (accounts, merchants, devices; edges sender-receiver, sender-device, device-receiver).
  - Method: eigsh of D^-1/2 A D^-1/2 with k=17, filtered by mean(lambda^1..5).
  - Per row: the sender, receiver and device vectors, their cosines and their node degrees.
  - Note: the top eigenvalues are 1.0 x4, because the graph has several components. This variant was not tuned further, since the strong embeddings in (a) are already null.
- **(c) Per-sender sequence features.**
  - Leave-one-out deviation and z-score vs the sender's other rows for: log amount, log balance, login_to_txn_sec, pin_retries, sim_age, device_age.
  - Circular hour similarity and the concentration of the sender's hours.
  - Share of the sender's other rows that use the same device, device_city, txn_type, receiver_type, merchant_category, receiver and app.
  - History position, gaps before and after, gap vs the sender's median gap.
  - First-time ranks for the pair, receiver_type and device.
  - Same device / city / receiver as the previous row; amount/balance.
  - Continuity checks: balance (both directions), sim_age and device_age vs wall time, and sender_account_age_days vs (date - opened_on).

## Balance continuity: checked carefully, no chain
- balance_before is redrawn independently on every row. Relative error vs prev_balance - prev_amount for the same sender:
  - abs(err) < 1e-3: L .00105, G .00106. abs(err) < 1e-2: L .0105, G .0111. abs(err) < 0.1: L .103, G .105.
  - These rates match a uniform spread, so there is no conserved chain to break.
  - The forward check (next row vs this row) gives the same picture.
- sim_age_hours does not carry over between rows: 0 of the rows hold within 0.1 h.
- device_age_days holds within 0.01 d on about 2 % of rows: L .0225, G .0215.
- acct_age_err (reported age minus the true age) is 0 on 78.2 % of L rows and 78.0 % of G rows.
- The generator does not keep per-sender state that camouflage injection could break.

## Exact-duplicate probes (no lift)
Rate of another row with the same value, L vs G:

| value | L | G |
|---|---|---|
| amount | .915 | .916 |
| balance | .560 | .559 |
| (sender, amount) | .0112 | .0116 |
| (receiver, amount) | .0672 | .0697 |
| (amount, balance) | .00092 | .00083 |
| ts | .236 | .234 |

## Conclusion
- None of these is a usable signal: unsupervised entity embeddings, edge plausibility, multi-hop spectral position, sender-habit deviation or state continuity.
- This agrees with camo.md: camouflage rows are independent draws from the legit generator for the same sender.
- Recall on camouflage stays capped. The headroom is on core rows: type confusion and precision.
- Not done here: out-of-time label propagation over the graph. camo a3 already shows that direct label linkage is null, and the hop agent covers multi-hop label linkage.
