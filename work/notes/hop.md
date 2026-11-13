# hop: multi-hop graph linkage on camouflage rows (hop agent, 2026-11-13)

Scripts: `work/scratch_hop/h1.py` (builds the features into `frame.parquet`), `h2.py` (G vs L test, output in `h2.txt`), `h3.py` (positive control, output in `h3.txt`).
**Result: null. No feature module was written**, because nothing came close to the 0.56 bar.

## Setup
- Graph: undirected. Nodes are accounts (AC), merchants (MR) and devices (DV), 802k in total. Edges are sender-receiver and sender-device pairs from all train+test rows.
- Two graph variants: `all`, and `p2p`, which drops merchant edges so merchant hubs cannot dominate.
- "Hot" rows: the tun_best model output p_fraud = 1 - p_legit > 0.5. These come from the OOF predictions for Aug/Sep and the test predictions for Oct. Jul has no OOF, so it has no hot rows.
  - Counts: Aug 7,691, Sep 7,473, Oct 3,586.
  - Every feature uses model outputs only. No labels are used.
- G = clean_labels fraud with p_legit > 0.9: Aug 12,274, Sep 11,858.
- L = clean legit with p_legit > 0.9: about 500k per month.

## Features tested (180 per row, each on sender / receiver / device)
- Hot-row count and pf-sum at 1, 2 and 3 hops (A·h, A²·h), raw and normalised by walk counts (2-hop and 3-hop neighbourhood size).
- Degree, plus 2-hop and 3-hop walk sizes.
- Personalized PageRank from the hot mass (restart 0.15, 30 iterations), raw and divided by degree.
- Shortest hop distance to a node touched by a hot row, capped at 6.
- Device-account bipartite component: size and hot count. Also component size and hot share of the non-merchant graph.
- Time-windowed versions (±1 d and ±7 d, both graph variants): hot counts at 0, 1 and 2 hops, degree, 2-hop size, and normalised 2-hop.
- Accounts opened on the same day at the same bank as hot-row accounts: cell hot count, cell hot share, and day hot share, for sender and receiver.

## Results (`h2.txt`)
- Univariate Sep AUC, G vs L: the best of the 180 features is **0.5046** (`all_pf2n_r`). All others fall in 0.497-0.503. Means are identical, for example `w1a_h0_s` hot share 0.0057 vs 0.0041.
- LightGBM fit on Aug G vs L, AUC on Sep (fixed rounds, no early stopping on Sep):
  - 100 rounds: **0.4995** (train AUC 0.72, so it is pure overfit).
  - 300 rounds: **0.5018**.
  - Per class, AUC is 0.49-0.53. The highest is loan at 0.527, on n=443, which is noise level.
  - Top-1 % G rate is 0.025, against a base rate of 0.024.
- Positive control (`h3.txt`): the same features clearly separate core fraud (C = clean fraud, p_legit < 0.5) from L, with the row's own hot contribution removed.
  - Sender ±1 d: 0.73.
  - Device ±1 d: 0.72.
  - Receiver ±1 d: 0.80.
  - Sender, all time: 0.79.
  - On G, the same features give 0.500-0.501.
  - So the graph code works, and G rows simply sit in the legit part of the graph.

## Conclusion
Camouflage rows have no multi-hop link to model-flagged fraud. This holds at 1, 2 and 3 hops, with or without merchants, for ±1 d, ±7 d and all time, and for PageRank, hop distance, components and the same-day/same-bank cohort.
Together with camo.md (direct linkage, row features and the sender profile), this means G rows are independent draws from the legit generator. Their only link to fraud is the label.
Recommendation: stop the graph search on camouflage. Graph embeddings (node2vec and similar) would encode the same adjacency and reach the same null. The positive control shows the neighbourhoods of G and L are statistically identical.
