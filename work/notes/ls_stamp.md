# ls_stamp: clean the fit target with the fraud-path stamp (fresh_judge #3). FAILED the gate, closed

Idea (fresh_columns.md #2, #3): legit-labelled train rows that carry the fraud-path stamp are ~88% fraud. The stamp is
(non-QR & qr_vpa_mismatch == 0) or (merchant QR & qr_vpa_mismatch null). Either drop these rows from the fit set or
relabel them.

## Data (train, s1.py)
- non-QR mm=0: 7454 rows, fraud share .877, legit 914 (Jul 316, Aug 280, Sep 311, Oct 7).
- merchant-QR mm null: 678 rows, fraud share .885, legit 78 (Jul 29, Aug 25, Sep 24).
- Total legit-stamped: 992 rows.

## Variants
Code: `relabel.py` letters T and U. Both are additive and used only when requested.
- **T (lsT6):** drop the 992 rows from the fit set.
- **U (lsU6):** relabel 640 Aug/Sep rows to the raw argmax over the fraud columns of the c2s6 OOF.
  - The folds are forward-only, so the label never comes from the row's own fit.
  - Jul and Oct rows have no OOF, so their 352 rows are dropped.
  - In val, U differs from T only on ~305 Aug rows in fold 1. Sep rows are val rows there and keep their raw labels.
  - U targets: phishing 137, mule 97, ..., refund 44, sim 28, collusion 20, fc 18, loan 16.

Recipe is identical to c2s6, so the comparison is paired:
- Same matrix: `feats_cache/matrix_core2swap.parquet`.
- Same modules, rounds 150, seeds 0-5, threads 3, lgb.
- Only the relabel differs: CT or CU instead of C.

Logs: notes/train_lsT6_val.log, notes/train_lsU6_val.log. The gate is scratch_ls_stamp/gate.py, a copy of
scratch_idea_zclean/gate.py. It uses the identical judge decode: qr_null + pooled bias + zone/bait. The sd column is a
sender-cluster bootstrap of the paired diff.

## Gate vs c2s6 judge (reproduced .5101 / .5228 / .5091)
| tag | relabel | pooled biased | Sep noisy | cleaned | OOS Aug->Sep | OOS diff sd |
|---|---|---|---|---|---|---|
| c2s6 | C | .4800 | .5101 | .5228 | .5091 | - |
| lsT6 | CT (drop) | .4798 | .5096 (-.0005) | .5223 (-.0005) | .5078 (-.0013) | .0004, P(<=0) 1.00 |
| lsU6 | CU (OOF relabel) | .4793 | .5089 (-.0012) | .5216 (-.0011) | .5076 (-.0014) | .0004, P(<=0) 1.00 |

The train zone edges are unchanged.

Per-class cleaned diff:
- T: qr -.003, phishing -.002, loan -.002.
- U: qr -.005, refund/fc/loan -.003.

## Verdict
Both variants are negative on all three metrics, beyond seed noise (.0005). No full run, no sub.

Likely why (hypothesis, not tested):
- The 88% purity is no better than the base noise rate. Every fraud-labelled row is also ~88% clean.
- The stamp rows are therefore a sample of the general 12% fraud->legit noise, and the model already absorbs that
  noise uniformly.
- Removing or relabelling the stamp rows shifts the model's legit prior only on stamped rows, and these are exactly the
  ambiguous QR/merchant cells.
- The losses concentrate in qr_swap, which matches this.
