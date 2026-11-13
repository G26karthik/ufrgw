# embed review (adversarial, 2026-11-13)

**Verdict: the null is CONFIRMED.** No feature module exists, so there was no macro-F1 run and no Oct transfer check to do.

## What I checked
- **Label leakage: none.**
  - e1_seq.py drops `label` and `y` before it builds any feature.
  - e2_emb.py reads only the sender, receiver and device ids.
  - The embeddings are fit on all rows, including Sep and Oct. This is unsupervised and uses no labels, so it is acceptable.
  - Labels enter only gl.parquet, through clean() and the G/L target.
- **Circularity: none.** `pl` (p_legit), `cls` and `g` are excluded from the feature columns; rv_check.py asserts this.
- **Out of time: yes.** The model is fit on Aug, and early stopping uses an inner 20 % split of Aug. AUC is measured on Sep.
- **Join integrity** (`scratch_embed/rv_check.py`):
  - txn_id is a String in all files, with no duplicates.
  - seq and emb each have 1,824,642 rows, and every G/L row matches. Max NaN rate is 0.156; no row is all-NaN.
  - G/L counts match the note: Aug 12,274/505,808, Sep 11,858/489,842.
  - So the null is not caused by a broken join.
- **Selection-bias check per class.** For each class, I picked the top 3 of the 160 features by Aug AUC and scored them on Sep, keeping the Aug sign.
  - On Aug, the best features reach 0.47-0.53. On Sep they fall to 0.47-0.51, and the direction often flips (cls5 s_sd0: .531 to .493; cls8 r_sr1: .524 to .480).
  - This is noise, not signal.
- **Embeddings-only positive control** (ev.py on ctrl/gl.parquet with only emb.parquet; core fraud vs L): Sep AUC **0.899**, top-1 % fraud rate 0.45 against a base rate of 0.014, and 0.81-0.98 for each class.
  - So the 120 graph features carry real signal, and they are not just riding on the sequence features.
  - Caveat: the top feature by gain is deg_r, the receiver degree (merchant vs P2P). Part of this control therefore reflects the row type, not ring communities.

## Weaknesses (none of them changes the verdict)
- The embeddings are global and low-rank: k=16 over 802k nodes. They cannot resolve small ring communities. On top of that, the spectral part is degenerate: four eigenvalues equal 1, so those eigenvectors only mark which component a node is in.
- Local multi-hop structure is covered elsewhere, and was null there too. hop.md tested hot counts at 1-3 hops, PPR, hop distance and components, against a working control (AUC 0.72-0.80).
- In ev.py, the per-class AUC line is convoluted (`ps[...] if False else`), but it indexes the same rows in the same order, so it is correct.

## Recommendation
Accept the null and stop hunting for camouflage with unsupervised graph and sequence features. Move effort to core-row type confusion and precision.
