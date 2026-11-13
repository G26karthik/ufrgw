# postproc: post-processing of class probabilities (postproc agent, 2026-11-13)

Code: `work/postproc.py` (`apply(P, ids_df, tx)` + CLI). Scratch: `work/scratch_postproc/e*.py`.
Run: `python postproc.py TAG [--qr-null]` (cwd=work). It applies the transform to `oof/TAG.npy` and `preds/TAG_test.npy`,
re-tunes biases on the post-processed pooled OOF, prints the eval numbers and writes `subs/TAG_pp.csv` (`TAG_ppq.csv` with `--qr-null`) through `train.write_sub`.

## Result on m5dm

Biases re-tuned on the pooled OOF for both rows, so decoding is identical. "cleaned" = `clean_labels.clean(tx)` labels on the val rows.

| | pooled | Sep | cleaned pooled | cleaned Sep |
|---|---|---|---|---|
| m5dm raw | 0.4574 | 0.4650 | 0.4700 | 0.4772 |
| **m5dm + postproc** | **0.4641** | **0.4721** | **0.4764** | **0.4845** |

Optimism check (`e8.py`): biases tuned on Aug only and scored on Sep give raw .4627 / cleaned .4749 against post .4690 / cleaned .4814 (+0.006). With the old m5dm bias left unchanged, post scores .4626 / .4751 pooled. The gain does not come from bias tuning.

What `apply` keeps:
1. **Takeover archetype forces the type** (the whole gain): `takeover__arch_sim=1` gives sim_swap and `arch_ato=1` gives account_takeover, as one-hot probs.
   - Inside each archetype the sim/ATO labels are about 30% swapped at event level (notes/takeover.md). `clean()` does not fix this.
   - The model splits these rows. On cleaned Sep the arch_sim hits are 593 sim against 239 ATO, and the model predicts 685 sim and 202 ATO.
   - Forcing the majority gives per-class cleaned Sep F1 of sim .34 to .39 and ATO .36 to .39.
   - Hard one-hot is as good as moving the sim+ATO mass to the archetype class (.4641 vs .4640).
   - After forcing, the re-tuned sim/ATO biases rise to about 2. They now act only on rows without an archetype flag.
2. **Nov-dated rows are never legit.** This is a no-op on m5dm: all 46 rows are already predicted fraud (36 refund, 6 loan, 3 layering, 1 fake_cashback). It is kept as a guard for other models.
3. **`--qr-null` (optional, cannot be validated):** QR to P2P with qr_vpa_mismatch null gives qr_swap.
   - It touches 140 test rows. All of them have merchant_category set.
   - m5dm already decodes 101 of them as qr_swap. The flag moves **39 rows from legit to qr_swap** (subs/m5dm_ppq.csv against m5dm_pp.csv).
   - On Sep, P2P-QR with mismatch=1 is 84% qr_swap on cleaned labels, and P2P-QR with category set is 83%.
   - If the null rows follow that rate, expect about +33 TP and +6 FP for qr_swap, roughly +0.001 macro. **Recommended on**, but it is a judgement call.
   - Test: on m5dm_pp the per-class change against raw m5dm is 95 rows (mostly sim/ATO swaps). The flag changes 39 more.

## Tested and dropped

**(1) Event smoothing** gives near zero because the module features already make predictions event-coherent.
- Prob mixing `P' = a*P + (1-a)*mean_segment(P)`. Segments are consecutive rows on a key within a gap, built inside `density.halves`, so train rows see Oct density. Hub cap 200.
- With the m5dm bias fixed, the best single keys add +0.0007 pooled: device or sender, gap 1-6 h, a=0.5. a=0 at 6 h hurts by -0.002.
- Receiver P2P and sender|merchant keys add about +0.0002.
- Unordered P2P pair (bait pairs, 6 h/48 h) adds -0.0004 to -0.001.
- With biases re-tuned on top of arch forcing, dev+snd smoothing adds -0.0022 to +0.0002 (geometric mean, gap 1 h/6 h, a in .3/.5/.7). Arithmetic mean adds 0.
- Not worth the code.

Why: on cleaned Sep, the remaining errors are FN to legit, about 2/3 of each class. These are the camouflage singletons, with no event to pool over. The fixable pair confusions are mostly event-level type noise: velocity-labelled mule bursts (141 rows), mule-labelled layering chains (96 rows), and phishing/fake_cashback bait ambiguity (89/53). Pooling cannot fix those either.

**(3) Structural rules: the model already gets them.** Forcing the rule label (with the m5dm bias fixed) changes pooled F1 by:

| rule | hits (Aug+Sep) | cleaned precision | model already predicts it | change in pooled F1 from forcing |
|---|---|---|---|---|
| layering chain_size>=3 | 1583 | .78 | .99 | +0.0001 |
| mule burst rule (flow.md) | 1810 | .79 (velocity .14) | .94 | -0.0019 |
| velocity device session n>=20, int, cat | 2010 | .997 | .999 | 0 |
| refund episode n>=3, <=10 d | 571 | .87 | 1.00 | 0 |
| QR->P2P with merchant_category | 1386 | .83 | .99 | +0.0001 |
| QR->P2P mismatch=1 | 846 | .84 | .99 | +0.0001 |
| new-payee under-round COLLECT -> phishing | 1290 | .72 | .995 | +0.0002 |
| arch_sim / arch_ato | 889 / 892 | .67 / .66 | .77 / .90 | **+0.0029 / +0.0021** (kept) |

Gotcha: matrix features are NaN on rows where they do not apply, and in polars `NaN >= 3` is **True**. Use `fill_nan(None)` before rules. Without it, chain_size>=3 hit 660k rows.

**(5) Prior shift (Saerens EM)** on m5dm test probs, with the train prior taken as the mean of the OOF probs. Estimated test prior / OOF prior:

| class | ratio |
|---|---|
| velocity | 0.48 |
| qr | 0.75 |
| layering | 0.89 |
| legit, ATO, phishing | 0.99-1.01 |
| refund, sim, fake_cashback, loan, collusion | 1.19-1.28 |
| mule | 1.38 |

- The velocity and qr drops match notes/drift.md: fewer bot bursts, and the QR mismatch evasion.
- With the m5dm bias, EM changes only 65 test decodes: +29 sim, -31 ATO, -17 qr, +10 mule.
- It cannot be validated, and the change is small. **Kept off.**

## Next ideas
- The model is close to the structure ceiling. The remaining wins are in training, not post-processing:
  - Train on archetype-cleaned sim/ATO labels (notes/takeover.md). This should beat forcing, because the non-arch rows also learn the right type.
  - Event-level type relabel for mule-labelled layering chains and velocity-labelled mule bursts.
- Re-run `python postproc.py <newtag>` on any new model or blend. The arch forcing should transfer, as long as the model's sim/ATO split stays imperfect.
