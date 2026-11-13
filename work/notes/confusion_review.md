# Review of confusion_pairsfd (adversarial reviewer, 13 Nov)

Verdict: gain confirmed. Upload `subs/confusion_pairsfd_ppq.csv` in place of `tun_best_ppq.csv`. Scripts: `work/scratch_confusion_rev/h1-h8.py`.

## Reproduced
- `scratch_confusion/ev.py tun_best confusion_pairsfd` gives exactly the claimed numbers.
  - pairsfd post: .4780 pooled / .4839 Sep / .4903 cPooled / .4963 cSep / OOS .4819 / cOOS .4942.
  - tun_best post: .4721 / .4797 / .4843 / .4920 / OOS .4791 / cOOS .4913.
- h2 refit of pairsfd (seeds 0-2, both folds) gives an OOF identical to `oof/confusion_pairsfd.npy` (max abs diff 0.0). Ids align with the tun_best OOF.

## Attacks
- **Leakage: none found.**
  - `feats/pairs.py` reads no label.
  - The `--qr-null` rule touches P2P receivers only, and pairs touches merchant receivers only, so they do not interact.
  - Oct QR-null share among candidate rows is the same as train: 67% vs 66%.
- **Density mismatch (full-density build in train vs Oct at half events): not a problem.**
  - h1 builds an Oct-like version: density.halves, plus candidate fraud rows linked per (merchant, cleaned class) with a 24 h gap.
  - Collusion segments shrink only slightly (median 19 -> 18). Refund and qr are almost unchanged.
  - The fd model scored on Oct-like features gives .4784 / OOS .4822, no loss.
  - The honest build end to end gives .4786 / OOS .4833 / cOOS .4957, which is within noise of fd.
- **Sampling noise: larger than the claim implies.**
  - A receiver-cluster bootstrap of the paired OOS Aug->Sep diff gives +.0026, sd .0019, P(d<=0) = 8.7%.
  - So the gain is 1.4 sd of row-sampling noise, not "2-4x noise". Seed noise is the wrong yardstick for transfer.
- **Selection: small.** The 24 h gap and the round-amount set were picked from labelled analysis over all train months, including both val folds. Only 2 builds were scored (dm, fd).
- **Oct transfer: supports the change.**
  - Oct candidate rows in n 6-8 segments: 66, against 62-125 per train month. In train these segments are 75-100% refund and 0% collusion.
  - tun_best calls 59 of the Oct ones collusion, but only 20 of the Sep ones (85% true refund). This fits ring seg_* under-sizing bursts in the dm build.
  - The pairs model moves those 59 rows to refund.
  - The collusion drop 132 -> 71 is all n>=12 rows. Oct has 71 such rows, against 124 / 232 / 251 in Jul / Aug / Sep. At half events that is the low end of the plausible range, not an artefact.
