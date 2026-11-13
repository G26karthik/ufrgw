# octcal: Oct vs Sep per-class calibration check (octcal agent, 13 Nov)

Scripts: `work/scratch_octcal/` (outputs in `s*.txt`).
- `prep.py`: halves_rsb + `postproc.apply(qr_null=True)`, bias re-tuned on the post OOF. The decode reproduces `subs/halves_rsb_ppq.csv` exactly (1.0 match).
  It writes a frame with the Sep OOF rows (fold 1, dm features) and the Oct test rows.
- `s1.py`: per-class rates and structural signatures.
- `s2_coll.py`, `s3_mid.py`: small-merchant segments.
- `s4_bias.py`: bias sensitivity and the loan cross-tab.
- `s5_soft.py`: soft rates and the EM prior estimate.

## Verdict: no fix. The model is calibrated on Oct structure; the rate gaps are real event-count gaps.
For every class, the Oct/Sep ratio of predicted rows tracks the Oct/Sep ratio of label-free structural rows.
The model decodes the structural rows of each class as that class at the same rate (~92-100%) in both months.
Per-class bias shifts barely move the test decode: ±0.2 changes ≤12 test rows per class.
No feature fix or decode fix is justified. No candidate sub was written, because it would be identical to `halves_rsb_ppq`.

## Per class (per 10k rows; Sep = OOF fold 1 at Oct density, 509,996 rows; Oct = test, 260,582 rows)
The structural signature is label-free (definitions in `s1.py`, the same rules as notes/postproc.md). `cprec` = cleaned-label precision of the signature on Sep.
"own" = share of signature rows that the model decodes as the class.

| class | signature | pred S | pred T | pred T/S | struct S | struct T | struct T/S | own S / T | cprec S |
|---|---|---|---|---|---|---|---|---|---|
| mule | flow mule-burst rule | 23.0 | 28.0 | 1.22 | 17.7 | 21.8 | 1.23 | .92 / .95 | .80 |
| velocity | device session n>=20, int, gaming/ecom | 21.4 | 10.4 | **0.48** | 20.9 | 10.3 | **0.49** | 1.00 / 1.00 | 1.00 |
| layering | chain_size>=3 | 20.8 | 21.5 | 1.03 | 14.7 | 16.5 | 1.12 | 1.00 / 1.00 | .80 |
| sim_swap | arch_sim | 9.8 | 10.9 | 1.11 | 8.5 | 9.4 | 1.11 | 1.00 / 1.00 | .69 |
| ATO | arch_ato | 9.5 | 11.2 | 1.18 | 8.5 | 9.5 | 1.12 | 1.00 / 1.00 | .70 |
| phishing | under-round new-payee P2P COLLECT | 18.6 | 18.4 | 0.99 | 12.1 | 12.5 | 1.03 | 1.00 / 1.00 | .71 |
| qr_swap | QR->P2P with category | 19.3 | 19.0 | 0.98 | 13.0 | 11.9 | 0.91 | 1.00 / 1.00 | .84 |
| refund | REFUND, rs_n>=3 within 10 d | 9.3 | 10.2 | 1.09 | 5.8 | 7.1 | 1.22 | 1.00 / 1.00 | .85 |
| fake_cashback | payback with 2 baits or dt>180 min | 6.5 | 7.4 | 1.14 | 1.3 | 1.5 | 1.16 | 1.00 / 1.00 | .92 |
| loan | sender s4_np_int_eve>=2, new-payee int>=1000 evening | 3.8 | 3.8 | 1.02 | 4.9 | 5.9 | 1.21 | .73 / .64 | .63 |
| collusion | round-set merchant segment n>=12, >=2 senders | 4.9 | 2.7 | **0.56** | 4.9 | 2.7 | **0.55** | .94 / 1.00 | .83 |

The remaining rows that are not structural give similar rates in both months, for example mule 6.8 vs 7.3 and phishing 6.5 vs 5.9.
- Soft rates (mean prob per 10k) agree: velocity is .82, collusion .89, and every other class is 1.00-1.06.
- Saerens EM on the post-processed test probs gives a prior ratio of velocity .44 and collusion .56; every other class is .96-1.28.
- Applying the EM shift to the bias changes **8** test rows.
- The model leaves 0.04 per 10k Oct rows decoded legit with p_legit<.5, against 0.18 in Sep, so it does not under-call.

## Why the collusion rate is 2.7 vs 4.9: Oct has 3 collusion bursts, and the model calls all 3
Round-set small-merchant segments, using the pairs logic (24 h gap), counted per month:

| size | Jul | Aug | Sep | Oct (test) | train cleaned label mix |
|---|---|---|---|---|---|
| 4-5 | 12 | 8 | 10 | 2 | refund .74, qr .08 |
| 6-8 | 10 | 15 | 9 | 11 | **refund .84, collusion 0** |
| 9-11 | 1 | 2 | 0 | 0 | collusion .78 |
| >=12 | 6 | 12 | 14 | 4 (3 test + 1 Sep-started event that sits in train) | collusion .72-.87 |

- Every train segment with n<=8 is refund (39 of 39, `s3.txt`), including n=8 segments with 2 senders and 4 rows per sender. Every train segment with n>=10 is collusion. The boundary is sharp.
- The Oct segments split the same way:
  - The 9 segments with n 6-8 have p_ref .70-.92 and p_coll .01-.16, and decode to refund.
  - The 3 segments with n>=20 (MR014892 28 rows, MR010487 23, MR002433 20) have p_coll .92-.96. Those are exactly the 71 collusion rows.
- At half events the expected count is about 5.3 bursts (train mean 10.7 per month). Observing <=3 has Poisson p = .22. Sep (14) is the high month, and Jul (6) would give about 2.3 per 10k.
- Variant B (whole 1000-9999 amounts, not the round set): the only Oct segments with n>=6 are at big merchants (MR007972, 19k rows) with 1 row per sender. They are legit background, and the model leaves them legit.
- The Oct 6-8 bucket is somewhat over-full (11 against about 6 expected). Train labels say all of those are refund, and the model follows them. Moving them to collusion would go against 39/39 train segments, so it is not proposed.

## Velocity
- There are 7 Oct device bursts against 25 in Sep: 0.55x per row, or about 0.28x in absolute count (notes/adapt.md, notes/drift.md).
- Every Oct velocity prediction lies inside a structural burst (pred 10.36 equals structural-and-pred 10.32 per 10k), and every burst row is decoded velocity.
- A bias shift of -0.7 changes 1 test row, and +0.4 adds 9. There is no lever: this is a real prior shift with confident decodes.

## Other classes
- **loan:** 64% of the Oct signature rows are decoded loan, against 73% in Sep. The difference is ATO: 40/155 against 49/251. Oct signature rows are 46% P2P against 35%, so there are more P2P drains.
  - On Sep, the ATO decodes of signature rows are mostly true ATO/sim (cleaned: ATO 36, sim 10 out of 49).
  - Loan rows decode at 3.8 against 3.8 per 10k. Nothing to fix.
- **mule:** the 1.22x Oct rate is matched by structure (1.23x events per row; adapt.md counted 66 against 118 bursts).
- **qr_swap:** the 140 null-mismatch P2P-QR rows are all qr_swap through `--qr-null`. The QR->P2P-with-category structure is 0.91x, and the decodes are 0.98x.
- **refund:** the structure is 1.22x and the decodes 1.09x. The round-set segments with n 4-11 are 2.9 vs 2.1 per 10k, and all 76 Oct rows decode to refund.

## Bias sensitivity (s4.txt)
Columns: test rows changed / Sep noisy F1 delta / Sep cleaned F1 delta.

| class | -0.2 | +0.2 |
|---|---|---|
| mule | 2 / 0 / 0 | 3 / 0 / 0 |
| velocity | 1 / 0 / 0 | 1 / 0 / 0 |
| layering | 2 / 0 / 0 | 1 / +.0001 / +.0001 |
| sim_swap | 0 / 0 / 0 | 2 / 0 / 0 |
| ATO | 3 / 0 / 0 | 2 / 0 / 0 |
| phishing | 10 / -.0006 / -.0006 | 12 / -.0002 / -.0002 |
| qr_swap | 1 / -.0001 / -.0001 | 0 / -.0001 / -.0001 |
| refund | 0 / -.0001 / -.0001 | 0 / -.0002 / -.0002 |
| fake_cashback | 7 / -.0003 / -.0003 | 5 / -.0006 / -.0006 |
| loan | 2 / -.0001 / -.0001 | 1 / 0 / 0 |
| collusion | 0 / 0 / 0 | 0 / 0 / 0 |

The decode is saturated on structural rows. Bias moves only camouflage-margin rows, and on Sep each shift is flat or negative.

## Not checked
- Velocity "spread-out" bot variants beyond adapt.md's scans (30 min / 1 h / 6 h device and sender sessions).
- Possible Oct-only structures with no train analogue. adapt.md found only the QR-null evasion.
