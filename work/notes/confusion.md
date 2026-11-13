# confusion: core-row type confusion (confusion agent, 13 Nov)

Code: `work/feats/pairs.py` (8 features `pairs__rseg_*`). Scratch: `work/scratch_confusion/`:
- `a1.py` confusion matrix (output `a1.txt`); `a2`-`a8` pair analyses; `a9.py` triad proxy; `a10.py` triad cells; `a11.py` density split check; `a12.py` Oct drift.
- `mk_matrix.py` dm matrix; `mk_matrix_fd.py` full-density matrix (the one that wins); `ev.py TAG` eval protocol (outputs `ev_*.txt`).

## Result: tag `confusion_pairsfd` beats tun_best on every metric
Matrix: `feats_cache/matrix_m5dm_pairsfd.parquet` = matrix_m5dm + full-density `pairs` (190 feats).
Run: `train.py --modules flow,burst,takeover,scam,ring,pairs --matrix feats_cache/matrix_m5dm_pairsfd.parquet --relabel C --rounds 150 --seeds 3 --tag confusion_pairsfd` (val, then full), then `postproc.py confusion_pairsfd --qr-null`.

Protocol (`scratch_confusion/ev.py`):
- Bias tuned on the pooled OOF, then used to decode pooled, Sep, cleaned pooled and cleaned Sep.
- OOS: bias tuned on the Aug fold only, then scored on Sep.
- Both rows use 3 seeds with the same seed draws.

| tag | variant | pooled | Sep | cPooled | cSep | OOS Aug->Sep | cOOS |
|---|---|---|---|---|---|---|---|
| tun_best | raw | .4719 | .4793 | .4842 | .4916 | .4787 | .4909 |
| tun_best | post | .4721 | .4797 | .4843 | .4920 | .4791 | .4913 |
| confusion_pairs (dm build, 2 seeds) | post | .4724 | .4798 | .4846 | .4921 | .4777 | .4899 |
| **confusion_pairsfd** (full-density build) | raw | **.4778** | **.4834** | **.4901** | **.4958** | **.4815** | **.4938** |
| **confusion_pairsfd** | post | **.4780** | **.4839** | **.4903** | **.4963** | **.4819** | **.4942** |

- Gain: +.0059 pooled, +.0042 Sep, +.006 cleaned pooled, +.0028 OOS. That is 2-4x seed noise (±.0013).
- Per class, cleaned Sep: refund .478 -> .513, collusion .389 -> .400, qr .485 -> .489. Every other class moves by .002 or less.
- Triad cells on candidate rows, pooled OOF, cleaned labels:
  - True refund: 81 -> 194 correct. ref->qr drops 84 -> 24, and ref->mc drops 53 -> 3.
  - True collusion: 407 -> 417 correct.
- Sub: `subs/confusion_pairsfd_ppq.csv` (validated, 260582 rows). Also `subs/confusion_pairsfd.csv`, `preds/confusion_pairsfd_test.npy` and `oof/confusion_pairsfd*`.
- Versus `tun_best_ppq`, the new sub changes 94 test rows:
  - 59 move collusion -> refund. These are Oct segments with a median of 8 rows, 3 senders, 15 h and 2.7 rows per sender. That is refund-sized in train (refund p90 is 8 rows; collusion p10 is 12 rows / 22 h).
  - The 71 rows still predicted collusion are large segments: median 23 rows, 42 h.
- Oct drift check (`a12.py`): Oct has 332 candidate rows, against 584-728 per train month, which fits half the event count. The size mix is the same: n>=6 is .41 in Oct against .35-.49 in train, and n==1 is .078 against .069-.12.

## Confusion matrix on core rows (`a1.txt`)
Model: tun_best plus postproc arch forcing. Core rows are rows with p_legit<=.9 or a fraud prediction (17,066 rows). Labels are cleaned. Top fraud-fraud pairs (a->b / b->a):

| pair | rows | separable? |
|---|---|---|
| sim_swap / ATO | 237 / 244 | No. This is the ~30% event-level label swap inside one archetype (notes/takeover.md), and arch forcing already takes the majority. |
| phishing / fake_cashback | 169 / 113 | No (see below). |
| velocity -> mule | 276 | No. Mule-shaped P2P bursts labelled velocity: rin_n12h 13 vs 12, new_payee .92 vs .81, amount 3729 vs 3711, onward forwarding .66 vs .63. Identical to mule. |
| mule -> layering | 248 | No. Mule-labelled fast chains (below). |
| refund -> qr, collusion -> qr/refund, qr -> refund | 87, 37/53, 14 | **Yes, by burst size. Fixed by pairs.** |
| loan -> phishing | 44 | No. Loan-labelled under-round new-payee COLLECTs. |

Partial rows (p_legit<=.9 and p_own<=.5): 2,428 on cleaned labels. Almost all are the pair confusions above. The rest have legit as the top class: phishing 120, qr 75, and about 30 each for the other classes.

## Per-pair findings (labelled train events)
- **mule vs layering chains** (`a3`, `a4`, `a5`): there are two layering variants.
  - Slow chains (hop gap 1-6 h, cspan over 2 h) are 100% layering.
  - Fast chains (gap 2-30 min) are ~20% mule-labelled, at a flat rate in every gap, size and amount bucket.
  - Chain heads have no inbound mule burst in either case.
  - The model already predicts layering for fast chains, which is the majority. Nothing to gain.
- **phishing vs fake_cashback** (`a6`, `a7`):
  - Payback rows with 2 baits, or with a payback more than 180 min after the bait, are 100% fake_cashback. The model gets them.
  - With 1 bait and a payback within 180 min, the split is ~60% phishing / 40% fake_cashback. This holds flat over bait amount, PAY vs COLLECT, dt, new_payee, hour, fraudster tiny-send count and receiver collect count.
  - A tiny bait with no payback is 95% phishing, and one with a payback is 74% fake_cashback. The model gets both.
  - Under-round new-payee P2P COLLECTs split ph .69 / fc .074 / loan .033 / legit .175. The fc and loan share is flat in every feature, including victim tiny-ins (none) and repeat payments. These rows give the fc->ph 96 and loan->ph 44 errors, and they cannot be fixed.
- **qr / refund / collusion at small merchants** (`a8`, `a9`):
  - Rows: amount in {1000, 2000, 2500, 5000, 10000}, merchant, non-REFUND, mismatch != 1.
  - Grouped into 24 h-gap segments per merchant on those rows only, the label follows segment size:

    | label | rows | senders | duration |
    |---|---|---|---|
    | qr | 3 | 2 | 2.8 h |
    | refund | 5 | 3 | 10 h |
    | collusion | 17 | 4 | 45 h |

  - Rows per sender: 1 gives 87% qr, 4 gives 62% collusion, 5 or more gives 100% collusion. Merchant category does not separate them.
  - A 3-class LGB on these rows, fit on 2 months and scored on the third, gets accuracy .86 with matrix features and .95-.96 with the matrix plus these segment features. Refund F1 goes .58 -> .90.
  - ring's segments (all rows, 12 h gap) do not capture this.

## Bug found: density.halves splits multi-sender merchant bursts (affects ring too)
- `density.halves` links fraud rows into events only by sender, by P2P receiver or sender|merchant, and by device. A merchant burst paid by 3-5 different senders is therefore several events, and each lands in a random half.
- Effect in the dm build (`a11.py`, pairs features): 81% of refund rows and 92% of collusion rows see a smaller segment. Collusion goes from a median of 19 rows / 4 senders to 12 rows / 2 senders. Oct test bursts stay whole.
- That is why the dm build of pairs gave nothing (.4724) while the full-density build gives +.006. The round-only segments are almost all fraud rows, so the full-density build already matches Oct.
- **ring's seg_* collusion features have the same mismatch.** That is a likely cause of the Oct collusion under-call: ring.md shows 4.9 -> 3.2 per 10k.
- Suggested fix for the density.py owner: in `fraud_events`, also link fraud rows on the merchant receiver_id with a short gap (for example 24 h), then rebuild ring__dm. The risk is giant components at big merchants. Cap them by linking only merchants with low volume, or only round-amount rows.

## Notes / risks
- `pairs` must be built with `common.run_feature` (full density), not `run_feature_matched`. train.py without `--matrix` would build it dm, which is the weaker variant. Use the `matrix_m5dm_pairsfd` matrix.
- `feats_cache/matrix_m5dm_pairs.parquet` (the dm variant) is kept only as the negative control.
- In-sample bias: the OOS decode also wins (+.0028), so the gain does not come from bias fitting.
