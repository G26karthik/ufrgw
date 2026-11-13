# adapt: how the October rings adapted (adapt agent, 2026-11-13)

Module: `work/feats/adapt.py`, 2 row-level features. Labels are never read. Cache: `feats_cache/adapt__dm.parquet`.
Scratch scripts: `work/scratch_adapt/` (a1-a19 analysis, `ev.py` eval, `cmp.py` test diff, `slim.parquet` analysis frame).

## Method
- Compared Oct (test) with Jul-Sep for each fraud type, using structure only.
- Sep sides used: Sep labels cleaned with `clean_labels.clean`, and Sep OOF predictions of m5dm.
- Oct side used: m5dm test predictions with the m5dm bias, plus raw structural signatures counted per month.
- Expected Oct/Sep ratio is about 0.5, because Oct has half the events. An adaptation is any shift away from that ratio or any new value pattern.

## Findings per type (Oct vs Sep)
| type | Oct/Sep events | shape change | verdict |
|---|---|---|---|
| mule | 66/118 inflow bursts (.56) | Same size (10.5 rows), senders, new_payee .93, r10 1.0, span 8.5-9 h. Forward rate .50 vs .44, out/in ratio .71 vs .59. | no adaptation |
| velocity_bot | 7 vs 25 device bursts (.28), 241 vs 857 login<=6 bot rows | Bursts identical: 30-58 rows, 5-13 min, login 3-4 s (1/7 "humanised" 46 s). No device-spread or merchant-spread variant: 0 multi-device merchant bursts; device/sender sessions at 1 h and 6 h gaps hide no extra events. | only fewer events, about 1/3 rate instead of 1/2. This is a prior shift, not a feature issue. Do not raise the velocity bias. |
| layering | 335 vs 654 hop pairs (.51) | Hop ratio still only in (0.96, 0.995], same histogram. Gaps and amounts the same. | no adaptation |
| sim_swap / ATO | S 86 vs 154, D 87 vs 161 drain sessions (.55) | Archetypes identical (away, night, pin, login, span, share). "SD" (sim+dev) has 7 sessions that look like D. | no adaptation |
| phishing | COLLECT 1999/4999/9999 at ratio 1.0-1.1 | No new under-round limits: no 2999, 7999 or 14999 cluster. 24999 slightly rarer (.6). | no adaptation |
| qr_swap | P2P-QR-with-category 309 vs 663 (.47) | **mismatch flag blanked**: 140/309 Oct rows are null (train has 0). mm=0 and mm=1 both shrank (57 vs ~121 expected, 112 vs ~190), so the flag is blanked regardless of value. Also 17 merchant-QR rows with null mismatch and a **non-integer** amount at tiny merchants (deg ~5); train has 0. In train, merchant-QR null occurs only with integer round amounts. | **adapted (evasion)** |
| refund_abuse | episodes with n>=4: 28 vs 54 | Same span (7 d), gap and amount. Receivers per episode 1.61 vs 1.35 (slightly more spread). | minor; burst counts per sender are already robust |
| fake_cashback | 114 vs 223 bait to payback pairs | Same dt, amount and COLLECT share; evening .34 vs .22 (Jul was .30). | no adaptation |
| loan | 18 vs 37 victims | Same n, evening share, amount, b11 and span. | no adaptation |
| merchant_collusion | 11 vs 14 small-merchant bursts | Same shape. | no adaptation |

Scans for "bursts just outside module windows" found nothing in Oct that the model leaves as legit:
- device sessions with 30 min, 1 h and 6 h gaps;
- P2P receiver sessions with a 6 h gap;
- small-merchant integer sessions with a 24 h gap;
- fast-login sessions by device and by receiver.
The only Oct leftovers are 3 mule-like inflow bursts without onward transfer (4-7 senders). They get p~.5. Train has the same pattern: those train bursts are mostly labelled layering, not mule.

Conclusion: the only structural evasion is the blanking of the QR mismatch flag. The other "adaptation" is fewer velocity events, which is a prior shift that features cannot fix.

## Features
- `adapt__qr_p2p_cat` = QR & P2P & merchant_category set. Invariant to the evasion. In noisy train it is 82% qr_swap, about 100% after the 16% label noise.
- `adapt__qr_mm` = qr_vpa_mismatch, with null set to 1 when the null pattern is unseen in train: QR & null & (P2P or non-integer amount). Merchant-QR null with an integer amount keeps null, as in train.

## Validation
Setup (`scratch_adapt/ev.py`):
- train.py protocol on the cached m5dm matrix plus the adapt join; lgb, lr .05, legit_frac .2, early stop.
- 2 seeds; folds Jul->Aug and Jul+Aug->Sep.
- Clean proxy = `clean_labels.clean` labels on the val rows.
- "evaded" = the same val rows with qr_vpa_mismatch blanked on a random 45% of P2P-QR-cat rows. This simulates the Oct evasion on Sep.

| variant | pooled noisy (own bias) | Sep noisy | pooled clean | Sep clean | Sep noisy, evaded | qr F1 Sep, m5dm bias: clean / evaded |
|---|---|---|---|---|---|---|
| base (m5dm feats, 2 seeds) | 0.4603 | 0.4671 | 0.4725 | 0.4792 | 0.4668 | .484 / **.463** |
| + adapt | 0.4613 | 0.4690 | 0.4736 | 0.4810 | 0.4688 | .485 / .482 |
| + adapt, drop raw `qr_vpa_mismatch` | **0.4614** | 0.4688 | **0.4736** | 0.4808 | **0.4688 (= unevaded)** | .482 / .482 |

Under the identical m5dm bias:
- pooled noisy: base .4592, adapt .4607, adapt_drop .4606;
- pooled clean: .4715, .4731, .4729.
Gains on un-evaded data (+.001) are within seed noise. The real gain is robustness: the base model loses .021 qr F1 when the flag is blanked, adapt_drop loses 0.

Oct effect (full fit, 2 seeds, `cmp.py`):
- 127 Oct rows change label vs a base refit (adapt: 147). Most of this is colsample churn among sim/ATO/phishing.
- Under the same m5dm bias, the base refit labels 124/140 null P2P-QR rows as qr_swap; adapt and adapt_drop label 140/140.
- The m5dm submission itself had 109/140 (31 left as legit).
- The 17 non-integer merchant-QR null rows stay legit in every variant. That is expected: in train, mm=1 at tiny merchants is only about 20% qr.

## Recommendation
- Add the module and drop the raw flag: `train.py --modules flow,burst,takeover,scam,ring,adapt --drop qr_vpa_mismatch`.
- This is evasion-proof by construction, costs nothing on pooled or clean OOF, and recovers about 16-31 qr_swap rows in Oct.
- Velocity: expect about a third of Sep's rate in Oct. Bias tuning on Sep will over-predict velocity a little, but the burst rows themselves are unchanged.
