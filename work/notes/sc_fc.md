# sc_fc: fake_cashback Oct structure (13 Nov)

Code: `work/sc_rules_fc.py` (`apply(sub_df, tx)`). Candidate: `subs/sc_fc.csv` = `subs/v3_final.csv` with 5 rows changed. Scratch: `work/scratch_sc_fc/s1-s13.py` (`judge_h6.pkl` = cached `ts_judge_post.get("h6")`, 78 MB, can be deleted).

## (a) Test fc rows in v3_final: 200
| source | rows | Sep precision of the same source (judge, cleaned) |
|---|---|---|
| fc zone override (474496-474514) | 12 | .74 |
| bait rule 2 (`nf>=2`, `dt>180` or known payee) | 104 | .86 |
| bait rule 3 (ambiguous; h6 leans fc, which is F1-optimal per typesplit_bait) | 85 | .34-.42 |
| other | 0 | - |

- Sep cleaned fc: 816 true rows, recall by source: zone 1.0, r2 .99, r3 .88, other 571 rows .00.
- The "other" rows are camouflage: merchant QR/PAY and P2P PAY with no bait link. Only 10 of 1,842 train non-bait fc rows touch a bait account within 7 days. Under-round COLLECTs are the ph .69 / fc .074 cell. Nothing to gain there.

## (b) Oct-specific structure for fc: none drifted
- Bait volume in Oct vs train: r2 .180 (106/588), r3 .181. The block scale is .1687 and the row scale .1666, so fc runs at about 1.08x. Bait amounts {1, 2, 5, 10}: same mix. Payback type and new_payee mix: same. Column null rates in bait rows: same.
- Partial decodes: none.
  - Every Oct rule-2 row is decoded fc (104/104).
  - Bait events with one zone row and one off-zone row have consistent decodes in test (2 fc/fc, 2 ph/ph). The train partner label matches the zone 26/34 (fc) and 23/29 (ph), so propagation is already done.
- Split of rule 2 by trigger:

  | trigger | Jul+Aug rows | Sep rows | Oct rows | train fc share |
  |---|---|---|---|---|
  | nf2 | 329 | 163 | 84 | .87-.88 |
  | dt180 | 61 | ... | 20 | .885 |
  | v_np==0 only | 2 | 2 | 0 | 1.0 |

  The Oct COLLECT new_payee=0 excess comes from nf2 and dt180 events. The lead is null.
- Generic zone-row linking (same account within 1 h or 1 d, off-zone): train label == zone only .3-2.7%, against 95-98% legit. No propagation lever.

## fc zone edges (the .33 agreement)
- The .33 is 1 of 3 non-legit model preds. The 2 misses (474507 COLLECT 9999, 474511 COLLECT 1990) are under-round COLLECTs that h6 calls ph.
- Train fc zone COLLECT P2P rows: 6/9 labelled fc, 0 ph. OOF decode of the fc-labelled ones: 3 fc, 1 ph. So the low agreement is a model blind spot, not an edge error.
- Structural brackets from model evidence:
  - refund|fc lies in [474496, 474505]: the last REFUND row is 474495 and the first rule-2 payback is 474505. 7 model-legit rows ride on it: 474496, 497, 498, 500, 502, 503, 504.
  - fc|loan lies in [474506, 474517]: the first lending row is 474517. 4 rows ride on it: 474507, 508, 511, 512.
  - Row types do not separate refund from fc: PAY P2P .27 vs .35, PAY MERCHANT .21 vs .18, AUTOPAY .04 vs .04.
- The current edges (474496 / 474515) are the max-fc choice: fc length 19, against an expected 24.3 (block scale) to 26.7 (Oct fc volume x1.08-1.16). Refund is 34 vs 35.6-37.7, loan 14 vs 16-17.
- The DP is a MAP estimate. The marginal length-prior posterior of each row being fc:

  | row(s) | P(fc) |
  |---|---|
  | 474496 | .28-.29 |
  | 474497 | .51 |
  | 474498 | .69 |
  | 474500+ | >= .89 |
  | 474512 | .89 |

- Break-even P(fc) for fc vs refund is .44, because fc has the higher per-row leverage: P+T ~640 vs ~810.
  - Only 474496 flips, for about +.00007 macro expected. Not shipped: below noise, and it depends on the variance model.
- Train thinned-block simulation (Aug-only / Sep-only OOF preds): the DP recovers refund|fc within +2..+6 and fc|loan within +1..+4 of the truth. The DP is accurate wherever evidence exists.

## (c) Rule shipped: bait-aware zone edges (`sc_rules_fc.py`)
- Train fact:
  - Bait-event rows (ts_bait rule > 0) on zone devices are in the ph or fc zone in 71/72 cases (Jul+Aug 60/61, the miss is 1 ATO; Sep 26/26).
  - 0/230 qr-zone rows are bait rows.
  - Rule-3 rows in the ph zone: 14/15 labelled ph.
- Oct: the answered-bait event AC345694 -> AC201560 (Rs 5, then 10461 back after 2 h; rule 3; h6 fc .38 / .56) sits on consecutive fresh devices 474358/474359. The judge DP put ph|qr at 474352, so both rows were decoded qr_swap.
- The DP with a bait emission (-8 outside ph/fc), run on the same model evidence (subs/h6_ppq.csv == judge pt, asserted), gives:
  - ph|qr = 474360;
  - ATO|ph moves 474284 -> 474285 (no device at 474284);
  - every other edge unchanged.
- The rule moves zone rows whose label is still the old zone type. Result: exactly 5 rows qr_swap -> phishing_collect:
  - the 2 bait rows;
  - 474352, 474353, 474356, which contiguity places in the ph zone.
- None is a QRN row: 474353 has mm=0.
- Out-of-time check: the bait emission changes no train edge in the Aug-only or Sep-only simulation. So it shows no harm, but no gain is measurable on train. The evidence is the cell count above.
- Purity of model-legit ph-zone rows: cleaned ph .84 (Aug), .91 (Sep).
- Expected macro gain, from test predicted counts (ph 501, qr 542) and Sep-cleaned T/P and F1:

  | candidate | rows | expected macro |
  |---|---|---|
  | 2-row floor (bait rows only) | 2 | +.0002 |
  | full candidate | 5 | +.0005 |

- This is far short of the +.004 target. fc has no lever larger than this: bait structure is fully decoded, camouflage is 70% of fc, and the ph/fc ambiguous core is irreducible.
