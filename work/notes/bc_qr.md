# bc_qr: qr_swap Oct structure on sc_final (13 Nov)

Scratch: `work/scratch_bc_qr/s1..s13.py` (+ `.log`). Run from cwd=work with `PYTHONPATH=.`. Reuses `scratch_v3_cells/frame.parquet` and `scratch_sc_fc/judge_h6.pkl`.
Rule: `work/bc_rules_qr.py`. It is the identity: no rule is accepted. No candidate sub is written (`sc_rules_loan.py` precedent).

## Verdict
qr is saturated. No label-free rule clears break-even, so the expected gain is 0. qr cannot supply the +.004 target.
- Arithmetic: qr T_test is about 1202 and P is 537, with F1 about .5. One extra TP is worth about +7e-5 macro. One FP costs about -2.4e-5. Break-even precision is about .25.
- Reaching +.004 from qr alone would take about 55 net new TPs. The qr pool that is not yet decoded is camouflage: about 56% of qr rows sit in legit-dominant cells with a qr rate of 0.3-0.5%.

## (a) Test rows decoded qr in sc_final: 537
| source | rows | structure | Sep judge precision of the same cell |
|---|---|---|---|
| zone [474360, 474462) | 41 | 41 devices, one row each; all zone rows are qr | zone purity about .85 |
| P2P-QR null rule (postproc qr_null) | 136 | Oct-only blanked flag | no train rows; the mm=0/1 siblings are .74-.84 noisy qr, the rest is mostly noise |
| QRN (v3_rules) | 20 | merchant-QR, null flag, not a multiple of 100, isolated | about .8 (sibling mm=1 m10) |
| model, merchant-QR null round | 68 | triad segment n<=3 | about .8 |
| model, other | 272 | P2P-QR mm=1 or mm=0: 160; round merchant PAY / QR mm=0 triad: about 110 | P2P-QR about .85; round merchant about .8 |

Round-set merchant segments (RS definition, Oct, s13): n 1-3 hold 185 rows, all qr. n 5-8 hold 76 rows, all refund. n>=9 holds 71 rows, all collusion. This matches sc_refund / sc_collusion (train refund share at n=3 is 8%).

## (b) Oct structure: all null
- **Blanked or shifted qr cells** (s2, s9: every (tt, rt, mm, akind) cell with fraud share > .3, Oct vs train rate):
  - P2P-QR flag 1/0 cells are depleted by -184 rows. The null cells gain +140, and merchant-QR null non-m100 gains +21.
  - Net -23 rows against an expectation of 347, so about 1.2 sd. No qr volume is missing.
  - QR MERCHANT NA m1000 has an excess of +20 (z 2.1). It is decoded qr 57 / refund 23 / collusion 22 by segment size, and those are the known Oct collusion bursts.
  - The other big Oct depletion, PAY MERCHANT int (-300 rows, z -7), is the velocity pool. That is known: Oct has 7 bursts against 25 in Sep. Not qr.
- **Signature rows decoded legit**:
  - P2P-QR: 0 Oct rows decoded legit.
  - Merchant-QR null: 0.
  - Round merchant cells: 12 rows (PAY NA 6, QR mm=0 6).
  - In Aug and Sep the judge decodes 54 rows of these cells legit, and the cleaned label is legit on all but 1 (s6). Not a lever.
- **P2P-QR decoded phishing in Oct** (s8, s12): 10 rows, all int/under amounts with mm=0.
  - Train mm=0 int is phishing 24 of 31. Under is phishing 17 of 23. Aug/Sep judge precision in the same cell is about .8.
  - So these decodes are right.
  - The single null int row is left qr. The mm=1 int sibling is qr 7 of 9, and Oct phishing int/under volume is not depleted, so phishing was not blanked.
- **Multi-victim fake payees** (s4, s7, s11):
  - P2P fake payees are one-shot: 3217 of 3308 train receivers have exactly 1 qr row. Non-signature rows to signature receivers are qr at 0-5% and are already decoded qr when they are qr.
  - Merchant links within 1-72 h of a qr anchor: linked rows decoded legit have a qr label 0-6 times per month (rate about 0.1%).
  - Links through the same sender, sender-receiver pair or device within 1 h to 7 d: 0-7 qr labels per month among more than 1000 linked rows.
  - No partial decodes.
- **Zone versus hard signature** (s13):
  - In train, P2P-QR mm=1 rows sit on zone devices only in the qr zone (27 rows over Jul-Sep, all qr zone). The cell "hard qr signature on another type's zone device" has 0 train rows.
  - Oct has 1 such row: collusion zone, now collusion.
  - Moving it to qr gains +.00012 if it is qr and loses -.0003 if it is collusion. Break-even is P(qr) of about .71. Zone purity (.85) and signature purity (.84) put it near .5. Left alone.
  - Disclosure: an aggregate Oct print in s8 surfaced this one row.

## Not done
- Triad swaps in Sep (refund->qr 17, coll->qr 13 rows) were not chased through OOF deltas. Halved-density OOF splits segments, and full-density segment purity is already in sc_refund / sc_collusion.
