# v3_final: combined accepted v3 rules (13 Nov)

`subs/v3_final.csv` = `subs/ts_judge_h6_qrm.csv` + 12 rows. Built by `work/v3_rules.py` (reusable: any decoded sub + load_all).
Change log: `notes/v3_final_changes.csv`. Checks: `scratch_v3_final/` (chk_cells*.py, chk_zprop_oof.py, chk_final.py + logs).

## Verdicts
| key | verdict | test rows | why |
|---|---|---|---|
| cells (QRN) | ACCEPT | 4 legit -> qr_swap (rule hits 21; 17 = qrm rows already qr_swap) | re-verified below |
| zprop R2+R3 | ACCEPT | 4 sim -> ATO (R2), 4 lay -> mule (R3a 2, R3b 2) | Sep +.0005 noisy / +.0004 cleaned / +.0005 OOS, Aug +.0009; label-only purity 1.0 every month |
| zprop R1 | not shipped | 0 guarded (15 unguarded, a bet on the mule/vel band) | depends on the disputed edge |
| dpocc | REJECT | 0 | fails its preregistered acceptance (unperturbed loss, <1 sd) |

## Independent re-verification
**QRN** (`chk_cells.py`, `chk_cells2.py`, train aggregates + Oct counts only):
- Train isolated non-m100 null-mm merchant-QR rows = 0 under per-split, per-month and +-7 d isolation. Test = 21 under all three.
- Train non-isolated non-m100 null rows: 38 collusion, 1 legit, 1 phishing (bursts). Train isolated null rows (all m100): qr 71/92.
- Blanking hit fraud rows only. Oct nonint mm=0 is 91,175 rows with 17 null nonint rows, so the legit blanking rate is <= 2e-4. That predicts <= 0.02 legit null rows among Oct's 100 mm=0 m10 rows, and there are 4.
- Legit mm=1 nonint is normal in Oct: 261 vs about 280 expected.
- mm=1 m10: 9/4/6 per month in train (15 qr, 4 legit), 0 in Oct (about 3.2 expected). Null m10: 4 in Oct.
- So the 4 rows look like blanked mm=1 m10 rows. Precision is about 0.8 (train sibling) or higher, since the blanking hit fraud only.
- Sep sibling forced to qr_swap: Sep noisy +.0001, cleaned +.0001, OOS +.0002, Aug +.0001.

**zprop** (re-implemented in `v3_rules.zprop`):
- OOF replication (`chk_zprop_oof.py`, judge decode, judge zones): Sep noisy .5101 / cleaned .5228 / OOS .5087 (judge .5096 / .5224 / .5082). Aug cleaned +.0009. Changed rows: Aug 16 (14 correct), Sep 6 (5 correct). This matches the proposer's numbers.
- Reproduces `subs/v3_zprop.csv` exactly.
- DP edges recomputed from the qrm decode equal the judge `et`.
- Windows were chosen on Aug and scored on Sep. Decode-filtered Aug/Sep for the shipped direction types:
  - lay -> mule: 8/8 correct;
  - sim -> ATO: 10/11 correct, the miss is cleaned-legit.
- R2 test sources sit 20 and 145 ids inside the ATO zone. The sim|ATO edge 473987 is agreed by both estimators.
- R3 sources are below the disputed band.

## Combination
- Rows are disjoint: QRN hits merchant-QR rows decoded legit/qr_swap; zprop hits rows decoded mule/vel/lay/sim/ATO. Overlap is asserted.
- Precedence if overlap ever occurs: QRN last.
- `v3_rules.py subs/ts_judge_h6.csv` (pre-qrm) gives labels identical to v3_final, with QRN 20 legit -> qr_swap. QRN therefore replaces the separate qrm override.

## Expected gain
- QRN: about +.0002 (4 rows at precision about .8).
- zprop: about +.0005 to +.001 (Oct classes are about half Sep's size, so each row counts roughly double).
- Total: about +.0007 to +.0012 test macro.
- Public LB: fewer changed rows land in public (about 1/3), but each counts more there, so the expected public delta is about the same as the full-test figure, only noisier. A public result near 0 would not refute it.

| candidate | rows vs qrm | expected test delta |
|---|---|---|
| ts_judge_h6_qrm (base) | 0 | public .5459 |
| v3_cells only | 4 | about +.0002 |
| v3_zprop only | 8 | about +.0005 to +.001 |
| **v3_final** | 12 | about +.0007 to +.0012 |

- **Against the goal:** the leader is at .5488, a gap of .0029. v3_final covers roughly 25-40% of that gap and does not take first place by itself. More gains are needed.
- Rebuild zones on the final edges before re-running if the mule|vel band is moved. Drop BAND where the estimators then agree.
