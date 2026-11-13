# sc_final: judge pass over class sweeps (13 Nov)

`subs/sc_final.csv` = `subs/v3_final.csv` + RS (3 rows) + FCB (5 rows). Built by `work/sc_final.py`. Checks: `work/scratch_sc_final/v1.py`, `v2.py` (+ logs).

| rule | source | rows | change | verdict | EV macro |
|---|---|---|---|---|---|
| RS (= SEGR) | sc_rules_collusion.py | 3 | collusion -> refund | accept | +.0003 at P(refund)=.8 (break-even .64) |
| FCB | sc_rules_fc.py | 5 | qr_swap -> phishing_collect | accept | +.0004 (about +.0001 per row at zone purity .85) |
| SC1 | sc_rules_takeover.py | 1 | loan -> ATO | reject | about 0 |
| loan | sc_rules_loan.py | 0 | - | null | 0 |

Total EV is about +.0006 (range +.0004 to +.0009). The target was +.004. About 2 of the 8 rows are public, so the leaderboard will not show this.

## Re-verification
- **RS**: round-set merchant segments are cut on whole data with a 24 h gap.
  - Train 4-8 cell: collusion labels are 0 in Jul, Aug and Sep. Refund is 81/108 in Sep.
  - 9-12 cell: refund is 0 in every month.
  - The smallest collusion-majority segment is n 11, 10 and 12 for Jul, Aug and Sep.
  - The hit merchant's last round row is 485 h before the test end, so the segment is not truncated at the end of the data.
  - SEGR changes the same 3 ids, so it is applied once.
- **FCB**: bait rows in the train qr zone are 0 (72/86/72 rows per month). Bait rows sit in the ph and fc zones (15/15/17 and 4/11/9) and once in ATO.
  - The edge moves to 474360 at every penalty from -1 to -16.
  - The 3 contiguity rows (474352/353/356) are model-legit and carry no hard qr signature (merchant with mm null or 1).
- **SC1 rejected** for three reasons:
  - The rule was spotted from a test session.
  - Its only decode-conditioned OOF hit is labelled legit.
  - The loan E_npie2 decode precision (.77-.81) contradicts the label-only ATO .86.
  - EV: +.00024 if ATO, -.00036 if loan, +.00006 if legit. The sign flips at P(loan) of about .35.

Gain arithmetic uses T_test = Sep T x (n_test / n_Sep): ph 1105, qr 1202, refund 549, collusion 430. P comes from v3_final.
