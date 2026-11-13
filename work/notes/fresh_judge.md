# fresh_judge: ranking of the fresh_* lenses (13 Nov)

Scratch scripts: `work/scratch_fresh_judge/j1-j5.py` (cwd=work, PYTHONPATH=.).

## Verified
- **A (testside), Oct merchant-QR rows with a null mismatch and a non-integer amount -> qr_swap.**
  - Train: 0 of 546k non-integer merchant-QR rows have a null mismatch. Oct has 17. ts_judge decodes 16 as legit, 1 as qr.
  - Train merchant-QR null rows (678, all integer): qr 275, collusion 206, refund 106, legit 78.
  - Share of rows at receiver degree <= 10:

    | group | share |
    |---|---|
    | Oct 17 | .82 |
    | train qr mm=1 non-int | .59 |
    | train qr null | .29 |
    | refund null | .23 |
    | collusion null | 0 |
    | legit mm=1 non-int | .085 |
    | legit mm=0 non-int | .085 |
    | train null rows that h6 argmax calls legit | .07 |

  - Rows on the sender's own device: Oct 17 .88, train qr mm=1 .90, legit .93. The two are consistent.
  - Reconcile with columns #3. In Aug/Sep, merchant-QR null rows that h6 argmax calls legit number n=14, with 2 fraud (14%). Their degree profile is legit-like (.07), so they are a different population from the 17. Non-QR mm=0 rows that h6 argmax calls legit: n=189, 39% fraud.
  - h6 is blind to this cell on train too. Train qr mm=1 non-int merchant rows: argmax legit 38/38, mean p_qr .09.
  - F1 simulation, with Sep h6 qr counts scaled x.5 (16 rows):

    | precision | macro delta |
    |---|---|
    | .77 | +.0008 |
    | .50 | +.0004 |
    | .25 | +.0000 |
    | 0 | -.0004 |

- **errors #1 (zone edges): the occupancy changepoint null does not transfer to test.**
  - The train block holds 3 months of devices. Any one month's devices are a ~1/3-thinned subset, so the single-month changepoint fails.
  - The Oct block is a complete one-month block, like the full train block. On the full train block the changepoint was exact (423660 vs 423658).
  - The test changepoint for mule|velocity is 473757. It lies inside the disputed range [473749, 473783).
  - Occupancy is therefore an untested emission for the test DP.

## Ranking
1. A as a post-rule (16 rows), uploaded alone on ts_judge_h6.
2. Occupancy-changepoint emission / prior inflation for the test zone DP (gated on the e22 harness, full-block simulation).
3. Columns stamp relabel (retrain, +0 to +.001).

Rejected:
- `--mv`: 1 row, already inside final1 (.5435).
- The 5 non-QR stamped rows: the Sep analogue is 39% fraud.
- Scoping-only items.
