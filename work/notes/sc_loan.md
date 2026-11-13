# sc_loan: loan_app_extortion Oct-specific search (13 Nov)

Verdict: **NULL.** No rule is proposed and `work/sc_rules_loan.py` is the identity. No `subs/sc_loan.csv` was written, because it would equal v3_final.

Scratch: `work/scratch_sc_loan/` (cwd=work, PYTHONPATH=.):
- `p0` caches the judge h6 arrays (`judge_h6.npz`).
- `a1` builds the loan frame `fr.parquet`.
- `a2`-`a8` are the scans below.

Definitions (label-free):
- npi = new_payee, integer amount >= 1000.
- npie = npi with hour >= 20.
- lnpi = npi at a lending merchant.
- core = row is lnpi and its sender has >= 2 distinct lending lnpi receivers within ±30 d.
- E_npie2 = sender has >= 2 npie rows within 7 d, not core.

## (a) Test loan decodes in v3_final: 103 rows
| source | rows | structure |
|---|---|---|
| model | 80 | core |
| model | 16 | E_npie2, own device, P2P |
| zone | 3 | core |
| zone | 4 | none; model decoded them legit |
| rule | 0 | |

- Every one of the 83 test core rows is decoded loan.
- Non-core lnpi rows: 2, both decoded legit. Train has 0 loan among 32 such rows.
- Core siblings (other rows of core victims): 21, all decoded legit. Train is 95-99% legit, 0 loan.

## Train evidence (by month; OOF fold 0 = Aug, 1 = Sep)
- Core label share is loan .87/.82/.87 (Jul/Aug/Sep), the rest legit (noise). OOF recall is 1.00/.987, and precision of loan decodes is .82/.88.
- E_npie2 own-device loan decodes: precision .77/.81. Loan is never decoded as sim/ATO there.
- Non-structural loan labels (about 470 per month) are camouflage, recall .02.

## (b) Oct-specific structure: none found
- **Lending cells** (new_payee x amount kind x size x evening; rate per 1e5 rows):
  - Signature cell: 33.2/35.7/33.7 in train vs 33.8 in Oct.
  - Hour histogram of lnpi rows: same in Oct.
  - txn_type mix at lending merchants: same in Oct.
  - No blanked or shifted cell.
- **Receiver-kind flip** (multi-npie senders paying non-lending merchants): 1 train row in 3 months, 0 in test.
- **npie P2P excess in Oct** (multi 28 vs 19-24; single PAY 259 vs 212-225 per 1e5):
  - It sits in amounts 1000-2000, while amounts <= 500 are depleted. That is the festival x1.17 legit amount scaling.
  - Decode shares match Sep: legit .76 vs .73.
- **Partial decodes:**
  - Core events: none.
  - Own-device P2P loan-variant events: 1 test event mixes decodes (ATO, ATO, loan).
  - OOF has no fraud-type-mixed decode event to calibrate a smoothing rule. Not actionable (1 row).
- **Loan vs phishing on under-round new-payee P2P COLLECTs:**
  - Loan share .038/.037/.029 per month.
  - It is flat at .035 against every sender and receiver key: sender lending core, sender npie count, receiver npie P2P inflow, receiver repeat COLLECTs, evening.
  - Oct cell rate is 125 vs 121-136 per 1e5. Test decodes are 324 ph and 2 fc. Irreducible; keep ph.
- **Zone edges** (fc|loan, loan|coll): not reopened. fresh_errors.md shows extra edge evidence was null to negative.

## Gain translation
- P = 103 predicted loan rows. Actual Oct loan rows A is about 250-290 (83 core / .29, or 7 zone rows / .031). F1 is about .46.
- One net TP gives about +.0003-.0004 macro. One FP taken from legit costs about -.0001.
- Break-even precision is about .25 from legit, higher from another fraud class.
- No candidate reaches 3 net TP. Expected gain: 0.
