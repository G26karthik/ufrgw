# v3_cells: systematic Oct-overrepresented cell scan (13 Nov)

Scratch: `work/scratch_v3_cells/` (cwd=work, `PYTHONPATH=.`).
- `f.py` builds `frame.parquet`: 21 discrete attributes per row, plus y, judge (qrm), and h6 p_legit/argmax.
- Attributes: txn_type, receiver_type, category, category consistency, mm {0,1,NA}, amount kind {nonint, int, m10, m100, m1000, just-under-round}, cents, magnitude, new_payee, pin, hour band, app, kyc, os, sim<48h, device<1d, away, login band, receiver degree, balance state, device kind {own, base_other, pool, zone, fresh_legit}.
- `s.py` is the first scan (all rows, 1-3 way).
- `s2.py` is the main scan:
  - It removes handled rows first: pool, zone, and QR + null mm + (P2P or non-integer).
  - It keeps a cell only if its ratio is at least 1.5x the best ratio among its (k-1)-subcells.
  - It reports distinct senders/receivers, the share decoded legit in the target, and the base fraud rate.
  - Two runs: Oct vs Jul-Sep, and a **Sep vs Jul+Aug calibration** (the null band).
  - k<=3 output is `s2_*.parquet`; k=4 output is `k4b/`.
- `nb.py` searches for the winning-rule shape. It takes train signature cells (n>=30, top class >=.7) and flips one attribute. A pair is flagged when the signature is depleted in Oct and the flipped neighbour has an excess.
- `post.py` builds the candidate.

## Results
- **Excess scan (k<=3, k=4):** every Oct cell with p<1e-5 is one of two kinds.
  - (a) Already handled:
    - P2P-QR null;
    - pool velocity at night (fewer, shifted bursts);
    - two collusion bursts with round 1000s amounts at merchants of degree >100, all decoded collusion (`m1.py`).
  - (b) Entity-diverse cells (>=5 senders and receivers) decoded legit, comparable to the Sep calibration null band:
    - k<=3: Oct's best is log10p -5.5; Sep's is -5.0.
    - k=4, cells with base rows: Oct's best is -7.6; Sep's is -8.6. Oct has 48 such cells, Sep 146.
    - k=4, cells with 0 base rows: the log10p comes from the 1e-3 floor and means nothing, so compare raw counts. Oct's largest is 5 rows (mm=0, amag=4, APP_C, dev<1d); Sep has cells of up to 8 rows.
    - These are mostly amag=4/5 cells.
    - They come from the festival x1.17 legit amount scaling: with amounts deflated by 1.17, the big-amount share per category returns to Sep's level.
    - Big amounts on fresh devices (<1 d): own devices, Oct 11 rows vs ~5.3 expected. Not significant after multiplicity.
- **Basic cell rates per 1e5 rows, by month:** very stable. Merchant category always matches the merchant (never false). The only nullable columns are category and mm.
  - mm is non-null outside QR only on fraud rows, and those cells are not depleted in Oct. So blanking touched only QR.
  - P2P-QR total is 122/1e5 in Oct vs 135-145: about -40 rows, ~2.4 sd.
- **Flip search (`nb.py`):** the only Oct pairs beyond the Sep null band are the known P2P-QR mm->NA (the positive control), pool, and the merchant-QR null family. Every legit-decoded neighbour scores weaker than -6.6; the Sep null reaches -16.
- **app_version / app within Oct:** zone/pool vs legit rows have the same version mix (.33/.33/.33) and app mix. Null.

## The one actionable cell: merchant-QR, mm null, amount a multiple of 10 but not of 100 (4 Oct rows, decoded legit)
Evidence comes only from aggregates and train sibling cells:
- **Paired shift.** Merchant-QR mm=1 with an m10 amount: 1.7 / .8 / 1.2 per 1e5 in Jul / Aug / Sep, 0 in Oct (~3.2 expected). Null mm with m10: 4 rows in Oct (1.5/1e5) vs 2 train rows in 3 months. This is the same shape as the P2P evasion.
- **Amount mix fits blanked qr_swap.** Train qr_swap merchant-QR mm=1 rows are nonint 58 and m10 15, and m10 is commoner at tiny merchants (d<=3: 5 of 11). The 17 nonint null rows (qrm rule) predict ~4.4 m10 null rows; observed 4. Legit mm=1 rows are m10 in only 4 of 1,602.
- **Not collusion.** The 40 train merchant-QR null rows whose amount is not a multiple of 100 all sit in bursts (>=2 null rows at one receiver), 38 of them collusion. The 4 Oct rows are 4 distinct receivers and 4 senders on own devices. Their receivers have no other judged-fraud test rows.
- **Isolation is a qr signature in train.** Train merchant-QR null rows that are the only null row at their receiver (all m100/m1000) are qr_swap 77% of the time:
  - over the whole train split: 71/92;
  - per month: 124/164;
  - within ±7 days: 155/201.
  - The rest is mostly legit (noise) and refund.
- **The non-m100 train rows stay in bursts at every window.** All 40 are in bursts whether isolation is counted over the whole split, per month or within ±7 days. So the 0-train-rows result does not depend on the longer train window.
- **Sep-testable sibling (`sepval.py`):** merchant-QR, mm=1, m10 amount, train. This is the strongest contrast.
  - Labels: qr 15/19 noisy (79%), legit 4. Per month .89 / .75 / .67. All 19 rows are isolated.
  - h6 OOF decodes 9 of the 10 Aug/Sep rows as legit.
  - Forcing these rows to qr_swap on the judge decode moves:
    - Sep noisy .5096 -> .5097;
    - Sep cleaned .5224 -> .5225;
    - OOS .5082 -> .5084;
    - Aug cleaned .5146 -> .5147.
  - That is positive, though tiny. The same sibling for non-integer amounts is only .22 (fresh_testside).
  - Oct has 0 mm=1 m10 rows, so this does not apply to test directly. It is the precision evidence for the null m10 rows.
- **Profile.** Grocery, receiver degree <=3/10/10/100. This matches the qrm 17 (small physical shops).

**Rule** (`post.py`, label-free, 0 train rows, asserted): QR & MERCHANT & mm null & amount % 100 != 0 & no other null-mm merchant-QR row at that receiver in the same split -> qr_swap.
- In test it hits 21 rows: the qrm 17, already qr_swap, plus 4 new. It subsumes the qrm rule.

**Candidate:** `subs/v3_cells.csv` = ts_judge_h6_qrm + 4 rows legit -> qr_swap.
- Expected gain: about +.0002 macro if precision matches the qrm rows (the fresh_judge simulation gives +.0008 for 16 rows at .77). Break-even precision is about .25.
- About 1 of the 4 rows is expected to be public, so the LB will not measure it. Bundle it into the combined candidate; do not spend an upload slot on it alone.
- The rule cell itself has 0 train rows. The mm=1 m10 sibling validates at +.0001 on Sep and +.0002 on OOS.
- Disclosure: `m1.py` printed row-level ids of the m1000 collusion bursts, which informed no rule. Every other Oct drill-down was aggregate.

## Verdict
The cell lens is saturated. Beyond the known evasion family (P2P-QR null, merchant-QR null non-m100), no Oct-only or Oct-overrepresented cell survives the Sep null calibration with a structural story. The large remaining gap is camouflage, not new cells.
