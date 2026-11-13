# fresh_testside: test-file-specific structure (13 Nov)

Scratch: `work/scratch_fresh_testside/`. `prep.py` builds `frame.parquet`: all rows, h6 OOF/test probs and ts_judge_h6 labels.
`a1`-`a8` are the analyses; each has a `.log` next to it. Run from cwd=work with `PYTHONPATH=.` (a5 also needs `scratch_art_zonemax`).

## Verdict
The test side is close to saturated. There is one small new lever (A); everything else is null or already used.

## A. 17 Oct merchant-QR rows with a null mismatch flag and a non-integer amount -> qr_swap (NEW, small)
In train, a null `qr_vpa_mismatch` on a merchant-QR row means fraud:
- 600/678 rows are fraud. The 78 legit rows are 11.5%, which is the fraud->legit noise rate.
- All 678 rows have integer amounts.
- Of 546k non-integer merchant-QR train rows, 0 have a null flag.

Oct has 17 such rows with non-integer amounts, a pattern train never shows (`a7b` triple scan: the only Oct-only cell besides the known P2P-QR evasion). Current judge decodes 16 of them as legit.

These are not ordinary legit draws:
- Receiver merchant degree is <=10 for 14 of 17 rows. The base rate for non-integer merchant QR in Oct is 8.4% (binomial p ~1e-12).
- One of the 17 rows (DV0474375) sits on a test qr_swap-zone device.
- Categories are food, grocery, pharmacy and fuel: physical shops, which fits "fake QR at a shop".

Reading: the Oct QR evasion (blank the flag) is applied by label. It also hit some qr_swap rows at merchants that otherwise look legit. The train qr_swap G rows at merchants already show a weak small-merchant excess: deg<=10 share .122 vs .085 for legit (4.5 sd).

Rule: `txn_type=QR & receiver_type=MERCHANT & qr_vpa_mismatch is null & amount % 1 != 0` -> qr_swap. It has 0 train rows, so it can sit in the existing `--qr-null` post-rule. It changes 16 test rows.
- Expected gain: qr F1 about +.005 to +.012, i.e. **macro +.0004 to +.0010**. Break-even precision is about .25.

Source check on train (`a9`). Non-integer merchant-QR rows with mm=1 are the core "fake QR at a small shop" variant:
- qr_swap has 58 such rows, 34 of them (59%) at merchant degree <=10.
- The 17 Oct rows are 82% at degree <=10, so they fit blanked qr mm=1 rows.
- They do not fit uniform blanking of all mm=1 rows, which would put about 10% at degree <=10. Nor do they fit qr camouflage, which puts 11% there.
- Expected Oct qr mm=1 non-integer merchant rows: about 11 (clean). 17 blanked rows means the flag on those rows was blanked at nearly 100%, plus a few others. Poisson noise allows this.
- No depletion is visible: the Oct mm=1 cell at degree <=6 has 17 rows against 17.5 expected (sd ~4), so the check is inconclusive.
- A sibling rule (mm=1, degree <=6, non-integer -> qr) gives only .22 noisy precision on both Aug and Sep. That is at break-even, and Oct's mm=1 cell has lost its qr rows to blanking. Not proposed.

Upload advice: stack A on `subs/ts_judge_h6.csv` alone. Do not stack it on final1 or bundle other changes. P(none of the 16 rows is public) = .7^16 = .3%.
- If the rows are right, public rises and this upload becomes the pick.
- If they are wrong, public falls and the incumbent stays.
So it is a near-free option worth one upload slot.

## Nulls (numbers)
- **Camouflage (G) in global time** (`a1`): G rows are no closer in time to same-class core rows than legit rows are. G/L ratio is .9-1.1 at windows of 1 min to 6 h, Aug and Sep. Timestamp-collision rates are equal (.0029 vs .0029).
- **Consistency bookkeeping** (`a6`): new_payee vs observed pair history and device_age vs first (sender, device) use are identical for G and L (np1_seen .0124 vs .0129, np0_first .707 vs .704, devage_short .094 vs .094).
- **Festival x1.17 amount scaling applies to camouflage too.** Oct zone rows with p_legit>.9 have log(amount / sender train profile) of .33 ± .10. Legit fresh-phone rows give .15 and unscaled would be 0. So the scaling gives no camouflage tell.
- **Split boundary** (`a3`): Oct test rows linked to Sep-started train fraud events (rows from 28 Sep on, plus the 147 Oct-dated train rows) via receiver, sender or device within 1-48 h. At most 4 such rows are judged fraud. Nothing to fix.
- **Rows per day**: Oct has 8.2-8.6k rows/day (Sep 16.7-17.3k), with judged fraud at 80-214/day. The 46 Nov rows (1-9 Nov) are all decoded fraud.
- **Entities first seen in Oct** (receiver/sender): fraud share 1.75%/1.87% vs 1.52% base. Sep has the same mild lift. Nothing new.
- **Oct-only value cells** (`a7`, `a7b`): pairs and triples of 19 discretised columns (txn/receiver type, mm, category presence/consistency, int/round amount, pin, new_payee, login, sim_age, device_age, home city, hour, kyc, os, receiver degree, overdraw, drain-all). Only the QR-null cells are flagged: the known 140 P2P-QR rows plus the 17 in A.
- **Non-QR rows with a non-null mismatch flag**: 7.5k train rows, 88% fraud (= 100% minus noise). They are a core-fraud generator signature, and h6 already uses it: OOF mean p_legit .10, and only 5 of 1,134 test rows are decoded legit.
- **Collusion splitting into 6-8-row segments in Oct?** (`a8`): no. The Oct 6-8 segments last 8-20 h. That matches train refund segments (median 14.6 h); collusion segments run a median 45.6 h. No segment has REFUND claims, in either class.

## Unchecked
- Predicted fraud by hour, Oct vs Sep.
- Senders of Oct zone rows vs their earlier-month fraud labels (art_acct found account label history null on Sep).

## Decision layer (per-class Oct counts)
- **Zone widths:** the Oct per-class widths are .48-.53 of a train month, but that is mostly by construction (the train layout is scaled into the block, then DP-refined). Only two things are independent:
  - the total block: 920 ids vs 1,818/month;
  - the used devices per class.
  Velocity has 28 used devices vs 60 in Sep (.47), a normal half month, while pool rows dropped to 269 (vs ~1,000/month). So only the velocity BURSTS shrank; the fresh-phone/camouflage velocity volume did not.
- **Unused/used id ratio** inside zones (sim .42/.42, ATO 2.81/2.81) is unverified as an event-mix signal. What consumes the unused ids is unknown, and the sim|ATO edge comes from the prior. The own-phone default of sim rests on train base rates only.
- **Per-class F1-optimal threshold** (F1/2 on a clean probability) is about .19-.24 for every class in Oct. Zone-share truth estimates (`a5`) are too noisy to set per-class biases: the macro estimate .449 is well below the public .544. This agrees with octcal: bias is not a lever.
- **Public/private**: the final pick is the best PUBLIC upload. A candidate that differs from ts_judge_h6 only in a few rows (A: ~5 public rows) is chosen only if its public score is >= .5443. A candidate whose changed rows all land in private ties at .5443, and the tie-break rule is unknown.
