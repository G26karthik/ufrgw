# sc_takeover: sim_swap + account_takeover on top of v3_final (13 Nov)

Scratch: `work/scratch_sc_takeover/`, with a1-a7 (`frame.parquet` = e2 judge dumps + v3_final test decode + session/drain cols). Rule: `work/sc_rules_takeover.py`. Candidate: `subs/sc_takeover.csv` (1 row vs v3_final).

## Verdict
**Takeover is saturated for Oct.** The best rule changes 1 test row, for about +.0001 expected. It is not worth an upload on its own, so stack it only if convenient.

## (a) Test sim/ATO decodes in v3_final
| type | source | rows | in drain session (>=2 drains) | sim<48h | dev<3d |
|---|---|---|---|---|---|
| sim | model (own phone) | 273 | 247 | 241 | 2 |
| sim | zone override | 92 | 81 | 10 | 76 |
| ATO | model | 45 | 45 | 1 | 2 |
| ATO | R2 rule (v3) | 4 | 4 | 2 | 0 |
| ATO | zone override | 190 | 174 | 3 | 173 |

Zone rows by model decode:
- sim zone: 73 ATO, 8 sim, 11 legit.
- ATO zone: 172 ATO, 2 sim, 16 legit.

Non-drain zone rows are pure in train: .84-.91 per month, which is the same as drain zone rows. The override stands.

## (b) Oct-specific structure: all null
- **Partial decodes.** Oct sessions holding a sim/ATO decode: 232 sessions, 605 rows. 0 rows in them are decoded legit, 0 sessions mix sim and ATO, and 1 row is decoded another class (loan). Aug/Sep have 26/15 legit-decoded drain rows in such sessions, labelled legit 25/26 and 13/15. That is the density.halves artefact, which is absent in test.
- **Fully missed drain sessions.** Off-zone Oct sessions with >=2 drains and no sim/ATO decode: 7 rows, all decoded fraud (loan/phishing/mule). No legit-decoded drain session exists.
- **Archetype mix** (off-zone, Aug/Sep; Jul has no OOF decode):
  - SIM own-phone: 70/22 sim/ATO.
  - "none": the model already splits it, at 42/48 ATO and 22/22 sim in Sep.
  - SIM on a non-own (base) device: sim/ATO 21/14 in Aug and 12/17 in Sep. This flips between months and is a coin flip over about 20 events. The 13 Oct rows are not flipped.
- **F1 marginal check.** Moving own-phone SIM events to ATO loses: the ATO gain is (p_a - F1a/2)/(Pa+Ta) ≈ .06/919, against a sim loss of .39/1045.
- **sim|ATO edge.** An id-occupancy changepoint (2-segment Bernoulli MLE; sim zone occupancy .70 vs ATO .26) gives Oct 473988, against the DP's 473987.
  - On train, the same MLE gives 425033 against the truth 425039, so accuracy is about ±6 ids.
  - This is a third estimator agreeing with the DP. The 1-id difference covers 1 non-drain row (DV0473987).
- **Oct cells** (P2P, integer, share > .4, sim48 or dev3, by new_payee):
  - The new_payee=0 S cell rises to 16.5 per 100k in Oct, against 9.5-11.8 in train. The excess rows look legit: single-row sessions, daytime, no Rs 500 rows, share median .56. This is festival share drift (takeover.md), not evasion.
  - new_payee=0 drain sessions (all flags 0) appear at the train rate: 8 Oct rows, all decoded sim.
  - No column has nulls in Oct.
- **Device reuse.** 0 Oct rows sit on train zone or train fresh-legit devices; only the velocity pool is reused.
- **Single drain-signature rows** (sd <= 1, sim48 or dev3, off-zone) decoded legit: 30/31 are labelled legit in Aug+Sep. Null; 12 Oct rows are left alone.

## (c) Rule SC1: session type consistency (`sc_rules_takeover.py`)
**Definition.** A drain row is eligible when:
- it is off the test zone block;
- it is decoded a fraud class other than sim/ATO;
- its session has >= 2 OTHER drain rows, all decoded one type t in {sim, ATO}.

Such a row is set to t.

**Train evidence, label-only.** A drain row next to >= 2 drain rows labelled t carries:

| | Jul | Aug | Sep |
|---|---|---|---|
| sim sessions: label t | .867 | .871 | .871 |
| ATO sessions: label t | .86 | .859 | .855 |

- A non-sim/ATO fraud label appears at .02-.03.
- Those labels spread over 8 types (sim sessions: mule 8, loan 6, phishing 5, ...; ATO sessions: lay 5, ph 5, ...). That is noise, not a concentrated type.

**Not validated out of time.** Decode-conditioned OOF hits are Aug 1 (phishing-decoded, labelled legit) and Sep 0. The rule was also spotted from a test session, then stated generally.

**Test.** 1 row, TXae079683, changes loan_app_extortion -> account_takeover. Its session has 4 drains in 15 min at 23h; the other 3 rows are decoded ATO.

**Expected macro.** The ATO gain is 2/(239+~680)·(p - .2)/12 ≈ +7e-5 at p = .6. Removing a probable loan FP from a 103-row class adds ≈ +2e-5. Total **≈ +.0001**.

## Not done
- No new upload candidate beyond the 1-row file.
- The S archetype (own phone, ~46% of events) stays at ~70% by design. All hunters, plus this pass, found no label-free separator.
