# bc_phishing: phishing_collect Oct dive (13 Nov)

Verdict: **saturated. No rule is shipped. subs/bc_phishing.csv is not written.** The class cannot supply any of the +.004 target.
Code: `work/bc_rules_phishing.py` (MIX rule; it has `apply()` but is rejected). Scratch: `work/scratch_bc_phishing/`.
- `prep.py` builds `frame.parquet` (judge_h6 pickle, h6 probs, sc_final decode, ts_bait rule).
- `s1`-`s14` are the analyses. `chk_mix*.py` check the rule; their `.log` files sit next to them.

## (a) Test rows decoded ph in sc_final: 506
| source | rows |
|---|---|
| model, off-zone (rule 0) | 369 |
| model, unanswered bait (rule 1) | 85 |
| model, ambiguous bait (rule 3) | 9 |
| ph zone override (model said other or legit) | 21 |
| ph zone, model agrees | 15 |
| off-zone, not model: FCB edge move (sc_final) | 5 |
| off-zone, not model: bait rule-1 override | 2 |

By row type: COLLECT P2P 368, PAY P2P 114, QR P2P 10, merchant rows 12.

Segment table (Sep cleaned T / P / TP versus Oct P; Oct is about half a month):

| segment | Sep T | Sep P | Sep TP | Oct P |
|---|---|---|---|---|
| under-round COLLECT P2P np=1 | 421 | 597 | 421 | 317 (of 326 rows) |
| bait rule 1 | 147 | 155 | 132 | 87 |
| zone | 77 | 81 | 73 | 41 |
| rest | 1374 | 73 | 61 | 44 |

The rest bucket is camouflage: of 1,374 Sep rows, 1,305 decode legit.
- Every segment's Oct/Sep ratio is .5-.6, which is volume-consistent.
- Exception: bait rule 3 has Oct P 9 against Sep 36. This is the model's fc lean, which is F1-optimal (typesplit_bait).

## (b) Oct structure: all null
- **Core cell** (COLLECT P2P, new_payee=1, amount R-1/R-10/R-50 for R in 2k, 5k, 10k, 25k):
  - Train 718/673/617 per month, Oct 326, which is the expected half volume.
  - 324 of the 326 Oct rows decode ph and 2 decode fc. No core row is decoded legit.
  - The amount mix is unchanged. There are no null-rate shifts in COLLECT P2P columns, and txn/receiver type shares are identical.
- **Balance artefact** (balance_before = 1.1 x amount; 2,341 train rows, 87% fraud, ph 892):
  - Oct rate is .00157 (train .0015).
  - All 410 Oct rows decode fraud, 0 legit.
  - The type split follows the cell, and decodes match the train majority in every cell.
- **Oct-only amount cells** (COLLECT np 0/1, PAY P2P np=1): only festival-scaling shifts in legit amounts. No ph value is new or depleted.
- **Under-round PAY P2P np=1**: Oct decodes ph on 2 of 34 rows, against 13 of 51 in Sep.
  - The Sep ph rows are mostly x999 amounts carrying the 1.1 balance artefact.
  - The Oct legit-decoded rows are 1950/1990 with ordinary balances and p_ph < .22. In train, 1950/1990 PAY rows are the legit/mule pattern.
  - Only 2 Oct x999 rows exist, against about 4 expected, and both are decoded ph. Nothing is decoded legit there that should flip.
- **Receiver fan-in**, among rows decoded legit that enter a receiver of a core row within 7 d: clean ph share 0/96 (Aug), 2/108 (Sep). Break-even for legit -> ph is about .23. Null. The victim's other rows (within 1 h, and inbound within 1 d) give 0 ph. Fraud receivers rarely repeat: 119 linked core rows in 3 months.
- **ATO|ph zone edge (474284)**: single-row devices are ~.95 "old" (device_age > 1.5 d) in every zone, ATO included. Device age cannot place the 3 legit-decoded single rows at 474278/281/283. Only multi-row takeover devices are fresh.
- **Bait rule 1** (unanswered bait): Oct ph share among events .26, Sep .23. Consistent.

## (c) Candidate rule MIX: rejected
Rule: in a rule-3 bait event (off the zone block) whose decode splits ph/fc, set both rows to ph.
- Test: 7 rows fc -> ph.
- Found on Sep: 13 guarded rows, clean ph 10 / fc 2 / legit 1, macro +.00045.
- Aug (the real holdout): 0 events survive the zone guard. All 8 unguarded Aug rows are zone-partner propagation, which is already handled.
- Prob-level Aug check (`chk_mix2`): off-zone rule-3 events where a row has p_ph > p_fc are ph 19 / fc 19 rows (k>=1). That is below the break-even.
- Break-even P(ph | ph or fc) = .645.
  - Per row: ph truth +.000135, fc truth -.000245, legit +.00003 (Sep T/P scaled to test).
  - EV for 7 rows: -.0004 at p .5, 0 at .64, +.0004 at .8.
  - Realistic range: about -.0002 to +.0004. Not worth an upload slot or the variance.

## Per-class leverage (Sep cleaned, scaled to test)
| class | T | P | TP | F1 |
|---|---|---|---|---|
| ph | 1095 | 498 | 372 | .467 |
| fc | 417 | 188 | 121 | .399 |
| loan | 326 | 104 | 90 | .421 |

Ph recall is capped by camouflage: rest-segment recall is .04, and the model already gets core rows at precision ~.7-.85.
