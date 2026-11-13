# typesplit_bait: phishing_collect vs fake_cashback vs loan_app_extortion (13 Nov)

Verdict: **no hidden label-free rule found.** The ambiguous cores are irreducible from the data. A small consistency rule exists, but the model already does it: test rows changed 2, Sep clean +.0003.
Code: `work/scratch_ts_bait/b1..b11.py`, `make_sub.py`. Module: `work/feats/ts_bait.py` (`ts_bait__rule`). Sub: `work/subs/ts_bait.csv` (2 rows differ from h6_ppq).

## Bait events (pair f->v with a tiny int P2P, plus every v->f row)
Train events by shape (nf = f->v rows, nb = v->f rows), event-majority label:

| nf/nb | events | ph | fc | all-legit |
|---|---|---|---|---|
| 1/0 | 1499 | 273 | 1 | 1197 |
| 2/0 | 336 | 152 | 1 | 173 |
| 1/1 | 372 | 152 | 146 | 66 |
| 2/1 | 192 | 1 | 161 | 28 |

- Inside an event the two rows agree: ph/ph 115 and fc/fc 103, against only 2 ph/fc mixes. So the type is drawn per event, it is not per-row noise.
- The fc generator is complete without the ph-answered events:
  - fc with nf=1 and dt<=180 is 104 events; fc with nf=2 and dt<=180 is 108.
  - fc dt>180 share is .30 for nf=1 and .33 for nf=2.
- The phishing "answered bait" path only ever has nf=1 and dt<=180 min (dt 8-178).
- So two genuine generator paths overlap on nf=1, nb=1, dt<=180. That is the ambiguous core: ~250 train events, 96 test rows, fc share .40.
- Scanned on the ambiguous core with no separation (fc share flat):
  - Payback: txn_type, amount and its quantiles, amount/balance (.909 artefact is the same in both), last digit.
  - Timing: bait and payback hour, dt shape inside 0-180, seconds-of-minute.
  - Devices and apps: own device for f and for v, sim_age, device_age, login_to_txn, pin, app, os, kyc.
  - Cities: home and device city relations.
  - Activity: fraudster degree and repeat events, other fraud rows of f and v within ±3 d.
  - Fraudster accounts are shared across types: fc fraudsters also receive under-round phishing COLLECTs. Receiver-ring identity therefore does not give the type.
- LGB on all 190 matrix features, fit Jul+Aug and scored on Sep, ph vs fc on payback rows: AUC .614. Nearly all of that comes from `scam__bait_before_min` (dt>180 = fc), which the model already uses.
- The only stable extra signal is bait amount 5: fc share is .42 vs .25 for ph in Jul, Aug and Sep. That gives a likelihood ratio of 1.7, i.e. posterior ~.54 fc. It is not enough to cross a decision.
- Sep ambiguous rows (163, clean labels):
  - Predicting all ph scores accuracy .601.
  - h6 accuracy is .436, because it predicts fc ~80% of the time.
  - Forcing ph there LOSES macro-F1 (-.002), so h6's fc lean is F1-optimal (fc is the smaller, lower-F1 class).

## Under-round new-payee P2P COLLECTs (ph .69 / fc .074 / loan .033 / legit .175)
- Type noise to the 8 other fraud types is ~0.5% each, matching uniform 4% noise. fc (10.6% of ph) and loan (4.8%) are 10-20x above that, so they are not uniform noise.
- Their amount pivot (1950/1990/1999/4950/.../24999) and hour histogram are the same shape as ph.
- They are singletons: no loan or fc sibling rows on the victim or the receiver within ±7 d.
- LGB on matrix features, Sep AUC: ph vs fc+loan .506, ph vs loan .42. Irreducible. They are either structured mislabels (the clean test says ph, which we already predict) or a variant identical to ph.

## Loan core (for context)
- Loan core = a victim paying 2-6 different `lending` merchants within 30 d. h6 recall there is .95-1.0 (snd_loan30>=2), and .977 on the lending category.
- 931 of 1283 Sep-pooled loan rows are singletons at ordinary merchants or mega-merchants. They are camouflage, with recall .005.
- Loan rows show no pair repetition (pair_n14 = 0 for 97%).

## Rule tested (out-of-time; no fitted parameters)
`ts_bait__rule`:
- 1 = unanswered bait -> ph.
- 2 = answered with nf>=2 or dt>180 or a known payee -> fc.
- 3 = ambiguous.

| fold | rows in rule-1/2 events | rule acc | h6 acc | macro noisy | macro clean | rows changed |
|---|---|---|---|---|---|---|
| Aug | 327 | .997 | .933 | .4730 -> .4738 | .4853 -> .4861 | 11 |
| Sep | 332 | .994 | .931 | .4855 -> .4857 | .4979 -> .4982 | 11 |

- The override applies only to rows decoded ph, fc or loan. Most h6 misses in those events are legit predictions.
- Forcing legit-decoded rule-2 rows to fc is neutral (4 rows per fold).
- Oct rule counts (rows) are 273 / 105 / 96 for rules 1 / 2 / 3. Sep is 608 / 196 / 198, so Oct runs at about half, matching Oct's half event volume. Consistent.
- Test: **2 rows change** (fc -> ph). Not worth an upload slot on its own. It can be folded in if another post-rule sub is made.
