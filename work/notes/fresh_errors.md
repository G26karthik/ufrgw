# fresh_errors: error anatomy of ts_judge_h6 on trusted labels (13 Nov)

Scripts: `work/scratch_fresh_errors/e1..e17*.py` (e1 dumps judge arrays + row context to `e1_oof.parquet` / `e1_test.parquet`).
Truth `yt` = zone type on zone-device rows, else cleaned label `yc`. Decode = `ts_judge_post.get("h6")` (qo / qt).
Device classes: own (DV == sender number), base_other (DV < 420000, != sender; 4.8% of rows), pool, zone, fresh_legit.

## 1. Sep error buckets (ts_judge_h6; Sep macro on yt .5321, on yc .5224)
| bucket | rows | oracle macro gain | what it is |
|---|---|---|---|
| FN -> legit | 11,640 | +.424 | 11,588 (99.6%) are truth-singletons (no linked fraud row); 11,538 have p_legit .90-.99. Camouflage. Only 102 FN have p_legit < .9 (oracle +.0054). |
| type swaps | 601 | +.0446 | all on own-device rows (zone rows are exact by construction) |
| FP legit -> fraud | 518 | +.0095 | mostly residual label noise, see below |
Swap pairs (rows, oracle): ATO->sim 101 (+.0092), fc->ph 54 (+.0064), ph->fc 81 (+.0057), vel->mule 116 (+.0043),
mule->lay 91 (+.0037), loan->ph 18 (+.0030), coll->qr 13 (+.0017), refund->qr 17 (+.0015). Event-majority smoothing would fix 23 of 601.
- base_other devices: fraud rate 3.4%, the same as own devices. No signal. fresh_legit block: 11 fraud rows in Sep (noise).
- FP = noise: zone rows decoded fraud carry a legit label 7.1% (Sep) / 8.4% (Aug). Applied to non-zone fraud-decoded rows that gives
  400 (Sep) / 505 (Aug) expected noise rows against 518 / 517 cleaned-legit observed. Real FP is about 12-118 rows (<= ~.002 macro).
- Zone propagation (rows sharing sender / P2P receiver / reversed role with a zone row within 1 h-1 d): the model already gets them
  (pred_ok .93-.97). Overriding them to the zone type: -.0034..+.0002. Null.

## 2. Swaps are generator design, not noise
- Fan-in template (>=5 senders -> one P2P receiver in a 12 h segment, round-10 amounts): noisy labels mule 61%, layering 31%,
  velocity 8%, never any of the 8 other types (uniform 4% type noise would spread there). So a structured label choice.
  Velocity-labelled fan-ins are 15/16 decoded mule; they are slightly larger (median 18 vs 15 rows). Oct: 55 fan-ins, 42 decoded
  mule, 13 layering, 0 velocity.
- The other pairs match notes/typesplit_*.md (per-event type, no label-free separator).

## 3. Oct vs Sep (test-specific failure modes)
Zone rows are a label-free volume estimator: zone rows / labelled rows per class is stable Jul-Sep (about 3.1% for every
non-takeover class, sim .14, ATO .29, velocity .019 = 2.8% of non-pool velocity).
| class | Oct zone rows | Oct pred / est. volume | Sep pred / labelled |
|---|---|---|---|
| mule | 79 (57 expected) | .36 | .37 |
| refund | 22 (15 expected) | .39 | .44 |
| collusion | 12 (12.8 expected) | .19 | .32 |
| velocity | 28 (non-pool) | .20 | .42 (pool bursts 7 vs 25) |
| others | - | within .05 of Sep | |
- Base-decode recall on zone rows (clean truth, both months): non-takeover zones Sep .333 vs Oct .348 fraud-any. No sign of an
  Oct adaptation that the model misses.
- Caveat: Oct zone counts depend on edges that the prior partly sets (§5). The own-device decode rates are mule 1.20x, refund 1.06x,
  qr .97x Sep, so "mule/refund volume up" is weak.
- octcal open question (Oct 6-8 row round-amount merchant segments are over-full, 11 vs 6): 4 of them hold refund-zone devices
  (474462-474495) and none hold a collusion-zone device. That is consistent with refund, not proof: those rows are decoded refund
  and pull the edge themselves, and 474493 is 3 ids from the refund|fc edge. The independent support is octcal's 39/39 train segments.
- Velocity: device bursts (>=5 rows in 10 min) exist only on pool devices, 25 in Sep and 7 in Oct. No bots moved to other devices.

## 4. Artefacts found in the validation itself
- Hypothesis (inferred, not shown row by row): density.halves splits noise-flipped rows from their events in train, so OOF
  decodes them legit. This also deflates the .071 noise rate, so "12-118 real FP" is an upper range. Example: sim/ATO zone rows on
  multi-row devices decoded legit in Sep carry a legit label 31/31 (3-4 row devices). In Oct the model decodes 0 of 244 such rows legit.
  So Sep FN/FP buckets slightly flatter the noisy labels; the test is not hurt.
- Zone noise looks feature-dependent (camo zone rows unreported .163 vs core .078, rising with amount). This comes from the same
  halves artefact, not from the generator.
- Zone id order is not time order (Spearman dn vs first ts |rho| < .2 per zone), and occupancy is about 50%. There is no gap -> time leak.

## 5. The one test-specific bucket: zone edges (e18-e22)
Caveat on §1: `yt` on zone rows uses the judge's own DP edges, so the zone bucket is 0 by construction. Against the label-fit
TR_EDGES, the train DP differs on 30 Aug+Sep rows (16 Sep). Mostly mule|velocity: DP 423707 vs 423658, 17 rows, noisy label
velocity .94. art_zonemax: the oracle-edge ceiling on Sep is +.0006.
- Test, rows placed by the length prior alone (strictly between the last evidence device of zone k and the first of zone k+1):
  mule|vel 17, ph|qr 7, refund|fc 7, fc|loan 4, loan|coll 4, sim|ATO 171 (h6 cannot split sim/ATO; the edge was cross-checked by
  the victim gap-density changepoint), all others <= 2.
- DP on other models' test decodes vs h6: c2s6 / tun_best 0 rows changed, confusion_pairsfd 1, xgbC / gpu_xgb 7 (qr->ph 4),
  ts_comb 32 (ph->qr 10, refund->fc 6, qr->refund 5, mule->vel 4). ts_comb uses zone features, so it is partly circular.
- mule|vel disputed range [473749, 473783) holds 11 test rows (h6: 9 legit, 1 mule, 1 layering), all currently velocity.
  Occupancy changepoint: train full block 423660 (true 423658), but single-month device subsets are useless (423442..424004).
  Test changepoint 473757.
- Wrong-prior simulation (e20; Sep devices, 50% subsample, DP prior widths perturbed by the TEST widths' Poisson sd, 12 reps):
  correct prior +.0003 (sd .0004), perturbed prior **-.0030 (sd .0016, worst -.0056)**, mean edge error 22 ids (4 with the
  correct prior). This is pessimistic, because the sim's own prior is tighter in relative terms than the test prior.
  On test, the edges are the prior-driven part of the judge.
- Raw-column zone evidence (e21; P(txn_type|receiver_type|merchant_category | zone) fit Jul+Aug, scored on Sep zone rows, in nats
  per row): ATO|ph 2.09, ph|qr 1.11, qr|refund 1.44, refund|fc .87, fc|loan .52, vel|lay .44, lay|sim .25, mule|vel .18,
  loan|coll .15, sim|ATO -.01. It covers every zone row, including the ~65% that the model decodes legit and the DP now ignores.
- Adding that emission to the DP (e22; same perturbed-prior harness, paired reps, w = 0 / .5 / 1): **null to negative.** Sep cleaned
  delta: w=0 mean -.0040 over 4 reps, w=.5 -.0044, w=1 no better in any rep. Edge errors at fc/loan/coll got larger. Stopped
  after 4 reps (4/4 not better). The prior-error risk is real (about -.003 expected in the pessimistic sim), but this evidence
  does not fix it. Untested next options: a looser DP length prior (variance scaled up to the test Poisson sd), or pooling the
  decodes of several models as evidence.

## Verdict
After ts_judge_h6, the fixable headroom on Sep is at most ~.01-.015 macro: swaps .045 oracle but mostly design-random, true FP
<= .002, boundary FN .005. Camouflage (.42 oracle) is the only large bucket, and earlier work closed it. The one test-specific risk
is zone edges set by the prior: about 40 non-sim/ATO rows plus 171 sim/ATO rows, worth about +-.003. No validated fix exists yet.
The gap to the leader (.0045) is inside the public LB sd (~.010).
