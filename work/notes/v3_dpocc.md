# v3_dpocc: occupancy emission + length-prior variance inflation for the zone DP (13 Nov)

Implements fresh_judge.md item 2. **Verdict: REJECTED. No candidate sub written** (`subs/v3_dpocc.csv` does not exist).

Scripts: `work/scratch_v3_dpocc/` (cwd=work, `PYTHONPATH=.;scratch_v3_dpocc`)
- `dpocc.py`: DP with generic evidence. Adds a Bernoulli occupancy emission w*log P(used | zone k) and a prior variance var_k. Has a self-check.
- `sim.py MODE`: the harness.
- `post.py W F [--write]`: test re-decode on top of ts_judge_h6_qrm.
- Logs: `repro.log`, `o1.log`, `simA_s0.log`, `simB_s0.log`, `simC_s0.log`. Results: `sim_*_s0.parquet`.

## Setup
- The w=0, var=exp DP reproduces both rows of `scratch_fresh_errors/edges.npy` exactly (train and test).
- Occupancy is the share of ids used per zone.

  | zone | mule | vel | lay | sim | ATO | ph | qr | ref | fc | loan | coll |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | train | .53 | .35 | .45 | .70 | .26 | .57 | .34 | .49 | .55 | .50 | .46 |
  | test (judge edges) | .68 | .33 | .49 | .70 | .26 | .53 | .42 | .65 | .63 | .50 | .52 |

  - Logit gaps after the level match have sd .26. That sd is the per-rep occupancy perturbation.
  - Non-takeover zone devices hold exactly 1 row each. No device spans two months.
  - Monthly new-device counts per zone are about Poisson (sd/sqrt(mean) .1-1.4). There is no train basis for f>1.
- Variance is set in test units: var_k = f*exp_k on test, and f*exp_k*N_block/N_test in the sim. f=1 is the current test judge.
- The e20 "stiff" prior (var=exp at train scale) is 5.9x tighter, yet it does not explain the old -.003. In A, stiff scores -.00081 and is better than w0f1 (-.00095), with 7 wins and 0 losses. The visible difference from e20 is evidence density: e20 used a 50% Sep-device subsample, A uses all Aug+Sep decodes.
- Fixed grid: (w, f) in {(0,1) baseline, (1,1), (0,3), (1,3)}.
- Acceptance:
  - paired mean of Sep cleaned delta > 1 sd of the paired differences;
  - and the unperturbed case does not lose.

## Results (paired vs w0f1)

| harness | reps | w1f1 paired mean / sd | w1f3 | w0f3 | unperturbed w1 |
|---|---|---|---|---|---|
| A: full block 423000-428454, used = all train devices, preds Aug+Sep, prior perturbed by test Poisson rel sd, occ logit +-.26 | 12 | +.00078 / .00116 (0.67 sd), wins 10, loses 2 of 12 | +.00077 / .00115 | 0 / 0 (identical) | -.00013 (misAll 30->32) |
| B: as A, used = Aug+Sep devices only (bracket) | 12 | +.00091 / .00095 (0.96 sd), wins 11, loses 1 of 12 | +.00085 / .00101 | 0 / 0 | w1f1 -.00022, w1f3 +.00018 |
| C: test scale, random windows of Poisson(test width) ids per zone, proportional prior | 16 | +.00006 / .00033 (0.2 sd), lost 5/16; **TAU=0 rerun** (windows already carry binomial occupancy noise, logit sd about .2-.5 at 14-117 ids, so TAU double-counted): +.00012 / .00033 (0.36 sd), won 12, lost 4; misassigned rows -5.0 +- 8.3 | +.00003 / .00041 (TAU=0: +.00006 / .00036) | -.00001 / .00006 | -.00026 (misAll 0->11; the baseline is error-free by construction here, so a candidate can only tie or lose) |

- Every cell fails the >1 sd bar. The unperturbed case loses in A and C.
- A is the required harness. There w1 fails both conditions independently: 0.67 sd, and the unperturbed run loses -.00013.
  - If "1 sd" is read as standard error, A passes (t about 2.3). The unperturbed loss still rejects it.
- The effect is real but small: w1 wins 10/12 (A) and 11/12 (B), +.0008-.0009 each.
- At test scale (C) the result is inconclusive: 0.2-0.36 sd, with 12/16 wins at TAU=0.
- C is the closest analogue of Oct (920 ids). Misassigned rows there are -3.3 +- 7.3.
- Edge errors:
  - Occupancy helps the edges with occupancy contrast: qr|ref, err 8 -> 3-4.
  - It hurts the small zones: ref|fc 9 -> 10-12, loan|coll 8 -> 9.
  - mule|vel stays at 49-51 even with occupancy. Mule predictions inside the velocity zone dominate the 0.07 nats/id of occupancy contrast. The `--mv` grouping (art_zonemax) is the real fix there.
- Variance inflation does nothing useful:
  - w0f3 is identical to w0f1 in A and B.
  - In C it is -.00001 / .00006.
  - With occupancy on, w1f3 is no better than w1f1.
- With full Aug+Sep evidence, the perturbed-prior cost on the full block is about -.001 (A, w0f1 -.00095), not -.003.

## Test stakes (informational only; run after rejection, not used for selection)
- `post.py 1 1` moves the edges by 1-4 ids: [0, 2, 3, 4, 1, -1, 2, 0, 0, -1, 0, 0].
- 8 test rows would change:
  - qr->ph 2
  - vel->mule 2
  - sim->lay 1
  - ATO->ph 1
  - lay->vel 1
  - ATO->sim 1
- 7 of the 8 rows are decoded legit by h6. The upside is at most about +-.0005.
- sim|ATO moves 473987 -> 473988 (1 id).

## Next
- The zone-edge line is closed for the judge grouping: art_zonemax, e22 and this note all came out null.
- Remaining zone headroom on test is <= ~.001.
- Untested (post hoc, left to the orchestrator): occupancy emission combined with the `--mv` mule+velocity grouping.
  - mule|vel stays at 49-51 ids of error here, because the judge grouping lets mule predictions drag the edge.
  - The stake is at most the 11 test rows in the disputed range [473749, 473783).
