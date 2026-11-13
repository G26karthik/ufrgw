# idea_prior: Oct class priors from device-zone counts. NULL, gate FAILED

Scripts: `work/scratch_idea_prior/` (`z1.py` zone chronology, `p1.py` counts + kill check -> `base.npz`, `p2.py` gate; log `p2.log`).
Baseline: ts_judge_c2s6 (Sep noisy .5101, cleaned .5228, OOS .5091).

## The idea as stated does not exist
- Train zone ids are NOT chronological: the corr of device id vs first use inside each zone is -.14 to +.09.
  Jul, Aug and Sep share one zone block, so no zone has a per-month width.
- Usable analogue: first-use fresh devices per zone per month (DP edges `eo`/`et`), normalised by month row volume.
- Velocity is excluded. Most bots live in the reused pool 420000-422999, so the fresh zone gives 1.02x Oct/AugSep. octcal measured 0.48x per row.

## Kill check (labels, Sep vs Aug): the zone ratio does not track the true class-rate ratio
- Non-zone label-rate ratios Sep/Aug are .94-1.06 for every class, so the true priors barely move.
- Zone-count ratios scatter from .80 to 1.29. This is Poisson noise: 7-140 devices per class per month.
- Spearman zone vs non-zone label ratio: .21 (p .53); .40 (p .26) without velocity.

## Gate (pre-set: lam in {0.5, 1}, log ratio clipped to +-.5, non-zone rows only, then zone + bait override)
| | Sep noisy | cleaned | OOS Aug->Sep | paired OOS diff (200 boot) | test rows vs ts_judge_c2s6 |
|---|---|---|---|---|---|
| ts_judge_c2s6 | .5101 | .5228 | .5091 | - | - |
| lam .5 | .5100 | .5227 | .5088 | -.00024 sd .00020 | 4 |
| lam 1 | .5099 | .5226 | .5086 | -.00043 sd .00023 | 7 |

Both settings lose on every metric. `subs/idea_prior.csv` (lam 1) is written for the record only: 16 rows differ from ts_judge_h6, 7 from ts_judge_c2s6. Do NOT upload.
This agrees with octcal.md: the Saerens EM prior shift changed 8 rows, and bias shifts of +-.2 are flat or negative. Decode priors are not a lever.
