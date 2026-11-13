# idea_bag (13 Nov evening): probability-level variance reduction / diversity. Verdict: NULL, nothing passes the gate

Scripts: `scratch_idea_bag/bag.py` (extra c2s6-recipe seeds, saved per seed in `scratch_idea_bag/seeds/`),
`scratch_idea_bag/ev.py` (blend -> ts_judge_post.get, i.e. identical qr_null decode + pooled bias + zones + bait;
paired bootstrap of the OOS diff, 300 reps, row-level and sender-cluster). Log: `scratch_idea_bag/ev1.log`.

Baseline c2s6 judge: Sep noisy .5101, cleaned .5228, OOS Aug->Sep .5091.

| candidate | noisy | cleaned | OOS | OOS boot sd (row/sender) | test rows vs ts_judge_h6 |
|---|---|---|---|---|---|
| sB6 = 6 new seeds (6-11), same recipe | -.0001 | -.0000 | -.0005 | .0003/.0003 | 26 |
| c2s12 = c2s6 + sB6 (arith) | -.0002 | -.0002 | -.0002 | .0003/.0003 | 19 |
| c2s12 geometric mean | -.0002 | -.0002 | -.0005 | .0003/.0003 | 19 |
| c2s6 + h6 (1:1) | -.0002 | -.0002 | +.0001 | .0003/.0003 | 11 |
| c2s6 + xgbC (3:1) | +.0001 | +.0001 | +.0000 | .0004/.0004 | 23 |
| c2s6 + xgbC (1:1) | -.0008 | -.0009 | -.0012 | .0008/.0009 | 53 |
| c2s6 + xgbC rank-average | -.32 | -.32 | -.32 | - | 258k (breaks the log-prob bias decode) |
| c2s6 + gpu_cat (3:1) | -.0003 | -.0004 | -.0013 | .0008/.0009 | 26 |
| c2s6 + confusion_pairsfd (1:1) | -.0003 | -.0003 | -.0007 | .0005/.0004 | 18 |

- Two independent 6-seed sets (c2s6 vs sB6) differ by <=.0005; 12 seeds do not beat 6. Seed variance is already averaged out.
- Same-feature / older-feature model blends (xgb, catboost, older matrices) are null or worse, consistent with bl_1_1.
- Test zone edges are unchanged for every prob-average candidate (only rank-average moved them).
- Not run: ff/bagging-diverse seeds (d100+), CPU xgb on core2, dart. Reason: 12-vs-6 seeds is null, so prob variance
  is not the bottleneck; each 6-seed set costs ~45 min under 5-agent CPU contention; dart needs more rounds at lr .05.
- subs/idea_bag.csv = judge on c2s6+xgbC 3:1 (best-looking, still a FAIL: OOS +.0000). 23 rows differ from ts_judge_h6,
  18 from ts_judge_c2s6. Not recommended for upload; it is noise-level relative to public-LB sd .01.
