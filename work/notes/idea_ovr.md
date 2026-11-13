# idea ovr: one-vs-rest specialists for weak classes (13 Nov) -- NULL, gate failed

Scripts: `work/scratch_idea_ovr/` (yfit.py caches relabel C target; fit_ovr.py binary LGB per class on
matrix_core2swap, same folds as train.py, fixed 200 rounds, LGB dict, legit frac .2; eval_ovr.py geometric mix
logP_k <- (1-w)logP_k + w log s_k on c2s6 raw probs, w per class greedy on Aug only (Aug bias, Aug score, baseline
label-free zones), then the unchanged ts_judge pipeline; write_sub.py). Logs: k1.log, eval_k1.log.

Kill test: fake_cashback(9), loan(10), collusion(11), 1 seed.
- Specialist ranks the same as the multiclass column. Sep AUC multi/ovr: fc .656/.652, loan .661/.653,
  collusion .631/.632. AP within +-.005.
- Aug greedy picked w fc=.75, loan=0, collusion=0. Aug .5033 -> .5034, which is noise.

| | Sep noisy | Sep cleaned | OOS Aug->Sep |
|---|---|---|---|
| ts_judge c2s6 (baseline, reproduced) | .5101 | .5228 | .5091 |
| ovr_k1 + judge | .5101 | .5228 | .5090 |

Per class (cleaned Sep): phishing .467 -> .470, fc .399 -> .396, all others unchanged.
Test: 10 rows differ vs the c2s6 judge, 21 vs subs/ts_judge_h6.csv. Sub: subs/idea_ovr.csv (do not upload).

Verdict: the OvR specialist sees the same information as the 12-class model, so it adds nothing.
Stopped before mule, phishing and a second seed set, because the AUC tie leaves nothing to tune.
sim and ATO were skipped: archetype forcing and the zone override overwrite those columns.
