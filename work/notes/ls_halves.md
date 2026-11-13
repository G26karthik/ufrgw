# ls_halves: link legit-labelled event rows into their event's half (13 Nov). FAILED the gate, closed

Idea: `density.halves` keeps cleaned fraud events whole, but legit-labelled rows that belong to an event go to a random half.
About 12% of fraud rows are flipped to legit, so these rows lose their event context about half the time
(scratch_cleanest/d_leak.py).
Fix: a label-free link. A cleaned-legit train row takes its event's half when it meets one of these:
- same sender within the event span +- 1 h;
- same device within the event span +- 1 h;
- same P2P receiver key (`rk`) within the span.

## Code
- `density.py`: `halves(..., link=False)` and `run_feature_matched(..., link=False)` -> cache `<name>__dm3.parquet`.
  - New helper `_legit_links`, constant `LINK_W = 3600`.
  - The link is an override applied after all RNG draws. With link=False the output is byte-identical (asserted against
    the pre-edit output for mlink 0 and 24 h, scratch_ls_halves/check.py).
  - The mtime of density.py was restored, so the existing __dm/__dm2 caches stay valid.
- Caches: feats_cache/{flow,takeover}__dm3 (mlink 0) and {burst,scam,ring}__dm3 (mlink 24 h).
- Matrix: feats_cache/matrix_core2swap_dm3.parquet. It is matrix_core2swap with those module columns swapped.
  Column order is unchanged and the core2 full-density columns are untouched (scratch_ls_halves/mk_matrix.py).
- Training: tag `ls_halves`, the exact c2s6 recipe (relabel C, 150 rounds, 6 seeds 0-5, threads 3).
  Logs: notes/train_ls_halves_{val,full}.log.

## Linkage facts
- 2783 legit rows are linked. 1423 of them change half (mlink 24 h), spread evenly over Jul/Aug/Sep (~460 each).
- Precision is low. Of the moved rows with a known structural type (scratch_cleanest ys), only 13% are fraud.
  Most linked rows are genuine legit traffic of the event's sender/device.
- Moving 1.4k rows nudges hub/merchant count features on ~1/3 of all rows (median relative change ~3e-4).
  Test rows: 88k rows changed by tiny amounts. Test zone edges are unchanged. Judge test decode differs on 16 rows.

## Gate (ts_judge_post.get decode, paired with c2s6; scratch_ls_halves/gate.py, gate.log)
| | Sep noisy | cleaned | OOS Aug->Sep |
|---|---|---|---|
| c2s6 judge | .5101 | .5228 | .5091 |
| ls_halves judge | .5090 (-.0010) | .5216 (-.0012) | .5078 (-.0013, paired boot sd .0005) |
| Sep moved rows masked: c2s6 | .5115 | .5242 | .5105 |
| Sep moved rows masked: ls_halves | .5116 (+.0001) | .5242 (-.0001) | .5103 (-.0001) |

Raw model results:
- pooled biased .4793 vs .4800;
- oos.py mean .4783 vs .4793.

Moved Sep rows (452):
- Decoded-fraud share rose from .144 to .272.
- Catch on the truly-fraud rows (n=59) rose from .42 to .88.
- The extra fraud calls on the genuine-legit majority are false positives against both the noisy and the cleaned labels.

## Verdict
The fit effect is null: the masked deltas are about 0. The literal gate fails for an evaluation reason:
- Re-attached rows now look like fraud.
- The moved rows are cleaned-legit by construction, so they are legit in both `y` and `yc`.
  Every extra fraud call on them is a false positive in all three metrics, including the 59 truly-fraud ones.
- Test events are never split, so test features barely move.

No sub was produced. train.py's raw decode was moved out of subs/ to scratch_ls_halves/raw_decode_ls_halves.csv.
The 16 changed test rows are within the seed-set range (19-26 rows, idea_bag.md), so the change is LB-neutral.
Not checked: the OOS sd comes from a row-level bootstrap, not a sender-cluster one. Only Sep moved rows are masked;
Aug moved rows still feed the Aug-tuned bias. `link` stays in density.py as an off-by-default option; nothing uses it.
A tighter link (sim/ATO/layering only) would touch fewer than 100 truly-fraud Sep rows, so it is not worth a retrain.
