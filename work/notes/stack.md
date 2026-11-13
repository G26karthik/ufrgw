# stack: second-stage stacking on entity-aggregated stage-1 probs (stack agent, 13 Nov)

Code: `work/stack.py` (`s1`, `s2`). Scratch: `work/scratch_stack/`:
- `p1.npy` + `p1_ids.parquet`: stage-1 probs for all rows, in matrix order.
- `p1_tail_m{7,8,9}.npy`: test-tail probs per LOMO model.
- `top.json`: top 30 raw features by stage-1 gain.
- `oof_<variant>_r<N>.npy`: stage-2 pooled OOF probs.
- `results_*.json`, `s1.log`, `s2_v1.log`, `s2r.log`, `calib.py`.

**Verdict: no gain. Stacking does not beat tun_best. No stack artifacts or sub were written; keep `subs/tun_best_ppq.csv`.**

## Setup
- **Stage 1 (`s1`, 15 min):** leave-one-month-out tun_best models.
  - Setup: matrix_m5dm, relabel C, train.LGB, 150 rounds, seeds 0-2.
  - Each model is fit on the other two of Jul/Aug/Sep, plus the 147 Oct train rows.
  - Jul, Aug and Sep rows each get probs from the model that never saw that month.
  - Test gets the mean of the 3 LOMO models.
- **Stage 2 features (178):**
  - Own 12 probs.
  - The top 30 raw features.
  - Aggregates of OTHER rows' stage-1 probs on 4 keys: sender, device, P2P receiver and merchant.
    - Windows: ±1h, ±1d, ±15d.
    - Each window has n, max p_fraud, mean p_fraud and the count with p_fraud > .5.
    - The 1d and 15d windows also have the mean prob of each of the 11 classes.
    - The neighbour pool is month x density half (`density.halves`) for train rows, and all test rows for test. Every neighbour prob therefore comes from a model that never saw that month, and the density matches Oct.
    - `_selftest()` checks the windowed n, max and mean against brute force.
- **Stage 2 model:** LGB (train.LGB), legit downsample .2 (`train.sample_fit`), relabel C target, `train.FOLDS`, 3 seeds.
  - Plain: fit from scratch.
  - `r` variants: boost from `init_score = log(stage-1 probs)`, so the trees only learn corrections.

## Calibration (train OOF vs test)
- Test stage-1 means, avg-of-3-LOMO vs the full model (`preds/tun_best_test.npy`): equal to within .0001 per class.
- Sharpness also matches:
  - q99.5 p_fraud: .943 for avg-of-3-LOMO, .949 for the full model.
  - Mean max-fraud prob where p_fraud > .5: .816 for avg-of-3-LOMO, .824 for the full model.
- Jul/Aug/Sep OOF compared with test: same per-class means, except velocity (.0052 vs .0043) and qr (.0047 vs .0042). Both drops are the known Oct drift.
- No mismatch fix is needed.

## Results
- All rows use the same protocol: 3 seeds, bias tuned on pooled OOF.
- "oos" means the bias is tuned on the Aug fold and the Sep fold is scored with it.
- c_ = cleaned proxy (`clean_labels.clean`).
- "+pp" = after `postproc.apply` (arch forcing; qr-null cannot be validated, so it is not applied).

| candidate | pooled | Sep | c_pooled | c_Sep | oos Sep | c_oos Sep |
|---|---|---|---|---|---|---|
| tun_best | .4719 | .4793 | .4842 | .4916 | **.4787** | **.4909** |
| tun_best +pp | .4721 | .4797 | .4843 | .4920 | **.4791** | **.4913** |
| lomo_s1 (stage 1 alone) | .4722 | .4791 | .4845 | .4914 | .4773 | .4896 |
| noagg r150 (scratch, own+raw) | .4670 | .4745 | .4793 | .4869 | .4700 | .4824 |
| stack r150 (scratch, +aggs) | .4686 | .4756 | .4807 | .4877 | .4726 | .4849 |
| stack r100 / r250 | .4682 / .4682 | .4756 / .4745 | .4803 / .4803 | .4878 / .4867 | .4717 / .4716 | .4839 / .4839 |
| rnoagg r30 (resid, own+raw) | .4721 | .4788 | .4844 | .4911 | .4735 | .4857 |
| **rstack r30** (resid, +aggs) | .4724 | .4796 | .4848 | .4920 | .4762 | .4885 |
| rstack r30 +pp | .4727 | .4800 | .4850 | .4925 | .4765 | .4889 |
| rstack r75 / r150 | .4714 / .4709 | .4776 / .4773 | .4838 / .4832 | .4901 / .4897 | .4743 / .4741 | .4867 / .4864 |

Reading:
- **A stage 2 trained from scratch loses about .005.** Re-learning the decision surface from probs on noisy labels is worse than the stage-1 model.
- **The aggregates add a little.**
  - Scratch: +.0016 pooled and +.0026 oos over noagg.
  - Residual: +.0003 pooled and +.0027 oos over rnoagg.
- **The best variant does not win.**
  - rstack r30 is +.0005 pooled and +.0004 cleaned over tun_best, inside seed noise (±.0013).
  - Its out-of-sample decode is **-.0025 (noisy) and -.0024 (cleaned) below tun_best**.
  - Extra rounds only overfit: r30 > r75 > r150.
- This fails the rule (pooled AND oos must win), so nothing is promoted.
- The result matches `camo.md` and `postproc.md`:
  - Missed (camouflage) rows have no fraud-predicted neighbours. Within ±1d, sender .0024 against .0027 for legit.
  - Caught core rows are already coherent through the module features.
  - Learned aggregation therefore has almost nothing left to add on top of event smoothing.
- Side note: lomo_s1 Aug rows come from a Jul+Sep model, so lomo and stacked pooled numbers on the Aug fold are not strictly comparable with tun_best (Jul-only Aug fold). The Sep fold is comparable, and Sep shows no gain either.

## Reproduce (cwd=work, PYTHONUTF8=1 POLARS_MAX_THREADS=3)
```
python stack.py s1 --seeds 3 --threads 3                                                    # 15 min, ~3 GB
python stack.py s2 --seeds 3 --threads 3                                                    # base,noagg,stack @100/150/250, ~64 min
python stack.py s2 --seeds 3 --threads 3 --variants rnoagg,rstack --checkpoints 30,75,150   # ~40 min
```
Peak RAM of s2: about 3.0 GB (178-col frame over 1.82M rows).

## Not tried
- 2-hop (graph) aggregation, i.e. aggregates of neighbours' aggregates.
- Label-propagation features: neighbours' noisy train labels. Those would leak across folds.
- Feeding the aggregates into the main 182-feature model as extra columns. That needs a third cross-fit level to keep the probs OOF.
