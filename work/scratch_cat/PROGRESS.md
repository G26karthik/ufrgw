# Categorical handling experiment (scratch_cat) - PROGRESS

Started 14 Nov 23:27 IST. One agent, no subagents. Limits: 4 CPU threads total, GPU only if < 2000 MiB used (checked before each launch),
no new fit after 05:30 IST 15 Nov, report by 06:00.

## Declared before any fit (14 Nov 23:32 IST)
Arms (benchmark N, seeds / rows / weights / target identical inside each pair):
- `cLGBnum`: reference recipe C refit here with 4 threads (numrep.py A_C uses 8). If bit-identical to `PN_C_*`, reference = `C`.
- `cLGBcat`: same, nine `CAT_COLS` as int32 pandas columns with `categorical_feature=CAT_COLS`. All other settings default/unchanged.
- Second LightGBM arm: none declared (August halves differ by ~.010, August cannot choose between close arms).
- `cCBnum` / `cCBcat`: CatBoost, gbdt_n.py settings (MultiClass, depth 8, lr .1, l2 3, 1000 iterations, seed 0), GPU with
  `gpu_ram_part=.5` if the card is free at both launches, else both on CPU with 4 threads and fewer iterations. cCBcat = `cat_features=CAT_COLS`
  with CatBoost defaults for categorical handling (what actually ran is read from `get_all_params()`).
- Blends: `cBLGBcat`, `cBCBnum`, `cBCBcat` = 0.65 C + 0.35 arm (probabilities, before the decoder).
- Benchmark D: `cLGBcat` seeds 0,1 against `A_C` (or a 4-thread refit if the N refit is not bit-identical), plus its blend.
Status rule (same as scratch_gap/numrep_rule.txt): "supported" only if the September after-decoder change is positive with the
paired event-bootstrap 5%..95% interval above zero on BOTH N and D and the August after-decoder change is >= 0 on both. One of two =
"promising but unconfirmed". Interval across zero = "no detected difference". Interval wholly below zero = "measurably worse".
An arm with N only can reach "promising but unconfirmed" at most.

## Steps
- [done 23:30] step 0: `inspect_cols.py` -> `inspect_cols.log`. Nine columns stored as Float32 integer codes, no NaN/null, 2..30 levels,
  no held-only codes; 'NA' level used by merchant_category, rcv_kyc_level, mer_category.
- [done 23:32] scripts `lgb_cat.py`, `cb_cat.py` written.
- [done 23:33] `prod_schema.py` -> `prod_schema.log`: every cached production matrix stores the nine columns as Float32.
- [running since 23:32] `cb_cat.py cCBnum N gpu 1000` -> `cb_cCBnum_N.log` (GPU was at 0 MiB at launch).
- [23:34] COORDINATOR INSTRUCTION: GPU is needed by the main job until about 03:00 IST. The running cCBnum GPU fit may finish; no further
  GPU fit until nvidia-smi shows < 500 MiB used. LightGBM arms and CPU work first.
  Plan change (declared before any CatBoost score is read): (a) LightGBM arms next, on CPU, after cCBnum ends (4-thread total limit).
  (b) CatBoost CPU pair `cCBnumCPU` / `cCBcatCPU` (4 threads, fewer iterations, same count in both arms) is fitted after the LightGBM work.
  (c) If the GPU shows < 500 MiB before 05:30, `cCBcat` is fitted on GPU to complete the GPU pair with `cCBnum`; otherwise the GPU pair is
  reported as not evaluated. One FREEZE and one September reading for all CatBoost arms.
- [done 23:35] cCBnum N GPU (fold0 57 s, fold1 95 s) -> scratch_gap/PN_cCBnum_{aug,sep}.npy. August argmax .4670 (C .4680). GPU 49 MiB after it ended.
- [running since 23:36] LightGBM chain: lgb_cat.py cLGBnum N, cLGBcat N, cLGBcat D -> lgb_<arm>_<BM>.log
- [23:41] STEP 1 RESULT (reference recipe, N fold0, seeds 0 and 1; `lgb_cLGBnum_N.log`, `models/N_cLGBnum_f0_s*_inspect.json`):
  `pandas_categorical=None`; all 54,000 splits are `<=`; the nine columns have numeric feature infos (`values: []`), 3,467 splits,
  3.0% of total gain (seed 0). They are ordered numbers in the fitted model. 4-thread refit is bit-identical to `PN_C_aug.npy`
  (max abs diff 0) -> reference = `C` on N (`A_C` on D) if September also matches.
- [done 23:50] cLGBnum N: bit-identical to `PN_C_aug.npy` and `PN_C_sep.npy` (max abs diff 0) -> reference is `C` on N and `A_C` on D; no D refit needed.
- [done 23:58] cLGBcat N -> `scratch_gap/PN_cLGBcat_{aug,sep}.npy`. Booster check: `==` splits on all nine columns (6,735 of 54,000 splits, fold0 seed0),
  `pandas_categorical=[]`, no `<=` split on them. LightGBM dropped some rare levels from its category list (see `lgb_cLGBcat_N.log`).
- [done 00:05] cLGBcat D -> `scratch_gap/PD_cLGBcat_{aug,sep}.npy`.
- [00:05] GPU check: 5228 MiB, 99% (main job). No GPU fit. Blends `P{N,D}_cBLGBcat_*` written (0.65 C/A_C + 0.35 cLGBcat).
- [running 00:06] August: `aug_dec.py N C,cLGBcat,cBLGBcat` -> `aug_N_lgb.log`; `aug_dec.py D A_C,cLGBcat,cBLGBcat` -> `aug_D_lgb.log`.
- [done 00:11] August after decoder (before decoder in brackets), harness check C .4978 / A_C .5004 reproduced:
  N: C .4978 (.4696) | cLGBcat .4975 (.4693), change -.0003 | cBLGBcat .4987 (.4709), change +.0009
  D: A_C .5004 (.4716) | cLGBcat .5001 (.4713), change -.0003 | cBLGBcat .4996 (.4701), change -.0008
- FREEZE (LightGBM) Sun Nov 15 00:11:20 IST 2026: files scratch_gap/P{N,D}_cLGBcat_{aug,sep}.npy and P{N,D}_cBLGBcat_{aug,sep}.npy are final; no setting was or will be
  changed after this line; no second LightGBM arm. September is read once with:
  `bench_eval.py N C,cLGBcat,cBLGBcat _cLGB` + `dec_diff.py dec_N_cLGB.npz C cLGBcat,cBLGBcat`
  `bench_eval.py D A_C,cLGBcat,cBLGBcat _cLGB` + `dec_diff.py dec_D_cLGB.npz A_C cLGBcat,cBLGBcat`
  Status rule: as declared at the top (numrep_rule.txt form). August change of cLGBcat is -.0003 on both benchmarks, so by the declared
  rule cLGBcat cannot reach "supported" whatever September shows.
- [done 00:17] SEPTEMBER, LightGBM arms (`sep_N_lgb.log`, `diff_N_lgb.log`, `sep_D_lgb.log`, `diff_D_lgb.log`, `scratch_gap/dec_{N,D}_cLGB.npz`):
  N (ref C .4815 / .5070): cLGBcat .4814 / .5072, change +.0003 (-.0005..+.0012), rows 72 / corrected 27 / newly wrong 35, missed fraud recovered 12, new legit called fraud 24.
     cBLGBcat .4813 / .5068, change -.0002 (-.0008..+.0005), rows 22 / 10 / 10, recovered 3, new legit called fraud 0.
  D (ref A_C .4824 / .5078): cLGBcat .4819 / .5073, change -.0005 (-.0023..+.0014), rows 114 / 45 / 56, recovered 1, new legit called fraud 5.
     cBLGBcat .4836 / .5088, change +.0010 (+.0002..+.0018), rows 62 / 23 / 29, recovered 12, new legit called fraud 23 (fake_cashback +.009).
  Status by the declared rule: cLGBcat alone = no detected difference (N and D). cBLGBcat = no detected difference (N interval across zero;
  D September interval above zero but D August change is -.0008, so D does not pass either).
- [done 00:17] `pooled.py` -> `pooled.log`: in cLGBcat LightGBM leaves the rarest level out of its category list for txn_type (5), merchant_category (8),
  mer_category (8) and, in 6 of 8 boosters (not fold1 seed1 on N or D), snd_bank_code (26): 0.7% to 1.0% of rows each. Nine-column gain share: numeric 3.0%, categorical 5.1%.
- [00:17] GPU check: 2351 MiB, 100% -> no GPU fit.
- [running 00:18] `cb_cat.py cCBnumCPU N cpu 1000` -> `cb_cCBnumCPU_N.log` (timing first; if 1000 iterations is too slow on CPU, both CPU arms get the same lower count).
- [00:35] stopped my own cCBnumCPU process (pid 19916/27904, 17 min, no fold finished, output buffered so speed unknown). Relaunching unbuffered (python -u) to measure the speed.
- [00:35] CPU speed: first iteration 3.4 s on fold0 (121k rows), projected 56 min for 1000 iterations on fold0 alone. Stopped my process (pid 23252/35956).
  DECISION (before any CPU CatBoost output exists): CPU pair runs 250 iterations in both arms (all other settings unchanged: depth 8, lr .1, l2 3, seed 0, 4 threads).
- [running 00:36] chain `cb_cat.py cCBnumCPU N cpu 250` then `cb_cat.py cCBcatCPU N cpu 250` -> `cb_cCBnumCPU_N.log`, `cb_cCBcatCPU_N.log`.
- [done 01:03] cCBnumCPU N (250 iterations; fold0 10m50s, fold1 16m51s) -> scratch_gap/PN_cCBnumCPU_{aug,sep}.npy. August argmax .4579 (C .4680). cCBcatCPU running since 01:03.
- [01:33] cCBcatCPU fold0 done (30 min): CPU defaults that ran: one_hot_max_size 2, simple_ctr Borders(TargetBorderCount 11)+Counter, combinations up to 4 (info json). August argmax .4594 (cCBnumCPU .4579). fold1 running.
- [done 02:27] cCBcatCPU N (250 iterations; fold0 30 min, fold1 53 min) -> scratch_gap/PN_cCBcatCPU_{aug,sep}.npy. Blends PN_cBCBnum, PN_cBCBnumCPU, PN_cBCBcatCPU written. GPU 2523 MiB -> still no GPU fit.
- [running 02:28] August: aug_dec.py N C,cCBnumCPU,cCBcatCPU,cBCBnumCPU,cBCBcatCPU -> aug_N_cbcpu.log ; aug_dec.py N cCBnum,cBCBnum -> aug_N_cbgpu_num.log
- [done 02:31] August after decoder (before decoder in brackets), N, C = .4978 (.4696):
  cCBnumCPU .4917 (.4627) | cCBcatCPU .4915 (.4627) | cBCBnumCPU .4980 (.4696) | cBCBcatCPU .4978 (.4698)
  cCBnum (GPU, 1000 it) .4945 (.4665) | cBCBnum .4979 (.4699)      (logs aug_N_cbcpu.log, aug_N_cbgpu_num.log)
- [02:31] GPU 2523 MiB, 99%. WAITING for the GPU (< 500 MiB) to fit `cCBcat` N on GPU (about 3 min), then, if time allows, a GPU pair on D
  (`cb_cat.py cCBnum D gpu 1000`, `cb_cat.py cCBcat D gpu 1000`, about 6 min). Cutoff 04:15: if the GPU is not free by then, FREEZE and score
  September with the CPU pair only and report the GPU pair as not evaluated. No September CatBoost score has been read.
  NEXT STEPS to resume: (1) GPU fit(s) if free; (2) `blend.py N C cCBcat cBCBcat`; (3) `aug_dec.py N cCBcat,cBCBcat`; (4) FREEZE line;
  (5) `bench_eval.py N C,cCBnumCPU,cCBcatCPU,cBCBnumCPU,cBCBcatCPU[,cCBnum,cCBcat,cBCBnum,cBCBcat] _cCB`, `dec_diff.py dec_N_cCB.npz C <all>`;
  (6) `bench_eval.py N cCBnumCPU,cCBcatCPU _cCBcpu` (+ `cCBnum,cCBcat _cCBgpu`) and dec_diff for categorical vs numeric CatBoost directly.
- [02:39] GPU 100 MiB (< 500, two readings 02:38:48 and now) -> launching cb_cat.py cCBcat N gpu 1000 -> cb_cCBcat_N.log
- [done 02:41] cCBcat N GPU (fold0 ~50 s, fold1 ~65 s) -> scratch_gap/PN_cCBcat_{aug,sep}.npy; params in N_cCBcat_f*_info.json. August argmax .4674 (cCBnum .4670).
- [running 02:42] GPU pair on D (GPU checked before each launch): cb_cat.py cCBnum D gpu 1000, cb_cat.py cCBcat D gpu 1000 -> cb_cCBnum_D.log, cb_cCBcat_D.log
- [done 02:44] GPU pair on D: `PD_cCBnum_*`, `PD_cCBcat_*` (August argmax .4679 / .4687). GPU defaults that ran in cCBcat: one_hot_max_size 2,
  simple_ctr Borders(TargetBorderCount 1) + FeatureFreq, combinations up to 4.
- [02:45] DECLARED before any CatBoost September score: replicate GPU pair with seed 1 on N (`SEED0=1` -> `PN_cCBnums1_*`, `PN_cCBcats1_*`), reported as a
  second paired reading of categorical vs numeric (GPU fits take about 2 min; one seed leaves CatBoost seed noise unmeasured). No blend is built from seed 1.
- [02:46] COORDINATOR: GPU queue finished, GPU may be used (100 MiB at 02:39). Deadlines unchanged.
- [done 02:48] seed-1 GPU pair on N: PN_cCBnums1_*, PN_cCBcats1_* (August argmax .4673 / .4672). Blends PN_cBCBcat, PD_cBCBnum, PD_cBCBcat written. ALL FITS DONE; no further fit planned.
- [running 02:49] August: aug_dec.py N cCBcat,cBCBcat,cCBnums1,cCBcats1 -> aug_N_cbgpu.log ; aug_dec.py D cCBnum,cCBcat,cBCBnum,cBCBcat -> aug_D_cb.log
- [done 02:52] August after decoder (before decoder in brackets); logs aug_N_cbgpu.log, aug_D_cb.log:
  N (C .4978): cCBcat .4959 (.4676) vs cCBnum .4945 (.4665) = +.0014 | cBCBcat .4978 vs cBCBnum .4979 | seed 1: cCBcats1 .4961 (.4683) vs cCBnums1 .4943 (.4659) = +.0018
     CPU 250 it: cCBcatCPU .4915 vs cCBnumCPU .4917 = -.0002
  D (A_C .5004): cCBcat .4974 (.4685) vs cCBnum .4946 (.4650) = +.0028 | cBCBcat .4981 (.4690) vs cBCBnum .4992 (.4696) = -.0011
- FREEZE (CatBoost) Sun Nov 15 02:52:18 IST 2026: all CatBoost files scratch_gap/P{N,D}_cCB*_{aug,sep}.npy and P{N,D}_cBCB*_{aug,sep}.npy are final; no CatBoost setting was
  chosen on any score; no further fit. September is read once with scratch_cat/sep_cb.sh (bench_eval N with reference C for all ten N arms, D with reference
  A_C for the four D arms; then categorical vs numeric directly: bench_eval with the numeric arm as reference for GPU seed 0, GPU seed 1, CPU, and the two
  blends on N; GPU seed 0 and the two blends on D; dec_diff for each).
  Status rule as declared at the top. For "categorical vs numeric CatBoost" the reference is the numeric CatBoost arm.
- [done 03:09] SEPTEMBER, CatBoost arms. Logs: sep_N_cb.log, diff_N_cb.log, diff_N_cb_direct.log, sep_N_cb_direct_{gpu,s1,cpu,blend}.log, sep_D_cb.log, diff_D_cb.log,
  diff_D_cb_direct.log, sep_D_cb_direct_{gpu,blend}.log; decode files scratch_gap/dec_{N,D}_cCB.npz (+ _cCBgpu, _cCBs1, _cCBcpu, _cCBbl).
  Against C on N (C .4815 / .5070), September before / after decoder, change after decoder (5%..95%), rows changed / corrected / newly wrong:
    cCBnum     .4791 / .5049  -.0020 (-.0038..-.0003)  191 / 70 / 102      cCBcat     .4783 / .5044  -.0026 (-.0047..-.0005)  219 / 81 / 114
    cCBnums1   .4772 / .5032  -.0038 (-.0068..-.0013)  271 / 83 / 157      cCBcats1   .4777 / .5035  -.0035 (-.0062..-.0013)  234 / 77 / 134
    cCBnumCPU  .4789 / .5041  -.0029 (-.0045..-.0012)  202 / 62 / 121      cCBcatCPU  .4778 / .5036  -.0034 (-.0052..-.0014)  201 / 66 / 115
    cBCBnum    .4818 / .5076  +.0006 (-.0008..+.0023)  126 / 68 / 41       cBCBcat    .4813 / .5071  +.0001 (-.0017..+.0020)  153 / 79 / 52
    cBCBnumCPU .4816 / .5069  -.0001 (-.0010..+.0009)   71 / 33 / 30       cBCBcatCPU .4808 / .5064  -.0006 (-.0015..+.0003)   71 / 34 / 31
  Against A_C on D (A_C .4824 / .5078):
    cCBnum     .4789 / .5051  -.0027 (-.0050..-.0001)  239 / 99 / 117      cCBcat     .4776 / .5038  -.0040 (-.0065..-.0012)  264 / 94 / 147
    cBCBnum    .4817 / .5078  -.0000 (-.0017..+.0020)  122 / 75 / 39       cBCBcat    .4787 / .5051  -.0027 (-.0054..+.0001)  185 / 65 / 98
  Categorical vs numeric CatBoost directly (reference = numeric arm), after decoder:
    N GPU seed 0  -.0005 (-.0020..+.0009)  146 / 61 / 62     N GPU seed 1  +.0003 (-.0012..+.0019)  148 / 73 / 56     N CPU 250 it  -.0005 (-.0023..+.0018)  169 / 82 / 72
    N blends      -.0005 (-.0014..+.0005)   91 / 41 / 41     D GPU seed 0  -.0013 (-.0028..+.0000)  167 / 57 / 92     D blends      -.0027 (-.0051..-.0010)  119 / 17 / 86
  Status: categorical vs numeric CatBoost = no detected difference on N (three pairs) and on D alone (leaning worse); the D blend with the categorical arm is
  measurably worse than the D blend with the numeric arm. Every CatBoost arm alone is measurably worse than C. Blends vs C: no detected difference.
- NEXT: final report. No job is running. Nothing in work/subs, oof, preds, feats_cache was written.
- [03:12] Notes for the report: CatBoost numeric seed spread on N .5049 (seed 0) vs .5032 (seed 1), categorical .5044 vs .5035: larger than any categorical-vs-numeric
  difference. August favoured categorical in the three GPU pairs (+.0014, +.0018, D +.0028); September did not. cCBnum is not a reproduction of PN_sCAT.
  Files written outside the PN_c/PD_c prefix by bench_eval.py: scratch_gap/dec_N_cLGB.npz, dec_D_cLGB.npz, dec_N_cCB.npz, dec_D_cCB.npz, dec_N_cCBgpu.npz,
  dec_N_cCBs1.npz, dec_N_cCBcpu.npz, dec_N_cCBbl.npz, dec_D_cCBgpu.npz, dec_D_cCBbl.npz. September readings added by this run: 12 variants on N, 6 on D.
  subs/dd_final.csv SHA-256 8e04353d...29cd unchanged. DONE 03:12 IST; final report delivered.
