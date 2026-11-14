# September, once, all CatBoost arms (run after the FREEZE line in PROGRESS.md). cwd = work. Two sequences, 2 polars threads each.
export PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=2
PY=../.venv/Scripts/python.exe; S=scratch_cat
NALL=cCBnum,cCBcat,cBCBnum,cBCBcat,cCBnumCPU,cCBcatCPU,cBCBnumCPU,cBCBcatCPU,cCBnums1,cCBcats1
(
  $PY scratch_gap/bench_eval.py N C,$NALL _cCB > $S/sep_N_cb.log 2>&1 &&
  $PY scratch_gap/dec_diff.py dec_N_cCB.npz C $NALL > $S/diff_N_cb.log 2>&1 &&
  $PY scratch_gap/dec_diff.py dec_N_cCB.npz cCBnum cCBcat > $S/diff_N_cb_direct.log 2>&1 &&
  $PY scratch_gap/dec_diff.py dec_N_cCB.npz cCBnumCPU cCBcatCPU >> $S/diff_N_cb_direct.log 2>&1 &&
  $PY scratch_gap/dec_diff.py dec_N_cCB.npz cCBnums1 cCBcats1 >> $S/diff_N_cb_direct.log 2>&1 &&
  $PY scratch_gap/dec_diff.py dec_N_cCB.npz cBCBnum cBCBcat >> $S/diff_N_cb_direct.log 2>&1
  echo "seqA exit $?" > $S/sep_cb_seqA.done
) &
$PY scratch_gap/bench_eval.py D A_C,cCBnum,cCBcat,cBCBnum,cBCBcat _cCB > $S/sep_D_cb.log 2>&1 &&
$PY scratch_gap/dec_diff.py dec_D_cCB.npz A_C cCBnum,cCBcat,cBCBnum,cBCBcat > $S/diff_D_cb.log 2>&1 &&
$PY scratch_gap/dec_diff.py dec_D_cCB.npz cCBnum cCBcat > $S/diff_D_cb_direct.log 2>&1 &&
$PY scratch_gap/dec_diff.py dec_D_cCB.npz cBCBnum cBCBcat >> $S/diff_D_cb_direct.log 2>&1 &&
$PY scratch_gap/bench_eval.py D cCBnum,cCBcat _cCBgpu > $S/sep_D_cb_direct_gpu.log 2>&1 &&
$PY scratch_gap/bench_eval.py D cBCBnum,cBCBcat _cCBbl > $S/sep_D_cb_direct_blend.log 2>&1 &&
$PY scratch_gap/bench_eval.py N cCBnum,cCBcat _cCBgpu > $S/sep_N_cb_direct_gpu.log 2>&1 &&
$PY scratch_gap/bench_eval.py N cCBnums1,cCBcats1 _cCBs1 > $S/sep_N_cb_direct_s1.log 2>&1 &&
$PY scratch_gap/bench_eval.py N cCBnumCPU,cCBcatCPU _cCBcpu > $S/sep_N_cb_direct_cpu.log 2>&1 &&
$PY scratch_gap/bench_eval.py N cBCBnum,cBCBcat _cCBbl > $S/sep_N_cb_direct_blend.log 2>&1
echo "seqB exit $?" > $S/sep_cb_seqB.done
wait
