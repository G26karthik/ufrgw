#!/bin/bash
# Build the second exploratory candidate end to end (scratch_wt/DECLARED_sq35.md). Needs the fits first:
#   BM=D SEED0=2 python scratch_wt/wt.py wSQ ; BM=D SEED0=4 python scratch_wt/wt.py wSQ   (seeds 0,1 = PD_wSQ, already fitted)
#   python scratch_wt/wt_oct.py 0 1 2 ; python scratch_wt/wt_oct.py 3 4 5 ; python scratch_wt/sq6.py assess
# usage (cwd=work): bash scratch_wt/build_sq35.sh ; every step refuses to overwrite. Log: scratch_wt/build_sq35.log
export PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4; PY=../.venv/Scripts/python.exe; L=scratch_wt
{ $PY $L/sq6.py build && $PY postproc.py sq35 --qr-null && $PY ts_judge_post.py sq35 \
  && $PY scratch_mix/chain_fb.py subs/ts_judge_sq35.csv sq35_dd_final && $PY $L/sq6.py counts; } > $L/build_sq35.log 2>&1 || echo "sq35 FAILED"
$PY $L/check_sq35.py > $L/check_sq35.log 2>&1; tail -n 60 $L/check_sq35.log
