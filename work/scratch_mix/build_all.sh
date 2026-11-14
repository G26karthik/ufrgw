#!/bin/bash
# Build the three declared mixtures end to end (DECLARED.md, Build): blend -> postproc --qr-null -> ts_judge_post -> chain_fb (alt_c2 chain + declared fallback).
# usage (cwd=work): bash scratch_mix/build_all.sh ; every step refuses to overwrite. Logs: scratch_mix/build_cmb<W>.log
export PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4; PY=../.venv/Scripts/python.exe; L=scratch_mix
for w in 50 65 80; do
  { $PY $L/mix_blend.py build $w && $PY postproc.py cmb$w --qr-null && $PY ts_judge_post.py cmb$w \
    && $PY scratch_mix/chain_fb.py subs/ts_judge_cmb$w.csv cmb${w}_dd_final; } > $L/build_cmb$w.log 2>&1 || echo "cmb$w FAILED"
done
$PY $L/check.py > $L/check.log 2>&1; tail -n 40 $L/check.log
