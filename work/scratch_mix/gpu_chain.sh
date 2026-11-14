#!/bin/bash
# GPU queue of the combined submission (serial): TabICLv2 test + validation, then TabM checkpoints, selection, refits.
export PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4; PY=../.venv/Scripts/python.exe; L=scratch_mix
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 20; done
for f in 2 0 1; do $PY $L/mix_icl.py $f > $L/icl_f$f.log 2>&1; done
for s in 0 1; do $PY $L/mix_tabm.py $s 0 6,12,20,30 > $L/tabm_s${s}_f0.log 2>&1; done
for s in 0 1; do
  $PY scratch_gap/aug_dec.py D xTck${s}e6,xTck${s}e12,xTck${s}e20,xTck${s}e30 > $L/tabm_aug$s.log 2>&1
  e=$($PY $L/pick_ep.py $s); echo "seed $s epoch $e" >> $L/tabm_pick.log
  $PY $L/mix_tabm.py $s 2 $e > $L/tabm_s${s}_f2.log 2>&1
done
for s in 0 1; do e=$(grep "seed $s " $L/tabm_pick.log | awk '{print $4}'); $PY $L/mix_tabm.py $s 1 $e > $L/tabm_s${s}_f1.log 2>&1; done
echo gpu chain finished
