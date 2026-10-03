#!/bin/zsh
# Waits for the two datagen runs, then Newton (tiny, small) + analysis per grid, then timing with nothing else of ours running.
cd "$(dirname "$0")/../.."
T=work/texas/tmo; PY=.venv/bin/python
for n in 2000 10000; do
  until grep -q "total$" work/goc$n/logs/datagen.log; do sleep 10; done
  for v in tiny small; do
    $T 20000 $PY -u work/goc$n/02_newton.py --variant $v --procs $([ $n = 2000 ] && echo 3 || echo 5) > work/goc$n/logs/newton_${v}_seed0.log 2>&1
    $T 3000 $PY work/goc$n/03_analysis.py --variant $v > work/goc$n/logs/analysis_newton_${v}_seed0.log 2>&1
  done
done
for n in 2000 10000; do
  for v in tiny small; do
    $T 6000 $PY -u work/goc$n/04_timing.py --variant $v > work/goc$n/logs/timing_$v.log 2>&1
  done
done
date > work/goc10000/logs/ALL_DONE
