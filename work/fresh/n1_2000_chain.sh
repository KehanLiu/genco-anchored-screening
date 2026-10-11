#!/bin/zsh
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
export GOC_DATA=$F/data2000
$T 9000 $PY -u analysis/analysis_full_n1.py run    --grid 2000 --procs 4 --bases 2 6 10 14 18 22 --tag _fresh --policy $F/policy_frozen.json > $L/fulln1_run.log 2>&1
$T 600  $PY -u analysis/analysis_full_n1.py timing --grid 2000 --tag _fresh > $L/fulln1_timing.log 2>&1
echo N1_RUN_DONE >> $L/fulln1_timing.log
