#!/bin/zsh
# Extension registered in POLICY_LOG.md A2 before any fresh N-1 outcome was inspected: all 24 design ranks of the fresh pool.
cd "$(dirname "$0")/../.."
set -e; set -u; set -o pipefail   # fail fast: the first failed command ends the chain with its status; every marker below is reached only on success
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
until grep -q N1_RUN_DONE $L/fulln1_timing.log 2>/dev/null; do sleep 20; done
export GOC_DATA=$F/data2000
$T 9000 $PY -u analysis/analysis_full_n1.py run --grid 2000 --procs 4 --bases 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --tag _fresh --policy $F/policy_frozen.json > $L/fulln1_run_24.log 2>&1
echo N1_EXTEND_DONE >> $L/fulln1_run_24.log
