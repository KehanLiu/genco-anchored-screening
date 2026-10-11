#!/bin/zsh
# Runs only when the machine is quiet: after the 24-rank N-1 extension, the GOC 10000 runs and the GOC 500 pool.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
until grep -q N1_EXTEND_DONE $L/fulln1_run_24.log 2>/dev/null && [ -f $L/datagen500.DONE ]; do sleep 60; done
sleep 30
echo "quiet chain start $(date); load: $(uptime)" > $L/quiet_chain.log
# keep the timing taken under load for the record, then redo it
cp work/fulln1/timing_2000_fresh.json work/fulln1/timing_2000_fresh_under_load.json
export GOC_DATA=$F/data2000
$T 900  $PY -u analysis/analysis_full_n1.py timing --grid 2000 --tag _fresh > $L/fulln1_timing_quiet.log 2>&1
$T 9000 $PY -u analysis/analysis_full_n1.py e2e --grid 2000 --tag _fresh --policy $F/policy_frozen.json --bases 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 > $L/fulln1_e2e.log 2>&1
echo "E2E_DONE $(date); load: $(uptime)" >> $L/quiet_chain.log
