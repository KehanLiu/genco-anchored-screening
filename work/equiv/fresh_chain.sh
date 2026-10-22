#!/bin/zsh
# Equivalent-circuit pilot, fresh pools (SPEC.md section 9). Starts after the fresh end-to-end timing benchmark so timing is not disturbed.
cd "$(dirname "$0")/../.."
set -e; set -u; set -o pipefail   # fail fast: the first failed command ends the chain with its status; every marker below is reached only on success
PY=.venv/bin/python; T=work/texas/tmo; E=work/equiv; L=$E/logs
until grep -q E2E_DONE work/fresh/logs/quiet_chain.log 2>/dev/null; do sleep 60; done
echo "equiv fresh chain start $(date)" > $L/fresh_chain.log
for t in none permute split_random_1 split_all split_gen split_all+refvm split_random_1+refvm; do
  $T 7200 $PY -u $E/run_equiv.py run --grid 500 --data work/fresh/data500 --variant tiny --seed 0 --transform $t --procs 4 --tag _fresh > $L/fresh500_$t.out 2>&1
done
for s in 1 42; do for t in split_random_1 split_all; do
  $T 7200 $PY -u $E/run_equiv.py run --grid 500 --data work/fresh/data500 --variant tiny --seed $s --transform $t --procs 4 --tag _fresh > $L/fresh500_seed${s}_$t.out 2>&1
done; done
echo "GOC500 done $(date)" >> $L/fresh_chain.log
for t in none permute split_random_1 split_all split_gen split_all+refvm split_random_1+refvm; do
  $T 14400 $PY -u $E/run_equiv.py run --grid 2000 --data work/fresh/data2000 --variant tiny --seed 0 --load-ids work/fresh/fresh_ids.json --transform $t --procs 4 --tag _fresh > $L/fresh2000_$t.out 2>&1
done
for t in split_random_1 split_all; do
  $T 14400 $PY -u $E/run_equiv.py run --grid 2000 --data work/fresh/data2000 --variant small --seed 0 --load-ids work/fresh/fresh_ids.json --transform $t --procs 4 --tag _fresh > $L/fresh2000_small_$t.out 2>&1
done
echo "EQUIV_FRESH_DONE $(date)" >> $L/fresh_chain.log
