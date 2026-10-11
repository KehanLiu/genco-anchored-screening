#!/bin/zsh
# DRAFT - not executed yet. Fresh-operating-point confirmation run for the new screening policies.
# Recipe, cost and the freeze checklist: work/fresh/README.md. Needs AC power (caffeinate), ~1 h 15 min wall.
#
# Preconditions (all enforced below):
#   FRESH_IDS_FROZEN=1                     set by hand only after the checklist in README.md section 5 is ticked
#   work/fresh/policy_frozen.json          lambda per grid/checkpoint, tau per workflow, c grid, tau1 grid, ranks, seed
#   work/fresh/datagen{2000,10000}.yaml    copies of work/finetune/datagen<N>.yaml with scenarios/data_dir/seed changed
#   code changes listed in README.md 2.2 (GOC_DATA override, --lam, fresh scorer, e2e phase) merged and tested
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs

if [ "${FRESH_IDS_FROZEN:-0}" != "1" ]; then
  echo "run_fresh.sh: refusing to run. Freeze the policy first (work/fresh/policy_frozen.json, README.md section 5)," >&2
  echo "then set FRESH_IDS_FROZEN=1 explicitly. Nothing was executed." >&2
  exit 1
fi
for f in $F/policy_frozen.json $F/datagen2000.yaml $F/datagen10000.yaml; do
  [ -f "$f" ] || { echo "run_fresh.sh: missing $f" >&2; exit 1; }
done
pmset -g batt | grep -q "AC Power" || { echo "run_fresh.sh: not on AC power" >&2; exit 1; }
mkdir -p $L
SEED=$($PY -c "import json;print(json.load(open('$F/policy_frozen.json'))['datakit_seed'])")
RANKS=$($PY -c "import json;print(' '.join(map(str,json.load(open('$F/policy_frozen.json'))['n1_base_ranks'])))")
echo "fresh seed $SEED, N-1 base ranks: $RANKS"

# 1. fresh pools (hours never used by the test pool (seed 0) nor the fine-tune pool (seed 1234); see README.md 2.1)
[ -f $L/datagen2000.DONE ]  || { ( time $T 20000 $PY $F/../finetune/00_datagen_skip.py $F/datagen2000.yaml 1500 )  > $L/datagen2000.log 2>&1;  date > $L/datagen2000.DONE; }
[ -f $L/datagen10000.DONE ] || { ( time $T 43200 $PY $F/../finetune/00_datagen_skip.py $F/datagen10000.yaml 500 ) > $L/datagen10000.log 2>&1; date > $L/datagen10000.DONE; }
# disjointness against both earlier pools, and the fresh id list, before anything is scored
for n in 2000 10000; do
  $PY work/finetune/01_split.py --net case${n}_goc --all $F/data$n --test work/goc$n/data        --out $F/data${n}_check_test --first $([ $n = 2000 ] && echo 1500 || echo 500) --pool_from $([ $n = 2000 ] && echo 2000 || echo 700) > $L/disjoint${n}_test.log 2>&1
  $PY work/finetune/01_split.py --net case${n}_goc --all $F/data$n --test work/finetune/data${n}_all --out $F/data${n}_check_ft --first $([ $n = 2000 ] && echo 1500 || echo 500) --pool_from $([ $n = 2000 ] && echo 2000 || echo 700) > $L/disjoint${n}_ft.log 2>&1
done
$PY - <<'EOF'
import json, pandas as pd
ids = {n: sorted(map(int, pd.read_parquet(f"work/fresh/data{n}/case{n}_goc/raw/branch_data.parquet", columns=["load_scenario_idx"]).load_scenario_idx.unique())) for n in (2000, 10000)}
json.dump(ids, open("work/fresh/fresh_ids.json", "w"), indent=1); print({n: len(v) for n, v in ids.items()})
EOF

# 2. GOC 2000 full N-1 on the pre-registered ranks of the fresh design (chunks resumable), unit costs, end-to-end timing
export GOC_DATA=$F/data2000
$T 9000 $PY -u analysis/analysis_full_n1.py run    --grid 2000 --procs 4 --bases ${=RANKS} --tag _fresh --policy $F/policy_frozen.json > $L/fulln1_run.log 2>&1
$T 600  $PY -u analysis/analysis_full_n1.py timing --grid 2000 --tag _fresh                            > $L/fulln1_timing.log 2>&1
$T 3000 $PY -u analysis/analysis_full_n1.py e2e    --grid 2000 --tag _fresh --policy $F/policy_frozen.json > $L/fulln1_e2e.log 2>&1

# 3. hybrid start on the fresh GOC 10000 (and 2000) load scenarios: pre-outage control, four released checkpoints
for n in 10000 2000; do
  export GOC_DATA=$F/data$n
  $T 3000 $PY -u analysis/analysis_prev_control.py run --grid $n --tag _fresh > $L/prev_$n.log 2>&1
  for ck in "tiny 0" "tiny 1" "small 0" "small 1"; do set -- ${=ck}
    [ $n = 2000 ] && [ "$1 $2" != "tiny 0" ] && continue
    $T 3000 $PY -u analysis/analysis_hybrid_start.py run --grid $n --variant $1 --seed $2 --tag _fresh --policy $F/policy_frozen.json > $L/hybrid_${n}_$1_seed$2.log 2>&1
  done
done
unset GOC_DATA

# 4. scoring with the frozen policy only (misses, false flags, share to AC, iterations, measured e2e in one table)
$PY $F/score_fresh_n1.py     --policy $F/policy_frozen.json > $L/score_n1.log 2>&1
$PY $F/score_fresh_hybrid.py --policy $F/policy_frozen.json > $L/score_hybrid.log 2>&1
date > $L/ALL_DONE
