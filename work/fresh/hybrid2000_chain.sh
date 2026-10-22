#!/bin/zsh
cd "$(dirname "$0")/../.."
set -e; set -u; set -o pipefail   # fail fast: the first failed command ends the chain with its status; every marker below is reached only on success
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
$PY work/finetune/01_split.py --net case2000_goc --all $F/data2000 --test work/goc2000/data --out $F/data2000_check_test --first 1500 --pool_from 2000 > $L/disjoint2000_test.log 2>&1
$PY work/finetune/01_split.py --net case2000_goc --all $F/data2000 --test work/finetune/data2000_all --out $F/data2000_check_ft --first 1500 --pool_from 2000 > $L/disjoint2000_ft.log 2>&1
$PY - <<'PYEOF'
import json, pandas as pd, pathlib
p=pathlib.Path("work/fresh/fresh_ids.json"); d=json.loads(p.read_text()) if p.exists() else {}
d["2000"]=sorted(map(int,pd.read_parquet("work/fresh/data2000/case2000_goc/raw/branch_data.parquet",columns=["load_scenario_idx"]).load_scenario_idx.unique()))
p.write_text(json.dumps(d,indent=0)); print("fresh ids 2000:",len(d["2000"]))
PYEOF
export GOC_DATA=$F/data2000
$T 3000 $PY -u analysis/analysis_prev_control.py run --grid 2000 --procs 4 --tag _fresh > $L/prev_2000.log 2>&1
$T 3000 $PY -u analysis/analysis_hybrid_start.py run --grid 2000 --variant tiny --seed 0 --procs 4 --tag _fresh --policy $F/policy_frozen.json > $L/hybrid_2000_tiny_seed0.log 2>&1
echo HYBRID2000_DONE >> $L/hybrid_2000_tiny_seed0.log
