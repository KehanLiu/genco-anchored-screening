#!/bin/zsh
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
until [ -f $L/datagen10000.DONE ]; do sleep 30; done
$PY work/finetune/01_split.py --net case10000_goc --all $F/data10000 --test work/goc10000/data --out $F/data10000_check_test --first 500 --pool_from 700 > $L/disjoint10000_test.log 2>&1
$PY work/finetune/01_split.py --net case10000_goc --all $F/data10000 --test work/finetune/data10000_all --out $F/data10000_check_ft --first 500 --pool_from 700 > $L/disjoint10000_ft.log 2>&1
$PY - <<'PYEOF'
import json, pandas as pd, pathlib
p=pathlib.Path("work/fresh/fresh_ids.json"); d=json.loads(p.read_text()) if p.exists() else {}
d["10000"]=sorted(map(int,pd.read_parquet("work/fresh/data10000/case10000_goc/raw/branch_data.parquet",columns=["load_scenario_idx"]).load_scenario_idx.unique()))
p.write_text(json.dumps(d,indent=0)); print("fresh ids 10000:",len(d["10000"]))
PYEOF
export GOC_DATA=$F/data10000
$T 3000 $PY -u analysis/analysis_prev_control.py run --grid 10000 --procs 4 --tag _fresh > $L/prev_10000.log 2>&1
for ck in "tiny 0" "tiny 1" "small 0" "small 1"; do set -- ${=ck}
  $T 3000 $PY -u analysis/analysis_hybrid_start.py run --grid 10000 --variant $1 --seed $2 --procs 4 --tag _fresh --policy $F/policy_frozen.json > $L/hybrid_10000_$1_seed$2.log 2>&1
done
unset GOC_DATA
echo HYBRID10000_DONE >> $L/hybrid_10000_small_seed1.log
# GOC 500 pool for the equivalent-circuit pilot (POLICY_LOG.md A4)
( time $T 14400 $PY $F/../finetune/00_datagen_skip.py $F/datagen500.yaml 6000 ) > $L/datagen500.log 2>&1 && date > $L/datagen500.DONE
