#!/bin/zsh
# fresh pools, sequential: GOC 2000 (~10 min) then GOC 10000 (~37 min, ~15 GB RAM). Policy frozen first (policy_frozen.json).
cd "$(dirname "$0")/../.."
set -e; set -u; set -o pipefail   # fail fast: the first failed command ends the chain with its status; every marker below is reached only on success
PY=.venv/bin/python; T=work/texas/tmo; F=work/fresh; L=$F/logs
( time $T 20000 $PY $F/../finetune/00_datagen_skip.py $F/datagen2000.yaml 1500 ) > $L/datagen2000.log 2>&1
date > $L/datagen2000.DONE
( time $T 43200 $PY $F/../finetune/00_datagen_skip.py $F/datagen10000.yaml 500 ) > $L/datagen10000.log 2>&1
date > $L/datagen10000.DONE
echo ALL_DONE_DATAGEN >> $L/datagen10000.log
