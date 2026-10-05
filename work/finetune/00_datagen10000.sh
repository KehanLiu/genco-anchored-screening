#!/bin/zsh
# GOC 10000 fine-tuning data: generate load scenarios 200-499 (seed 1234; test set: 0-199, seed 0), split + disjointness
# proof against work/goc10000/data, then graphkit's preprocess cache. Stages are skipped via DONE markers.
# usage: nohup work/finetune/00_datagen10000.sh > work/finetune/logs/datagen10000.out 2>&1 &
# NOTE: the generation itself is NOT resumable (datakit overwrites its output folder); never rerun it while
# logs/datagen10000.DONE is missing AND a generation is still running.
cd "$(dirname "$0")/../.."
F=work/finetune; L=$F/logs; PY=.venv/bin/python; T=work/texas/tmo
if [ ! -f $L/datagen10000.DONE ]; then
  pgrep -f "00_datagen_skip.py $F/datagen10000.yaml" > /dev/null && { echo "generation already running" >&2; exit 1; }
  ( time $T 43200 $PY $F/00_datagen_skip.py $F/datagen10000.yaml 200 ) > $L/datagen10000.log 2>&1 || { echo "generation failed, see $L/datagen10000.log" >&2; exit 1; }
  date > $L/datagen10000.DONE
fi
if [ ! -f $L/split10000.DONE ]; then
  $PY $F/01_split.py --net case10000_goc --all $F/data10000_all --test work/goc10000/data --out $F/data10000_ft --first 200 --pool_from 500 > $L/split10000.log 2>&1 || { echo "split failed, see $L/split10000.log" >&2; exit 1; }
  date > $L/split10000.DONE
fi
if [ ! -f $L/preprocess10000.DONE ]; then
  $T 14400 $PY $F/01b_preprocess.py case10000_goc $F/data10000_ft > $L/preprocess10000.log 2>&1 || { echo "preprocess failed, see $L/preprocess10000.log" >&2; exit 1; }
  date > $L/preprocess10000.DONE
fi
