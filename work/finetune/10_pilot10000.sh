#!/bin/zsh
# GOC 10000 device pilot: STEPS (default 150) steps on DEVICE (default mps), released hyper-parameters (batch 2) ->
# seconds per step in logs/train_pilot10000_<device>.log and models/pf_case10000_goc_pilot10000_<device>/train_log.csv.
# usage: nohup work/finetune/10_pilot10000.sh [DEVICE] [STEPS] > work/finetune/logs/pilot10000.out 2>&1 &
# Needs the fine-tuning data (00_datagen10000.sh: logs/split10000.DONE); waits for it. Queues behind running trainings.
cd "$(dirname "$0")/../.."
DEV=${1:-mps}; STEPS=${2:-150}; G=10000; NET=case${G}_goc
F=work/finetune; L=$F/logs; D=$F/data${G}_ft; INIT=work/goc$G/pf_${NET}_tiny
[ -n "$DRY" ] && { echo "dry: DEV=$DEV STEPS=$STEPS"; exit 0; }
source $F/queue.sh
until [ -f $L/split$G.DONE ]; do sleep 60; done
queue_enter pilot$G
if [ ! -f $L/preprocess$G.DONE ]; then
  while pgrep -f "01b_preprocess.py $NET" > /dev/null; do sleep 30; done   # 00_datagen10000.sh may be building it
  if [ ! -f $L/preprocess$G.DONE ]; then
    work/texas/tmo 14400 .venv/bin/python $F/01b_preprocess.py $NET $D > $L/preprocess$G.log 2>&1 || { qlog "pilot$G: preprocess failed"; exit 1; }
    date > $L/preprocess$G.DONE
  fi
fi
wait_training
$F/run_train.sh 7200 pilot${G}_$DEV --net $NET --data $D --splits $D/splits_base --init $INIT --hparams $INIT/released_training.yaml \
  --max_steps $STEPS --val_every $(( STEPS / 3 )) --limit_val 10 --accelerator $DEV || { qlog "pilot$G $DEV rc=$?"; exit 1; }
date > $L/pilot$G.DONE
