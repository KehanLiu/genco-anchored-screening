#!/bin/zsh
# GOC 2000 device pilot: 300 steps on DEVICE (default mps), released hyper-parameters -> seconds per step in
# logs/train_pilot2000_<device>.log and models/pf_case2000_goc_pilot2000_<device>/train_log.csv.
# usage: nohup work/finetune/10_pilot2000.sh [DEVICE] > work/finetune/logs/pilot2000.out 2>&1 &
# Queues behind running trainings (queue.sh); builds the preprocess cache first if missing.
cd "$(dirname "$0")/../.."
DEV=${1:-mps}; G=2000; NET=case${G}_goc
F=work/finetune; L=$F/logs; D=$F/data${G}_ft; INIT=work/goc$G/pf_${NET}_tiny
[ -n "$DRY" ] && { echo "dry: DEV=$DEV"; exit 0; }
source $F/queue.sh
queue_enter pilot$G
if [ ! -f $L/preprocess$G.DONE ]; then
  work/texas/tmo 7200 .venv/bin/python $F/01b_preprocess.py $NET $D > $L/preprocess$G.log 2>&1 || { qlog "pilot$G: preprocess failed"; exit 1; }
  date > $L/preprocess$G.DONE
fi
wait_training
$F/run_train.sh 3600 pilot${G}_$DEV --net $NET --data $D --splits $D/splits_base --init $INIT --hparams $INIT/released_training.yaml \
  --max_steps 300 --val_every 100 --limit_val 25 --accelerator $DEV || { qlog "pilot$G $DEV rc=$?"; exit 1; }
date > $L/pilot$G.DONE
