#!/bin/zsh
# GOC 10000: fine-tune released Tiny seed0 -> evaluate on the existing test set (GENCO start and hybrid start) -> summary.
# usage: nohup work/finetune/20_pipeline10000.sh STEPS DEVICE [LR] [SEEDS] > work/finetune/logs/pipeline10000.out 2>&1 &
#   STEPS/DEVICE from the pilot (10_pilot10000.sh); LR 0 = released value; SEEDS = training seeds, default "0".
# Needs the fine-tuning data (00_datagen10000.sh: logs/split10000.DONE); waits for it. Queues behind running trainings
# (queue.sh). Trainings go through run_train.sh (gate, tmo <= 8 h). Resumable: rerun this script (DONE markers / result
# files; an interrupted training resumes from its last checkpoint; if 8 h were not enough, rerun to continue).
cd "$(dirname "$0")/../.."
STEPS=${1:?steps}; DEV=${2:?device}; LR=${3:-0}; SEEDS=${4:-0}
G=10000; NET=case${G}_goc
F=work/finetune; L=$F/logs; D=$F/data${G}_ft; M=$F/models; R=$F/results; PY=.venv/bin/python; T=work/texas/tmo
INIT=work/goc$G/pf_${NET}_tiny; PROCS=3
[ -n "$DRY" ] && { echo "dry: STEPS=$STEPS DEV=$DEV LR=$LR SEEDS=($SEEDS) data ready: $([ -f $L/split$G.DONE ] && echo yes || echo no)"; exit 0; }
source $F/queue.sh
until [ -f $L/split$G.DONE ]; do sleep 60; done
queue_enter pipeline$G
if [ ! -f $L/preprocess$G.DONE ]; then
  while pgrep -f "01b_preprocess.py $NET" > /dev/null; do sleep 30; done   # 00_datagen10000.sh may be building it
  if [ ! -f $L/preprocess$G.DONE ]; then
    $T 14400 $PY $F/01b_preprocess.py $NET $D > $L/preprocess$G.log 2>&1 || { qlog "pipeline$G: preprocess failed"; exit 1; }
    date > $L/preprocess$G.DONE
  fi
fi
VAL=$(val_every $STEPS $D/splits_base $INIT/released_training.yaml)  # released batch size 2 -> one epoch = (training scenarios / 2) steps
train() {  # tag seed
  [ -f $M/pf_${NET}_$1/DONE ] && return 0
  wait_training
  $F/run_train.sh 28800 $1 --net $NET --data $D --splits $D/splits_base --init $INIT --hparams $INIT/released_training.yaml \
    --seed $2 --max_steps $STEPS --val_every $VAL --accelerator $DEV --lr $LR || { qlog "training $1 failed/blocked (rc=$?)"; exit 1; }
}
evaluate() {  # tag
  [ -f $R/lines_newton_${G}_$1.parquet ] || $T 14400 $PY -u $F/02_eval.py --grid $G --model $M/pf_${NET}_$1 --tag $1 --procs $PROCS > $L/eval_$1.log 2>&1 \
    || { qlog "pipeline$G: 02_eval $1 failed"; exit 1; }
  [ -f $R/lines_hybrid_${G}_$1.parquet ] || $T 14400 $PY -u $F/04_hybrid_eval.py --grid $G --model $M/pf_${NET}_$1 --tag $1 --procs $PROCS >> $L/hybrid_$1.log 2>&1 \
    || { qlog "pipeline$G: 04_hybrid_eval $1 failed"; exit 1; }
}
[ -f $R/lines_hybrid_${G}_released.parquet ] || $T 14400 $PY -u $F/04_hybrid_eval.py --grid $G --procs $PROCS >> $L/hybrid_${G}_released.log 2>&1
a1=(released=work/goc$G/lines_newton_tiny_seed0.parquet); a2=(released=$R/lines_hybrid_${G}_released.parquet)
for s in ${=SEEDS}; do
  tag=ft${G}_seed$s
  train $tag $s
  evaluate $tag
  a1+=($tag=$R/lines_newton_${G}_$tag.parquet); a2+=($tag=$R/lines_hybrid_${G}_$tag.parquet)
done
$PY analysis_finetune.py --grid $G $a1 > $L/summary$G.log 2>&1
$PY analysis_finetune_hybrid.py $a2 > $L/summary_hybrid$G.log 2>&1
date > $L/pipeline$G.DONE
qlog "pipeline$G: done, see $L/summary$G.log and $L/summary_hybrid$G.log"
