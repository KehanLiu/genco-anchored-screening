#!/bin/zsh
# GOC 500: fine-tune (2 seeds) -> evaluate on the existing test set -> active-learning round -> evaluate -> summary.
# usage: nohup work/finetune/20_pipeline500.sh STEPS DEVICE [LR] &      (STEPS/DEVICE from the pilot, 10_pilot500.sh)
# Every training goes through run_train.sh (AC-power + sleep-prevention gate, tmo <= 8 h, resumable: just rerun this script;
# finished stages are skipped via their DONE markers / result files).
cd "$(dirname "$0")/../.."
STEPS=${1:?steps}; DEV=${2:?device}; LR=${3:-0}
F=work/finetune; L=$F/logs; D=$F/data500_ft; M=$F/models; R=$F/results; PY=.venv/bin/python; T=work/texas/tmo
VAL=$(( STEPS / 10 )); [ $VAL -gt 1000 ] && VAL=1000  # must validate before the first epoch ends (1,121 steps): the LR scheduler needs the metric
K=1500
train() {  # tag init splits seed steps
  [ -f $M/pf_case500_goc_$1/DONE ] && return 0
  $F/run_train.sh 14400 $1 --net case500_goc --data $D --splits $D/$3 --init $2 --hparams work/goc500/pf_case500_goc_tiny/released_training.yaml \
    --seed $4 --max_steps $5 --val_every $VAL --accelerator $DEV --lr $LR || { echo "training $1 failed/blocked (rc=$?)" >> $L/LOG.txt; exit 1; }
}
evaluate() {  # tag [extra args]
  tag=$1; shift
  out=$R/lines_newton_500_${tag}${POOL:+}.parquet
  [ -f $R/lines_newton_500_${EVTAG:-$tag}.parquet ] && return 0
  $T 7200 $PY -u $F/02_eval.py --grid 500 --model $M/pf_case500_goc_$tag --tag ${EVTAG:-$tag} --procs 4 "$@" > $L/eval_${EVTAG:-$tag}.log 2>&1
}
# B. fine-tune released seed0 (training seed 0) and released seed1 (training seed 1)
train ft_seed0 work/goc500/pf_case500_goc_tiny       splits_base 0 $STEPS
evaluate ft_seed0
train ft_seed1 work/goc500/pf_case500_goc_tiny_seed1 splits_base 1 $STEPS
evaluate ft_seed1
# D. active learning from ft_seed0: label-free scores on the pool, then +K scenarios and the same number of extra steps
EVTAG=pool_ft_seed0 evaluate ft_seed0 --data $D --role pool
[ -f $D/al_selection.csv ] || $PY $F/03_select.py --data $D --pool_tag pool_ft_seed0 --k $K > $L/select.log 2>&1
for c in mis1 random nearlimit; do
  train al_${c}_seed0 $M/pf_case500_goc_ft_seed0 splits_al_$c 0 $STEPS
  evaluate al_${c}_seed0
done
# control: the same extra steps without any new scenarios
train al_none_seed0 $M/pf_case500_goc_ft_seed0 splits_base 0 $STEPS
evaluate al_none_seed0
args=(released_seed0=work/goc500/lines_newton_tiny_seed0.parquet released_seed1=work/goc500/lines_newton_tiny_seed1.parquet)
for t in ft_seed0 ft_seed1 al_none_seed0 al_random_seed0 al_mis1_seed0 al_nearlimit_seed0; do args+=($t=$R/lines_newton_500_$t.parquet); done
$PY analysis_finetune.py --grid 500 $args > $L/summary500.log 2>&1
date > $L/pipeline500.DONE
