#!/bin/zsh
# device pilot: 300 steps each on CPU (4 threads) and MPS, released hyper-parameters
cd "$(dirname "$0")/../.."
for dev in cpu mps; do
  work/finetune/run_train.sh 1800 pilot500_$dev --net case500_goc --data work/finetune/data500_ft --splits work/finetune/data500_ft/splits_base \
    --init work/goc500/pf_case500_goc_tiny --max_steps 300 --val_every 100 --limit_val 25 --accelerator $dev || echo "pilot $dev rc=$?" >> work/finetune/logs/LOG.txt
done
date > work/finetune/logs/pilot500.DONE
