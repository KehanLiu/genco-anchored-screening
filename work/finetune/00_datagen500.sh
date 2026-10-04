#!/bin/zsh
cd "$(dirname "$0")/../.."
( time work/texas/tmo 14400 .venv/bin/gridfm_datakit generate work/finetune/datagen500.yaml ) > work/finetune/logs/datagen500.log 2>&1
date > work/finetune/logs/datagen500.DONE
