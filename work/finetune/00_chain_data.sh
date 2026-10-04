#!/bin/zsh
# after the GOC 500 generation: build the 500 fine-tuning dataset, then generate + build the 2000 one
cd "$(dirname "$0")/../.."
L=work/finetune/logs
until [ -f $L/datagen500.DONE ]; do sleep 15; done
.venv/bin/python work/finetune/01_split.py --net case500_goc --all work/finetune/data500_all --test work/goc500/data --out work/finetune/data500_ft --first 3000 --pool_from 5000 > $L/split500.log 2>&1
date > $L/split500.DONE
( time work/texas/tmo 20000 .venv/bin/python work/finetune/00_datagen_skip.py work/finetune/datagen2000.yaml 1000 ) > $L/datagen2000.log 2>&1
.venv/bin/python work/finetune/01_split.py --net case2000_goc --all work/finetune/data2000_all --test work/goc2000/data --out work/finetune/data2000_ft --first 1000 --pool_from 1500 > $L/split2000.log 2>&1
date > $L/split2000.DONE
