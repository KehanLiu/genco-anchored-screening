#!/bin/zsh
# usage: run_train.sh SECONDS TAG train.py-args...
# Gate required before EVERY training launch: on AC power and a sleep-prevention assertion active (caffeinate).
# The lid may be closed. Result is appended to logs/LOG.txt.
# Exit 3 (and logs/BLOCKED) if the gate fails; otherwise runs train.py under work/texas/tmo SECONDS (<= 28800).
cd "$(dirname "$0")/../.."
L=work/finetune/logs; S=$1; TAG=$2; shift 2
[ "$S" -le 28800 ] || { echo "timeout $S > 28800" >&2; exit 2; }
BATT=$(pmset -g batt | head -1); LID=$(ioreg -r -k AppleClamshellState -d 4 | grep AppleClamshellState | head -1 | sed 's/.*= //')
AWAKE=$(pmset -g assertions | awk '/PreventUserIdleSystemSleep|PreventSystemSleep/ {s+=$2} END {print (s>0) ? "yes" : "no"}')
echo "$(date '+%F %T') gate for $TAG: [$BATT] lid closed=$LID sleep prevented=$AWAKE" >> $L/LOG.txt
if [[ "$BATT" != *"AC Power"* || "$AWAKE" != "yes" ]]; then
  echo "$(date '+%F %T') BLOCKED $TAG: training not started" | tee -a $L/LOG.txt > $L/BLOCKED; exit 3
fi
rm -f $L/BLOCKED
echo "$(date '+%F %T') START $TAG ($S s limit): $*" >> $L/LOG.txt
work/texas/tmo $S .venv/bin/python -u work/finetune/train.py --tag $TAG "$@" >> $L/train_$TAG.log 2>&1
RC=$?
echo "$(date '+%F %T') END $TAG rc=$RC" >> $L/LOG.txt
exit $RC
