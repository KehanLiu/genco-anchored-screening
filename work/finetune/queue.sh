# sourced by 10_pilot2000.sh / 20_pipeline2000.sh / 10_pilot10000.sh / 20_pipeline10000.sh (cwd = repo root).
# One training at a time on this machine:
#   queue_enter NAME   blocks (polling every 60 s) until (1) this script holds logs/train.lock (a directory; taken over
#                      if its owner PID is dead), (2) no 20_pipeline500.sh / 10_pilot500.sh is running (they do not use
#                      the lock and start their trainings without looking), (3) no work/finetune/train.py is running.
#                      The lock is held until the script exits (trap), so its trainings AND evaluations run undisturbed.
#   wait_training      (3) only; called again before every single training.
# Scripts waiting for the lock are served in arbitrary order, not strictly first come first served.
QLOCK=work/finetune/logs/train.lock
qlog() { echo "$(date '+%F %T') $*" >> work/finetune/logs/LOG.txt; }
training_running() { pgrep -f "work/finetune/train.py" > /dev/null; }
foreign_pipeline_running() { pgrep -f "work/finetune/(20_pipeline500|10_pilot500)\.sh" > /dev/null; }
wait_training() { while training_running; do sleep 60; done; }
# release on exit of the main shell (zsh: an EXIT trap set inside a function would already fire when the function returns)
zshexit() { [ "$(cat $QLOCK/pid 2> /dev/null)" = "$$" ] && { rm -f $QLOCK/pid $QLOCK/name; rmdir $QLOCK 2> /dev/null; } }
queue_enter() {
  local name=$1 told=0
  while true; do
    if mkdir $QLOCK 2> /dev/null; then echo $$ > $QLOCK/pid; echo $name > $QLOCK/name; break; fi
    local owner=$(cat $QLOCK/pid 2> /dev/null)
    if [ -n "$owner" ] && ! kill -0 $owner 2> /dev/null; then
      qlog "$name: removing stale lock of dead PID $owner ($(cat $QLOCK/name 2> /dev/null))"; rm -f $QLOCK/pid $QLOCK/name; rmdir $QLOCK 2> /dev/null; continue
    fi
    [ $told = 0 ] && { qlog "$name: queued behind $(cat $QLOCK/name 2> /dev/null) (PID $owner)"; told=1; }
    sleep 60
  done
  trap 'exit 130' INT TERM HUP
  told=0
  while foreign_pipeline_running || training_running; do
    [ $told = 0 ] && { qlog "$name: waiting for the running GOC 500 pipeline / training to finish"; told=1; }
    sleep 60
  done
  # a pipeline between two of its stages has no train.py for a moment: require two quiet checks in a row
  sleep 20
  while foreign_pipeline_running || training_running; do sleep 60; done
  qlog "$name: queue passed, starting"
}
# steps between validations: at most STEPS/10 and at most 90% of one epoch (the LR scheduler needs the validation metric
# before the first epoch ends); usage: val_every STEPS SPLITS_DIR RELEASED_YAML
val_every() {
  .venv/bin/python - "$@" <<'PY'
import sys, torch, yaml
steps, splits, hp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
n = len(torch.load(f"{splits}/train.pt"))
batch = yaml.safe_load(open(hp))["training"]["batch_size"]
print(max(1, min(steps // 10, int(0.9 * (n // batch)))))
PY
}
