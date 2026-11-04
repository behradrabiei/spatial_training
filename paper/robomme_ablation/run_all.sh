#!/usr/bin/env bash
# Run every arm of the RoboMME memory ablation (paper/robomme_ablation/arms.py) sequentially on
# the single GPU: one JAX policy server + the simulator per arm (~19 GB), Counting suite only.
#
#   tmux new -d -s robomme_abl 'bash paper/robomme_ablation/run_all.sh'
#   MODEL=perceptual-framesamp-context tmux new -d -s robomme_abl_ctx 'MODEL=perceptual-framesamp-context bash paper/robomme_ablation/run_all.sh'
#   ARMS="full k1" bash paper/robomme_ablation/run_all.sh      # a subset, in the given order
#
# Results: $SAVE_DIR/<arm>/ckpt79999/seed7/{episodes.jsonl,progress.json,log.json,policy_calls.jsonl,videos/}
# An arm whose log.json exists is skipped; an interrupted arm resumes from its progress.json.
# Pause between arms: touch $SAVE_DIR/STOP (checked before each arm starts; remove it before relaunching).
# Then: python paper/robomme_ablation/aggregate.py && <vln python> paper/robomme_ablation/plot.py   (all models: per-model + combined figures)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOMME="$HERE/../../robomme_benchmark"
MODEL="${MODEL:-perceptual-framesamp-modul}"
case "$MODEL" in
  perceptual-framesamp-modul)   default_save=/mnt/vault/robomme/runs/ablation ;;
  perceptual-framesamp-context) default_save=/mnt/vault/robomme/runs/ablation_context ;;
  *) default_save=/mnt/vault/robomme/runs/ablation_$MODEL ;;
esac
export SAVE_DIR="${SAVE_DIR:-$default_save}"
SEED=7
TASKS=BinFill,PickXtimes,SwingXtimes,StopCube
mkdir -p "$SAVE_DIR"
DRIVER="$SAVE_DIR/driver.log"

arms_list=$(python3 "$HERE/arms.py" list)
if [[ -n "${ARMS:-}" ]]; then
  selected=()
  for a in $ARMS; do
    grep -q "^$a	" <<<"$arms_list" || { echo "unknown arm $a" >&2; exit 1; }
    selected+=("$(grep "^$a	" <<<"$arms_list")")
  done
  arms_list=$(printf '%s\n' "${selected[@]}")
fi

while IFS=$'\t' read -r arm family spec; do
  [[ -z "$arm" ]] && continue
  dir="$SAVE_DIR/$arm/ckpt79999/seed$SEED"
  if [[ -f "$dir/log.json" ]]; then echo "[skip] $arm done" | tee -a "$DRIVER"; continue; fi
  if [[ -f "$SAVE_DIR/STOP" ]]; then echo "[stop] $(date -Is) STOP sentinel present, not starting $arm" | tee -a "$DRIVER"; exit 0; fi
  echo "[start] $(date -Is) $arm ($family) $spec" | tee -a "$DRIVER"
  t0=$(date +%s)
  ARM="$arm" ABLATION="$spec" "$ROBOMME/run_mme_vla_eval.sh" "$MODEL" "$SEED" \
      --args.only_tasks="$TASKS" > "$SAVE_DIR/${arm}_driver.log" 2>&1
  rc=$?
  echo "[end]   $(date -Is) $arm rc=$rc $(( $(date +%s) - t0 )) s" | tee -a "$DRIVER"
  pkill -f "serve_policy.py --seed=$SEED" 2>/dev/null; sleep 3
done <<<"$arms_list"
echo "[all done] $(date -Is)" | tee -a "$DRIVER"
