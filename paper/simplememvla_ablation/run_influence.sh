#!/usr/bin/env bash
# Velocity-field influence of the context window, SimpleMemVLA on a RoboMME suite (default:
# Permanence, test episodes 0-11 of each task).
#
#   bash paper/simplememvla_ablation/run_influence.sh
#   EPISODES_PER_TASK=8 KS=4,8,16,32 bash paper/simplememvla_ablation/run_influence.sh
#
# One full-context rollout (the unmodified model). At every decision the probe
# (robomme_sim/influence_probe.py, INFLUENCE_KS in robomme_benchmark/run_simplememvla_eval.sh)
# also runs the model with only the newest K history frames, for each K in KS, and records the
# squared L2 distance between the two velocity fields on the full-context sampler's states.
# Raw run: $SAVE_ROOT/<suite>/$ARM/{influence.jsonl,episodes.jsonl,run_config.json,results.json,eval.log}.
# A finished run (results.json) is skipped, a partial one resumes by episode.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOMME="$ROOT/robomme_benchmark"
SUITE="${SUITE:-permanence}"
case "$SUITE" in
  permanence) TASKS_DEFAULT="VideoUnmask ButtonUnmask VideoUnmaskSwap ButtonUnmaskSwap" ;;
  counting)   TASKS_DEFAULT="BinFill PickXtimes SwingXtimes StopCube" ;;
  reference)  TASKS_DEFAULT="PickHighlight VideoRepick VideoPlaceButton VideoPlaceOrder" ;;
  imitation)  TASKS_DEFAULT="MoveCube InsertPeg PatternLock RouteStick" ;;
  *) echo "unknown SUITE=$SUITE" >&2; exit 1 ;;
esac
export TASKS="${TASKS:-$TASKS_DEFAULT}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-12}" GROUP_SIZE="${GROUP_SIZE:-4}" SEED="${SEED:-0}"
export INFLUENCE_KS="${KS:-1,2,4,8,16,32,64}" INFLUENCE_NOISE="${NOISE:-8}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/vault/robomme/runs/simplememvla_ablation}"
ARM="${ARM:-influence_recent}"
OUT="$SAVE_ROOT/$SUITE"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/${ARM}_chain.log"; }

[[ -f "$OUT/$ARM/results.json" ]] && { log "[skip] $ARM (results.json exists)"; exit 0; }
log "[start] $ARM K=$INFLUENCE_KS noise=$INFLUENCE_NOISE episodes/task=$EPISODES_PER_TASK tasks: $TASKS"
t0=$(date +%s)
RUN_DIR="$OUT/$ARM" "$ROBOMME/run_simplememvla_eval.sh" > "$OUT/${ARM}_driver.log" 2>&1
rc=$?
log "[done] $ARM exit=$rc wall=$(( $(date +%s) - t0 ))s $(grep -a 'OVERALL' "$OUT/${ARM}_driver.log" | tail -1)"
exit $rc
