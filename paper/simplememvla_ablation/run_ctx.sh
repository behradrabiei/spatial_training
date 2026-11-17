#!/usr/bin/env bash
# Context-window and layer-depth ablations of SimpleMemVLA on a RoboMME suite (default: Permanence).
#
#   bash paper/simplememvla_ablation/run_ctx.sh                 # full, k32, k16, k8, k4, k1
#   ARMS="k8 k4" bash paper/simplememvla_ablation/run_ctx.sh    # a subset, in the given order
#   bash paper/simplememvla_ablation/run_depth.sh               # blind_l28 ... blind_l8
#
# Arm k<K>: the model sees the newest K sampled history frames (CONTEXT_FRAMES=K, see
# robomme_benchmark/run_simplememvla_eval.sh); arm blind_l<L>: visual tokens hidden from decoder
# layer L on (BLIND_LAYER=L); `full` leaves the model untouched.
# Raw runs: $SAVE_ROOT/<suite>/<arm>/{results.json,episodes.jsonl,run_config.json,eval.log,videos/}.
# Finished arms (results.json) are skipped, partial arms resume, `touch $SAVE_ROOT/<suite>/STOP`
# stops the driver before the next arm.
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
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}" GROUP_SIZE="${GROUP_SIZE:-4}" SEED="${SEED:-0}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/vault/robomme/runs/simplememvla_ablation}"
OUT="$SAVE_ROOT/$SUITE"
ARMS="${ARMS:-full k32 k16 k8 k4 k1}"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/driver.log"; }

for arm in $ARMS; do
  [[ -f "$OUT/STOP" ]] && { log "[stop] STOP file present, not starting $arm"; exit 0; }
  [[ -f "$OUT/$arm/results.json" ]] && { log "[skip] $arm (results.json exists)"; continue; }
  case "$arm" in
    full) ctx="" blind="" ;;
    k[0-9]*) ctx="${arm#k}" blind="" ;;
    blind_l[0-9]*) ctx="" blind="${arm#blind_l}" ;;
    *) log "[error] unknown arm $arm"; exit 1 ;;
  esac
  log "[start] $arm (context_frames=${ctx:-native} blind_layer=${blind:-none}) tasks: $TASKS"
  t0=$(date +%s)
  CONTEXT_FRAMES="$ctx" BLIND_LAYER="$blind" RUN_DIR="$OUT/$arm" "$ROBOMME/run_simplememvla_eval.sh" > "$OUT/${arm}_driver.log" 2>&1
  rc=$?
  log "[done] $arm exit=$rc wall=$(( $(date +%s) - t0 ))s $(grep -a 'OVERALL' "$OUT/${arm}_driver.log" | tail -1)"
  [[ $rc -ne 0 ]] && { log "[error] $arm failed; see $OUT/${arm}_driver.log"; exit $rc; }
done
log "[all done] $ARMS"
