#!/usr/bin/env bash
# One worker of the parallel sweep: runs the given arms in order through run_r1.sh / run_uninavid.sh,
# claiming each with a lock (runs/.locks/<arm>.lock) so several workers can share one arm list
# without ever running the same arm twice. Finished arms are skipped by the arm scripts; the list
# is walked twice so an arm that failed or was held on the first pass is retried.
#
#   nohup bash paper/hybrid_prune/run_worker.sh U uninavid_hybrid_b8 ... r1_pyramiddrop_b1 ... &
#   EXTRA_OVERRIDES="resources.osm_gb=10" nohup bash paper/hybrid_prune/run_worker.sh R r1_hybrid_b8 ... &
#
# One GPU: a LongNav-R1 arm takes ~6.5 GB (the no-window arms more) and ~50% of an RTX 5090, a
# Uni-NaVid arm ~15 GB, so run at most two workers and let only ONE of them hold Uni-NaVid arms.
# Give concurrent R1 arms a smaller Ray object store (EXTRA_OVERRIDES="resources.osm_gb=10"; two
# 20 GB stores exceed /dev/shm). Concurrent arms share the GPU, so their vlm_latency_ms is not
# comparable with serial runs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export DATASET="${DATASET:-hm3d}"  # hm3d | mp3d, read by the arm scripts
DEFAULT_RUNS="$ROOT/paper_results/hybrid_prune/runs"
[[ "$DATASET" == "mp3d" ]] && DEFAULT_RUNS="$ROOT/paper_results/hybrid_prune_mp3d/runs"
RUNS="${OUT_ROOT:-$DEFAULT_RUNS}"
WORKER="${1:?usage: run_worker.sh <worker name> <arm> [<arm> ...]}"; shift
mkdir -p "$RUNS/.locks"
{
  echo "$(date '+%F %T')  worker $WORKER start: $*"
  for PASS in 1 2; do
    for ARM in "$@"; do
      FAM="${ARM%%_*}"
      flock -n -E 99 "$RUNS/.locks/$ARM.lock" env ARMS="$ARM" bash "paper/hybrid_prune/run_${FAM}.sh"
      [[ $? -eq 99 ]] && echo "$(date '+%F %T')  worker $WORKER: $ARM is held by another worker, skipped"
    done
  done
  echo "$(date '+%F %T')  worker $WORKER done"
} >> "$RUNS/driver.log" 2>&1
