#!/usr/bin/env bash
# One worker of a parallel sweep: runs the given arms in order through run_r1.sh / run_uninavid.sh,
# claiming each with a lock (runs/.locks/<arm>.lock) so several workers can share one arm list
# without ever running the same arm twice. Finished arms are skipped by the arm scripts; the list
# is walked twice so an arm that failed or was held on the first pass is retried.
#
#   VARIANT=flat nohup bash paper/token_prune/run_worker.sh A r1_divprune_b32 r1_dart_b32 ... &
#   VARIANT=flat nohup bash paper/token_prune/run_worker.sh B uninavid_random_b8 ... r1_random_b16 ... &
#
# One GPU: a LongNav-R1 arm takes ~6.5 GB and ~50% of an RTX 5090, a Uni-NaVid arm ~15 GB, so run
# at most two workers and let only ONE of them hold Uni-NaVid arms. Give concurrent R1 arms a
# smaller Ray object store (EXTRA_OVERRIDES="resources.osm_gb=10"; two 20 GB stores exceed /dev/shm).
# Concurrent arms share the GPU, so their vlm_latency_ms is not comparable with serial runs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# No default on purpose: a worker started without VARIANT would run (and write) the other experiment.
export VARIANT="${VARIANT:?set VARIANT=window or VARIANT=flat}"
case "$VARIANT" in
  window) RUNS="$ROOT/paper_results/token_prune/runs" ;;
  flat)   RUNS="$ROOT/paper_results/token_prune_flat/runs" ;;
  *) echo "Unknown VARIANT '$VARIANT' (expected window or flat)" >&2; exit 2 ;;
esac
WORKER="${1:?usage: run_worker.sh <worker name> <arm> [<arm> ...]}"; shift
mkdir -p "$RUNS/.locks"
{
  echo "$(date '+%F %T')  worker $WORKER start (variant=$VARIANT): $*"
  for PASS in 1 2; do
    for ARM in "$@"; do
      FAM="${ARM%%_*}"
      flock -n -E 99 "$RUNS/.locks/$ARM.lock" env ARMS="$ARM" bash "paper/token_prune/run_${FAM}.sh"
      [[ $? -eq 99 ]] && echo "$(date '+%F %T')  worker $WORKER: $ARM is held by another worker, skipped"
    done
  done
  echo "$(date '+%F %T')  worker $WORKER done"
} >> "$RUNS/driver.log" 2>&1
