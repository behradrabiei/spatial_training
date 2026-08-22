#!/usr/bin/env bash
# Run strict Uni-NaVid context windows over the 100-episode HM3D-v2 split.
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a WINDOWS <<< "${WINDOWS:-32 16 8 4 1}"

for WINDOW in "${WINDOWS[@]}"; do
  OUTPUT="$BASELINE_ROOT/results/hm3d_v2_100_win${WINDOW}"
  echo "Starting Uni-NaVid context window ${WINDOW} (+ current frame)"
  bash "$BASELINE_ROOT/run.sh" eval \
    --resume \
    --context-window "$WINDOW" \
    --output "$OUTPUT"
done

"$BASELINE_ROOT/.envs/uninavid-hm3d/bin/python" \
  "$BASELINE_ROOT/plot_context_window_ablation.py"
