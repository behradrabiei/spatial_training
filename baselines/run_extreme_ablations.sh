#!/usr/bin/env bash
# Run current-frame-only and text-only Uni-NaVid ablations on HM3D-v2.
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$BASELINE_ROOT/run.sh" eval \
  --resume \
  --context-window 0 \
  --output "$BASELINE_ROOT/results/hm3d_v2_100_win0"

bash "$BASELINE_ROOT/run.sh" eval \
  --resume \
  --no-visual-tokens \
  --output "$BASELINE_ROOT/results/hm3d_v2_100_no_visual"

bash "$BASELINE_ROOT/run.sh" eval \
  --resume \
  --current-frame-only-64 \
  --output "$BASELINE_ROOT/results/hm3d_v2_100_current64"
