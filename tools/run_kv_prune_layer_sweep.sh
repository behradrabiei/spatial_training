#!/usr/bin/env bash
# Layer-selective KV-prune sweep: every selector in METHODS at every prune layer in LAYERS and
# every budget in BUDGETS, on the ~100-ep HM3D-v2 split (SCREEN=true: the 25-ep seed-17 subset).
# The budget applies to decoder layers [layer, 28); layer 0 is uniform pruning (today's arms).
#
#   METHODS="random attn" LAYERS="0 7 14 21" BUDGETS="1799" bash tools/run_kv_prune_layer_sweep.sh
#   METHODS="random" LAYERS="0 14" BUDGETS="1799" SCREEN=true bash tools/run_kv_prune_layer_sweep.sh
#
# Run names: <PREFIX_BASE>_<method>_w32_b<budget> (layer 0) and
#            <PREFIX_BASE>_<method>_w32_l<layer>_b<budget> (layer > 0);
# plot with tools/plot_kv_prune_layers.py, compare with tools/compare_runs.py. Other knobs of
# tools/run_kv_prune_ablation.sh (SCORE_LAYERS, VOXEL=..., RESOURCES, ...) pass through the env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
FULL_JSON="${FULL_JSON:-$ROOT/dump/hm3d_v2_100_labels.json}"
SCREEN_JSON="${SCREEN_JSON:-$ROOT/dump/hm3d_v2_25_seed17_labels.json}"
SCREEN="${SCREEN:-false}"
WINDOW="${WINDOW-32}"
PREFIX_BASE="${PREFIX_BASE:-kvprune_layer}"
read -r -a METHODS <<< "${METHODS:-random attn}"
read -r -a LAYERS <<< "${LAYERS:-0 7 14 21}"
read -r -a BUDGETS <<< "${BUDGETS:?set BUDGETS, e.g. BUDGETS=\"1187 1799\"}"

EPISODE_JSON="$FULL_JSON"
if [[ "$SCREEN" == "true" ]]; then
  EPISODE_JSON="$SCREEN_JSON"
  if [[ ! -f "$SCREEN_JSON" ]]; then
    "$PYTHON" tools/sample_episode_manifest.py "$FULL_JSON" "$SCREEN_JSON" --count 25 --seed 17
  fi
  PREFIX_BASE="${PREFIX_BASE}_screen"
fi

for method in "${METHODS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    echo "### layer sweep: method=${method} layer_start=${layer} budgets=${BUDGETS[*]}"
    PYTHON="$PYTHON" EPISODE_JSON="$EPISODE_JSON" WINDOW="$WINDOW" \
      IMPORTANCE="$method" LAYER_START="$layer" BUDGETS="${BUDGETS[*]}" \
      PREFIX="${PREFIX_BASE}_${method}" \
      bash tools/run_kv_prune_ablation.sh
  done
done

echo "Layer sweep complete. Plot with:"
echo "  /home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_prune_layers.py --prefix ${PREFIX_BASE}"
