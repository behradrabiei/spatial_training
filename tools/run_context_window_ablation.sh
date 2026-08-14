#!/usr/bin/env bash
# Run LongNav eval across context-window sizes on the ~100-ep HM3D V2 subset.
#
#   bash tools/run_context_window_ablation.sh
#   MODE=recompute PREFIX=hm3d_v2_100_recompwin WINDOWS="32" bash tools/run_context_window_ablation.sh
#
# MODE=evict slices old turns' K/V out of the cache; MODE=recompute rebuilds the window's
# K/V from scratch, so evicted frames cannot reach the decision through the surviving keys.
# Skip the "full" tag under recompute: nothing is ever evicted at full context, so the two
# modes are identical there and the evict run is the shared anchor.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EPISODE_JSON="${EPISODE_JSON:-$ROOT/dump/hm3d_v2_100_labels.json}"
RESOURCES="${RESOURCES:-single}"
SHARD_SIZE="${SHARD_SIZE:-5}"
MODE="${MODE:-evict}"
PREFIX="${PREFIX:-hm3d_v2_100_win}"

# Tags: "full" omits vlm.context_window; integers set vlm.context_window=N
read -r -a WINDOWS <<< "${WINDOWS:-full 64 32 16 8 4 1}"

VIZ_OFF=(
  rollout.visualize_token_filtering=false
  rollout.visualize_attention=false
  rollout.visualize_attention_heads=false
  rollout.visualize_attention_3d=false
  sim.add_top_down_map=false
  sim.visualize_3d=false
  sim.visualize_attn3d=false
)

for TAG in "${WINDOWS[@]}"; do
  RUN_NAME="${PREFIX}${TAG}"
  echo "========================================"
  echo "Starting context-window ablation: ${RUN_NAME} (mode=${MODE})"
  echo "========================================"

  EXTRA=()
  if [[ "${TAG}" != "full" ]]; then
    EXTRA+=(vlm.context_window="${TAG}" vlm.context_window_mode="${MODE}")
  elif [[ "${MODE}" != "evict" ]]; then
    echo "Skipping ${RUN_NAME}: full context never evicts, so mode=${MODE} has nothing to do."
    continue
  fi

  python -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources="${RESOURCES}" \
    task.run_name="${RUN_NAME}" \
    task.subset_label="" \
    task.episode_json="${EPISODE_JSON}" \
    task.shard_size="${SHARD_SIZE}" \
    "${VIZ_OFF[@]}" \
    "${EXTRA[@]}"

  echo "Finished ${RUN_NAME}"
done

echo "All context-window ablation runs complete."
echo "Plot with: python tools/plot_context_window_ablation.py"
