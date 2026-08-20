#!/usr/bin/env bash
# Run LongNav eval with context_window_mode='prune' (KV token budget) on the ~100-ep
# HM3D V2 subset. One run per budget in BUDGETS.
#
#   BUDGETS="1000" bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000 1800" WINDOW=32 bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000" WINDOW=32 MERGE=true PREFIX=hm3d_v2_100_prunemerge bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000" WINDOW=32 IMPORTANCE=random PREFIX=hm3d_v2_100_prunerand bash tools/run_kv_prune_ablation.sh
#
# WINDOW="" (default 32) drops the frame pool entirely: budget-only over the whole episode.
# Baselines with the same kv_len logging come from run_context_window_ablation.sh
# (MODE=reindex). Compare with tools/compare_runs.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EPISODE_JSON="${EPISODE_JSON:-$ROOT/dump/hm3d_v2_100_labels.json}"
RESOURCES="${RESOURCES:-single}"
SHARD_SIZE="${SHARD_SIZE:-5}"
WINDOW="${WINDOW-32}"  # explicit WINDOW="" means no frame pool (budget-only)
MERGE="${MERGE:-false}"
IMPORTANCE="${IMPORTANCE:-attn}"
GRANULARITY="${GRANULARITY:-slot}"
RECENT="${RECENT:-2}"
BETA="${BETA:-0.7}"
PREFIX="${PREFIX:-hm3d_v2_100_prune}"

read -r -a BUDGETS <<< "${BUDGETS:?set BUDGETS, e.g. BUDGETS=\"1000 1800\"}"

VIZ_OFF=(
  rollout.visualize_token_filtering=false
  rollout.visualize_attention=false
  rollout.visualize_attention_heads=false
  rollout.visualize_attention_3d=false
  sim.add_top_down_map=false
  sim.visualize_3d=false
  sim.visualize_attn3d=false
)

for BUDGET in "${BUDGETS[@]}"; do
  if [[ -n "${WINDOW}" ]]; then
    RUN_NAME="${PREFIX}_w${WINDOW}_b${BUDGET}"
    WINDOW_ARGS=(vlm.context_window="${WINDOW}")
  else
    RUN_NAME="${PREFIX}_nowin_b${BUDGET}"
    WINDOW_ARGS=()
  fi
  echo "========================================"
  echo "Starting KV-prune ablation: ${RUN_NAME} (merge=${MERGE}, importance=${IMPORTANCE})"
  echo "========================================"

  python -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources="${RESOURCES}" \
    task.run_name="${RUN_NAME}" \
    task.subset_label="" \
    task.episode_json="${EPISODE_JSON}" \
    task.shard_size="${SHARD_SIZE}" \
    vlm.context_window_mode=prune \
    vlm.kv_budget="${BUDGET}" \
    vlm.kv_prune_merge="${MERGE}" \
    vlm.kv_prune_importance="${IMPORTANCE}" \
    vlm.kv_prune_granularity="${GRANULARITY}" \
    vlm.kv_prune_recent_turns="${RECENT}" \
    vlm.kv_prune_ema_beta="${BETA}" \
    "${VIZ_OFF[@]}" \
    "${WINDOW_ARGS[@]}"

  echo "Finished ${RUN_NAME}"
done

echo "All KV-prune ablation runs complete."
echo "Compare with: python tools/compare_runs.py <runs...> --baseline <baseline>"
