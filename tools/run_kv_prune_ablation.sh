#!/usr/bin/env bash
# Run LongNav eval with context_window_mode='prune' (KV token budget) on the ~100-ep
# HM3D V2 subset. One run per budget in BUDGETS.
#
#   BUDGETS="1000" bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000 1800" WINDOW=32 bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000" WINDOW=32 MERGE=true PREFIX=hm3d_v2_100_prunemerge bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1000" WINDOW=32 IMPORTANCE=random PREFIX=hm3d_v2_100_prunerand bash tools/run_kv_prune_ablation.sh
#   BUDGETS="1799" LAYER_START=14 IMPORTANCE=random PREFIX=kvprune_layer_random bash tools/run_kv_prune_ablation.sh
#     -> run kvprune_layer_random_w32_l14_b1799: budget on decoder layers 14..27 only
#     (tools/run_kv_prune_layer_sweep.sh loops methods x layers x budgets)
#
# WINDOW="" (default 32) drops the frame pool entirely: budget-only over the whole episode.
# Baselines with the same kv_len logging come from run_context_window_ablation.sh
# (MODE=reindex). Compare with tools/compare_runs.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EPISODE_JSON="${EPISODE_JSON:-$ROOT/dump/hm3d_v2_100_labels.json}"
RESOURCES="${RESOURCES:-single}"
SHARD_SIZE="${SHARD_SIZE:-5}"
WINDOW="${WINDOW-32}"  # explicit WINDOW="" means no frame pool (budget-only)
MERGE="${MERGE:-false}"
IMPORTANCE="${IMPORTANCE:-attn}"
GRANULARITY="${GRANULARITY:-slot}"
RECENT="${RECENT:-2}"
BETA="${BETA:-0.7}"
SCOPE="${SCOPE:-all}"
SEED="${SEED:-17}"
FISHER_POOL_FACTOR="${FISHER_POOL_FACTOR:-2.0}"
PREFIX="${PREFIX:-hm3d_v2_100_prune}"
# Voxel selectors (IMPORTANCE=voxel_dedup|voxel_strat) need the sim to attach per-patch world
# voxels and the VLM to keep stock position ids: VOXEL=true POS_ID_MODE=standard.
VOXEL="${VOXEL:-false}"
POS_ID_MODE="${POS_ID_MODE:-}"
VOXEL_CAP="${VOXEL_CAP:-1}"
VOXEL_SCALE="${VOXEL_SCALE:-2}"
VOXEL_2D="${VOXEL_2D:-true}"
# Layer-selective pruning: the budget applies to decoder layers [LAYER_START, LAYER_END) only
# (LAYER_END="" = through the last layer). The run name gains "_l<start>[-<end>]" whenever the
# range is not the default "every layer", so existing run names are unchanged.
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-}"
SCORE_LAYERS="${SCORE_LAYERS:-pruned}"  # attn selector: pruned | all | boundary
# VISUAL_BLIND=true: hide and drop EVERY visual slot on layers [LAYER_START, LAYER_END) -- the
# information-horizon hypothesis test. No budget is involved; BUDGETS may be left unset.
VISUAL_BLIND="${VISUAL_BLIND:-false}"

if [[ "${VISUAL_BLIND}" == "true" ]]; then
  read -r -a BUDGETS <<< "${BUDGETS:-}"
  if [[ ${#BUDGETS[@]} -eq 0 ]]; then
    BUDGETS=("")
  fi
else
  read -r -a BUDGETS <<< "${BUDGETS:?set BUDGETS, e.g. BUDGETS=\"1000 1800\"}"
fi

LAYER_TAG=""
LAYER_ARGS=(vlm.kv_prune_layer_start="${LAYER_START}" vlm.kv_prune_score_layers="${SCORE_LAYERS}")
if [[ "${LAYER_START}" != "0" || -n "${LAYER_END}" ]]; then
  LAYER_TAG="_l${LAYER_START}"
  if [[ -n "${LAYER_END}" ]]; then
    LAYER_TAG+="-${LAYER_END}"
    LAYER_ARGS+=(vlm.kv_prune_layer_end="${LAYER_END}")
  fi
fi
if [[ "${VISUAL_BLIND}" == "true" ]]; then
  LAYER_ARGS+=(vlm.kv_prune_visual_blind=true)
  LAYER_TAG="_vblind${LAYER_TAG}"
fi

EXTRA_ARGS=()
if [[ "${VOXEL}" == "true" ]]; then
  # '+' appends keys to the (struct) voxel_kwargs dict; a bare dict override is rejected.
  EXTRA_ARGS+=(+sim.voxel_kwargs.patch_size=32 +sim.voxel_kwargs.resolution=0.15
               +sim.voxel_kwargs.fov_degrees=79 sim.output_schema.obs.patch_coords=true)
fi
if [[ -n "${POS_ID_MODE}" ]]; then
  EXTRA_ARGS+=(rollout.pos_id_mode="${POS_ID_MODE}")
fi
# Any further Hydra overrides, space-separated (e.g. EXTRA_HYDRA="vlm.kv_prune_log_layer_influence=true").
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  read -r -a EXTRA_HYDRA_ARR <<< "${EXTRA_HYDRA}"
  EXTRA_ARGS+=("${EXTRA_HYDRA_ARR[@]}")
fi

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
  BUDGET_TAG=""
  BUDGET_ARGS=()
  if [[ -n "${BUDGET}" ]]; then
    BUDGET_TAG="_b${BUDGET}"
    BUDGET_ARGS=(vlm.kv_budget="${BUDGET}")
  fi
  if [[ -n "${WINDOW}" ]]; then
    RUN_NAME="${PREFIX}_w${WINDOW}${LAYER_TAG}${BUDGET_TAG}"
    WINDOW_ARGS=(vlm.context_window="${WINDOW}")
  else
    RUN_NAME="${PREFIX}_nowin${LAYER_TAG}${BUDGET_TAG}"
    WINDOW_ARGS=()
  fi
  echo "========================================"
  echo "Starting KV-prune ablation: ${RUN_NAME} (merge=${MERGE}, importance=${IMPORTANCE}, scope=${SCOPE}, seed=${SEED}, layers=${LAYER_START}..${LAYER_END:-end}, score_layers=${SCORE_LAYERS})"
  echo "========================================"

  "$PYTHON" -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources="${RESOURCES}" \
    task.run_name="${RUN_NAME}" \
    task.subset_label="" \
    task.episode_json="${EPISODE_JSON}" \
    task.shard_size="${SHARD_SIZE}" \
    vlm.context_window_mode=prune \
    "${BUDGET_ARGS[@]}" \
    vlm.kv_prune_merge="${MERGE}" \
    vlm.kv_prune_importance="${IMPORTANCE}" \
    vlm.kv_prune_granularity="${GRANULARITY}" \
    vlm.kv_prune_recent_turns="${RECENT}" \
    vlm.kv_prune_ema_beta="${BETA}" \
    vlm.kv_prune_candidate_scope="${SCOPE}" \
    vlm.kv_prune_seed="${SEED}" \
    vlm.kv_prune_fisher_pool_factor="${FISHER_POOL_FACTOR}" \
    vlm.kv_prune_voxel_cap="${VOXEL_CAP}" \
    vlm.kv_prune_voxel_scale="${VOXEL_SCALE}" \
    vlm.kv_prune_voxel_2d="${VOXEL_2D}" \
    "${LAYER_ARGS[@]}" \
    "${VIZ_OFF[@]}" \
    "${WINDOW_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "Finished ${RUN_NAME}"
done

echo "All KV-prune ablation runs complete."
echo "Compare with: python tools/compare_runs.py <runs...> --baseline <baseline>"
