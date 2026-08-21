#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESOURCES="${RESOURCES:-single}"
SHARD_SIZE="${SHARD_SIZE:-6}"
NUM_VLMS="${NUM_VLMS:-2}"
NUM_SIMS="${NUM_SIMS:-3}"
VLM_GPU_FRACTION="${VLM_GPU_FRACTION:-0.4}"
SIM_GPU_FRACTION="${SIM_GPU_FRACTION:-0.05}"
RUN_SUFFIX="${RUN_SUFFIX:-_oracle_stop}"

for MODE in stop_then_next stop_only found_only found_then_next; do
  RUN_NAME="onemap_multi_all3_${MODE}${RUN_SUFFIX}"
  echo "Starting ${RUN_NAME}"
  python -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=onemap_multi +experiment=eval_multi +resources="${RESOURCES}" \
    task.run_name="${RUN_NAME}" \
    task.shard_size="${SHARD_SIZE}" \
    resources.num_vlms="${NUM_VLMS}" \
    resources.num_sims="${NUM_SIMS}" \
    resources.vlm_gpu_fraction="${VLM_GPU_FRACTION}" \
    resources.sim_gpu_fraction="${SIM_GPU_FRACTION}" \
    rollout.reveal_all_goals=true \
    rollout.multi_goal_transition="${MODE}" \
    rollout.flush_on_goal_switch=false \
    rollout.post_goal_stop_veto=0 \
    sim.fn_guard=true \
    sim.fp_guard=false \
    sim.minimal_logging=true
done
