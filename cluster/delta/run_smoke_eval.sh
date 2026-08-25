#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this command inside a Delta interactive GPU session." >&2
    exit 2
fi
longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_require_data
longnav_require_file "${LONGNAV_MODEL_MANIFEST}"

longnav_enable_offline_mode
export LONGNAV_LOCAL_CHECKPOINT
export LONGNAV_LOCAL_BASE_MODEL
LONGNAV_LOCAL_CHECKPOINT="$(
    "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path adapter
)"
LONGNAV_LOCAL_BASE_MODEL="$(
    "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path base
)"
"${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --validate-only

export LONGNAV_RAY_ROOT="${LONGNAV_RAY_ROOT:-/tmp/${USER}/longnav-${SLURM_JOB_ID}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${LONGNAV_RAY_ROOT}/tmp}"
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-${LONGNAV_RAY_ROOT}/spill}"
mkdir -p "${RAY_TMPDIR}" "${RAY_OBJECT_SPILL_DIR}" "${LONGNAV_OUTPUT_ROOT}"

RUN_NAME="${LONGNAV_RUN_NAME:-delta_smoke5_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${LONGNAV_OUTPUT_ROOT}/${RUN_NAME}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

stop_ray() {
    "${LONGNAV_VLM_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true
}
trap stop_ray EXIT

cd "${LONGNAV_REPO_ROOT}"
"${LONGNAV_VLM_ENV}/bin/python" -m longnav.scripts.eval \
    +checkpoint=longnav \
    +dataset=hm3d_v2_val \
    +experiment=eval \
    +resources=single \
    "training.checkpoint=${LONGNAV_LOCAL_CHECKPOINT}" \
    "vlm.model_id=${LONGNAV_LOCAL_BASE_MODEL}" \
    vlm.attn_impl=sdpa \
    "sim.workspace=${LONGNAV_REPO_ROOT}" \
    "sim.config_path=${LONGNAV_REPO_ROOT}/habitat_configs/objectnav_hm3d_v2.yaml" \
    "sim.dataset_path=${LONGNAV_HM3D_EPISODES}" \
    "sim.scenes_dir=${LONGNAV_HM3D_SCENES}" \
    sim.minimal_logging=true \
    sim.add_top_down_map=false \
    sim.visualize_3d=false \
    sim.visualize_attn3d=false \
    rollout.max_steps=50 \
    rollout.visualize_token_filtering=false \
    rollout.visualize_attention=false \
    rollout.visualize_attention_heads=false \
    rollout.visualize_attention_3d=false \
    "task.run_name=${RUN_NAME}" \
    "task.output_dir=${LONGNAV_OUTPUT_ROOT}" \
    task.wandb_project=null \
    task.subset_label= \
    task.shard_size=5 \
    "task.episode_json=${LONGNAV_SMOKE_EPISODES}" \
    "resources.vlm_conda_env=${LONGNAV_VLM_ENV}" \
    "resources.habitat_conda_env=${LONGNAV_HABITAT_ENV}" \
    "resources.object_spilling_directory=${RAY_OBJECT_SPILL_DIR}" \
    resources.osm_gb=12 \
    hydra.job.chdir=false

"${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/validate_results.py" \
    "${RUN_DIR}" "${LONGNAV_SMOKE_EPISODES}"
