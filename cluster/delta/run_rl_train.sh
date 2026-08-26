#!/usr/bin/env bash
# Launch the HAMLET second-stage RL run (HM3D v1 train, 4 VLM actors) on the
# current Delta GPU allocation. Works inside `sbatch cluster/delta/train_hamlet.sbatch`
# and inside an interactive session. Extra Hydra overrides are passed through:
#   RUN_NAME=hamlet_mini cluster/delta/run_rl_train.sh rollout.max_steps=60 training.rl_config.n_rollout=4
# Environment knobs: RUN_NAME, OSM_GB (64), SAVE_STEP (4), MAX_WALLCLOCK_HOURS (7.6),
# WANDB_API_KEY (online logging; otherwise wandb runs offline under LONGNAV_OUTPUT_ROOT),
# reward shaping COLLISION_PENALTY (0.02) / EXPLR_BONUS (0.03) / FPSTOP_PENALTY (null) --
# the stage-1 run's values (its saved config), not the schema defaults 0.05/0.13/0.3 the
# first HAMLET runs silently used -- and the LoRA schedule LR (1.25e-6, the lr stage 1
# ended at after 7424 scheduler steps) / WARMUP_STEPS (32) / TOTAL_STEPS (20000).
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this inside a Delta GPU allocation (sbatch or an interactive session)." >&2
    exit 2
fi
longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_require_data
longnav_require_file "${LONGNAV_MODEL_MANIFEST}"
longnav_require_file "${LONGNAV_HM3D_TRAIN_EPISODES}"
if [[ ! -d "${LONGNAV_HM3D_SCENES}/hm3d_v0.2/train" || ! -L "${LONGNAV_HM3D_SCENES}/hm3d" \
      || ! -L "${LONGNAV_SIM_WORKSPACE}/data/scene_datasets/hm3d" ]]; then
    echo "HM3D train scenes are not prepared; run cluster/delta/prepare_hm3d_train.sh first." >&2
    exit 2
fi

longnav_enable_offline_mode
LONGNAV_LOCAL_CHECKPOINT="$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path adapter)"
RESUME_OVERRIDES=()
# START_CHECKPOINT=<dir with adapter_model.safetensors>: resume from a previous run's
# checkpoint (LoRA + HAMLET + frozen reference + optimizer + scheduler) instead
# of the staged stage-1 adapter. Older checkpoints without hamlet_reference.pt
# still load, but cannot reproduce their original HAMLET reference policy.
if [[ -n "${START_CHECKPOINT:-}" ]]; then
    longnav_require_file "${START_CHECKPOINT}/adapter_model.safetensors"
    longnav_require_file "${START_CHECKPOINT}/optimizer.pt"
    longnav_require_file "${START_CHECKPOINT}/scheduler.pt"
    LONGNAV_LOCAL_CHECKPOINT="${START_CHECKPOINT}"
    RESUME_OVERRIDES+=(training.load_optim=true training.load_sched=true)
fi
LONGNAV_LOCAL_BASE_MODEL="$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path base)"
"${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --validate-only

# wandb: online when a key is present, otherwise an offline run that can be synced later.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi
export WANDB_DIR="${LONGNAV_OUTPUT_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

export LONGNAV_RAY_ROOT="${LONGNAV_RAY_ROOT:-/tmp/${USER}/longnav-${SLURM_JOB_ID}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${LONGNAV_RAY_ROOT}/tmp}"
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-${LONGNAV_RAY_ROOT}/spill}"
mkdir -p "${RAY_TMPDIR}" "${RAY_OBJECT_SPILL_DIR}" "${LONGNAV_OUTPUT_ROOT}"
export TOKENIZERS_PARALLELISM=false HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

RUN_NAME="${RUN_NAME:-hamlet_rl_$(date -u +%Y%m%dT%H%M%SZ)}"
echo "[run_rl_train] run=${RUN_NAME} job=${SLURM_JOB_ID} node=$(hostname) wandb=${WANDB_MODE}"
echo "[run_rl_train] adapter=${LONGNAV_LOCAL_CHECKPOINT}"
echo "[run_rl_train] base=${LONGNAV_LOCAL_BASE_MODEL}"

stop_ray() {
    "${LONGNAV_VLM_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true
}
trap stop_ray EXIT

# objectnav_hm3d_v2.yaml is the stage-1 rgbd_semantic config minus the semantic
# sensor: nothing in training reads it (fp/fn guards off, output schema has no
# semantic field) and on Delta's numpy-2 `vln` env habitat-lab's semantic-sensor
# gym Box overflows (uint32 max into int32) before the first reset.
# Growing per-episode KV caches + full-episode training replays fragment the CUDA
# caching allocator; expandable segments keep the 40 GB A100s from spurious OOMs.
# Ray workers inherit the driver's environment, so this reaches every actor.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${LONGNAV_REPO_ROOT}"
"${LONGNAV_VLM_ENV}/bin/python" -m longnav.scripts.train_rl \
    +checkpoint=longnav \
    +dataset=hm3d_train \
    +experiment=train \
    +resources=quad \
    +training=hapo \
    +hamlet=on \
    "training.checkpoint=${LONGNAV_LOCAL_CHECKPOINT}" \
    "vlm.model_id=${LONGNAV_LOCAL_BASE_MODEL}" \
    vlm.attn_impl=sdpa \
    "sim.workspace=${LONGNAV_SIM_WORKSPACE}" \
    "sim.config_path=${LONGNAV_REPO_ROOT}/habitat_configs/objectnav_hm3d_v2.yaml" \
    "sim.dataset_path=${LONGNAV_HM3D_TRAIN_EPISODES}" \
    "sim.scenes_dir=${LONGNAV_HM3D_SCENES}" \
    sim.split=train \
    sim.minimal_logging=true \
    sim.add_top_down_map=false \
    sim.visualize_3d=false \
    sim.visualize_attn3d=false \
    rollout.visualize_token_filtering=false \
    rollout.visualize_attention=false \
    rollout.visualize_attention_heads=false \
    rollout.visualize_attention_3d=false \
    "task.run_name=${RUN_NAME}" \
    "task.output_dir=${LONGNAV_OUTPUT_ROOT}" \
    task.wandb_project=longnav_training \
    "resources.vlm_conda_env=${LONGNAV_VLM_ENV}" \
    "resources.habitat_conda_env=${LONGNAV_HABITAT_ENV}" \
    "resources.object_spilling_directory=${RAY_OBJECT_SPILL_DIR}" \
    "resources.osm_gb=${OSM_GB:-64}" \
    "sim.collision_penalty=${COLLISION_PENALTY:-0.02}" \
    "sim.explr_bonus=${EXPLR_BONUS:-0.03}" \
    "sim.fpstop_penalty=${FPSTOP_PENALTY:-null}" \
    "training.learning_rate=${LR:-1.25e-6}" \
    "training.warmup_steps=${WARMUP_STEPS:-32}" \
    "training.total_optimization_steps=${TOTAL_STEPS:-20000}" \
    "training.save_step=${SAVE_STEP:-4}" \
    "training.max_wallclock_hours=${MAX_WALLCLOCK_HOURS:-7.6}" \
    "${RESUME_OVERRIDES[@]}" \
    hydra.job.chdir=false \
    "$@"
