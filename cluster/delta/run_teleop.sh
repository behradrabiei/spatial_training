#!/usr/bin/env bash
# Tele-operate one HM3D-v2 val episode with an adapter checkpoint, then hand control
# to the model (longnav.scripts.teleop_eval: W/A/D move, X stop, Space = model, Q abort).
# Needs a TTY on a GPU node, i.e. run it inside an interactive session:
#   cluster/delta/request_interactive.sh      (or: srun --jobid=<alloc> --overlap --pty bash --login)
#   TELEOP_CHECKPOINT=<adapter dir> EPISODE_INDEX=0 cluster/delta/run_teleop.sh
# Knobs: TELEOP_CHECKPOINT (default: staged stage-1 adapter), TELEOP_HAMLET (on|off, default on --
#        must match how the checkpoint was trained), EPISODE_INDEX (0): position in the full
#        HM3D-v2 val set = position in src/longnav/conf/episode_jsons/hm3d_v2_val.json (1000
#        episodes, scenes alphabetical, label <scene>_<0..27>; index 28*k+j = scene k, episode j),
#        or into EVAL_EPISODES (optional JSON label list, e.g. cluster/delta/hm3d_v2_val36.json),
#        RUN_NAME (teleop_<utc>), TELEOP_FPS (4), EVAL_MAX_STEPS (350), OSM_GB (12).
#        Extra Hydra overrides may follow as arguments.
# Output: $LONGNAV_OUTPUT_ROOT/<RUN_NAME>/teleop/<EPISODE_INDEX>_<label>/{current.png,episode.mp4,trace.json}
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Delta GPU allocation (interactive session)." >&2; exit 2; }
[[ -t 0 ]] || { echo "teleop needs an interactive terminal (stdin is not a TTY); use srun --pty." >&2; exit 2; }
longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_require_data
longnav_enable_offline_mode
LONGNAV_LOCAL_BASE_MODEL="$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path base)"
TELEOP_CHECKPOINT="${TELEOP_CHECKPOINT:-$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path adapter)}"
longnav_require_file "${TELEOP_CHECKPOINT}/adapter_model.safetensors"
EPISODES="${EVAL_EPISODES:-}"   # empty = index the whole dataset loaded by habitat
[[ -z "${EPISODES}" ]] || longnav_require_file "${EPISODES}"
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-/tmp/${USER}/longnav-${SLURM_JOB_ID}/spill}"
mkdir -p "${RAY_OBJECT_SPILL_DIR}" "${LONGNAV_OUTPUT_ROOT}"
RUN_NAME="${RUN_NAME:-teleop_$(date -u +%Y%m%dT%H%M%SZ)}"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
stop_ray() { "${LONGNAV_VLM_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true; }
trap stop_ray EXIT
echo "[run_teleop] run=${RUN_NAME} hamlet=${TELEOP_HAMLET:-on} episode_index=${EPISODE_INDEX:-0} of ${EPISODES:-full HM3D-v2 val} ckpt=${TELEOP_CHECKPOINT}"
cd "${LONGNAV_REPO_ROOT}"
"${LONGNAV_VLM_ENV}/bin/python" -m longnav.scripts.teleop_eval \
    +checkpoint=longnav \
    +dataset=hm3d_v2_val \
    +experiment=eval \
    +resources=single \
    "+hamlet=${TELEOP_HAMLET:-on}" \
    "training.checkpoint=${TELEOP_CHECKPOINT}" \
    "vlm.model_id=${LONGNAV_LOCAL_BASE_MODEL}" \
    "sim.workspace=${LONGNAV_SIM_WORKSPACE}" \
    "sim.config_path=${LONGNAV_REPO_ROOT}/habitat_configs/objectnav_hm3d_v2.yaml" \
    "sim.dataset_path=${LONGNAV_HM3D_EPISODES}" \
    "sim.scenes_dir=${LONGNAV_HM3D_SCENES}" \
    "rollout.max_steps=${EVAL_MAX_STEPS:-350}" \
    "task.run_name=${RUN_NAME}" \
    "task.output_dir=${LONGNAV_OUTPUT_ROOT}" \
    task.subset_label= \
    "task.episode_json=${EPISODES}" \
    "teleop.episode_index=${EPISODE_INDEX:-0}" \
    "teleop.fps=${TELEOP_FPS:-4}" \
    "resources.vlm_conda_env=${LONGNAV_VLM_ENV}" \
    "resources.habitat_conda_env=${LONGNAV_HABITAT_ENV}" \
    "resources.object_spilling_directory=${RAY_OBJECT_SPILL_DIR}" \
    "resources.osm_gb=${OSM_GB:-12}" \
    hydra.job.chdir=false \
    "$@"
