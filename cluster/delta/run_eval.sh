#!/usr/bin/env bash
# Evaluate an adapter checkpoint on a fixed HM3D-v2 val episode subset (1 GPU).
# Knobs: EVAL_CHECKPOINT (adapter dir; default = staged stage-1 adapter),
#        EVAL_HAMLET (on|off, default off), EVAL_EPISODES (JSON label list,
#        default hm3d_v2_val36.json), EVAL_SUBSET (explicit legacy constants-table
#        key), EVAL_MAX_STEPS (350), RUN_NAME, OSM_GB (12).
# Extra Hydra overrides may follow as arguments.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Delta GPU job." >&2; exit 2; }
longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_require_data
longnav_enable_offline_mode
LONGNAV_LOCAL_BASE_MODEL="$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path base)"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-$("${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path adapter)}"
longnav_require_file "${EVAL_CHECKPOINT}/adapter_model.safetensors"
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-/tmp/${USER}/longnav-${SLURM_JOB_ID}/spill}"
mkdir -p "${RAY_OBJECT_SPILL_DIR}" "${LONGNAV_OUTPUT_ROOT}"
RUN_NAME="${RUN_NAME:-eval_$(date -u +%Y%m%dT%H%M%SZ)}"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
stop_ray() { "${LONGNAV_VLM_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true; }
trap stop_ray EXIT
# EVAL_EPISODES takes precedence; EVAL_SUBSET is used only when explicitly set.
ACTIVE_EPISODES="${EVAL_EPISODES:-}"
if [[ -z "${ACTIVE_EPISODES}" && -z "${EVAL_SUBSET:-}" ]]; then ACTIVE_EPISODES="${SCRIPT_DIR}/hm3d_v2_val36.json"; fi
if [[ -n "${ACTIVE_EPISODES}" ]]; then longnav_require_file "${ACTIVE_EPISODES}"; SUBSET_OVERRIDE="task.subset_label="; EPISODES_OVERRIDE="task.episode_json=${ACTIVE_EPISODES}"
else SUBSET_OVERRIDE="task.subset_label=${EVAL_SUBSET}"; EPISODES_OVERRIDE="task.episode_json="; fi
echo "[run_eval] run=${RUN_NAME} hamlet=${EVAL_HAMLET:-off} episodes=${ACTIVE_EPISODES:-${EVAL_SUBSET}} ckpt=${EVAL_CHECKPOINT}"
cd "${LONGNAV_REPO_ROOT}"
"${LONGNAV_VLM_ENV}/bin/python" -m longnav.scripts.eval \
    +checkpoint=longnav \
    +dataset=hm3d_v2_val \
    +experiment=eval \
    +resources=single \
    "+hamlet=${EVAL_HAMLET:-off}" \
    "training.checkpoint=${EVAL_CHECKPOINT}" \
    "vlm.model_id=${LONGNAV_LOCAL_BASE_MODEL}" \
    vlm.attn_impl=sdpa \
    "sim.workspace=${LONGNAV_SIM_WORKSPACE}" \
    "sim.config_path=${LONGNAV_REPO_ROOT}/habitat_configs/objectnav_hm3d_v2.yaml" \
    "sim.dataset_path=${LONGNAV_HM3D_EPISODES}" \
    "sim.scenes_dir=${LONGNAV_HM3D_SCENES}" \
    sim.minimal_logging=true \
    sim.add_top_down_map=false \
    sim.visualize_3d=false \
    sim.visualize_attn3d=false \
    "rollout.max_steps=${EVAL_MAX_STEPS:-350}" \
    rollout.visualize_token_filtering=false \
    rollout.visualize_attention=false \
    rollout.visualize_attention_heads=false \
    rollout.visualize_attention_3d=false \
    "task.run_name=${RUN_NAME}" \
    "task.output_dir=${LONGNAV_OUTPUT_ROOT}" \
    task.wandb_project=null \
    "${SUBSET_OVERRIDE}" \
    "${EPISODES_OVERRIDE}" \
    task.shard_size=5 \
    "resources.vlm_conda_env=${LONGNAV_VLM_ENV}" \
    "resources.habitat_conda_env=${LONGNAV_HABITAT_ENV}" \
    "resources.object_spilling_directory=${RAY_OBJECT_SPILL_DIR}" \
    "resources.osm_gb=${OSM_GB:-12}" \
    hydra.job.chdir=false \
    ${EVAL_EXTRA:-} \
    "$@"
# A fixed evaluation is only valid when every requested label produced exactly
# one result. Fail the job instead of silently aggregating a partial run.
if [[ -n "${ACTIVE_EPISODES}" ]]; then
    "${LONGNAV_VLM_ENV}/bin/python" - "${ACTIVE_EPISODES}" "${LONGNAV_OUTPUT_ROOT}/${RUN_NAME}" <<'PY'
import glob, json, os, sys
requested = json.load(open(sys.argv[1]))
rows = [json.loads(line) for path in glob.glob(os.path.join(sys.argv[2], "rollout", "results_*"))
        for line in open(path) if line.strip()]
labels = [row["episode_label"] for row in rows]
missing, extra = sorted(set(requested) - set(labels)), sorted(set(labels) - set(requested))
if len(labels) != len(requested) or missing or extra:
    raise SystemExit(f"ERROR: incomplete evaluation: expected={len(requested)} got={len(labels)} "
                     f"missing={missing} extra={extra}")
print(f"[run_eval] validated {len(labels)} requested episodes")
PY
fi
echo "[run_eval] done: ${LONGNAV_OUTPUT_ROOT}/${RUN_NAME}"
