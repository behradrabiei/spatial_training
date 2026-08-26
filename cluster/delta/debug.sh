#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

usage() {
    cat <<'EOF'
Usage: cluster/delta/debug.sh COMMAND

Commands:
  gpu         Check CUDA, BF16, and SDPA
  imports     Check both Conda environments and pinned versions
  models      Validate the staged model bundle without network access
  habitat     Reset and step one real HM3D v2 episode
  eval-smoke  Run the repository's dummy-environment inference smoke test
  rl-smoke    Run the repository's one-step RL smoke test
  hamlet-smoke Run the HAMLET (moment tokens + memory) rollout/train/checkpoint smoke test
  hm3d        Run and validate the five-episode HM3D evaluation
EOF
}

command_name="${1:-}"
if [[ -z "${command_name}" || "${command_name}" == "-h" || "${command_name}" == "--help" ]]; then
    usage
    exit 0
fi

longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_enable_offline_mode
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-/tmp/${USER}/longnav-${SLURM_JOB_ID:-login}/spill}"
mkdir -p "${RAY_OBJECT_SPILL_DIR}"

model_path() {
    "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --print-path "$1"
}

require_gpu_session() {
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "This check requires the interactive GPU session." >&2
        exit 2
    fi
}

case "${command_name}" in
    gpu)
        require_gpu_session
        exec "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/gpu_smoke.py"
        ;;
    imports)
        "${LONGNAV_VLM_ENV}/bin/python" -c \
            'import longnav, peft, ray, torch, transformers, trl; print(f"longnav_vlm: torch={torch.__version__}, transformers={transformers.__version__}, ray={ray.__version__}")'
        "${LONGNAV_HABITAT_ENV}/bin/python" -c \
            'from importlib.metadata import version; import cv2, habitat, habitat_sim, longnav, ray, regex; habitat_sim_version = version("habitat-sim"); print("vln: habitat-sim={}, ray={}, cv2={}".format(habitat_sim_version, ray.__version__, cv2.__version__))'
        ;;
    models)
        exec "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --validate-only
        ;;
    habitat)
        require_gpu_session
        longnav_require_data
        exec "${LONGNAV_HABITAT_ENV}/bin/python" "${SCRIPT_DIR}/habitat_smoke.py"
        ;;
    eval-smoke)
        require_gpu_session
        export LONGNAV_MODEL_ID="$(model_path base)"
        cd "${LONGNAV_REPO_ROOT}"
        exec "${LONGNAV_VLM_ENV}/bin/python" tests/eval_smoke.py
        ;;
    rl-smoke)
        require_gpu_session
        export LONGNAV_MODEL_ID="$(model_path base)"
        cd "${LONGNAV_REPO_ROOT}"
        exec "${LONGNAV_VLM_ENV}/bin/python" tests/rl_smoke.py
        ;;
    hamlet-smoke)
        require_gpu_session
        export LONGNAV_MODEL_ID="$(model_path base)"
        cd "${LONGNAV_REPO_ROOT}"
        exec "${LONGNAV_VLM_ENV}/bin/python" tests/hamlet_smoke.py
        ;;
    hm3d)
        require_gpu_session
        exec "${SCRIPT_DIR}/run_smoke_eval.sh"
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        usage >&2
        exit 2
        ;;
esac
