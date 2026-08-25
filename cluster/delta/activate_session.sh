#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file so the environment remains active:" >&2
    echo "  source cluster/delta/activate_session.sh" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "No Slurm allocation detected. Run request_interactive.sh first." >&2
    return 1
fi
longnav_require_environment "${LONGNAV_VLM_ENV}" || return 1
longnav_require_environment "${LONGNAV_HABITAT_ENV}" || return 1
longnav_require_data || return 1

CONDA_BIN="$(longnav_find_conda)" || {
    echo "Could not locate conda." >&2
    return 1
}
CONDA_ROOT="$(cd -- "$(dirname -- "${CONDA_BIN}")/.." && pwd)"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${LONGNAV_VLM_ENV}"

longnav_enable_offline_mode
export TOKENIZERS_PARALLELISM=false
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LONGNAV_RAY_ROOT="${LONGNAV_RAY_ROOT:-/tmp/${USER}/longnav-${SLURM_JOB_ID}}"
export RAY_TMPDIR="${LONGNAV_RAY_ROOT}/tmp"
export RAY_OBJECT_SPILL_DIR="${LONGNAV_RAY_ROOT}/spill"
mkdir -p "${RAY_TMPDIR}" "${RAY_OBJECT_SPILL_DIR}"

_longnav_stop_ray() {
    ray stop --force >/dev/null 2>&1 || true
}
trap _longnav_stop_ray EXIT

cd "${LONGNAV_REPO_ROOT}" || return 1
echo "LongNav Delta session ready on $(hostname)."
nvidia-smi -L
echo "Debug commands: cluster/delta/debug.sh {gpu|imports|models|habitat|eval-smoke|rl-smoke|hm3d}"
