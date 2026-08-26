#!/usr/bin/env bash

# Shared paths for Delta setup and runtime scripts. Every value can be
# overridden before sourcing this file.
LONGNAV_DELTA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LONGNAV_REPO_ROOT="${LONGNAV_REPO_ROOT:-$(cd -- "${LONGNAV_DELTA_DIR}/../.." && pwd)}"
export HABITAT_DATA_ROOT="${HABITAT_DATA_ROOT:-/work/nvme/bgon/brabiei/habitat_data}"
export LONGNAV_RUNTIME_ROOT="${LONGNAV_RUNTIME_ROOT:-/work/nvme/bgon/brabiei/longnav_runtime}"

export LONGNAV_ENV_ROOT="${LONGNAV_ENV_ROOT:-${LONGNAV_RUNTIME_ROOT}/envs}"
export LONGNAV_VLM_ENV="${LONGNAV_VLM_ENV:-${LONGNAV_ENV_ROOT}/longnav_vlm}"
export LONGNAV_HABITAT_ENV="${LONGNAV_HABITAT_ENV:-${LONGNAV_ENV_ROOT}/vln}"
export LONGNAV_MODEL_ROOT="${LONGNAV_MODEL_ROOT:-${LONGNAV_RUNTIME_ROOT}/models}"
export LONGNAV_MODEL_MANIFEST="${LONGNAV_MODEL_MANIFEST:-${LONGNAV_MODEL_ROOT}/manifest.json}"
export LONGNAV_OUTPUT_ROOT="${LONGNAV_OUTPUT_ROOT:-${LONGNAV_RUNTIME_ROOT}/runs}"
export LONGNAV_MANIFEST_ROOT="${LONGNAV_MANIFEST_ROOT:-${LONGNAV_RUNTIME_ROOT}/manifests}"
export LONGNAV_LOG_ROOT="${LONGNAV_LOG_ROOT:-${LONGNAV_RUNTIME_ROOT}/logs}"
# Habitat working directory (sim.workspace): holds data/scene_datasets symlinks
# so the episodes' cwd-relative scene_dataset_config paths resolve on Delta.
export LONGNAV_SIM_WORKSPACE="${LONGNAV_SIM_WORKSPACE:-${LONGNAV_RUNTIME_ROOT}/workspace}"

export HABITAT_SIM_COMMIT="${HABITAT_SIM_COMMIT:-57ee4941dc4765240f0f91f70b2c97a919bf9038}"
export HABITAT_SIM_SOURCE="${HABITAT_SIM_SOURCE:-${LONGNAV_RUNTIME_ROOT}/sources/habitat-sim}"
export HABITAT_SIM_WHEEL_ROOT="${HABITAT_SIM_WHEEL_ROOT:-${LONGNAV_RUNTIME_ROOT}/wheels}"
export HABITAT_SIM_WHEEL="${HABITAT_SIM_WHEEL:-${HABITAT_SIM_WHEEL_ROOT}/habitat_sim-0.3.3-cp310-cp310-linux_x86_64.whl}"

export CONDARC="${CONDARC:-${LONGNAV_RUNTIME_ROOT}/config/condarc}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${LONGNAV_RUNTIME_ROOT}/conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LONGNAV_RUNTIME_ROOT}/cache/pip}"
export HF_HOME="${HF_HOME:-${LONGNAV_RUNTIME_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${LONGNAV_RUNTIME_ROOT}/cache/torch}"

export LONGNAV_HM3D_EPISODES="${LONGNAV_HM3D_EPISODES:-${HABITAT_DATA_ROOT}/evaluation_episodes/HM3D/objectnav_hm3d_v2/val/val.json.gz}"
export LONGNAV_HM3D_SCENES="${LONGNAV_HM3D_SCENES:-${HABITAT_DATA_ROOT}/scenes/HM3D/v2}"
export LONGNAV_SMOKE_EPISODES="${LONGNAV_SMOKE_EPISODES:-${LONGNAV_DELTA_DIR}/hm3d_v2_smoke5.json}"
# HM3D ObjectNav v1 train split (what the first RL stage trained on); scenes
# resolve through the `hm3d -> hm3d_v0.2` symlink created by prepare_hm3d_train.sh.
export LONGNAV_HM3D_TRAIN_EPISODES="${LONGNAV_HM3D_TRAIN_EPISODES:-${HABITAT_DATA_ROOT}/evaluation_episodes/HM3D/objectnav_hm3d_v1/train/train.json.gz}"

longnav_find_conda() {
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        printf '%s\n' "${CONDA_EXE}"
    elif command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [[ -x /sw/rh9.4/python/miniforge3/bin/conda ]]; then
        printf '%s\n' /sw/rh9.4/python/miniforge3/bin/conda
    else
        return 1
    fi
}

longnav_enable_offline_mode() {
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export PIP_NO_INDEX=1
    export WANDB_MODE=offline
}

longnav_require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Missing required file: %s\n' "$1" >&2
        return 1
    fi
}

longnav_require_environment() {
    if [[ ! -x "$1/bin/python" ]]; then
        printf 'Conda environment is not installed: %s\nRun cluster/delta/setup.sh on a login node first.\n' "$1" >&2
        return 1
    fi
}

longnav_require_data() {
    local scene_dir="${LONGNAV_HM3D_SCENES}/hm3d_v0.2/val/00877-4ok3usBNeis"
    longnav_require_file "${LONGNAV_HM3D_EPISODES}"
    longnav_require_file "${LONGNAV_HM3D_EPISODES%/*}/content/4ok3usBNeis.json.gz"
    longnav_require_file "${scene_dir}/4ok3usBNeis.basis.glb"
    longnav_require_file "${scene_dir}/4ok3usBNeis.basis.navmesh"
}

# Ray's conda runtime_env (utils/factories.py) shells out to `conda info --json`
# and `conda activate <prefix>` on every node. Compute nodes have no conda on
# PATH, so point Ray at the miniforge the envs were built with.
if [[ -z "${RAY_CONDA_HOME:-}" ]]; then
    _longnav_conda_bin="$(longnav_find_conda 2>/dev/null || true)"
    if [[ -n "${_longnav_conda_bin}" ]]; then
        export RAY_CONDA_HOME="$(cd -- "$(dirname -- "${_longnav_conda_bin}")/.." && pwd)"
    fi
    unset _longnav_conda_bin
fi
