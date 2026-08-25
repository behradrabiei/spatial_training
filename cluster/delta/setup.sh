#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

CONDA_BIN="$(longnav_find_conda)" || {
    echo "Could not find Delta's Miniforge installation." >&2
    exit 1
}

mkdir -p \
    "$(dirname -- "${CONDARC}")" \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${LONGNAV_ENV_ROOT}" \
    "${LONGNAV_MODEL_ROOT}" \
    "${LONGNAV_OUTPUT_ROOT}" \
    "${LONGNAV_MANIFEST_ROOT}" \
    "${HABITAT_SIM_WHEEL_ROOT}" \
    "$(dirname -- "${HABITAT_SIM_SOURCE}")"

if [[ ! -e "${CONDARC}" ]]; then
    printf '%s\n' \
        'channels:' \
        '  - conda-forge' \
        'channel_priority: strict' \
        'auto_activate_base: false' >"${CONDARC}"
fi

git -C "${LONGNAV_REPO_ROOT}" \
    -c url.https://github.com/.insteadOf=git@github.com: \
    submodule sync --recursive
git -C "${LONGNAV_REPO_ROOT}" \
    -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init --recursive

create_environment() {
    local prefix="$1"
    shift
    if [[ -e "${prefix}" && ! -f "${prefix}/conda-meta/history" ]]; then
        printf 'Refusing to replace non-Conda path: %s\n' "${prefix}" >&2
        exit 1
    fi
    if [[ ! -f "${prefix}/conda-meta/history" ]]; then
        "${CONDA_BIN}" create --yes --prefix "${prefix}" "$@"
    fi
}

echo "Creating/updating VLM environment at ${LONGNAV_VLM_ENV}"
create_environment "${LONGNAV_VLM_ENV}" python=3.10.16 pip
"${LONGNAV_VLM_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${LONGNAV_VLM_ENV}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0
"${LONGNAV_VLM_ENV}/bin/python" -m pip install \
    --constraint "${SCRIPT_DIR}/constraints.txt" \
    --editable "${LONGNAV_REPO_ROOT}" trl
"${LONGNAV_VLM_ENV}/bin/python" -m pip install \
    --no-dependencies --editable "${LONGNAV_REPO_ROOT}/verl"

echo "Creating/updating Habitat environment at ${LONGNAV_HABITAT_ENV}"
create_environment "${LONGNAV_HABITAT_ENV}" python=3.10.16 pip
"${CONDA_BIN}" install --yes --prefix "${LONGNAV_HABITAT_ENV}" \
    --channel conda-forge cmake=3.31 ninja \
    libegl-devel libgl-devel libglx-devel libglu
"${LONGNAV_HABITAT_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ ! -f "${HABITAT_SIM_WHEEL}" ]]; then
    if [[ -e "${HABITAT_SIM_SOURCE}" && ! -d "${HABITAT_SIM_SOURCE}/.git" ]]; then
        printf 'Refusing to replace non-Git path: %s\n' "${HABITAT_SIM_SOURCE}" >&2
        exit 1
    fi
    if [[ ! -d "${HABITAT_SIM_SOURCE}/.git" ]]; then
        git clone https://github.com/facebookresearch/habitat-sim.git \
            "${HABITAT_SIM_SOURCE}"
    fi
    git -C "${HABITAT_SIM_SOURCE}" fetch origin "${HABITAT_SIM_COMMIT}"
    git -C "${HABITAT_SIM_SOURCE}" checkout --detach "${HABITAT_SIM_COMMIT}"
    git -C "${HABITAT_SIM_SOURCE}" submodule update --init --recursive
    "${LONGNAV_HABITAT_ENV}/bin/python" -m pip install \
        'scikit-build-core>=0.10' 'pybind11>=2.10'
    env \
        "PATH=${LONGNAV_HABITAT_ENV}/bin:${PATH}" \
        "CMAKE_PREFIX_PATH=${LONGNAV_HABITAT_ENV}" \
        "LD_LIBRARY_PATH=${LONGNAV_HABITAT_ENV}/lib:${LD_LIBRARY_PATH:-}" \
        HABITAT_BUILD_GUI_VIEWERS=OFF \
        HABITAT_WITH_BULLET=ON \
        CMAKE_BUILD_PARALLEL_LEVEL="${HABITAT_BUILD_PARALLEL_LEVEL:-8}" \
        "${LONGNAV_HABITAT_ENV}/bin/python" -m pip wheel \
            --no-build-isolation --no-dependencies \
            --wheel-dir "${HABITAT_SIM_WHEEL_ROOT}" \
            "${HABITAT_SIM_SOURCE}"
fi

HABITAT_SIM_WHEEL_SHA256="$(sha256sum "${HABITAT_SIM_WHEEL}" | awk '{print $1}')"
HABITAT_SIM_INSTALL_MARKER="${LONGNAV_HABITAT_ENV}/.longnav-habitat-sim-sha256"
if [[ ! -f "${HABITAT_SIM_INSTALL_MARKER}" ]] || \
        [[ "$(<"${HABITAT_SIM_INSTALL_MARKER}")" != "${HABITAT_SIM_WHEEL_SHA256}" ]]; then
    "${LONGNAV_HABITAT_ENV}/bin/python" -m pip install \
        --force-reinstall --no-dependencies "${HABITAT_SIM_WHEEL}"
    printf '%s\n' "${HABITAT_SIM_WHEEL_SHA256}" \
        >"${HABITAT_SIM_INSTALL_MARKER}"
fi
"${LONGNAV_HABITAT_ENV}/bin/python" -m pip install \
    gym==0.23.0 hydra-core 'omegaconf>=2.2.3' \
    numpy==2.2.6 pillow==10.4.0 regex attrs \
    gitpython imageio imageio-ffmpeg matplotlib \
    'numba>=0.60.0' 'numpy-quaternion>=2024.0.0' \
    'scipy>=1.13.0' tqdm \
    opencv-python-headless==4.11.0.86 'ray[default]==2.53.0'
"${LONGNAV_HABITAT_ENV}/bin/python" -m pip install \
    --no-dependencies --editable "${LONGNAV_REPO_ROOT}/.habitat-lab/habitat-lab"
"${LONGNAV_HABITAT_ENV}/bin/python" -m pip install \
    --no-dependencies --editable "${LONGNAV_REPO_ROOT}"

"${LONGNAV_VLM_ENV}/bin/python" -c \
    'import longnav, peft, ray, torch, transformers, trl; print("VLM imports: OK")'
"${LONGNAV_HABITAT_ENV}/bin/python" -c \
    'import cv2, habitat, habitat_sim, longnav, ray, regex; print("Habitat imports: OK")'

"${CONDA_BIN}" list --explicit --prefix "${LONGNAV_VLM_ENV}" \
    >"${LONGNAV_MANIFEST_ROOT}/longnav_vlm.conda-explicit.txt"
"${LONGNAV_VLM_ENV}/bin/python" -m pip freeze \
    >"${LONGNAV_MANIFEST_ROOT}/longnav_vlm.pip-freeze.txt"
"${CONDA_BIN}" list --explicit --prefix "${LONGNAV_HABITAT_ENV}" \
    >"${LONGNAV_MANIFEST_ROOT}/vln.conda-explicit.txt"
"${LONGNAV_HABITAT_ENV}/bin/python" -m pip freeze \
    >"${LONGNAV_MANIFEST_ROOT}/vln.pip-freeze.txt"
{
    printf 'habitat-sim commit: %s\n' "${HABITAT_SIM_COMMIT}"
    printf 'habitat-sim source: %s\n' "${HABITAT_SIM_SOURCE}"
    printf 'habitat-sim wheel: %s\n' "${HABITAT_SIM_WHEEL}"
    sha256sum "${HABITAT_SIM_WHEEL}"
} >"${LONGNAV_MANIFEST_ROOT}/habitat-sim-source.txt"

echo "Environment setup complete. Next run: ${SCRIPT_DIR}/stage_models.sh"
