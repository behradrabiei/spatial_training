#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${DELTA_ACCOUNT:-}" ]]; then
    echo "Set DELTA_ACCOUNT to a GPU allocation before requesting a session." >&2
    echo "Available allocations are shown by the Delta 'accounts' command." >&2
    exit 2
fi

DELTA_PARTITION="${DELTA_PARTITION:-gpuA40x4-interactive}"
case "${DELTA_PARTITION}" in
    gpuA40x4-interactive|gpuA100x4-interactive|gpuH200x8-interactive) ;;
    *)
        echo "Unsupported interactive partition: ${DELTA_PARTITION}" >&2
        exit 2
        ;;
esac

command -v srun >/dev/null 2>&1 || {
    echo "srun is unavailable; run this command on a Delta login node." >&2
    exit 1
}
longnav_require_environment "${LONGNAV_VLM_ENV}"
longnav_require_environment "${LONGNAV_HABITAT_ENV}"
longnav_require_data
longnav_require_file "${LONGNAV_MODEL_MANIFEST}"

longnav_enable_offline_mode
"${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" --validate-only

echo "Requesting a one-hour ${DELTA_PARTITION} session."
echo "After the shell opens, run: source cluster/delta/activate_session.sh"
exec srun \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gpus-per-node=1 \
    --mem=64G \
    --time=01:00:00 \
    --pty bash --login
