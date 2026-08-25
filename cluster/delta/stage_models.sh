#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
longnav_require_environment "${LONGNAV_VLM_ENV}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE PIP_NO_INDEX
export HF_HUB_DISABLE_TELEMETRY=1

exec "${LONGNAV_VLM_ENV}/bin/python" "${SCRIPT_DIR}/stage_models.py" "$@"
