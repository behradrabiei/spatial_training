#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="$BASELINE_ROOT/.envs/uninavid-hm3d"
PYTHON="$ENV_PREFIX/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing isolated runtime. Run: bash $BASELINE_ROOT/setup.sh" >&2
  exit 1
fi

export PIP_CACHE_DIR="$BASELINE_ROOT/.cache/pip"
export HF_HOME="$BASELINE_ROOT/.cache/huggingface"
export TORCH_HOME="$BASELINE_ROOT/.cache/torch"
export XDG_CACHE_HOME="$BASELINE_ROOT/.cache/xdg"
export TMPDIR="$BASELINE_ROOT/.cache/tmp"
export PYTHONPATH="$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export DISPLAY=

NVIDIA_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [[ -f "$NVIDIA_EGL" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL"
fi

cd "$BASELINE_ROOT/Uni-NaVid"
COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  echo "Usage: $0 {download|validate|model-check|smoke|eval|full|teleop|multi-validate|multi-smoke|multi-full|multi-all-goals-smoke|multi-all-goals-full} [extra arguments]" >&2
  exit 2
fi
shift

case "$COMMAND" in
  download)
    exec "$PYTHON" -m uninavid_hm3d.download_models "$@"
    ;;
  validate)
    exec "$PYTHON" -m uninavid_hm3d.eval --validate-only "$@"
    ;;
  model-check)
    exec "$PYTHON" -m uninavid_hm3d.eval --model-load-only \
      --output "$BASELINE_ROOT/results/model_check" "$@"
    ;;
  smoke)
    exec "$PYTHON" -m uninavid_hm3d.eval --limit 1 --video \
      --output "$BASELINE_ROOT/results/hm3d_v2_100_smoke" "$@"
    ;;
  eval)
    exec "$PYTHON" -m uninavid_hm3d.eval "$@"
    ;;
  full)
    exec "$PYTHON" -m uninavid_hm3d.eval --resume \
      --output "$BASELINE_ROOT/results/hm3d_v2_100" "$@"
    ;;
  teleop)
    exec "$PYTHON" -m uninavid_hm3d.teleop "$@"
    ;;
  multi-validate)
    exec "$PYTHON" -m uninavid_hm3d.eval_multi --validate-only "$@"
    ;;
  multi-smoke)
    exec "$PYTHON" -m uninavid_hm3d.eval_multi --limit 1 --video \
      --output "$BASELINE_ROOT/results/onemap_multi_uninavid_sequential_smoke" "$@"
    ;;
  multi-full)
    exec "$PYTHON" -m uninavid_hm3d.eval_multi --resume \
      --output "$BASELINE_ROOT/results/onemap_multi_uninavid_sequential" "$@"
    ;;
  multi-all-goals-smoke)
    exec "$PYTHON" -m uninavid_hm3d.eval_multi --reveal-all-goals --limit 1 --video \
      --output "$BASELINE_ROOT/results/onemap_multi_uninavid_all_goals_smoke" "$@"
    ;;
  multi-all-goals-full)
    exec "$PYTHON" -m uninavid_hm3d.eval_multi --reveal-all-goals --resume \
      --output "$BASELINE_ROOT/results/onemap_multi_uninavid_all_goals" "$@"
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    exit 2
    ;;
esac
