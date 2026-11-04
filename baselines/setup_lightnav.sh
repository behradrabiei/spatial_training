#!/usr/bin/env bash
# Build the isolated LightNav-0 runtime: a pinned upstream clone, a Python 3.9 conda prefix
# for its Habitat env server and a Python 3.11 conda prefix for its GPU client. Nothing is
# installed into the existing `vln` / `longnav_vlm` environments. Mirrors baselines/setup.sh.
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LN="$BASELINE_ROOT/LightNav-0"
LN_SHA="${LIGHTNAV_SHA:-3015508b70fb6e30bbd66701e94c9b916dcb8df8}"
ENV_SERVER="$BASELINE_ROOT/.envs/lightnav-habitat"
ENV_CLIENT="$BASELINE_ROOT/.envs/lightnav"
CONDA="${CONDA_EXE:-/home/brabiei/miniconda3/bin/conda}"
CACHE_ROOT="$BASELINE_ROOT/.cache"

export CONDA_PKGS_DIRS="$CACHE_ROOT/conda"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONNOUSERSITE=1

mkdir -p "$BASELINE_ROOT/.envs" "$CACHE_ROOT"/{conda,pip,huggingface,torch,xdg,tmp}

# ---- upstream clone, pinned -------------------------------------------------------------
if [[ ! -d "$LN/.git" ]]; then
  git clone https://github.com/lightorigins/LightNav-0.git "$LN"
fi
git -C "$LN" fetch origin "$LN_SHA"
git -C "$LN" checkout --detach "$LN_SHA"

# ---- Habitat env server: python 3.9 + habitat-sim 0.3.1 (upstream's verified stack) -----
if [[ ! -x "$ENV_SERVER/bin/python" ]]; then
  "$CONDA" env create -y -p "$ENV_SERVER" -f "$LN/habitat_server/environment.yml" \
    || "$CONDA" env create -y -p "$ENV_SERVER" -f "$LN/habitat_server/environment.yml" \
         --override-channels -c aihabitat -c conda-forge -c defaults
fi
SPY="$ENV_SERVER/bin/python"
"$SPY" -m pip install --no-deps "habitat-lab==0.3.20231024"
"$SPY" -m pip install --force-reinstall "numpy>=1.20,<1.24"
"$SPY" -m pip install -e "$LN/habitat_server"
"$SPY" -c "import habitat, habitat_sim, numpy, lightnav_habitat.serve; print('server env ok: habitat-sim', habitat_sim.__version__, 'numpy', numpy.__version__)"

# ---- GPU client: python 3.11 + torch/transformers/vLLM pinned by upstream pyproject -------
if [[ ! -x "$ENV_CLIENT/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_CLIENT" python=3.11 pip
fi
CPY="$ENV_CLIENT/bin/python"
"$CPY" -m pip install -e "$LN[vllm,video,habitat]"
"$CPY" - <<'PY'
import torch, vllm, transformers, zmq, lightnav.cli.eval_habitat  # noqa: F401
archs = torch.cuda.get_arch_list()
print("client env ok: torch", torch.__version__, "vllm", vllm.__version__,
      "transformers", transformers.__version__, "archs", archs)
assert any(a.endswith("120") for a in archs), "torch build lacks sm_120 (RTX 5090)"
PY

mkdir -p "$LN/checkpoints"
echo "Isolated LightNav-0 runtime ready: $ENV_SERVER (server), $ENV_CLIENT (client)"
echo "Next: bash $BASELINE_ROOT/run_lightnav.sh download"
