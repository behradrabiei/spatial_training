#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASELINE_ROOT/.." && pwd)"
ENV_PREFIX="$BASELINE_ROOT/.envs/uninavid-hm3d"
SOURCE_VLN_PREFIX="${SOURCE_VLN_PREFIX:-/home/brabiei/miniconda3/envs/vln}"
CACHE_ROOT="$BASELINE_ROOT/.cache"
TOOLS_ROOT="$BASELINE_ROOT/.tools"

export CONDA_PKGS_DIRS="$CACHE_ROOT/conda"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONNOUSERSITE=1

mkdir -p "$BASELINE_ROOT/.envs" "$CACHE_ROOT"/{conda,pip,huggingface,torch,xdg,tmp}

if [[ ! -d "$BASELINE_ROOT/Uni-NaVid/.git" ]]; then
  git clone https://github.com/jzhzhang/Uni-NaVid.git "$BASELINE_ROOT/Uni-NaVid"
fi
git -C "$BASELINE_ROOT/Uni-NaVid" fetch origin 79ef5ea3fea14c205342d1ab070563d84c7a966a
git -C "$BASELINE_ROOT/Uni-NaVid" checkout --detach 79ef5ea3fea14c205342d1ab070563d84c7a966a

if [[ ! -d "$BASELINE_ROOT/habitat-lab/.git" ]]; then
  git clone https://github.com/facebookresearch/habitat-lab.git "$BASELINE_ROOT/habitat-lab"
fi
git -C "$BASELINE_ROOT/habitat-lab" fetch origin cdbb4880519505adf45fba0f0c0c3a3fd18a2a55
git -C "$BASELINE_ROOT/habitat-lab" checkout --detach cdbb4880519505adf45fba0f0c0c3a3fd18a2a55

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  python -m pip install --target "$TOOLS_ROOT" conda-pack==0.9.2
  PYTHONPATH="$TOOLS_ROOT" python -m conda_pack.cli \
    --prefix "$SOURCE_VLN_PREFIX" \
    --output "$CACHE_ROOT/vln-runtime.tar.gz" \
    --ignore-editable-packages --quiet --force
  mkdir -p "$ENV_PREFIX"
  tar -xzf "$CACHE_ROOT/vln-runtime.tar.gz" -C "$ENV_PREFIX"
  "$ENV_PREFIX/bin/conda-unpack"
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip uninstall -y habitat-lab longnav || true
"$PYTHON" -m pip install --no-deps -e "$BASELINE_ROOT/habitat-lab/habitat-lab"
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
"$PYTHON" -m pip install -r "$BASELINE_ROOT/requirements-inference.txt"
"$PYTHON" -m pip install --no-deps -e "$BASELINE_ROOT/Uni-NaVid"

mkdir -p "$BASELINE_ROOT/Uni-NaVid/data/scene_datasets" "$BASELINE_ROOT/Uni-NaVid/model_zoo"
if [[ ! -e "$BASELINE_ROOT/Uni-NaVid/data/scene_datasets/hm3d_v0.2" ]]; then
  ln -s /home/brabiei/vault/habitat_data/scenes/HM3D/v2/hm3d_v0.2 \
    "$BASELINE_ROOT/Uni-NaVid/data/scene_datasets/hm3d_v0.2"
fi

echo "Isolated runtime ready: $ENV_PREFIX"
