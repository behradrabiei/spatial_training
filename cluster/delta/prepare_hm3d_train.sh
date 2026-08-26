#!/usr/bin/env bash
# Extract the HM3D v0.2 *train* scenes (+ semantic annotations) on Delta and
# wire up the symlinks the ObjectNav episode files expect.
#
# Idempotent: already-extracted files are skipped, symlinks are re-pointed.
# Run once from the repo root, preferably on a CPU node:
#   sbatch -p cpu -A bgon-delta-cpu -N1 -c8 --mem=16G -t 02:00:00 \
#          -o /work/nvme/bgon/brabiei/longnav_runtime/logs/prep_%j.out \
#          cluster/delta/prepare_hm3d_train.sh
#
# Why the symlinks: the v1 train episodes carry scene ids of the form
# `hm3d/train/00744-1S7LAXRdDqK/...basis.glb` and a cwd-relative
# `./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`,
# while v2 episodes use `hm3d_v0.2/...`. Both must resolve against the single
# v0.2 scene tree on /work.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Under sbatch the script runs from SLURM's spool copy; env.sh lives in the repo.
if [[ ! -f "${SCRIPT_DIR}/env.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/cluster/delta"
fi
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

SCENES_V2="${LONGNAV_HM3D_SCENES}"                 # .../scenes/HM3D/v2
TREE="${SCENES_V2}/hm3d_v0.2"                      # contains val/ (and train/ after this)
TRAIN_DIR="${TREE}/train"
TAR_HABITAT="${SCENES_V2}/hm3d-train-habitat-v0.2.tar"
TAR_SEMANTIC="${SCENES_V2}/hm3d-train-semantic-annots-v0.2.tar"
TAR_CONFIGS="${SCENES_V2}/hm3d-train-semantic-configs-v0.2.tar"

for f in "${TAR_HABITAT}" "${TAR_SEMANTIC}" "${TAR_CONFIGS}"; do
    longnav_require_file "$f"
done

mkdir -p "${TRAIN_DIR}"
echo "[prep] extracting scene meshes into ${TRAIN_DIR} ($(date))"
tar -xf "${TAR_HABITAT}" -C "${TRAIN_DIR}" --skip-old-files
echo "[prep] extracting semantic annotations ($(date))"
tar -xf "${TAR_SEMANTIC}" -C "${TRAIN_DIR}" --skip-old-files
echo "[prep] extracting scene-dataset configs ($(date))"
tar -xf "${TAR_CONFIGS}" -C "${TREE}" --skip-old-files

# Episode files reference the v0.2 tree under two names.
ln -sfn "hm3d_v0.2" "${SCENES_V2}/hm3d"

# `scene_dataset_config` paths in the episodes are relative to sim.workspace.
# The repo's tracked data/scene_datasets/hm3d_v0.2 symlink points at the home
# machine, so Delta runs use a dedicated workspace directory instead
# (LONGNAV_SIM_WORKSPACE in env.sh) and pass every other path absolutely.
mkdir -p "${LONGNAV_SIM_WORKSPACE}/data/scene_datasets"
ln -sfn "${TREE}" "${LONGNAV_SIM_WORKSPACE}/data/scene_datasets/hm3d_v0.2"
ln -sfn "${TREE}" "${LONGNAV_SIM_WORKSPACE}/data/scene_datasets/hm3d"

echo "[prep] verifying ($(date))"
n_dirs="$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
n_glb="$(find "${TRAIN_DIR}" -name '*.basis.glb' | wc -l)"
n_nav="$(find "${TRAIN_DIR}" -name '*.basis.navmesh' | wc -l)"
n_sem="$(find "${TRAIN_DIR}" -name '*.semantic.glb' | wc -l)"
echo "[prep] train scene dirs=${n_dirs} glb=${n_glb} navmesh=${n_nav} semantic=${n_sem}"
test "${n_dirs}" -ge 800 && test "${n_glb}" -ge 800 && test "${n_nav}" -ge 800 \
    || { echo "[prep] ERROR: incomplete extraction" >&2; exit 1; }
for cfg in hm3d_annotated_basis hm3d_annotated_train_basis; do
    longnav_require_file "${TREE}/${cfg}.scene_dataset_config.json"
done
ls -la "${SCENES_V2}/hm3d" "${LONGNAV_SIM_WORKSPACE}/data/scene_datasets/"
du -sh "${TRAIN_DIR}"
echo "[prep] done ($(date))"
