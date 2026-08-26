#!/usr/bin/env bash
#SBATCH --job-name=hamlet_rl
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgon-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/work/nvme/bgon/brabiei/longnav_runtime/logs/hamlet_rl_%j.out
# HAMLET second-stage RL on one 4xA100-40GB node (8 h by default). Submit from the repo root:
#   sbatch cluster/delta/train_hamlet.sbatch            # optionally: RUN_NAME=... sbatch --export=ALL ...
#   12 h: sbatch --time=12:00:00 --export=ALL,RUN_NAME=<run>,MAX_WALLCLOCK_HOURS=11.4 cluster/delta/train_hamlet.sbatch
# The training loop stops itself after MAX_WALLCLOCK_HOURS (default 7.6 h) and
# writes checkpoints/checkpoint_final before the job limit. Pair with
# cluster/delta/watch_train.sh <job id> <run> to evaluate checkpoints as they land
# and cancel the job if the argmax success collapses.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/delta/env.sh
echo "[sbatch] job ${SLURM_JOB_ID} on $(hostname), $(date -u)"
nvidia-smi -L
df -h /dev/shm /tmp | tail -2
# Gate: the dummy-env HAMLET smoke test must pass on this node before the 8 h run
# starts (set SKIP_SMOKE=1 to bypass once it is known-good).
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    export RAY_OBJECT_SPILL_DIR="/tmp/${USER}/longnav-${SLURM_JOB_ID}/spill"
    mkdir -p "${RAY_OBJECT_SPILL_DIR}"
    echo "[sbatch] running HAMLET smoke gate $(date -u)"
    CUDA_VISIBLE_DEVICES=0 cluster/delta/debug.sh hamlet-smoke
    echo "[sbatch] smoke gate passed $(date -u)"
fi
exec cluster/delta/run_rl_train.sh "$@"
