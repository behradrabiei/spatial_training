#!/usr/bin/env bash
#SBATCH --job-name=hamlet_eval
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bgon-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bgon/brabiei/longnav_runtime/logs/hamlet_eval_%j.out
# Fixed-subset eval of one checkpoint. Submit with e.g.
#   sbatch --export=ALL,RUN_NAME=eval_ckpt11,EVAL_HAMLET=on,EVAL_CHECKPOINT=<dir> cluster/delta/eval_ckpt.sbatch
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "[eval] job ${SLURM_JOB_ID} on $(hostname), $(date -u)"
cluster/delta/run_eval.sh
