#!/usr/bin/env bash
#SBATCH --job-name=hamlet_ddp
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bgon-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/work/nvme/bgon/brabiei/longnav_runtime/logs/hamlet_ddp_%j.out
# 2-GPU pre-flight for the multi-actor path the 8 h job uses: two RL actors on two
# GPUs (DDP/NCCL gradient sync inside Ray actors), three sims, short real-HM3D
# episodes, two optimizer steps. Everything else identical to run_rl_train.sh.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/delta/env.sh
echo "[ddp] job ${SLURM_JOB_ID} on $(hostname), $(date -u)"
nvidia-smi -L
export RAY_OBJECT_SPILL_DIR="/tmp/${USER}/longnav-${SLURM_JOB_ID}/spill"
mkdir -p "${RAY_OBJECT_SPILL_DIR}"
t0=$(date +%s)
RUN_NAME="hamlet_ddp_${SLURM_JOB_ID}" MAX_WALLCLOCK_HOURS=0.3 SAVE_STEP=1 OSM_GB=12 \
    cluster/delta/run_rl_train.sh \
    resources.num_vlms=2 resources.num_sims=3 \
    rollout.max_steps=30 \
    training.rl_config.n_rollout=2 training.rl_config.n_adv=8 \
    training.total_optimization_steps=2
echo "[ddp] === done ($(( $(date +%s) - t0 ))s) ==="
cat "${LONGNAV_OUTPUT_ROOT}/hamlet_ddp_${SLURM_JOB_ID}/progress.json" || true
