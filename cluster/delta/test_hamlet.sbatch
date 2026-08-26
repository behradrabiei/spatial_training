#!/usr/bin/env bash
#SBATCH --job-name=hamlet_test
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bgon-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/nvme/bgon/brabiei/longnav_runtime/logs/hamlet_test_%j.out
# 1-GPU pre-flight for the HAMLET RL run: GPU/import checks, the HAMLET and RL
# smoke tests on the dummy env, then a short real-HM3D training cycle with one
# VLM actor (data loading, memory, checkpointing, timing).
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/delta/env.sh
echo "[test] job ${SLURM_JOB_ID} on $(hostname), $(date -u)"
nvidia-smi -L
export RAY_OBJECT_SPILL_DIR="/tmp/${USER}/longnav-${SLURM_JOB_ID}/spill"
mkdir -p "${RAY_OBJECT_SPILL_DIR}"
t0=$(date +%s)
cluster/delta/debug.sh gpu
cluster/delta/debug.sh imports
if [[ "${ONLY_MINI:-0}" != "1" ]]; then   # ONLY_MINI=1: skip the (already passed) smoke stages
    echo "[test] === rl-smoke ($(( $(date +%s) - t0 ))s) ==="
    cluster/delta/debug.sh rl-smoke 2>&1 | grep -E "actor/ppo_kl|Traceback|Error|Complete" | head -20
    echo "[test] === hamlet-smoke ($(( $(date +%s) - t0 ))s) ==="
    cluster/delta/debug.sh hamlet-smoke
fi
echo "[test] === mini real run ($(( $(date +%s) - t0 ))s) ==="
# MINI_STEPS / MINI_ROLLOUT / MINI_OPT_STEPS: episode cap, episodes per cycle, optimizer steps
# (e.g. MINI_STEPS=350 MINI_ROLLOUT=2 MINI_OPT_STEPS=2 to measure peak VRAM on full-length episodes)
RUN_NAME="hamlet_mini_${SLURM_JOB_ID}" MAX_WALLCLOCK_HOURS=0.3 SAVE_STEP=1 OSM_GB=12 \
    cluster/delta/run_rl_train.sh \
    resources.num_vlms=1 resources.num_sims=2 \
    "rollout.max_steps=${MINI_STEPS:-40}" \
    "training.rl_config.n_rollout=${MINI_ROLLOUT:-2}" training.rl_config.n_adv=8 \
    "training.total_optimization_steps=${MINI_OPT_STEPS:-8}"
echo "[test] === done ($(( $(date +%s) - t0 ))s) ==="
ls -la "${LONGNAV_OUTPUT_ROOT}/hamlet_mini_${SLURM_JOB_ID}/checkpoints" || true
cat "${LONGNAV_OUTPUT_ROOT}/hamlet_mini_${SLURM_JOB_ID}/progress.json" || true
