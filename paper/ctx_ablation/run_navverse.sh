#!/usr/bin/env bash
# LongNav-R1 context-window arms on NavVerse indoor ObjectNav (paper/ctx_ablation/README.md,
# section "NavVerse indoor replicate").
#
#   bash paper/ctx_ablation/run_navverse.sh                       # every arm
#   ARMS="nv_evict_k8 nv_evict_k1" bash paper/ctx_ablation/run_navverse.sh
#   ARMS=nv_full EPISODE_LABELS=grCommercial_scene15_0 OUT_ROOT=/tmp/nv_smoke \
#       bash paper/ctx_ablation/run_navverse.sh                   # 1-episode wiring smoke
#
# Per arm: our model is served from the longnav_vlm env (longnav.scripts.serve, the paper_ctx
# preset = argmax actions, sparse filter, frame cut, plus the arm's vlm.context_frames /
# context_window_mode), and NavVerse's evaluator runs in its Docker image with host networking
# so its LongNav client (one JPEG per step, goal on the first step, no client-side history)
# posts to that server. 100 episodes = src/longnav/conf/episode_jsons/navverse_indoor_100_seed17.json
# (seed-17 sample of benchmark500_indoor); the matching *_set.txt is written into the dataset's
# episodes/ folder once so the evaluator can select it.
#
# Workers: one simulator+server pair per arm by default (two pairs fit the GPU for the windowed
# arms but not host RAM, see K_WORKERS below). WORKERS_OVERRIDE / FULL_WORKERS / K_WORKERS set N; worker i posts to port PORT+i, so N servers are
# started and the Ray VRAM reservations are set so the evaluator allocates N slots on the GPU.
#
# Output: paper_results/ctx_ablation_navverse/runs/<arm>/ holding NavVerse's dump for the arm
# (<arm>/result.json, data/, logs), server_config.json, server_steps.jsonl (server_steps_<i>.jsonl
# per worker when N > 1), server*.log and driver.log. Arms resume (--resume_existing_results,
# same run name); a complete arm is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
NAVVERSE_ROOT="${NAVVERSE_ROOT:-$ROOT/NavVerse-Benchmark}"
IMAGE="${IMAGE:-navverse-release:local}"
DATA_DIR="${DATA_DIR:-/mnt/vault/navverse_data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/ctx_ablation_navverse/runs}"
MANIFEST="${MANIFEST:-$ROOT/src/longnav/conf/episode_jsons/navverse_indoor_100_seed17.json}"
EPISODE_SET="${EPISODE_SET:-navverse_indoor100_seed17}"   # <name>_set.txt in $DATA_DIR/episodes
EPISODE_LABELS="${EPISODE_LABELS:-}"                        # comma list overrides the set (smoke runs)
PORT="${PORT:-8000}"
MAX_STEPS="${MAX_STEPS:-500}"
# NavVerse's default episode limit is 300 s of simulated time; with its 2 s per-action limit a
# step costs ~1.3 s of simulated time, so 300 s ends episodes after ~230 steps, before the
# 500-step cap of the Habitat protocol. The limit is raised so the step cap is what binds.
SIM_TIMEOUT_S="${SIM_TIMEOUT_S:-1800}"
K_LIST="${K_LIST:-32 16 8 4 1}"
EXPERIMENT="${EXPERIMENT:-paper_ctx}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
FULL_WORKERS="${FULL_WORKERS:-1}"
# K_WORKERS=2 fits the GPU (~13 GB per pair) but not the 62 GB host: Isaac Sim takes 12-16 GB of RAM
# per process on some GRScenes, NavVerse's task-queue actor grows to ~13 GB over an arm and each
# agent worker holds 4-5 GB of frames, so Ray's memory monitor kills the pairs in a loop (2026-10-21).
K_WORKERS="${K_WORKERS:-1}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="nv_full"
  for K in $K_LIST; do ARMS+=" nv_evict_k${K}"; done
fi

complete() { "$PYTHON" -m longnav.analysis.results_io --complete "$1" --manifest "$MANIFEST" >/dev/null 2>&1; }

# The evaluator selects splits by <name>_set.txt inside EPISODE_FOLDER; derive it from the manifest once.
SET_FILE="$DATA_DIR/episodes/${EPISODE_SET}_set.txt"
if [[ -z "$EPISODE_LABELS" && ! -f "$SET_FILE" ]]; then
  "$PYTHON" - "$MANIFEST" "$SET_FILE" <<'PY'
import json, sys
labels = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write("".join(f"{l}\n" for l in labels))
print(f"wrote {len(labels)} labels to {sys.argv[2]}")
PY
fi

SERVER_PIDS=()
CONTAINER=""
cleanup() {
  [[ -n "$CONTAINER" ]] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || continue
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  done
  # Whatever still holds a server port (a server that survived the signals) must go too.
  for ((i = 0; i < ${WORKERS:-1}; i++)); do
    for p in $(ss -ltnp 2>/dev/null | grep ":$((PORT + i)) " | grep -oP 'pid=\K[0-9]+' | sort -u); do
      kill -9 "$p" 2>/dev/null && echo "$(date '+%F %T')  killed leftover server pid $p on port $((PORT + i))" || true
    done
  done
  SERVER_PIDS=(); CONTAINER=""
}
trap cleanup EXIT INT TERM

for ARM in $ARMS; do
  ARM_DIR="$OUT_ROOT/$ARM"
  if [[ -z "$EPISODE_LABELS" && -d "$ARM_DIR" ]] && complete "$ARM_DIR"; then
    echo "Skipping $ARM: complete."
    continue
  fi
  if [[ "$ARM" == "nv_full" ]]; then
    SETS=(); WORKERS="${WORKERS_OVERRIDE:-$FULL_WORKERS}"
  elif [[ "$ARM" =~ ^nv_(evict|recompute)_k([0-9]+)$ ]]; then
    SETS=(--set "vlm.context_frames=${BASH_REMATCH[2]}" --set "vlm.context_window_mode=${BASH_REMATCH[1]}")
    WORKERS="${WORKERS_OVERRIDE:-$K_WORKERS}"
  else
    echo "Unknown arm '$ARM' (expected nv_full, nv_evict_k<K> or nv_recompute_k<K>)" >&2
    exit 2
  fi
  # Ray reservations (logical, against the first GPU's total): one pair per worker slot.
  if [[ "$WORKERS" -le 1 ]]; then SIM_VRAM_GB="${SIM_VRAM_GB:-16}"; AGENT_VRAM_GB="${AGENT_VRAM_GB:-1}"
  else SIM_VRAM_GB="${SIM_VRAM_GB:-8}"; AGENT_VRAM_GB="${AGENT_VRAM_GB:-4}"; fi
  mkdir -p "$ARM_DIR"
  LOG="$ARM_DIR/driver.log"
  echo "$(date '+%F %T')  starting $ARM  workers=$WORKERS  (gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr -d ' '))" | tee -a "$LOG"

  # 1. model servers (our stack), one conversation each. The full-context cache reaches
  # ~100k slots by the step cap on NavVerse frames (the sparse filter keeps more tokens of JPEG
  # frames than of Habitat renders), so keep PyTorch's reserved memory close to the allocated.
  SERVER_PIDS=()
  for ((i = 0; i < WORKERS; i++)); do
    if [[ "$WORKERS" -le 1 ]]; then SUFFIX=""; else SUFFIX="_$i"; fi
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON" -m longnav.scripts.serve --host 127.0.0.1 --port "$((PORT + i))" --experiment "$EXPERIMENT" "${SETS[@]}" \
      --config-out "$ARM_DIR/server_config$SUFFIX.json" --step-log "$ARM_DIR/server_steps$SUFFIX.jsonl" \
      >> "$ARM_DIR/server$SUFFIX.log" 2>&1 &
    SERVER_PIDS+=($!)
  done
  for ((i = 0; i < WORKERS; i++)); do
    for ((t = 0; t < HEALTH_TIMEOUT_S; t += 5)); do
      kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null || { echo "server $i died; see $ARM_DIR/server*.log" | tee -a "$LOG"; exit 1; }
      curl -fsS "http://127.0.0.1:$((PORT + i))/health" >/dev/null 2>&1 && break
      sleep 5
    done
    curl -fsS "http://127.0.0.1:$((PORT + i))/health" | tee -a "$LOG"; echo | tee -a "$LOG"
  done

  # 2. NavVerse evaluator in its image; host networking reaches the servers at 127.0.0.1:PORT+i
  # ({port} in the URL template = --longnav_base_port + worker id). The one file of the NavVerse
  # branch ctx-ablation-timeout (the --sim_episode_timeout_s flag, forwarded to the simulator's
  # existing --controlled_episode_timeout_s) is bind-mounted over the image's copy, so the image
  # itself need not be rebuilt.
  if [[ -n "$EPISODE_LABELS" ]]; then SELECT=(--episode_labels "$EPISODE_LABELS"); else SELECT=(--split-set "$EPISODE_SET"); fi
  CONTAINER="navverse-ctx-$ARM-$$"
  set +e
  # No --ipc host / large --shm-size: Ray (orchestration only here) would place its object store in
  # the host's shared memory (~19 GB of RAM, 30% of the node) and, with two simulators (~8.5 GB
  # RSS each), its memory monitor then kills the worker pairs in a loop. A small /dev/shm makes Ray
  # fall back to a disk-backed store under /tmp, as in the compose-based runs.
  docker run --rm --name "$CONTAINER" --network host --runtime nvidia --gpus all --shm-size 256m \
    -v "$DATA_DIR:/data:ro" -v "$ARM_DIR:/opt/navverse/dump:rw" \
    -v "$NAVVERSE_ROOT/baselines/eval_navverse.py:/opt/navverse/baselines/eval_navverse.py:ro" \
    -e SCENE_FOLDER=/data -e EPISODE_FOLDER=/data/episodes -e NAVVERSE_RAY_TMP_DIR=/tmp/ray_tmp \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 \
    "$IMAGE" zsh -lc "navverse-run benchmark --preset benchmark_longnav --run-name $ARM ${SELECT[*]} \
      --longnav_server_url 'http://127.0.0.1:{port}/act' --longnav_base_port $PORT --longnav_max_steps $MAX_STEPS \
      --sim_episode_timeout_s $SIM_TIMEOUT_S --no-save_video \
      --gpu-ids 0 --sim_vram_gb $SIM_VRAM_GB --agent_vram_gb $AGENT_VRAM_GB --no-wandb --resume_existing_results ${EXTRA_ARGS:-}" \
    >> "$ARM_DIR/evaluator.log" 2>&1
  STATUS=$?
  set -e
  CONTAINER=""
  cleanup
  # The container runs as root; hand its output back to the user so git/rm work without sudo.
  docker run --rm -v "$ARM_DIR:/d" "$IMAGE" chown -R "$(id -u):$(id -g)" /d >/dev/null 2>&1 || true
  echo "$(date '+%F %T')  evaluator exit $STATUS for $ARM" | tee -a "$LOG"
  if [[ -n "$EPISODE_LABELS" ]]; then
    [[ "$STATUS" -eq 0 ]] || { echo "evaluator failed; see $ARM_DIR/evaluator.log" | tee -a "$LOG"; exit 1; }
  else
    if complete "$ARM_DIR"; then
      echo "$(date '+%F %T')  $ARM complete" | tee -a "$LOG"
    else
      "$PYTHON" -m longnav.analysis.results_io --complete "$ARM_DIR" --manifest "$MANIFEST" | tee -a "$LOG" || true
      echo "$ARM is incomplete; rerun to resume" | tee -a "$LOG"
      exit 1
    fi
  fi
done
echo "$(date '+%F %T')  all arms done: $ARMS"
