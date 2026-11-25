#!/usr/bin/env bash
# LightNav-0 "old blocks kept" context arms on NavVerse indoor ObjectNav (paper/ctx_ablation/README.md,
# section "NavVerse indoor replicate"). Sibling of run_navverse.sh (the LongNav-R1 arms): same
# evaluator, episodes, output layout and completion logic; only the model server differs.
#
#   bash paper/ctx_ablation/run_navverse_lightnav.sh                  # nvl_full + nvl_keep_k{32,16,8,4,1}
#   ARMS="nvl_keep_k8" bash paper/ctx_ablation/run_navverse_lightnav.sh
#   ARMS=nvl_full EPISODE_LABELS=grCommercial_scene15_0 OUT_ROOT=/tmp/nvl_smoke \
#       bash paper/ctx_ablation/run_navverse_lightnav.sh              # 1-episode wiring smoke
#
# Per arm: LightNav-0 is served from its isolated runtime (baselines/.envs/lightnav) by
# baselines/lightnav_hm3d/serve_navverse.py, which speaks the /act contract of our LongNav-R1 server,
# so NavVerse's LongNav client (one JPEG per step, goal on the first step, no client-side history,
# its own STOP at the 500-step cap) drives it unchanged. The server discretizes LightNav's 1 s
# velocity step into STOP / FORWARD / left / right (see the server's docstring); the client maps
# left/right to NavVerse's 15 deg TURN_*_SMALL moves (ACTION_MAP, TURN_DEG), the closest discrete
# turn to LightNav's own 18 deg rotate step (with the 30 deg turns of the LongNav-R1 arms the
# model over-rotated and oscillated left/right on the smoke episode; user decision 2026-10-28), and feeds
# the 640x480 frames at their native aspect ratio (aspect_mode keep: 384x288, 108 image tokens per
# frame, 30 per pooled history pair) -- the LongNav-R1 arms also saw the frames unstretched.
#
# nvl_full: the model's own full-episode SlowFast history on the compiled vLLM path (the HM3D
# lightnav_full protocol). nvl_keep_k<K>: eval_prune's recency rule under the eager masked kernel
# keeps the image tokens of the current pair plus the K-2 newest history frames and hides every
# older block's image tokens, leaving its timestamp and vision markers in the prompt (the HM3D
# lightnav_keep_k<K> = lightnav_recency_b<K-2> arms); budget = (K-2) x 15 tokens = (K-2)/2 pooled
# pairs of 30, so K must be even and >= 4; K = 2 = the current pair with no history (k = 0), K = 1 = no
# history and the current frame doubled (k = 0, --prune-current-frames 1 = upstream's step-0 layout; the
# HM3D lightnav_keep_k1 = lightnav_recency_b0_cur1 point).
#
# Output: paper_results_archive/ctx_ablation_navverse/runs/<arm>/ holding NavVerse's dump for the arm
# (<arm>/result.json, data/, logs), server_config.json, server_steps.jsonl, server.log, driver.log
# and prune/<launch>/ (the keep arms' per-episode pruning records, one sub-directory per server
# launch). Arms resume (--resume_existing_results, same run name); a complete arm is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"   # results_io (completion checks)
NAVVERSE_ROOT="${NAVVERSE_ROOT:-$ROOT/NavVerse-Benchmark}"
IMAGE="${IMAGE:-navverse-release:local}"
DATA_DIR="${DATA_DIR:-/mnt/vault/navverse_data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results_archive/ctx_ablation_navverse/runs}"
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
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
WORKERS=1   # one vLLM instance (~0.65 of the GPU at start-up) + one Isaac Sim worker fill the card

# LightNav-0 runtime (as baselines/run_lightnav.sh): isolated env, pinned caches, offline hub.
BASELINE_ROOT="$ROOT/baselines"
LN="$BASELINE_ROOT/LightNav-0"
CPY="$BASELINE_ROOT/.envs/lightnav/bin/python"
CKPT="${CKPT:-$LN/checkpoints/LightNav-0}"
CACHE_ROOT="$BASELINE_ROOT/.cache"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
ASPECT_MODE="${ASPECT_MODE:-keep}"
EXPECT_VIDEO_SIZE="${EXPECT_VIDEO_SIZE:-288 384}"   # the model frame under ASPECT_MODE for 640x480 input
TOKENS_PER_FRAME="${TOKENS_PER_FRAME:-15}"          # a pooled 2-frame history pair holds 30 tokens at 384x288
ACTION_MAP="${ACTION_MAP:-STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL}"   # client names for action ids 0-3
TURN_DEG="${TURN_DEG:-15}"                           # the turn those names execute (normalizes the server's rule)
SERVER="$BASELINE_ROOT/lightnav_hm3d/serve_navverse.py"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="nvl_full"
  for K in $K_LIST; do ARMS+=" nvl_keep_k${K}"; done
fi

[[ -x "$CPY" ]] || { echo "Missing isolated runtime ($CPY). Run: bash $BASELINE_ROOT/setup_lightnav.sh" >&2; exit 1; }
for f in config.json model.safetensors.index.json eval_config.json action_tokenizer/manifest.json; do
  [[ -e "$CKPT/$f" ]] || { echo "Checkpoint incomplete: missing $CKPT/$f (bash baselines/run_lightnav.sh download)" >&2; exit 1; }
done
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# PYTHONPATH: LightNav's package plus this repo (baselines.lightnav_hm3d.eval_prune for the keep arms).
CLIENT_ENV=(env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$LN/src:$ROOT" HF_HUB_OFFLINE=1)
"${CLIENT_ENV[@]}" "$CPY" -c "import fastapi, uvicorn, multipart, vllm, lightnav" || { echo "LightNav client runtime lacks a dependency" >&2; exit 1; }

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
  for ((i = 0; i < WORKERS; i++)); do
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
  if [[ "$ARM" == "nvl_full" ]]; then
    PRUNE=()
  elif [[ "$ARM" =~ ^nvl_keep_k([0-9]+)$ ]]; then
    K="${BASH_REMATCH[1]}"
    if (( K == 1 )); then  # no history, current frame doubled (HM3D lightnav_keep_k1 = recency_b0_cur1)
      PRUNE=(--prune-method recency --prune-k 0 --prune-tokens-per-frame "$TOKENS_PER_FRAME"
             --prune-current-frames 1 --prune-log-dir "$ARM_DIR/prune")
    elif (( K == 2 )); then  # the current pair, no history (HM3D lightnav_keep_k2 = random_b0)
      PRUNE=(--prune-method recency --prune-k 0 --prune-tokens-per-frame "$TOKENS_PER_FRAME"
             --prune-log-dir "$ARM_DIR/prune")
    elif (( K < 4 || K % 2 != 0 )); then
      echo "nvl_keep_k$K: K must be 1, 2 or even and >= 4 (current pair + (K-2)/2 pooled history pairs)" >&2; exit 2
    else
      PRUNE=(--prune-method recency --prune-k "$((K - 2))" --prune-tokens-per-frame "$TOKENS_PER_FRAME"
             --prune-log-dir "$ARM_DIR/prune")
    fi
  else
    echo "Unknown arm '$ARM' (expected nvl_full or nvl_keep_k<K>)" >&2
    exit 2
  fi
  # Ray reservations (logical, against the first GPU's total): one simulator slot; the server is
  # outside Ray's accounting.
  SIM_VRAM_GB="${SIM_VRAM_GB:-16}"; AGENT_VRAM_GB="${AGENT_VRAM_GB:-1}"
  mkdir -p "$ARM_DIR"
  LOG="$ARM_DIR/driver.log"
  echo "$(date '+%F %T')  starting $ARM  (gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr -d ' '))" | tee -a "$LOG"

  # 1. the LightNav-0 server, started before the simulator so vLLM's start-up memory check sees
  # the free card (it then holds ~weights + a fixed 2 GiB KV cache; Isaac Sim needs ~9 GB).
  SERVER_PIDS=()
  # shellcheck disable=SC2086
  "${CLIENT_ENV[@]}" "$CPY" "$SERVER" --host 127.0.0.1 --port "$PORT" --arm "$ARM" --model-path "$CKPT" \
      --backend vllm_local --gpu-memory-utilization "$GPU_MEM_UTIL" --aspect-mode "$ASPECT_MODE" \
      --expect-video-size $EXPECT_VIDEO_SIZE --turn-deg "$TURN_DEG" "${PRUNE[@]}" \
      --config-out "$ARM_DIR/server_config.json" --step-log "$ARM_DIR/server_steps.jsonl" \
      >> "$ARM_DIR/server.log" 2>&1 &
  SERVER_PIDS+=($!)
  for ((t = 0; t < HEALTH_TIMEOUT_S; t += 5)); do
    kill -0 "${SERVER_PIDS[0]}" 2>/dev/null || { echo "server died; see $ARM_DIR/server.log" | tee -a "$LOG"; exit 1; }
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS "http://127.0.0.1:$PORT/health" | tee -a "$LOG"; echo | tee -a "$LOG"

  # 2. NavVerse evaluator in its image; host networking reaches the server at 127.0.0.1:PORT
  # ({port} in the URL template = --longnav_base_port + worker id). The one file of the NavVerse
  # branch ctx-ablation-timeout (the --sim_episode_timeout_s flag, forwarded to the simulator's
  # existing --controlled_episode_timeout_s) is bind-mounted over the image's copy, so the image
  # itself need not be rebuilt.
  if [[ -n "$EPISODE_LABELS" ]]; then SELECT=(--episode_labels "$EPISODE_LABELS"); else SELECT=(--split-set "$EPISODE_SET"); fi
  CONTAINER="navverse-ctx-$ARM-$$"
  set +e
  # No --ipc host / large --shm-size: Ray (orchestration only here) would place its object store in
  # the host's shared memory; a small /dev/shm makes it fall back to a disk-backed store under /tmp.
  docker run --rm --name "$CONTAINER" --network host --runtime nvidia --gpus all --shm-size 256m \
    -v "$DATA_DIR:/data:ro" -v "$ARM_DIR:/opt/navverse/dump:rw" \
    -v "$NAVVERSE_ROOT/baselines/eval_navverse.py:/opt/navverse/baselines/eval_navverse.py:ro" \
    -e SCENE_FOLDER=/data -e EPISODE_FOLDER=/data/episodes -e NAVVERSE_RAY_TMP_DIR=/tmp/ray_tmp \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 \
    "$IMAGE" zsh -lc "navverse-run benchmark --preset benchmark_longnav --run-name $ARM ${SELECT[*]} \
      --longnav_server_url 'http://127.0.0.1:{port}/act' --longnav_base_port $PORT --longnav_max_steps $MAX_STEPS \
      --longnav_action_map $ACTION_MAP --sim_episode_timeout_s $SIM_TIMEOUT_S --no-save_video \
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
