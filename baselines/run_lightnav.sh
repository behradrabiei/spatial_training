#!/usr/bin/env bash
# LightNav-0 baseline on the 100 held-out HM3D ObjectNav v2 episodes.
#
# Usage: bash baselines/run_lightnav.sh <mode> [extra client args]
#   download        fetch the LightOriginsHQ/LightNav-0 checkpoint into LightNav-0/checkpoints/
#   subset          write the 100-episode dataset copy under .cache/datasets/
#   validate        import checks, subset, checkpoint files, boot the env server and probe it
#   smoke           1 episode, comparable protocol  -> results/lightnav_hm3d_v2_100_smoke
#   smoke-native    1 episode, LightNav's protocol   -> results/lightnav_hm3d_v2_100_smoke_native
#   full            100 episodes, success radius 1.0 m (as Uni-NaVid/LongNav) -> results/lightnav_hm3d_v2_100
#   full-native     100 episodes, 0.1 m + navmesh re-bake 0.05 (LightNav paper) -> results/lightnav_hm3d_v2_100_native
#   convert <dir>   rebuild episodes/ + summary.json from <dir>/raw/results.jsonl
#
# Knobs: BACKEND (vllm_local|hf, default vllm_local), GPU_MEM_UTIL (0.65), PORT (5555),
#        GPU (0), RUN_NAME (override the results directory name), RUN_DIR (absolute run
#        directory, overrides results/$RUN_NAME), READY_TIMEOUT_S (900),
#        CONTEXT_FRAMES (K: run the client through lightnav_hm3d/eval_ctx.py, which cuts the
#        visual history to the newest K frames, current included; unset = the model's own
#        full-episode history),
#        BLIND_LAYER (L: run the client through lightnav_hm3d/eval_blind.py, which hides every
#        visual token from decoder layer L on -- eager vLLM with a masked attention kernel on
#        every layer; L = 36 hides nothing and is the matched reference),
#        PRUNE_METHOD=M PRUNE_K=k (M in random|fastv|divprune|dart|recency: run the client through
#        lightnav_hm3d/eval_prune.py, which keeps k x 14 history visual tokens (k = 0: none, the
#        no-visual-history control) under rule M with a
#        per-episode deletion ledger -- same eager masked kernel as BLIND_LAYER, dropped columns
#        hidden on every layer; the records land in raw/prune/ and are renamed by the converter),
#        INFLUENCE=1 (run the client through lightnav_hm3d/eval_influence.py: normal navigation plus,
#        after every decision, the keep-one KL probes of paper/frame_influence -- same eager masked
#        kernel; records land in raw/influence/ and are renamed by the converter; MAX_AGE (33)
#        bounds the probed history tubelets' age. INFLUENCE_MEASURE=recent_k probes
#        KL(full || K most recent frames) instead, K <= MAX_K (32), both with the old blocks kept
#        (masked) and as a rebuilt K-frame window).
#        The four wrappers are not combined.
#        LABELS (a JSON list of episode labels, default the 100-episode file): the dataset subset
#        is built under .cache/datasets/objectnav_<labels stem> with that many episodes.
# The model, its camera (480x270, hfov 120), velocity_control actions, greedy decoding and the
# 500-step budget are LightNav's own; only the server's success flags differ between variants.
set -euo pipefail
set +m  # no job control: background jobs share our process group, so setsid execs without forking and $! is the real pid

BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASELINE_ROOT/.." && pwd)"
LN="$BASELINE_ROOT/LightNav-0"
ENV_SERVER="$BASELINE_ROOT/.envs/lightnav-habitat"
ENV_CLIENT="$BASELINE_ROOT/.envs/lightnav"
SPY="$ENV_SERVER/bin/python"
CPY="$ENV_CLIENT/bin/python"
CKPT="${CKPT:-$LN/checkpoints/LightNav-0}"
LABELS="${LABELS:-$PROJECT_ROOT/dump/hm3d_v2_100_labels.json}"
LABELS_STEM="$(basename "$LABELS" .json)"; LABELS_STEM="${LABELS_STEM%_labels}"
SUBSET_ROOT="${SUBSET_ROOT:-$BASELINE_ROOT/.cache/datasets/objectnav_$LABELS_STEM}"
EXPECTED_COUNT="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$LABELS")"
SCENES_DIR="${SCENES_DIR:-/home/brabiei/vault/habitat_data/scenes/HM3D/v2}"
HABITAT_CONFIG="$LN/habitat_server/configs/objectnav_hm3d_v2.yaml"
CACHE_ROOT="$BASELINE_ROOT/.cache"
PORT="${PORT:-5555}"
GPU="${GPU:-0}"
BACKEND="${BACKEND:-vllm_local}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-}"
if [[ -n "$CONTEXT_FRAMES" && ! "$CONTEXT_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTEXT_FRAMES must be a positive integer, got '$CONTEXT_FRAMES'" >&2
  exit 2
fi
BLIND_LAYER="${BLIND_LAYER:-}"
if [[ -n "$BLIND_LAYER" && ! "$BLIND_LAYER" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "BLIND_LAYER must be a non-negative integer, got '$BLIND_LAYER'" >&2
  exit 2
fi
if [[ -n "$BLIND_LAYER" && -n "$CONTEXT_FRAMES" ]]; then
  echo "BLIND_LAYER and CONTEXT_FRAMES are separate ablations; set only one" >&2
  exit 2
fi
PRUNE_METHOD="${PRUNE_METHOD:-}"
PRUNE_K="${PRUNE_K:-}"
PRUNE_CURRENT_FRAMES="${PRUNE_CURRENT_FRAMES:-}"  # 1: current frame doubled (upstream's K = 1 layout); default 2
if [[ -n "$PRUNE_METHOD" || -n "$PRUNE_K" ]]; then
  if [[ -z "$PRUNE_METHOD" || -z "$PRUNE_K" ]]; then
    echo "PRUNE_METHOD and PRUNE_K go together, got method='$PRUNE_METHOD' k='$PRUNE_K'" >&2
    exit 2
  fi
  if [[ ! "$PRUNE_METHOD" =~ ^(random|fastv|divprune|dart|recency)$ ]]; then
    echo "PRUNE_METHOD must be random|fastv|divprune|dart|recency, got '$PRUNE_METHOD'" >&2
    exit 2
  fi
  if [[ -n "$PRUNE_CURRENT_FRAMES" && ! "$PRUNE_CURRENT_FRAMES" =~ ^[12]$ ]]; then
    echo "PRUNE_CURRENT_FRAMES must be 1 or 2, got '$PRUNE_CURRENT_FRAMES'" >&2
    exit 2
  fi
  if [[ ! "$PRUNE_K" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "PRUNE_K must be a non-negative integer (0 = no visual history), got '$PRUNE_K'" >&2
    exit 2
  fi
  if [[ -n "$BLIND_LAYER" || -n "$CONTEXT_FRAMES" ]]; then
    echo "PRUNE_METHOD/PRUNE_K is a separate ablation; do not set BLIND_LAYER or CONTEXT_FRAMES with it" >&2
    exit 2
  fi
fi
INFLUENCE="${INFLUENCE:-}"
MAX_AGE="${MAX_AGE:-33}"
INFLUENCE_MEASURE="${INFLUENCE_MEASURE:-keep_one}"
MAX_K="${MAX_K:-32}"
if [[ -n "$INFLUENCE" ]]; then
  if [[ "$INFLUENCE" != 1 || ! "$MAX_AGE" =~ ^[1-9][0-9]*$ ]]; then
    echo "INFLUENCE must be 1 and MAX_AGE a positive integer, got INFLUENCE='$INFLUENCE' MAX_AGE='$MAX_AGE'" >&2
    exit 2
  fi
  if [[ ! "$INFLUENCE_MEASURE" =~ ^(keep_one|recent_k)$ || ! "$MAX_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "INFLUENCE_MEASURE must be keep_one|recent_k and MAX_K a positive integer, got '$INFLUENCE_MEASURE' / '$MAX_K'" >&2
    exit 2
  fi
  if [[ -n "$BLIND_LAYER" || -n "$CONTEXT_FRAMES" || -n "$PRUNE_METHOD" ]]; then
    echo "INFLUENCE is a separate experiment; do not set BLIND_LAYER, CONTEXT_FRAMES or PRUNE_METHOD with it" >&2
    exit 2
  fi
fi

for py in "$SPY" "$CPY"; do
  if [[ ! -x "$py" ]]; then
    echo "Missing isolated runtime ($py). Run: bash $BASELINE_ROOT/setup_lightnav.sh" >&2
    exit 1
  fi
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

# Server-side environment: habitat-sim picks the render GPU from HABITAT_SIM_GPU_ID and must
# see every GPU (it matches CUDA/EGL devices by UUID), so CUDA_VISIBLE_DEVICES is NOT set.
SERVER_ENV=(env -u CUDA_VISIBLE_DEVICES
  HABITAT_SIM_GPU_ID="$GPU" DISPLAY= HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
  PYTHONPATH="$LN/habitat_server"
  LD_LIBRARY_PATH="$ENV_SERVER/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
NVIDIA_EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -f "$NVIDIA_EGL" ]] && SERVER_ENV+=(__EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL")
CLIENT_ENV=(env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$LN/src" HF_HUB_OFFLINE=1)
# The blind and prune wrappers need eager vLLM (no torch.compile, no CUDA graphs) so their hooks run.
[[ -n "$BLIND_LAYER" || -n "$PRUNE_METHOD" || -n "$INFLUENCE" ]] && CLIENT_ENV+=(VLN_VLLM_ENFORCE_EAGER=1)

subset() {
  python3 -I "$BASELINE_ROOT/lightnav_hm3d/make_subset.py" --labels "$LABELS" --expected-count "$EXPECTED_COUNT" \
    --dest "$SUBSET_ROOT" --scenes-dir "$SCENES_DIR" --lightnav-repo "$LN" "$@"
}

check_checkpoint() {
  for f in config.json eval_config.json action_tokenizer/manifest.json; do
    [[ -f "$CKPT/$f" ]] || { echo "checkpoint incomplete: missing $CKPT/$f (run download)" >&2; return 1; }
  done
  ls "$CKPT"/*.safetensors >/dev/null 2>&1 || { echo "checkpoint has no safetensors" >&2; return 1; }
}

SERVER_PID=""
CLIENT_PID=""
cleanup() {
  for pid in "$CLIENT_PID" "$SERVER_PID"; do
    [[ -n "$pid" ]] && { kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; }
  done
  sleep 1
  for pid in "$CLIENT_PID" "$SERVER_PID"; do
    [[ -n "$pid" ]] && { kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true; }
  done
  return 0  # the EXIT trap's status must not replace the script's
}
trap cleanup EXIT INT TERM

# start_server <variant> <raw_dir>  (variant: compat | native)
start_server() {
  local variant="$1" raw="$2"
  local args=(-m lightnav_habitat.serve --task objectnav --config "$HABITAT_CONFIG" --split val
    --data-path "$SUBSET_ROOT" --scenes-dir "$SCENES_DIR" --max-steps 500
    --port "$PORT" --ready-file "$raw/server.ready")
  case "$variant" in
    compat) args+=(--success-distance 1.0) ;;
    native) args+=(--navmesh-cell-height 0.05) ;;
    *) echo "unknown variant $variant" >&2; return 1 ;;
  esac
  mkdir -p "$raw/logs"
  rm -f "$raw/server.ready"
  SERVER_CMD="$SPY ${args[*]}"
  echo "[lightnav] server: $SERVER_CMD"
  setsid "${SERVER_ENV[@]}" "$SPY" "${args[@]}" >"$raw/logs/server.log" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while [[ ! -f "$raw/server.ready" ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[lightnav] env server died before ready; tail of $raw/logs/server.log:" >&2
      tail -30 "$raw/logs/server.log" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[lightnav] env server not ready after ${READY_TIMEOUT_S}s" >&2
      tail -30 "$raw/logs/server.log" >&2
      return 1
    fi
    sleep 3
  done
  echo "[lightnav] env server ready (pid $SERVER_PID)"
}

stop_server() {
  # The server exits on the client's `close`; give it a moment, then kill the group.
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -- -"$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

# run_eval <variant> <run_name> <episodes> [extra client args]
run_eval() {
  local variant="$1" run_name="$2" episodes="$3"; shift 3
  local run_dir="${RUN_DIR:-$BASELINE_ROOT/results/$run_name}" raw
  raw="$run_dir/raw"
  if [[ -f "$raw/results.jsonl" ]]; then
    echo "[lightnav] $raw/results.jsonl exists (LightNav appends to it); move the run away first" >&2
    exit 1
  fi
  check_checkpoint
  subset >/dev/null
  mkdir -p "$raw/logs"
  local started; started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_server "$variant" "$raw"

  # Upstream entry point unless an ablation is requested; then a harness wrapper, which patches
  # upstream in-process (history sampler / attention kernel) and hands every other argument on.
  local entry=(-m lightnav.cli.eval_habitat)
  if [[ -n "$CONTEXT_FRAMES" ]]; then
    entry=("$BASELINE_ROOT/lightnav_hm3d/eval_ctx.py" --context-frames "$CONTEXT_FRAMES")
  elif [[ -n "$BLIND_LAYER" ]]; then
    entry=("$BASELINE_ROOT/lightnav_hm3d/eval_blind.py" --blind-layer "$BLIND_LAYER")
  elif [[ -n "$PRUNE_METHOD" ]]; then
    entry=("$BASELINE_ROOT/lightnav_hm3d/eval_prune.py" --prune-method "$PRUNE_METHOD" --prune-k "$PRUNE_K")
    [[ -n "$PRUNE_CURRENT_FRAMES" ]] && entry+=(--prune-current-frames "$PRUNE_CURRENT_FRAMES")
  elif [[ -n "$INFLUENCE" ]]; then
    entry=("$BASELINE_ROOT/lightnav_hm3d/eval_influence.py" --max-age "$MAX_AGE")
    [[ "$INFLUENCE_MEASURE" != keep_one ]] && entry+=(--measure "$INFLUENCE_MEASURE" --max-k "$MAX_K")
  fi
  local args=("${entry[@]}" --model_path "$CKPT" --server "tcp://localhost:$PORT"
    --backend "$BACKEND" --episodes "$episodes" --max_steps 500
    --gpu_memory_utilization "$GPU_MEM_UTIL" --zmq_timeout_ms 600000 --output_dir "$raw" "$@")
  CLIENT_CMD="$CPY ${args[*]}"
  echo "[lightnav] client: $CLIENT_CMD"
  set +e
  setsid "${CLIENT_ENV[@]}" "$CPY" "${args[@]}" > >(tee "$raw/logs/client.log") 2>&1 &
  CLIENT_PID=$!
  wait "$CLIENT_PID"; local rc=$?
  set -e
  CLIENT_PID=""
  stop_server
  local finished; finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local n_lines=0; [[ -f "$raw/results.jsonl" ]] && n_lines=$(wc -l <"$raw/results.jsonl")
  local n_rebaked; n_rebaked=$(grep -c '\[navmesh\] rebaked' "$raw/logs/server.log" || true)
  local n_rebake_fail; n_rebake_fail=$(grep -c 'rebake failed' "$raw/logs/server.log" || true)
  local n_server_err; n_server_err=$(grep -c -E 'Reset error|Step error' "$raw/logs/server.log" || true)
  echo "[lightnav] client rc=$rc episodes=$n_lines navmesh_rebaked=$n_rebaked rebake_failed=$n_rebake_fail server_errors=$n_server_err"

  "$CPY" - "$run_dir" "$variant" "$run_name" "$episodes" "$started" "$finished" "$rc" "$BACKEND" \
    "$GPU_MEM_UTIL" "$SERVER_CMD" "$CLIENT_CMD" "$n_rebaked" "$n_rebake_fail" "$n_server_err" <<'PY'
import hashlib, json, os, subprocess, sys
(run_dir, variant, run_name, episodes, started, finished, rc, backend, gmu,
 server_cmd, client_cmd, n_rebaked, n_rebake_fail, n_server_err) = sys.argv[1:]
root = os.environ["BASELINE_ROOT"]
ln = os.path.join(root, "LightNav-0")
ctx = os.environ.get("CONTEXT_FRAMES") or None
blind = os.environ.get("BLIND_LAYER") or None
prune = os.environ.get("PRUNE_METHOD") or None
prune_k = os.environ.get("PRUNE_K") or None
prune_meta = os.path.join(run_dir, "raw", "prune", "meta.json")
influence = os.environ.get("INFLUENCE") or None
influence_meta = os.path.join(run_dir, "raw", "influence", "meta.json")
def sha(repo):
    try: return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return ""
def ver(mod):
    try: return __import__(mod).__version__
    except Exception: return None
labels = os.environ.get("LABELS_PATH")
subset_manifest = os.path.join(os.environ["SUBSET_ROOT"], "manifest.json")
download = os.path.join(os.environ["CKPT"], "DOWNLOAD.json")
info = {
    "variant": variant, "run_name": run_name, "episodes_arg": int(episodes),
    "protocol": {"compat": "success_distance 1.0 m, shipped navmesh",
                 "native": "success_distance 0.1 m, navmesh re-baked at cell_height 0.05"}[variant],
    "started": started, "finished": finished, "client_exit_code": int(rc),
    "backend": backend, "gpu_memory_utilization": float(gmu),
    "server_cmd": server_cmd, "client_cmd": client_cmd,
    "client_entry": ("lightnav_hm3d/eval_ctx.py" if ctx else "lightnav_hm3d/eval_blind.py" if blind
                     else "lightnav_hm3d/eval_prune.py" if prune else "lightnav_hm3d/eval_influence.py" if influence
                     else "lightnav.cli.eval_habitat"),
    "context_frames": int(ctx) if ctx else None,
    "blind_layer": int(blind) if blind else None,
    "prune_method": prune,
    "prune_k": int(prune_k) if prune_k else None,
    "prune": json.load(open(prune_meta)) if prune and os.path.isfile(prune_meta) else None,
    "influence": json.load(open(influence_meta)) if influence and os.path.isfile(influence_meta) else None,
    "enforce_eager": bool(blind or prune or influence),
    "lightnav_sha": sha(ln),
    "lightnav_repo_sha": sha(root),
    "checkpoint": os.environ["CKPT"],
    "checkpoint_download": json.load(open(download)) if os.path.isfile(download) else None,
    "labels_path": labels, "labels_sha256": hashlib.sha256(open(labels, "rb").read()).hexdigest(),
    "subset_manifest": json.load(open(subset_manifest)),
    "client_packages": {m: ver(m) for m in ("torch", "vllm", "transformers", "lightnav")},
    "server_log_counts": {"navmesh_rebaked": int(n_rebaked), "rebake_failed": int(n_rebake_fail),
                          "server_errors": int(n_server_err)},
}
json.dump(info, open(os.path.join(run_dir, "run.json"), "w"), indent=2)
PY
  local allow=()
  (( episodes > 0 && episodes < EXPECTED_COUNT )) && allow=(--allow-missing)
  (cd "$PROJECT_ROOT" && python3 -m baselines.lightnav_hm3d.convert_results "$run_dir" --labels "$LABELS" \
    --expected-count "$EXPECTED_COUNT" "${allow[@]}")
  echo "[lightnav] done: $run_dir/summary.json (client rc=$rc)"
  return "$rc"
}
export LABELS_PATH="$LABELS" SUBSET_ROOT CKPT BASELINE_ROOT CONTEXT_FRAMES BLIND_LAYER PRUNE_METHOD PRUNE_K PRUNE_CURRENT_FRAMES INFLUENCE MAX_AGE

MODE="${1:-}"; shift || true
case "$MODE" in
  download)
    mkdir -p "$LN/checkpoints"
    env -u HF_HUB_OFFLINE "$CPY" - "$CKPT" <<'PY'
import json, sys
from huggingface_hub import snapshot_download, HfApi
dest = sys.argv[1]
path = snapshot_download("LightOriginsHQ/LightNav-0", local_dir=dest)
sha = HfApi().model_info("LightOriginsHQ/LightNav-0").sha
json.dump({"repo_id": "LightOriginsHQ/LightNav-0", "revision": sha, "local_dir": path},
          open(f"{dest}/DOWNLOAD.json", "w"), indent=2)
print("downloaded", path, "revision", sha)
PY
    check_checkpoint && echo "checkpoint OK: $CKPT"
    ;;
  subset)
    subset "$@"
    ;;
  validate)
    "${SERVER_ENV[@]}" "$SPY" -c "import habitat, habitat_sim, numpy, lightnav_habitat.serve; print('server env: habitat-sim', habitat_sim.__version__, 'numpy', numpy.__version__)"
    "${CLIENT_ENV[@]}" "$CPY" -c "import torch, vllm, transformers, zmq, lightnav.cli.eval_habitat; print('client env: torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
    subset --force
    check_checkpoint && echo "checkpoint OK: $CKPT"
    [[ -f "$NVIDIA_EGL" ]] && echo "EGL vendor: $NVIDIA_EGL" || echo "WARN: $NVIDIA_EGL missing"
    RAW="$BASELINE_ROOT/results/lightnav_validate/raw"; mkdir -p "$RAW"
    start_server compat "$RAW"
    grep -E "Dataset loaded|Split|navmesh" "$RAW/logs/server.log" || true
    "${CLIENT_ENV[@]}" "$CPY" "$BASELINE_ROOT/lightnav_hm3d/probe_server.py" --server "tcp://localhost:$PORT"
    stop_server
    ;;
  smoke)        run_eval compat "${RUN_NAME:-lightnav_hm3d_v2_100_smoke}" 1 --verbose --save_video "$@" ;;
  smoke-native) run_eval native "${RUN_NAME:-lightnav_hm3d_v2_100_smoke_native}" 1 --verbose --save_video "$@" ;;
  full)         run_eval compat "${RUN_NAME:-lightnav_hm3d_v2_100}" -1 "$@" ;;
  full-native)  run_eval native "${RUN_NAME:-lightnav_hm3d_v2_100_native}" -1 "$@" ;;
  convert)
    (cd "$PROJECT_ROOT" && python3 -m baselines.lightnav_hm3d.convert_results "$@")
    ;;
  *)
    echo "Usage: $0 {download|subset|validate|smoke|smoke-native|full|full-native|convert <run_dir>} [extra args]" >&2
    exit 2
    ;;
esac
