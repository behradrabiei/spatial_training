#!/usr/bin/env bash
# Uni-NaVid arms of the hybrid-pruning comparison (paper/hybrid_prune/README.md).
#
#   bash paper/hybrid_prune/run_uninavid.sh                            # every arm
#   ARMS="uninavid_hybrid_b8" bash paper/hybrid_prune/run_uninavid.sh
#
# Uses the isolated baseline runtime via baselines/run.sh with the official evaluator settings
# (seed 30, temperature 0.2, 2 actions per inference, 500 steps). Arm names:
# uninavid_<protocol>_<mp|m|b><k> (mp = the experiment: whole-model budget, --prune-budget-scope model
# -- k*4 bounds the history tokens attended averaged over all 32 layers, with a layer-0 pre-pass stage
# in front -- and --prune-schedule-anchor pool; m = the same anchored at the original token count,
# aborted; b = last-stage budget, the first, superseded run):
#   hybrid       --context-frames 32 --prune-method hybrid       (attn stages 8/16, random from 24)
#   pyramiddrop  --context-frames 64 --prune-method pyramiddrop  (stages 8/16/24; 64 frames is the
#                widest history the pruner supports -- the long-term tier starts at 65)
# Both protocols seed the per-call sampler (--sampling-seed 30), so the pair differs by the
# protocol and not by sampler noise.
#
# DATASET=mp3d runs the MP3D replicate (objectnav_mp3d.yaml, MP3D val episodes and scenes, the seed-17
# 100-episode manifest, data in paper_results/hybrid_prune_mp3d/).
#
# A pilot writes elsewhere: OUT_ROOT=<dir> MANIFEST=<labels.json> EVAL_EXTRA="--expected-count 3".
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
DATASET="${DATASET:-hm3d}"  # hm3d | mp3d
if [[ "$DATASET" == "mp3d" ]]; then
  MANIFEST="${MANIFEST:-$ROOT/src/longnav/conf/episode_jsons/mp3d_100_seed17.json}"
  OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/hybrid_prune_mp3d/runs}"
  DATA_ARGS=(--config "$ROOT/baselines/uninavid_hm3d/config/objectnav_mp3d.yaml"
             --dataset /home/brabiei/vault/habitat_data/evaluation_episodes/MP3D/objectnav_mp3d_v1/val/val.json.gz
             --scenes /home/brabiei/vault/habitat_data/scenes)
else
  MANIFEST="${MANIFEST:-$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json}"
  OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/hybrid_prune/runs}"
  DATA_ARGS=()
fi
K_LIST="${K_LIST:-8 4 16 2}"
SAMPLING_SEED="${SAMPLING_SEED:-30}"
read -r -a EVAL_EXTRA <<< "${EVAL_EXTRA:-}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS=""
  for K in $K_LIST; do ARMS+=" uninavid_hybrid_mp${K} uninavid_pyramiddrop_mp${K} uninavid_uniform_mp${K}"; done
fi

for ARM in $ARMS; do
  if [[ "$ARM" =~ ^uninavid_(hybrid|pyramiddrop|uniform)_(mp|m|b)([0-9]+)$ ]]; then
    FRAMES=32; [[ "${BASH_REMATCH[1]}" == "pyramiddrop" ]] && FRAMES=64
    EXTRA=(--context-frames "$FRAMES" --prune-method "${BASH_REMATCH[1]}" --prune-k "${BASH_REMATCH[3]}"
           --sampling-seed "$SAMPLING_SEED")
    [[ "${BASH_REMATCH[2]}" != "b" ]] && EXTRA+=(--prune-budget-scope model)
    [[ "${BASH_REMATCH[2]}" == "mp" ]] && EXTRA+=(--prune-schedule-anchor pool)
  else
    echo "Unknown arm '$ARM' (expected uninavid_<hybrid|pyramiddrop|uniform>_<mp|m|b><k>; uniform is mp only)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${EXTRA[*]}"
  echo "========================================"
  bash baselines/run.sh eval --resume \
    --labels "$MANIFEST" \
    --output "$OUT_ROOT/$ARM" \
    "${DATA_ARGS[@]}" "${EXTRA[@]}" "${EVAL_EXTRA[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested Uni-NaVid arms done. Next: python paper/hybrid_prune/aggregate.py${DATASET:+ --dataset $DATASET}"
