#!/usr/bin/env bash
# Uni-NaVid arms of the visual-token pruning experiment (paper/token_prune/README.md).
#
#   bash paper/token_prune/run_uninavid.sh                     # every arm
#   ARMS="uninavid_identity uninavid_fastv_b8" bash paper/token_prune/run_uninavid.sh
#
# Uses the isolated baseline runtime via baselines/run.sh with the ctx-ablation settings
# (official evaluator: seed 30, temperature 0.2, 2 actions per inference, 500 steps,
# --context-frames 32 = 31 history frames + the current one) plus the pruning flags. Arm names:
# uninavid_identity | uninavid_<method>_b<k>. Set SAMPLING_SEED to seed the per-call sampler
# (the ctx-ablation arms did not, so uninavid_identity matches uninavid_k32 only statistically).
#
# VARIANT=flat runs the flat variant (README "Flat variant"): --context-frames 64 (the widest
# window the pruner supports; at 65 frames Uni-NaVid's long-term tier starts) and --prune-layer 0
# (every method scores and deletes at decoder layer 0 by row removal), methods
# {random,fastv,divprune,dart}, k in {32,16,8,4}, output under paper_results/token_prune_flat/runs.
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
VARIANT="${VARIANT:-window}"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
case "$VARIANT" in
  window)
    OUT_ROOT="$ROOT/paper_results/token_prune/runs"
    CONTEXT_FRAMES=32; PRUNE_LAYER=""
    METHODS="${METHODS:-random fastv pyramiddrop divprune dart}"
    K_LIST="${K_LIST:-8 4 1 16}" ;;
  flat)
    OUT_ROOT="$ROOT/paper_results/token_prune_flat/runs"
    CONTEXT_FRAMES=64; PRUNE_LAYER=0
    METHODS="${METHODS:-random fastv divprune dart}"
    K_LIST="${K_LIST:-8 4 16 32}" ;;
  *) echo "Unknown VARIANT '$VARIANT' (expected window or flat)" >&2; exit 2 ;;
esac
SAMPLING_SEED="${SAMPLING_SEED:-}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="uninavid_identity"
  for K in $K_LIST; do
    for M in $METHODS; do ARMS+=" uninavid_${M}_b${K}"; done
  done
fi

for ARM in $ARMS; do
  EXTRA=(--context-frames "$CONTEXT_FRAMES")
  if [[ "$ARM" =~ ^uninavid_(random|fastv|pyramiddrop|divprune|dart)_b([0-9]+)$ ]]; then
    EXTRA+=(--prune-method "${BASH_REMATCH[1]}" --prune-k "${BASH_REMATCH[2]}")
    [[ -n "$PRUNE_LAYER" ]] && EXTRA+=(--prune-layer "$PRUNE_LAYER")
  elif [[ "$ARM" == "uninavid_identity" ]]; then
    EXTRA+=(--prune-method identity)
  else
    echo "Unknown arm '$ARM' (expected uninavid_identity or uninavid_<method>_b<k>)" >&2
    exit 2
  fi
  [[ -n "$SAMPLING_SEED" ]] && EXTRA+=(--sampling-seed "$SAMPLING_SEED")

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
    "${EXTRA[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested Uni-NaVid arms done. Next: python paper/token_prune/aggregate.py"
