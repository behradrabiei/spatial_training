#!/usr/bin/env bash
# LongNav-R1 arms of the visual-token pruning experiment (paper/token_prune/README.md).
#
#   bash paper/token_prune/run_r1.sh                          # every arm, in order
#   ARMS="r1_identity r1_random_b8" bash paper/token_prune/run_r1.sh
#
# Arm names encode the config: r1_identity | r1_<method>_b<k>, method in
# {random,fastv,pyramiddrop,divprune,dart}, k in {16,8,4,1} frames' worth of history tokens.
# Default order: identity, then k=8 for every method (feeds the retention figure), then 4, 1, 16,
# so partial sweeps are informative early. Finished arms (all manifest episodes logged) are
# skipped, so the script can be re-run after an interruption; an unfinished arm resumes through
# wandb (same task.run_name => already-logged episodes are skipped).
#
# VARIANT=flat runs the flat variant (README "Flat variant"): full episode context, every method
# at decoder layer 0, methods {random,fastv,divprune,dart}, k in {32,16,8,4}, output under
# paper_results/token_prune_flat/runs (preset paper_prune_flat, its own wandb project).
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_OVERRIDES:-}"  # e.g. resources.osm_gb=10 when two arms share the machine
VARIANT="${VARIANT:-window}"
MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_100.json"
case "$VARIANT" in
  window)
    PRESET=paper_prune
    OUT_ROOT="$ROOT/paper_results/token_prune/runs"  # must match task.output_dir in experiment/paper_prune.yaml
    METHODS="${METHODS:-random fastv pyramiddrop divprune dart}"
    K_LIST="${K_LIST:-8 4 1 16}" ;;
  flat)
    PRESET=paper_prune_flat
    OUT_ROOT="$ROOT/paper_results/token_prune_flat/runs"  # must match task.output_dir in experiment/paper_prune_flat.yaml
    METHODS="${METHODS:-random fastv divprune dart}"
    K_LIST="${K_LIST:-8 4 16 32}" ;;
  *) echo "Unknown VARIANT '$VARIANT' (expected window or flat)" >&2; exit 2 ;;
esac

if [[ -z "${ARMS:-}" ]]; then
  ARMS="r1_identity"
  for K in $K_LIST; do
    for M in $METHODS; do ARMS+=" r1_${M}_b${K}"; done
  done
fi

method_overrides() {  # <method> -> Hydra overrides selecting the pruning rule (see README arms table)
  if [[ "$VARIANT" == "flat" ]]; then
    case "$1" in
      random)   echo "vlm.kv_prune_importance=random" ;;
      divprune) echo "vlm.kv_prune_importance=divprune" ;;
      fastv)    echo "vlm.kv_prune_importance=attn vlm.kv_prune_score_layer=0" ;;
      dart)     echo "vlm.kv_prune_importance=dart" ;;
      *) return 1 ;;
    esac
    return
  fi
  case "$1" in
    random)      echo "vlm.kv_prune_importance=random" ;;
    divprune)    echo "vlm.kv_prune_importance=divprune" ;;
    fastv)       echo "vlm.kv_prune_importance=attn vlm.kv_prune_layer_start=2 vlm.kv_prune_score_layers=boundary" ;;
    dart)        echo "vlm.kv_prune_importance=dart vlm.kv_prune_layer_start=2" ;;
    pyramiddrop) echo "vlm.kv_prune_importance=attn vlm.kv_prune_stages=[7,14,21]" ;;
    *) return 1 ;;
  esac
}

for ARM in $ARMS; do
  OVERRIDES=()
  if [[ "$ARM" =~ ^r1_(random|fastv|pyramiddrop|divprune|dart)_b([0-9]+)$ ]]; then
    if ! METHOD_OVERRIDES="$(method_overrides "${BASH_REMATCH[1]}")"; then
      echo "Arm '$ARM' is not part of VARIANT=$VARIANT" >&2
      exit 2
    fi
    read -r -a OVERRIDES <<< "$METHOD_OVERRIDES"
    OVERRIDES+=("vlm.kv_prune_budget_frames=${BASH_REMATCH[2]}")
  elif [[ "$ARM" != "r1_identity" ]]; then
    echo "Unknown arm '$ARM' (expected r1_identity or r1_<method>_b<k>)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${OVERRIDES[*]:-}"
  echo "========================================"
  "$PYTHON" -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment="$PRESET" +resources="$RESOURCES" \
    task.run_name="$ARM" "${OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested LongNav-R1 arms done. Next: python paper/token_prune/aggregate.py --variant $VARIANT"
