#!/usr/bin/env bash
# LightNav-0 arms of the flat token-prune variant (paper/token_prune/README.md, "Flat variant").
#
#   VARIANT=flat bash paper/token_prune/run_lightnav.sh                       # identity + 16 arms
#   VARIANT=flat ARMS="lightnav_fastv_b8" bash paper/token_prune/run_lightnav.sh
#   VARIANT=flat K_LIST="32 16" METHODS="random dart" bash paper/token_prune/run_lightnav.sh
#   VARIANT=flat ARMS="lightnav_random_b2 lightnav_random_b1 lightnav_random_b0" bash ...   # layout check
#   VARIANT=flat ARMS="lightnav_recency_b2" bash ...                                        # window-content control
#   VARIANT=flat ARMS="lightnav_recency_b0_cur1" bash ...                                   # its K = 1 point
#
# Arms: lightnav_identity (imported from the information-horizon reference
# paper_results/info_horizon/runs/lightnav_ref: nothing hidden on the same eager masked-attention
# kernel the prune arms use) | lightnav_<method>_b<k>, method in random|fastv|divprune|dart,
# k in 32 16 8 4 (budget = k x 14 history visual tokens, deletion ledger per episode); the layout
# check adds random at k = 2 / 1 / 0 (0 = no visual history) and recency at k = 2 / 6 / 14 / 30 (the newest
# K - 2 history frames = the K = 4 / 8 / 16 / 32 window's visual content, skeleton kept) plus
# lightnav_recency_b0_cur1 (k = 0 with the current frame doubled = the K = 1 window's content,
# PRUNE_CURRENT_FRAMES=1), outside the default grid, reported in results/summary.csv only. Each arm is
# `RUN_DIR=... PRUNE_METHOD=M PRUNE_K=k bash baselines/run_lightnav.sh full`: the comparable 1.0 m
# protocol, the wrapper baselines/lightnav_hm3d/eval_prune.py, upstream clone untouched. HM3D
# only. LightNav must have the GPU to itself (env server + vLLM at 0.65 of the memory), so the
# arms run serially; run_worker.sh can drive this script (family "lightnav") with a lock per arm.
#
# LightNav cannot resume: its results.jsonl is append-only and upstream has no episode filter, so
# a complete arm is skipped and a partial one must be moved away before it is rerun.
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to retry it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VARIANT="${VARIANT:-}"
[[ "$VARIANT" == "flat" ]] || { echo "LightNav-0 runs the flat variant only; set VARIANT=flat" >&2; exit 2; }
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
OUT_ROOT="$ROOT/paper_results/token_prune_flat/runs"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
IDENTITY_SOURCE="${IDENTITY_SOURCE:-$ROOT/paper_results/info_horizon/runs/lightnav_ref}"
K_LIST="${K_LIST:-32 16 8 4}"
METHODS="${METHODS:-random fastv dart divprune}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="lightnav_identity"
  for K in $K_LIST; do
    for M in $METHODS; do ARMS+=" lightnav_${M}_b${K}"; done
  done
fi

complete() { "$PYTHON" -m longnav.analysis.results_io --complete "$1" --manifest "$MANIFEST" >/dev/null 2>&1; }

for ARM in $ARMS; do
  if [[ -d "$OUT_ROOT/$ARM" ]] && complete "$OUT_ROOT/$ARM"; then
    echo "Skipping $ARM: complete."
    continue
  fi
  if [[ "$ARM" == "lightnav_identity" ]]; then
    complete "$IDENTITY_SOURCE" || { echo "$IDENTITY_SOURCE is not a complete run; run paper/info_horizon/run_lightnav.sh (lightnav_ref) first" >&2; exit 1; }
    [[ -d "$OUT_ROOT/$ARM" ]] && { echo "$OUT_ROOT/$ARM exists but is incomplete; move it away" >&2; exit 1; }
    echo "$(date '+%F %T')  importing $ARM from $IDENTITY_SOURCE"
    mkdir -p "$OUT_ROOT/$ARM"
    cp -r "$IDENTITY_SOURCE/raw" "$IDENTITY_SOURCE/episodes" "$IDENTITY_SOURCE/summary.json" "$IDENTITY_SOURCE/run.json" "$OUT_ROOT/$ARM/"
    rm -f "$OUT_ROOT/$ARM/raw/server.ready"
    printf '{"imported_from": "%s", "imported_at": "%s"}\n' "${IDENTITY_SOURCE#$ROOT/}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_ROOT/$ARM/import.json"
    complete "$OUT_ROOT/$ARM" && echo "$(date '+%F %T')  imported $ARM"
    continue
  fi
  if [[ "$ARM" =~ ^lightnav_(random|fastv|divprune|dart|recency)_b([0-9]+)(_cur1)?$ ]]; then
    M="${BASH_REMATCH[1]}"; K="${BASH_REMATCH[2]}"; CUR=""
    [[ -n "${BASH_REMATCH[3]}" ]] && CUR=1
  else
    echo "Unknown arm '$ARM' (expected lightnav_identity or lightnav_<method>_b<k>)" >&2
    exit 2
  fi
  if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
    echo "$OUT_ROOT/$ARM is partial; LightNav cannot resume (append-only results.jsonl), move it away first" >&2
    exit 1
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  (PRUNE_METHOD=$M PRUNE_K=$K PRUNE_CURRENT_FRAMES=${CUR:-2})"
  echo "========================================"
  RUN_DIR="$OUT_ROOT/$ARM" PRUNE_METHOD="$M" PRUNE_K="$K" PRUNE_CURRENT_FRAMES="$CUR" RUN_NAME="$ARM" \
    bash baselines/run_lightnav.sh full 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested LightNav-0 arms done. Next: python paper/token_prune/aggregate.py --variant flat"
