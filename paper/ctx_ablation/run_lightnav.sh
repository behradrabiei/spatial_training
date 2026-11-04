#!/usr/bin/env bash
# LightNav-0 arms of the context-window ablation (paper/ctx_ablation/README.md).
#
#   bash paper/ctx_ablation/run_lightnav.sh                    # every arm
#   ARMS="lightnav_k8 lightnav_k1" bash paper/ctx_ablation/run_lightnav.sh
#
# Uses the isolated LightNav runtime via baselines/run_lightnav.sh in its comparable protocol
# (`full`: success radius 1.0 m, the model's own camera, velocity actions, greedy decoding,
# 500-step cap). K = frames visible per inference including the current one; the window is
# applied in the harness (baselines/lightnav_hm3d/eval_ctx.py), the upstream clone is untouched.
# `lightnav_full` is the comparable 100-episode run already under baselines/results/ (the
# model's own full-episode history); it is imported, not rerun. Outputs land next to the other
# arms in paper_results/ctx_ablation/runs/<arm> (raw/, run.json, episodes/, summary.json).
#
# LightNav cannot resume: its results.jsonl is append-only and upstream has no episode filter, so
# a complete arm is skipped and a partial one must be moved away before it is rerun.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
OUT_ROOT="$ROOT/paper_results/ctx_ablation/runs"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
FULL_SOURCE="${FULL_SOURCE:-$ROOT/baselines/results/lightnav_hm3d_v2_100}"
K_LIST="${K_LIST:-32 16 8 4 1}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="lightnav_full"
  for K in $K_LIST; do ARMS+=" lightnav_k${K}"; done
fi

complete() { "$PYTHON" -m longnav.analysis.results_io --complete "$1" --manifest "$MANIFEST" >/dev/null 2>&1; }

for ARM in $ARMS; do
  if [[ -d "$OUT_ROOT/$ARM" ]] && complete "$OUT_ROOT/$ARM"; then
    echo "Skipping $ARM: complete."
    continue
  fi
  if [[ "$ARM" == "lightnav_full" ]]; then
    complete "$FULL_SOURCE" || { echo "$FULL_SOURCE is not a complete run; run 'bash baselines/run_lightnav.sh full' first" >&2; exit 1; }
    [[ -d "$OUT_ROOT/$ARM" ]] && { echo "$OUT_ROOT/$ARM exists but is incomplete; move it away" >&2; exit 1; }
    echo "$(date '+%F %T')  importing $ARM from $FULL_SOURCE"
    mkdir -p "$OUT_ROOT/$ARM"
    cp -r "$FULL_SOURCE/raw" "$FULL_SOURCE/episodes" "$FULL_SOURCE/summary.json" "$FULL_SOURCE/run.json" "$OUT_ROOT/$ARM/"
    rm -f "$OUT_ROOT/$ARM/raw/server.ready"
    printf '{"imported_from": "%s", "imported_at": "%s"}\n' "${FULL_SOURCE#$ROOT/}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_ROOT/$ARM/import.json"
    complete "$OUT_ROOT/$ARM" && echo "$(date '+%F %T')  imported $ARM"
    continue
  fi
  if [[ "$ARM" =~ ^lightnav_k([0-9]+)$ ]]; then
    K="${BASH_REMATCH[1]}"
  else
    echo "Unknown arm '$ARM' (expected lightnav_full or lightnav_k<K>)" >&2
    exit 2
  fi
  if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
    echo "$OUT_ROOT/$ARM is partial; LightNav cannot resume (append-only results.jsonl), move it away first" >&2
    exit 1
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  (CONTEXT_FRAMES=$K)"
  echo "========================================"
  RUN_DIR="$OUT_ROOT/$ARM" CONTEXT_FRAMES="$K" RUN_NAME="$ARM" \
    bash baselines/run_lightnav.sh full 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested LightNav-0 arms done. Next: python paper/ctx_ablation/aggregate.py"
