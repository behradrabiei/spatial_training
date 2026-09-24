#!/usr/bin/env bash
# Uni-NaVid arms of the context-window ablation (paper/ctx_ablation/README.md).
#
#   bash paper/ctx_ablation/run_uninavid.sh                    # every arm
#   ARMS="uninavid_full uninavid_k1" bash paper/ctx_ablation/run_uninavid.sh
#
# Uses the isolated baseline runtime via baselines/run.sh (official evaluator settings:
# seed 30, temperature 0.2, 2 actions per inference, 500 steps). K = frames visible per
# inference including the current one (--context-frames). Outputs land next to the R1 arms in
# paper_results/ctx_ablation/runs/<arm> (tracked; traces are gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
OUT_ROOT="$ROOT/paper_results/ctx_ablation/runs"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
K_LIST="${K_LIST:-32 16 8 4 1}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="uninavid_full"
  for K in $K_LIST; do ARMS+=" uninavid_k${K}"; done
fi

for ARM in $ARMS; do
  EXTRA=()
  if [[ "$ARM" =~ ^uninavid_k([0-9]+)$ ]]; then
    EXTRA+=(--context-frames "${BASH_REMATCH[1]}")
  elif [[ "$ARM" != "uninavid_full" ]]; then
    echo "Unknown arm '$ARM' (expected uninavid_full or uninavid_k<K>)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${EXTRA[*]:-}"
  echo "========================================"
  bash baselines/run.sh eval --resume \
    --labels "$MANIFEST" \
    --output "$OUT_ROOT/$ARM" \
    "${EXTRA[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested Uni-NaVid arms done. Next: python paper/ctx_ablation/aggregate.py"
