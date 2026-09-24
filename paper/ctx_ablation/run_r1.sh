#!/usr/bin/env bash
# LongNav-R1 arms of the context-window ablation (paper/ctx_ablation/README.md).
#
#   bash paper/ctx_ablation/run_r1.sh                       # every arm, in order
#   ARMS="r1_full r1_evict_k8" bash paper/ctx_ablation/run_r1.sh
#
# Arm names encode the config: r1_full | r1_{evict,recompute}_k<K>, K = frames visible at
# decision time including the current one. Finished arms (all manifest episodes logged) are
# skipped, so the script can be re-run after an interruption; an unfinished arm resumes
# through wandb (same task.run_name => already-logged episodes are skipped).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
OUT_ROOT="$ROOT/paper_results/ctx_ablation/runs"  # must match task.output_dir in experiment/paper_ctx.yaml
MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_100.json"
K_LIST="${K_LIST:-32 16 8 4 1}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="r1_full"
  for MODE in evict recompute; do
    for K in $K_LIST; do ARMS+=" r1_${MODE}_k${K}"; done
  done
fi

for ARM in $ARMS; do
  OVERRIDES=()
  if [[ "$ARM" =~ ^r1_(evict|recompute)_k([0-9]+)$ ]]; then
    OVERRIDES+=("vlm.context_frames=${BASH_REMATCH[2]}" "vlm.context_window_mode=${BASH_REMATCH[1]}")
  elif [[ "$ARM" != "r1_full" ]]; then
    echo "Unknown arm '$ARM' (expected r1_full or r1_{evict,recompute}_k<K>)" >&2
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
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment=paper_ctx +resources="$RESOURCES" \
    task.run_name="$ARM" "${OVERRIDES[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested LongNav-R1 arms done. Next: python paper/ctx_ablation/aggregate.py"
