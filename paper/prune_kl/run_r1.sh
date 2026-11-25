#!/usr/bin/env bash
# LongNav-R1 arm of the pruning-KL experiment (paper/prune_kl/README.md).
#
#   conda activate longnav_vlm
#   bash paper/prune_kl/run_r1.sh
#
# One run (arm r1_prune_kl, +experiment=paper_prune_kl) on the 10-episode seed-17 manifest:
# full-context navigation plus, after every decision, KL(full || pruned) for one shadow pruner per
# (method, k) -- random / FastV / DivPrune / DART at k = 4, 8, 16, 32 frames' worth -- by leaky
# decision replay (longnav.utils.prune_kl_probe). Skipped when every manifest episode is logged; an
# unfinished run resumes through wandb (same task.run_name). Extra arguments are hydra overrides.
# Smoke runs: OUT_ROOT_OVERRIDE=<ignored dir> MANIFEST=<labels.json> ARM=<name> plus
# task.wandb_project=<its own project> (resume keys on project + run name).
set -uo pipefail  # exit status handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
ARM="${ARM:-r1_prune_kl}"
DEFAULT_OUT="$ROOT/paper_results/prune_kl/runs"  # must match task.output_dir in the experiment preset
DEFAULT_MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$DEFAULT_OUT}"
MANIFEST="${MANIFEST:-$DEFAULT_MANIFEST}"
EXTRA=()
[[ "$OUT_ROOT" != "$DEFAULT_OUT" ]] && EXTRA+=("task.output_dir=$OUT_ROOT")
[[ "$MANIFEST" != "$DEFAULT_MANIFEST" ]] && EXTRA+=("task.episode_json=$MANIFEST")

if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi

mkdir -p "$OUT_ROOT/$ARM"
echo "========================================"
echo "$(date '+%F %T')  starting $ARM"
echo "========================================"
"$PYTHON" -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment=paper_prune_kl +resources="$RESOURCES" \
  task.run_name="$ARM" "${EXTRA[@]}" "$@" \
  2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "$(date '+%F %T')  $ARM FAILED (exit $STATUS); see $OUT_ROOT/$ARM/launch.log"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
echo "Next: python paper/prune_kl/aggregate.py"
