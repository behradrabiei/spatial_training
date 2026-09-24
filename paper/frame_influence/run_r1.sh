#!/usr/bin/env bash
# LongNav-R1 arm of the leave-one-frame-out influence experiment (paper/frame_influence/README.md).
#
#   conda activate longnav_vlm
#   bash paper/frame_influence/run_r1.sh
#
# Arms (ARM=...): r1_influence (default; evict-style replay probes, 10 episodes,
# +experiment=paper_influence), r1_influence_recompute (leak-free full re-prefill probes on the
# 9 episodes shorter than 150 steps, +experiment=paper_influence_recompute) and
# r1_influence_keepone (leave-one-IN replay probes, 10 episodes, +experiment=paper_influence_keepone,
# output under paper_results/frame_influence_keepone/). Skipped when every
# manifest episode is already logged; an unfinished run resumes through wandb (same
# task.run_name). Never reuse a run name from another experiment in the same wandb project
# (e.g. r1_full): resume keys on the name and would skip the already-logged episodes.
set -uo pipefail  # exit status handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
OUT_ROOT="$ROOT/paper_results/frame_influence/runs"  # must match task.output_dir in the experiment preset
ARM="${ARM:-r1_influence}"
case "$ARM" in
  r1_influence)           EXPERIMENT=paper_influence;           MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json" ;;
  r1_influence_recompute) EXPERIMENT=paper_influence_recompute; MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_9_seed17_short.json" ;;
  r1_influence_keepone)   EXPERIMENT=paper_influence_keepone;   MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
                          OUT_ROOT="$ROOT/paper_results/frame_influence_keepone/runs" ;;
  *) echo "Unknown arm '$ARM' (expected r1_influence, r1_influence_recompute or r1_influence_keepone)" >&2; exit 2 ;;
esac

if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi

mkdir -p "$OUT_ROOT/$ARM"
echo "========================================"
echo "$(date '+%F %T')  starting $ARM"
echo "========================================"
"$PYTHON" -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment="$EXPERIMENT" +resources="$RESOURCES" \
  task.run_name="$ARM" "$@" \
  2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "$(date '+%F %T')  $ARM FAILED (exit $STATUS); see $OUT_ROOT/$ARM/launch.log"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
echo "Next: python paper/frame_influence/aggregate.py$([[ "$ARM" == r1_influence_keepone ]] && echo ' --measure keep_one')"
