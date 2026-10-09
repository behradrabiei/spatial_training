#!/usr/bin/env bash
# Selector-criterion vs keep-one-in KL probe of LongNav-R1 (paper/selector_kl/README.md), end to end:
#
#   conda activate longnav_vlm
#   bash paper/selector_kl/run.sh            # R1 probe run -> Uni-NaVid probe run -> aggregate -> showcase frame -> plot
#   SHOWCASE=wcojb4TFT35_0:120 bash paper/selector_kl/run.sh   # override the showcase decision
#
# The probe run is skipped when complete (results_io --complete) and resumes through wandb
# otherwise; aggregate / showcase / plot are re-run every time (seconds, minutes, seconds).
# Everything lands under paper_results/selector_kl/ (runs/, results/, figures/).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
VLN_PYTHON="${VLN_PYTHON:-/home/brabiei/miniconda3/envs/vln/bin/python}"
RESOURCES="${RESOURCES:-single}"
OUT="$ROOT/paper_results/selector_kl"
RUNS="$OUT/runs"
ARM="r1_selector_probe"
MANIFEST="src/longnav/conf/episode_jsons/hm3d_v2_10_short.json"
mkdir -p "$RUNS/$ARM"

if "$PYTHON" -m longnav.analysis.results_io --complete "$RUNS/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
else
  echo "$(date '+%F %T')  starting $ARM"
  "$PYTHON" -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_v2_val +experiment=paper_selector_probe +resources="$RESOURCES" \
    task.run_name="$ARM" 2>&1 | tee -a "$RUNS/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
    exit "$STATUS"
  fi
  echo "$(date '+%F %T')  finished $ARM"
fi

bash paper/selector_kl/run_uninavid.sh || exit $?

SHOWCASE_ARGS=()
if [[ -n "${SHOWCASE:-}" ]]; then SHOWCASE_ARGS=(--showcase "$SHOWCASE"); fi
"$PYTHON" paper/selector_kl/aggregate.py "${SHOWCASE_ARGS[@]}" || exit $?
"$PYTHON" paper/selector_kl/showcase_frames.py || exit $?
"$VLN_PYTHON" paper/selector_kl/plot.py || exit $?
echo "Results: $OUT/results   Figures: $OUT/figures"
