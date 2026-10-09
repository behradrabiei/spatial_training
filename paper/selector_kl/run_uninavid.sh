#!/usr/bin/env bash
# Uni-NaVid arm of the selector-criterion vs keep-one-in KL probe (paper/selector_kl/README.md):
#
#   bash paper/selector_kl/run_uninavid.sh              # the 10-episode arm, skipped when complete
#   LABELS=/abs/path/one.json COUNT=1 bash paper/selector_kl/run_uninavid.sh   # a smoke on one label
#
# Native evaluator, seeded official decoding (--sampling-seed 30, as paper/info_horizon), full
# native history, the per-token probe after every inference (--selector-probe). Resumes through
# --resume (finished labels are skipped). Absolute paths: baselines/run.sh cds into Uni-NaVid.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
ARM="${ARM:-uninavid_selector_probe}"
OUT="$ROOT/paper_results/selector_kl/runs/$ARM"
LABELS="${LABELS:-$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_short.json}"
COUNT="${COUNT:-10}"
SAMPLING_SEED="${SAMPLING_SEED:-30}"
PROBE_BATCH="${PROBE_BATCH:-12}"

if [[ -d "$OUT" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT" --manifest "$LABELS" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi
mkdir -p "$OUT"
echo "$(date '+%F %T')  starting $ARM"
bash baselines/run.sh eval --resume --labels "$LABELS" --expected-count "$COUNT" --output "$OUT" \
  --sampling-seed "$SAMPLING_SEED" --selector-probe --probe-batch "$PROBE_BATCH" \
  2>&1 | tee -a "$OUT/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ $STATUS -ne 0 ]]; then
  echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run to resume"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
