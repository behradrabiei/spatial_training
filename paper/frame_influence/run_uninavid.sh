#!/usr/bin/env bash
# Uni-NaVid arm of the leave-one-frame-out influence experiment (paper/frame_influence/README.md).
#
#   bash paper/frame_influence/run_uninavid.sh
#
# Runs the isolated baseline runtime (baselines/run.sh) on the 10-episode manifest with
# --frame-influence: after every inference the prompt is re-scored with each history frame
# removed in turn plus the joint probes, written to influence/<label>.jsonl. Official decoding
# (temperature 0.2, 2 actions per call, 500 steps) with per-call seeding (--sampling-seed) so the
# sampled trajectory is reproducible. Skipped when every manifest episode is done; --resume
# continues a partial run. MEASURE=keep_one runs the leave-one-IN variant instead (arm
# uninavid_influence_keepone under paper_results/frame_influence_keepone/, --influence-measures
# keep_one; same seed, so the same sampled trajectories).
set -uo pipefail  # exit status handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
MEASURE="${MEASURE:-leave_one}"  # leave_one | keep_one
KEEP_ONE_MODE="${KEEP_ONE_MODE:-mask}"  # keep_one only: mask (position-preserving, the paper's) | splice
case "$MEASURE" in
  leave_one) OUT_ROOT="$ROOT/paper_results/frame_influence/runs";         ARM="${ARM:-uninavid_influence}"; MEASURE_FLAG=leave_one ;;
  keep_one)  OUT_ROOT="$ROOT/paper_results/frame_influence_keepone/runs"; ARM="${ARM:-uninavid_influence_keepone}"
             case "$KEEP_ONE_MODE" in mask) MEASURE_FLAG=keep_one_mask ;; splice) MEASURE_FLAG=keep_one ;;
               *) echo "Unknown KEEP_ONE_MODE '$KEEP_ONE_MODE' (expected mask or splice)" >&2; exit 2 ;; esac ;;
  *) echo "Unknown MEASURE '$MEASURE' (expected leave_one or keep_one)" >&2; exit 2 ;;
esac
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
SAMPLING_SEED="${SAMPLING_SEED:-30}"
PROBE_BATCH="${PROBE_BATCH:-8}"

if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi

mkdir -p "$OUT_ROOT/$ARM"
echo "========================================"
echo "$(date '+%F %T')  starting $ARM"
echo "========================================"
bash baselines/run.sh eval --resume \
  --labels "$MANIFEST" --expected-count 10 \
  --output "$OUT_ROOT/$ARM" \
  --frame-influence --influence-measures "$MEASURE_FLAG" --probe-batch "$PROBE_BATCH" --sampling-seed "$SAMPLING_SEED" "$@" \
  2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "$(date '+%F %T')  $ARM FAILED (exit $STATUS); see $OUT_ROOT/$ARM/launch.log"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
echo "Next: python paper/frame_influence/aggregate.py$([[ "$MEASURE" == keep_one ]] && echo ' --measure keep_one')"
