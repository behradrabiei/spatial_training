#!/usr/bin/env bash
# LightNav-0 arm of the leave-one-IN (keep-one) influence experiment (paper/frame_influence/README.md).
#
#   bash paper/frame_influence/run_lightnav.sh
#
# Runs the isolated LightNav runtime (baselines/run_lightnav.sh, INFLUENCE=1) on the 10-episode
# manifest: normal navigation (eager vLLM, masked-SDPA kernel = lightnav_ref) plus, after every
# decision, the keep-one KL probes of baselines/lightnav_hm3d/eval_influence.py over the history
# tubelets of age <= MAX_AGE (default 33 = the fast tier), written to influence/<label>.jsonl.
# MEASURE=recent_k runs the recent-window variant instead (arm lightnav_influence_recent under
# paper_results/frame_influence_recent/runs): after every decision, KL(full || K most recent frames)
# for K <= MAX_K (default 32), both with the old blocks kept (visual columns masked) and with the
# prompt rebuilt as a K-frame window -- one run, both arms on the same decisions.
# No leave-one-out arm for this model. Skipped when every manifest episode is done;
# LightNav cannot resume an episode list (append-only results.jsonl), so a failed run is moved away
# by hand before relaunching.
set -uo pipefail  # exit status handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
MEASURE="${MEASURE:-keep_one}"
case "$MEASURE" in
  keep_one) ARM="${ARM:-lightnav_influence_keepone}"; OUT_ROOT="$ROOT/paper_results/frame_influence_keepone/runs" ;;
  recent_k) ARM="${ARM:-lightnav_influence_recent}";  OUT_ROOT="$ROOT/paper_results/frame_influence_recent/runs" ;;
  *) echo "Unknown MEASURE '$MEASURE' (expected keep_one or recent_k)" >&2; exit 2 ;;
esac
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$OUT_ROOT}"  # smoke runs: an ignored validation directory
MANIFEST="${MANIFEST:-$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json}"
MAX_AGE="${MAX_AGE:-33}"
MAX_K="${MAX_K:-32}"

if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi
if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
  echo "$OUT_ROOT/$ARM holds a partial run (LightNav cannot resume); move it away first" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/$ARM"
echo "========================================"
echo "$(date '+%F %T')  starting $ARM (measure $MEASURE, max_age $MAX_AGE, max_k $MAX_K)"
echo "========================================"
INFLUENCE=1 INFLUENCE_MEASURE="$MEASURE" MAX_AGE="$MAX_AGE" MAX_K="$MAX_K" LABELS="$MANIFEST" RUN_DIR="$OUT_ROOT/$ARM" \
  bash baselines/run_lightnav.sh full "$@" 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "$(date '+%F %T')  $ARM FAILED (exit $STATUS); see $OUT_ROOT/$ARM/launch.log"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
echo "Next: python paper/frame_influence/aggregate.py --measure $MEASURE"
