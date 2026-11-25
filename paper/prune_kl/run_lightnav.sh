#!/usr/bin/env bash
# LightNav-0 arm of the pruning-KL experiment (paper/prune_kl/README.md).
#
#   bash paper/prune_kl/run_lightnav.sh
#
# Runs the isolated LightNav runtime (baselines/run_lightnav.sh, INFLUENCE=1
# INFLUENCE_MEASURE=prune_kl) on the 10-episode manifest: normal navigation (eager vLLM,
# masked-SDPA kernel = lightnav_ref) plus, after every decision, KL(full || pruned) for one shadow
# pruner per (method, k) -- random / FastV / DivPrune / DART at k = 4, 8, 16, 32 frames' worth,
# eval_prune.py's rule on the full-context trajectory -- written to influence/<label>.jsonl.
# The random selector is seeded with each episode's index in the 100-episode pruning runs
# (LightNav serves a subset in sorted label order), so the shadow's random masks are those of
# paper_results/token_prune_flat/runs/lightnav_random_b<k> (checked by aggregate.py).
# GATE_RECENCY=1 adds recency:{2,6,14,30} shadows, which must reproduce the recent-window run's
# old-blocks-kept probes at K = 4, 8, 16, 32 (smoke runs).
# Skipped when every manifest episode is done; LightNav cannot resume an episode list (append-only
# results.jsonl), so a failed run is moved away by hand before relaunching. Needs the GPU alone.
set -uo pipefail  # exit status handled explicitly below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # completeness check + seed episodes
ARM="${ARM:-lightnav_prune_kl}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$ROOT/paper_results/prune_kl/runs}"  # smoke runs: an ignored validation directory
MANIFEST="${MANIFEST:-$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json}"
SEED_LABELS="${SEED_LABELS:-$ROOT/dump/hm3d_v2_100_labels.json}"  # the episode list of the pruning runs
ARMS="${PRUNE_KL_ARMS:-}"  # default: eval_influence.py's 4 x 4 grid
if [[ -n "${GATE_RECENCY:-}" ]]; then
  grid=""
  for m in random fastv divprune dart; do for k in 4 8 16 32; do grid+="$m:$k "; done; done
  ARMS="${ARMS:-${grid% }} recency:2 recency:6 recency:14 recency:30"
fi

if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
  echo "Skipping $ARM: complete."
  exit 0
fi
if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
  echo "$OUT_ROOT/$ARM holds a partial run (LightNav cannot resume); move it away first" >&2
  exit 1
fi
# Episode i of this run (sorted label order) seeds like episode SEED_EPISODES[i] of the 100-episode runs.
SEED_EPISODES="$(python3 - "$MANIFEST" "$SEED_LABELS" <<'PY'
import json, sys
labels, ref = sorted(json.load(open(sys.argv[1]))), sorted(json.load(open(sys.argv[2])))
print(" ".join(str(ref.index(l)) for l in labels))
PY
)" || { echo "cannot map $MANIFEST onto $SEED_LABELS" >&2; exit 1; }

mkdir -p "$OUT_ROOT/$ARM"
echo "========================================"
echo "$(date '+%F %T')  starting $ARM (arms: ${ARMS:-default 4 x 4}; seed episodes: $SEED_EPISODES)"
echo "========================================"
INFLUENCE=1 INFLUENCE_MEASURE=prune_kl PRUNE_KL_ARMS="$ARMS" PRUNE_KL_SEED_EPISODES="$SEED_EPISODES" \
  LABELS="$MANIFEST" RUN_DIR="$OUT_ROOT/$ARM" \
  bash baselines/run_lightnav.sh full "$@" 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "$(date '+%F %T')  $ARM FAILED (exit $STATUS); see $OUT_ROOT/$ARM/launch.log"
  exit "$STATUS"
fi
echo "$(date '+%F %T')  finished $ARM"
echo "Next: python paper/prune_kl/aggregate.py"
