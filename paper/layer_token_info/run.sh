#!/usr/bin/env bash
# Layer-wise visual-token information figure (paper/layer_token_info/README.md), end to end:
#
#   conda activate longnav_vlm
#   bash paper/layer_token_info/run.sh              # survey -> probe recompute -> probe mask -> plot
#   EPISODE=VBzV5z6i1WS_1 bash paper/layer_token_info/run.sh   # skip the survey's pick
#   LAYERS="0 4 8 12 16 20 24 27" bash paper/layer_token_info/run.sh   # probe a subset of layers
#
# Steps whose output already exists are skipped; delete the file to redo it. Everything lands
# under paper_results/layer_token_info/ (runs/, figures/, results/), log in runs/driver.log.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
VLN_PYTHON="${VLN_PYTHON:-/home/brabiei/miniconda3/envs/vln/bin/python}"
OUT="$ROOT/paper_results/layer_token_info"
RUNS="$OUT/runs"
LOG="$RUNS/driver.log"
mkdir -p "$RUNS"
DRIVER="paper/layer_token_info/probe_episode.py"
LAYER_ARGS=()
if [[ -n "${LAYERS:-}" ]]; then LAYER_ARGS=(--layers $LAYERS); fi

run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "$(date '+%F %T')  starting $label" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "$(date '+%F %T')  $label FAILED (exit $status); see $LOG" | tee -a "$LOG"
    exit "$status"
  fi
  echo "$(date '+%F %T')  finished $label" | tee -a "$LOG"
}

if [[ -z "${EPISODE:-}" ]]; then
  if [[ -f "$RUNS/selected.json" ]]; then
    echo "Skipping survey: $RUNS/selected.json exists."
  else
    run survey "$PYTHON" "$DRIVER" survey
  fi
  EPISODE="$("$PYTHON" -c "import json;print(json.load(open('$RUNS/selected.json'))['episode_label'])")"
elif [[ ! -f "$RUNS/$EPISODE/trajectory.json" ]]; then
  run "survey($EPISODE)" "$PYTHON" "$DRIVER" survey --episodes "$EPISODE"
fi
echo "Episode: $EPISODE"

for ARM in recompute mask; do
  if [[ -f "$RUNS/$EPISODE/token_info_$ARM.npz" ]]; then
    echo "Skipping probe $ARM: $RUNS/$EPISODE/token_info_$ARM.npz exists."
  else
    run "probe:$ARM" "$PYTHON" "$DRIVER" probe --arm "$ARM" --episode "$EPISODE" "${LAYER_ARGS[@]}"
  fi
done

run plot "$VLN_PYTHON" paper/layer_token_info/plot.py --episode "$EPISODE"
if ls "$RUNS"/hm3d_v2_100_w32_vblind* >/dev/null 2>&1; then
  run horizon "$VLN_PYTHON" paper/layer_token_info/horizon.py
else
  echo "Skipping horizon figure: no visual-blind arms under $RUNS (see README)."
fi
echo "Figures: $OUT/figures/$EPISODE and $OUT/figures/information_horizon.pdf"
