#!/usr/bin/env bash
# Layer-wise visual-token information WITHOUT visual history, several scenes and timesteps
# (paper/layer_token_info/README.md, "No-visual-history study"):
#
#   conda activate longnav_vlm
#   bash paper/layer_token_info/run_nohist.sh
#   PROBES="q5QZSEeHe5g_0:10 q5QZSEeHe5g_0:18" bash paper/layer_token_info/run_nohist.sh
#
# PROBES is a list of <episode>:<decision> pairs (1-based decision; frame step_<decision-1>.png).
# Two measures per token and layer, both with every history frame's visual tokens hidden on
# layers >= i (the past actions stay): keep_all (keep-one-in) and leave_nohist (leave-one-out).
# Panel (b) always shows layers 1 / 12 / 18. Order: drive the episodes (skipped when the
# trajectories exist), the cheap leaky arm for every pair, then the leak-free arm cheapest
# (earliest decision) first; figures are redrawn after every probe. Probe files that exist are
# skipped, so the script resumes. Leak-free cost on the 5090, all 28 layers: ~25 min at
# decision 8, ~1 h at 16, ~1.5 h at 21, ~2 h at 26 (one full re-prefill per probe).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
VLN_PYTHON="${VLN_PYTHON:-/home/brabiei/miniconda3/envs/vln/bin/python}"
STUDY=nohist
RUNS="$ROOT/paper_results/layer_token_info/$STUDY/runs"
LOG="$RUNS/driver.log"
DRIVER="paper/layer_token_info/probe_episode.py"
# per scene the decision whose observation moves the action most (survey step scores) in an
# early (5-15) and a late (16-30) window
PROBES="${PROBES:-Nfvxx8J5NCo_1:7 h1zeeAwLh9Z_0:8 zt1RVoi7PcG_0:9 zt1RVoi7PcG_0:16 h1zeeAwLh9Z_0:21 Nfvxx8J5NCo_1:26}"
MAX_STEPS="${MAX_STEPS:-40}"
PANEL_LAYERS="${PANEL_LAYERS:-1 12 18}"
LAYER_ARGS=()
if [[ -n "${LAYERS:-}" ]]; then LAYER_ARGS=(--layers $LAYERS); fi
mkdir -p "$RUNS"

run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "$(date '+%F %T')  starting $label" | tee -a "$LOG"
  "$@" 2>&1 | grep --line-buffered -v "Warning" | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "$(date '+%F %T')  $label FAILED (exit $status); see $LOG" | tee -a "$LOG"
    exit "$status"
  fi
  echo "$(date '+%F %T')  finished $label" | tee -a "$LOG"
}

MISSING=()
for P in $PROBES; do
  EP="${P%%:*}"
  [[ -f "$RUNS/$EP/trajectory.json" ]] || MISSING+=("$EP")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  run "survey(${MISSING[*]})" "$PYTHON" "$DRIVER" --study "$STUDY" survey --max-steps "$MAX_STEPS" \
    --step-scores --episodes $(printf '%s\n' "${MISSING[@]}" | sort -u)
fi

# earliest decision first: the leak-free arm's cost grows with the context
SORTED=$(for P in $PROBES; do echo "${P##*:} ${P%%:*}"; done | sort -n | awk '{print $2":"$1}')
for ARM in mask recompute; do
  for P in $SORTED; do
    EP="${P%%:*}"; STEP="${P##*:}"
    if [[ -f "$RUNS/$EP/token_info_${ARM}_t$(printf '%03d' "$STEP").npz" ]]; then
      echo "Skipping $ARM $EP t=$STEP: probe file exists."
      continue
    fi
    run "probe:$ARM $EP t=$STEP" "$PYTHON" "$DRIVER" --study "$STUDY" probe --arm "$ARM" --episode "$EP" \
      --steps "$STEP" --measures keep_all leave_nohist "${LAYER_ARGS[@]}"
    run "plot $EP t=$STEP" "$VLN_PYTHON" paper/layer_token_info/plot.py --study "$STUDY" --episode "$EP" \
      --steps "$STEP" --layers $PANEL_LAYERS
  done
done
echo "Figures: $ROOT/paper_results/layer_token_info/$STUDY/figures/<episode>/t<step>_<arm>_<measure>.{pdf,png}"
