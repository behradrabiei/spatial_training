#!/usr/bin/env bash
# LightNav-0 arms of the information-horizon experiment (paper/info_horizon/README.md).
#
#   bash paper/info_horizon/run_lightnav.sh                      # lightnav_ref then every blind arm
#   ARMS="lightnav_blind_l16" bash paper/info_horizon/run_lightnav.sh
#   LAYERS="8 16" bash paper/info_horizon/run_lightnav.sh
#   ARMS="lightnav_probe" bash paper/info_horizon/run_lightnav.sh   # blind-from-layer KL probe (not in the default set)
#
# Arms: lightnav_ref (BLIND_LAYER=36: nothing hidden, same eager masked-attention path as the
# blind arms -- the matched reference) | lightnav_blind_l<L> (every <|video_pad|> column hidden
# from decoder layer L on, 100 episodes). Each arm is `RUN_DIR=... BLIND_LAYER=L bash
# baselines/run_lightnav.sh full`: the comparable 1.0 m protocol of the context-window ablation,
# the wrapper baselines/lightnav_hm3d/eval_blind.py, upstream clone untouched. HM3D only (the
# LightNav harness has no MP3D subset).
# lightnav_probe (2026-11-23) is the KL arm: normal unblinded navigation on the 10 seed-17 episodes
# and, after every decision, KL(P_full || P_every visual token hidden from layer L on) for each L in
# PROBE_LAYERS (default every second layer) -- baselines/lightnav_hm3d/eval_influence.py
# --measure blind_layer, records in influence/<label>.jsonl.
#
# LightNav cannot resume: its results.jsonl is append-only and upstream has no episode filter, so
# a complete arm is skipped and a partial one must be moved away before it is rerun.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
OUT_ROOT="$ROOT/paper_results/info_horizon/runs"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
MANIFEST_PROBE="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"  # the KL-probe episodes of the other models
PROBE_LAYERS="${PROBE_LAYERS:-}"  # lightnav_probe: space-separated layers; empty = every second layer (0 2 ... 34)
N_LAYERS=36  # LightNav-0 (Qwen3-VL-4B) text decoder depth; BLIND_LAYER=36 hides nothing
LAYERS="${LAYERS:-12 16 18 20 22 24 26 28 30}"  # user, 2026-10-16; 22/26 added 2026-10-18, 28/30 2026-11-23

if [[ -z "${ARMS:-}" ]]; then
  ARMS="lightnav_ref"
  for L in $LAYERS; do ARMS+=" lightnav_blind_l${L}"; done
fi

complete() { "$PYTHON" -m longnav.analysis.results_io --complete "$1" --manifest "$2" >/dev/null 2>&1; }

for ARM in $ARMS; do
  ARM_MANIFEST="$MANIFEST"
  if [[ "$ARM" == "lightnav_ref" ]]; then
    L=$N_LAYERS
  elif [[ "$ARM" == "lightnav_probe" ]]; then
    L=probe
    ARM_MANIFEST="$MANIFEST_PROBE"
  elif [[ "$ARM" =~ ^lightnav_blind_l([0-9]+)$ ]]; then
    L="${BASH_REMATCH[1]}"
    (( L >= 0 && L < N_LAYERS )) || { echo "blind layer $L outside 0..$((N_LAYERS - 1))" >&2; exit 2; }
  else
    echo "Unknown arm '$ARM' (expected lightnav_ref, lightnav_blind_l<L> or lightnav_probe)" >&2
    exit 2
  fi
  if [[ -d "$OUT_ROOT/$ARM" ]] && complete "$OUT_ROOT/$ARM" "$ARM_MANIFEST"; then
    echo "Skipping $ARM: complete."
    continue
  fi
  if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
    echo "$OUT_ROOT/$ARM is partial; LightNav cannot resume (append-only results.jsonl), move it away first" >&2
    exit 1
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  (BLIND_LAYER=$L)"
  echo "========================================"
  if [[ "$ARM" == "lightnav_probe" ]]; then
    RUN_DIR="$OUT_ROOT/$ARM" RUN_NAME="$ARM" LABELS="$MANIFEST_PROBE" \
      INFLUENCE=1 INFLUENCE_MEASURE=blind_layer PROBE_LAYERS="$PROBE_LAYERS" \
      bash baselines/run_lightnav.sh full 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  else
    RUN_DIR="$OUT_ROOT/$ARM" BLIND_LAYER="$L" RUN_NAME="$ARM" \
      bash baselines/run_lightnav.sh full 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  fi
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested LightNav-0 arms done. Next: python paper/info_horizon/aggregate.py"
