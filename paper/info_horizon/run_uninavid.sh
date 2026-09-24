#!/usr/bin/env bash
# Uni-NaVid arms of the information-horizon experiment (paper/info_horizon/README.md).
#
#   bash paper/info_horizon/run_uninavid.sh                 # every arm, in order
#   ARMS="uninavid_ref uninavid_blind_l16" bash paper/info_horizon/run_uninavid.sh
#
# Arms: uninavid_ref (seeded, nothing hidden) | uninavid_blind_l<L> (every visual token hidden
# from decoder layer L on, 100 episodes) | uninavid_probe (layer-wise keep-one-in / leave-one-out
# KL of the current frame after every inference, the 10 seed-17 episodes). All arms use
# --sampling-seed 30 so the reference and the probe share trajectories with Exp 3's
# uninavid_influence. Finished arms are skipped; an unfinished arm resumes (--resume).
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
DATASET="${DATASET:-hm3d}"  # hm3d | mp3d (paper_results/info_horizon_mp3d, seed-17 manifests, MP3D scenes)
if [[ "$DATASET" == "mp3d" ]]; then
  OUT_ROOT="$ROOT/paper_results/info_horizon_mp3d/runs"
  MANIFEST_100="$ROOT/src/longnav/conf/episode_jsons/mp3d_100_seed17.json"
  MANIFEST_9="$ROOT/src/longnav/conf/episode_jsons/mp3d_10_seed17.json"
  PROBE_COUNT=10
  DATA_ARGS=(--config "$ROOT/baselines/uninavid_hm3d/config/objectnav_mp3d.yaml"
             --dataset /home/brabiei/vault/habitat_data/evaluation_episodes/MP3D/objectnav_mp3d_v1/val/val.json.gz
             --scenes /home/brabiei/vault/habitat_data/scenes)
else
  OUT_ROOT="$ROOT/paper_results/info_horizon/runs"
  MANIFEST_100="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
  MANIFEST_9="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
  PROBE_COUNT=10
  DATA_ARGS=()
fi
LAYERS="${LAYERS:-8 12 16 18 20 24 28}"  # L=0,4,22 skipped (user, 2026-09-16)
PROBE_LAYERS="${PROBE_LAYERS:-0,4,8,12,16,18,20,22,24,28}"  # the probe is cheap: keep the full grid
SAMPLING_SEED="${SAMPLING_SEED:-30}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS="uninavid_ref"
  for L in $LAYERS; do ARMS+=" uninavid_blind_l${L}"; done
  ARMS+=" uninavid_probe"
fi

for ARM in $ARMS; do
  EXTRA=(--sampling-seed "$SAMPLING_SEED")
  MANIFEST="$MANIFEST_100"
  COUNT=100
  if [[ "$ARM" =~ ^uninavid_blind_l([0-9]+)$ ]]; then
    EXTRA+=(--blind-layer "${BASH_REMATCH[1]}")
  elif [[ "$ARM" == "uninavid_probe" ]]; then
    EXTRA+=(--layer-probe --probe-layers "$PROBE_LAYERS")
    MANIFEST="$MANIFEST_9"
    COUNT=$PROBE_COUNT
  elif [[ "$ARM" != "uninavid_ref" ]]; then
    echo "Unknown arm '$ARM' (expected uninavid_ref, uninavid_blind_l<L> or uninavid_probe)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${EXTRA[*]}"
  echo "========================================"
  bash baselines/run.sh eval --resume \
    --labels "$MANIFEST" --expected-count "$COUNT" \
    --output "$OUT_ROOT/$ARM" \
    "${DATA_ARGS[@]}" "${EXTRA[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested Uni-NaVid arms done. Next: python paper/info_horizon/aggregate.py"
