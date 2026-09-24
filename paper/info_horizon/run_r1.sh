#!/usr/bin/env bash
# LongNav-R1 arms of the information-horizon experiment (paper/info_horizon/README.md).
#
#   bash paper/info_horizon/run_r1.sh                       # every arm, in order
#   ARMS="r1_probe_leaky r1_blind_l16" bash paper/info_horizon/run_r1.sh
#
# Arms: r1_blind_l<L> (every visual token hidden from decoder layer L on, 100 episodes;
# +experiment=paper_horizon) | r1_probe_leaky, r1_probe (layer-wise keep-one-in / leave-one-out
# KL of the current frame after every step on the 10 seed-17 episodes;
# +experiment=paper_horizon_probe, leaky = prune mode, leak-free = evict mode). Finished arms
# are skipped; an unfinished arm resumes through wandb (same task.run_name).
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
DATASET="${DATASET:-hm3d}"  # hm3d | mp3d (paper_results/info_horizon_mp3d, seed-17 manifests, +dataset=mp3d_val)
if [[ "$DATASET" == "mp3d" ]]; then
  OUT_ROOT="$ROOT/paper_results/info_horizon_mp3d/runs"
  MANIFEST_100="src/longnav/conf/episode_jsons/mp3d_100_seed17.json"
  MANIFEST_9="src/longnav/conf/episode_jsons/mp3d_10_seed17.json"
  HYDRA_DATASET=mp3d_val
else
  OUT_ROOT="$ROOT/paper_results/info_horizon/runs"  # must match task.output_dir in the presets
  MANIFEST_100="src/longnav/conf/episode_jsons/hm3d_v2_100.json"
  MANIFEST_9="src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
  HYDRA_DATASET=hm3d_v2_val
fi
LAYERS="${LAYERS:-8 12 14 16 18 20 24}"  # L=0,4 skipped: SR is 0 there (user, 2026-09-16)

if [[ -z "${ARMS:-}" ]]; then
  ARMS="r1_probe_leaky"
  [[ "$DATASET" == "mp3d" ]] && ARMS+=" r1_ref"  # HM3D reuses ctx_ablation/runs/r1_full
  for L in $LAYERS; do ARMS+=" r1_blind_l${L}"; done
  # r1_probe (leak-free re-prefill, ~3 h) is not part of the default set (user, 2026-09-16);
  # run it with ARMS=r1_probe if wanted.
fi

for ARM in $ARMS; do
  OVERRIDES=()
  MANIFEST="$MANIFEST_100"
  if [[ "$ARM" =~ ^r1_blind_l([0-9]+)$ ]]; then
    EXPERIMENT=paper_horizon
    OVERRIDES+=("vlm.kv_prune_layer_start=${BASH_REMATCH[1]}")
  elif [[ "$ARM" == "r1_probe" ]]; then
    EXPERIMENT=paper_horizon_probe
    MANIFEST="$MANIFEST_9"
  elif [[ "$ARM" == "r1_probe_leaky" ]]; then
    EXPERIMENT=paper_horizon_probe
    MANIFEST="$MANIFEST_9"
    OVERRIDES+=("vlm.context_window_mode=prune")
  elif [[ "$ARM" == "r1_ref" ]]; then
    EXPERIMENT=paper_ctx  # the unblinded full-context protocol of ctx_ablation/r1_full
  else
    echo "Unknown arm '$ARM' (expected r1_blind_l<L>, r1_probe or r1_probe_leaky)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${OVERRIDES[*]:-}"
  echo "========================================"
  # Under DATASET=mp3d the presets' HM3D manifest / output dir are overridden on the command line.
  [[ "$DATASET" == "mp3d" ]] && OVERRIDES+=("task.episode_json=$MANIFEST" "task.output_dir=./paper_results/info_horizon_mp3d/runs")
  "$PYTHON" -m longnav.scripts.eval \
    +checkpoint=longnav +dataset="$HYDRA_DATASET" +experiment="$EXPERIMENT" +resources="$RESOURCES" \
    task.run_name="$ARM" "${OVERRIDES[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested LongNav-R1 arms done. Next: python paper/info_horizon/aggregate.py"
