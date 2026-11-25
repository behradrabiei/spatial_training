#!/usr/bin/env bash
# LongNav-R1 arms of the hybrid-pruning comparison (paper/hybrid_prune/README.md).
#
#   bash paper/hybrid_prune/run_r1.sh                                  # every arm
#   ARMS="r1_hybrid_mp8 r1_pyramiddrop_mp8" bash paper/hybrid_prune/run_r1.sh
#
# Arm names: r1_<protocol>_<mp|m|b><k>, protocol in {hybrid, pyramiddrop, uniform}, k = frames' worth of
# history visual tokens. uniform (mp only): K=32 frame cut, one uniform random draw at layer 0 (layers
# 0-17 hold the same slots) and visual-blind from layer 18 on (no visual token, the current frame's
# included, reaches layers 18-27).
#   mp  the experiment: WHOLE-MODEL budget (k bounds the history visual slots averaged over all 28
#       layers; a stage at layer 0 is added and the schedule's ratio is solved each step), schedule
#       anchored at what the cache holds now (kv_prune_schedule_anchor=pool)
#   m   whole-model budget anchored at the ORIGINAL token count of the scope -- aborted: without a
#       window the anchor grows all episode and PyramidDrop's deep stages starve
#   b   last-stage budget (the first, superseded run: k bounds the deepest scheduled stage only)
# Finished arms are skipped; a partial arm resumes through wandb (same task.run_name =>
# already-logged episodes are skipped). K order 8, 4, 16, 2: the smallest budget runs the longest
# episodes, so it goes last.
#
# DATASET=mp3d runs the MP3D replicate: +dataset=mp3d_val, the seed-17 100-episode manifest, data in
# paper_results/hybrid_prune_mp3d/ and its own wandb project (the arm names repeat, and wandb resume
# keys on (project, run name)).
#
# A pilot writes elsewhere: OUT_ROOT=<dir> MANIFEST=<labels.json> EXTRA_OVERRIDES="task.output_dir=<dir>
# task.wandb_project=<scratch project> task.episode_json=<labels.json>".
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to resume it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
RESOURCES="${RESOURCES:-single}"
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_OVERRIDES:-}"  # e.g. resources.osm_gb=10 when two arms share the machine
DATASET="${DATASET:-hm3d}"  # hm3d | mp3d
DATASET_OVERRIDES=()
if [[ "$DATASET" == "mp3d" ]]; then
  MANIFEST="${MANIFEST:-src/longnav/conf/episode_jsons/mp3d_100_seed17.json}"
  OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/hybrid_prune_mp3d/runs}"
  HYDRA_DATASET=mp3d_val
  # the preset pins the HM3D manifest / output dir / wandb project; a key given in EXTRA_OVERRIDES
  # (a pilot) is left to it
  for KV in "task.episode_json=$MANIFEST" "task.output_dir=./paper_results/hybrid_prune_mp3d/runs" \
            "task.wandb_project=longnav_paper_hybrid_mp3d"; do
    [[ " ${EXTRA_OVERRIDES[*]:-} " == *" ${KV%%=*}="* ]] || DATASET_OVERRIDES+=("$KV")
  done
else
  MANIFEST="${MANIFEST:-src/longnav/conf/episode_jsons/hm3d_v2_100.json}"
  OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/hybrid_prune/runs}"
  HYDRA_DATASET=hm3d_v2_val
fi
K_LIST="${K_LIST:-8 4 16 2}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS=""
  for K in $K_LIST; do ARMS+=" r1_hybrid_mp${K} r1_pyramiddrop_mp${K} r1_uniform_mp${K}"; done
fi

protocol_overrides() {  # <protocol> <mp|m|b> -> Hydra overrides on top of +experiment=paper_hybrid
  case "$1$2" in
    hybridmp)      echo "vlm.kv_prune_stages=[0,7,14,20] vlm.kv_prune_stage_importance=[attn,attn,attn,random] vlm.kv_prune_tail_budget_frames=1 vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool" ;;
    pyramiddropmp) echo "vlm.context_frames=null vlm.context_cut=header vlm.kv_prune_stages=[0,7,14,21] vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool" ;;
    uniformmp)     echo "vlm.kv_prune_stages=[0,18] vlm.kv_prune_stage_importance=[random,random] vlm.kv_prune_visual_blind=true vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool" ;;
    hybridm)      echo "vlm.kv_prune_stages=[0,7,14,20] vlm.kv_prune_stage_importance=[attn,attn,attn,random] vlm.kv_prune_tail_budget_frames=1 vlm.kv_prune_budget_scope=model" ;;
    pyramiddropm) echo "vlm.context_frames=null vlm.context_cut=header vlm.kv_prune_stages=[0,7,14,21] vlm.kv_prune_budget_scope=model" ;;
    hybridb)      echo "vlm.kv_prune_stages=[7,14,20] vlm.kv_prune_stage_importance=[attn,attn,random] vlm.kv_prune_tail_budget_frames=1" ;;
    pyramiddropb) echo "vlm.context_frames=null vlm.context_cut=header vlm.kv_prune_stages=[7,14,21]" ;;
    *) return 1 ;;
  esac
}

for ARM in $ARMS; do
  if [[ "$ARM" =~ ^r1_(hybrid|pyramiddrop|uniform)_(mp|m|b)([0-9]+)$ ]]; then
    read -r -a OVERRIDES <<< "$(protocol_overrides "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")"
    OVERRIDES+=("vlm.kv_prune_budget_frames=${BASH_REMATCH[3]}")
  else
    echo "Unknown arm '$ARM' (expected r1_<hybrid|pyramiddrop|uniform>_<mp|m|b><k>; uniform is mp only)" >&2
    exit 2
  fi

  if [[ -d "$OUT_ROOT/$ARM" ]] && "$PYTHON" -m longnav.analysis.results_io --complete "$OUT_ROOT/$ARM" --manifest "$MANIFEST" >/dev/null 2>&1; then
    echo "Skipping $ARM: complete."
    continue
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  ${OVERRIDES[*]}"
  echo "========================================"
  "$PYTHON" -m longnav.scripts.eval \
    +checkpoint=longnav +dataset="$HYDRA_DATASET" +experiment=paper_hybrid +resources="$RESOURCES" \
    task.run_name="$ARM" "${OVERRIDES[@]}" "${DATASET_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  STATUS=${PIPESTATUS[0]}
  if [[ $STATUS -ne 0 ]]; then
    echo "$(date '+%F %T')  FAILED $ARM (exit $STATUS); re-run this script to resume it"
  else
    echo "$(date '+%F %T')  finished $ARM"
  fi
done

echo "All requested LongNav-R1 arms done. Next: python paper/hybrid_prune/aggregate.py${DATASET:+ --dataset $DATASET}"
