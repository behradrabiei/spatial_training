#!/usr/bin/env bash
# Clean cost benchmark for paper/hybrid_prune/plot_cost.py: the unpruned model and the protocol at
# k = 16, for LongNav-R1 and Uni-NaVid, each arm ALONE on the GPU (the sweep ran two arms at a time,
# so its latencies are not comparable), on the 17 HM3D-v2 val episodes that run >= 250 steps under
# either LongNav-R1 arm (src/longnav/conf/episode_jsons/hm3d_v2_17_long.json). Output:
# paper_results/hybrid_prune/runs/cost_bench/cost_<arm>/. ~35 min on one RTX 5090.
#
#   conda activate longnav_vlm && bash paper/hybrid_prune/run_cost_bench.sh
#   MANIFEST=src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json BENCH=cost_bench_seed17 bash ...   # another split
#   MANIFEST=src/longnav/conf/episode_jsons/hm3d_v2_100.json BENCH=cost_bench_100 ARMS="uniform pyramiddrop" bash ...
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
MANIFEST="$ROOT/${MANIFEST:-src/longnav/conf/episode_jsons/hm3d_v2_17_long.json}"
OUT="$ROOT/paper_results/hybrid_prune/runs/${BENCH:-cost_bench}"
# wandb resume keys on (project, run name) and skips already-logged episode labels, so a second
# benchmark on another split needs its own run names
RUN_SUFFIX=""; [[ "${BENCH:-cost_bench}" != "cost_bench" ]] && RUN_SUFFIX="_${BENCH#cost_bench_}"
SKIP_UNINAVID="${SKIP_UNINAVID:-0}"; SKIP_R1="${SKIP_R1:-0}"  # rerun one model family only
N=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$MANIFEST")
mkdir -p "$OUT"
COMMON=(+checkpoint=longnav +dataset=hm3d_v2_val +resources=single task.output_dir="$OUT"
        task.wandb_project=longnav_scratch_cost task.episode_json="$MANIFEST")
# ARMS: which protocols to benchmark, each on both model families (SKIP_R1 / SKIP_UNINAVID narrow it).
#   full         the unpruned models (LongNav-R1 r1_full; Uni-NaVid's native evaluator)
#   hybrid       the hybrid protocol at k = 16
#   uniform      the uniform protocol at k = 16 (random draw at layer 0, visual-blind from 18 / 20)
#   pyramiddrop  no-window PyramidDrop at k = 16 (Uni-NaVid: 64 frames)
ARMS="${ARMS:-full hybrid}"
PRUNE_COMMON=(vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool vlm.kv_prune_budget_frames=16)
UNI_PRUNE=(--prune-k 16 --sampling-seed 30 --prune-budget-scope model --prune-schedule-anchor pool)
r1_arm() {  # <arm> <experiment> [overrides...]
  local arm="$1" exp="$2"; shift 2
  echo "$(date '+%F %T') start $arm"
  "$PY" -m longnav.scripts.eval +experiment="$exp" "${COMMON[@]}" task.run_name="$arm$RUN_SUFFIX" "$@" \
    > "$OUT/$arm.log" 2>&1 || echo "$(date '+%F %T') FAILED $arm (see $OUT/$arm.log)"
}
uni_arm() {  # <arm> [eval flags...]
  local arm="$1"; shift
  echo "$(date '+%F %T') start $arm"
  bash baselines/run.sh eval --labels "$MANIFEST" --expected-count "$N" --output "$OUT/$arm" "$@" \
    > "$OUT/$arm.log" 2>&1 || echo "$(date '+%F %T') FAILED $arm (see $OUT/$arm.log)"
}
for A in $ARMS; do
  case "$A" in
    full)
      [[ "$SKIP_R1" == "0" ]] && r1_arm cost_r1_full paper_ctx
      [[ "$SKIP_UNINAVID" == "0" ]] && uni_arm cost_uninavid_full --sampling-seed 30 ;;
    hybrid)
      [[ "$SKIP_R1" == "0" ]] && r1_arm cost_r1_hybrid_mp16 paper_hybrid 'vlm.kv_prune_stages=[0,7,14,20]' \
        'vlm.kv_prune_stage_importance=[attn,attn,attn,random]' vlm.kv_prune_tail_budget_frames=1 "${PRUNE_COMMON[@]}"
      [[ "$SKIP_UNINAVID" == "0" ]] && uni_arm cost_uninavid_hybrid_mp16 --context-frames 32 --prune-method hybrid "${UNI_PRUNE[@]}" ;;
    uniform)
      [[ "$SKIP_R1" == "0" ]] && r1_arm cost_r1_uniform_mp16 paper_hybrid 'vlm.kv_prune_stages=[0,18]' \
        'vlm.kv_prune_stage_importance=[random,random]' vlm.kv_prune_visual_blind=true "${PRUNE_COMMON[@]}"
      [[ "$SKIP_UNINAVID" == "0" ]] && uni_arm cost_uninavid_uniform_mp16 --context-frames 32 --prune-method uniform "${UNI_PRUNE[@]}" ;;
    pyramiddrop)
      [[ "$SKIP_R1" == "0" ]] && r1_arm cost_r1_pyramiddrop_mp16 paper_hybrid vlm.context_frames=null vlm.context_cut=header \
        'vlm.kv_prune_stages=[0,7,14,21]' "${PRUNE_COMMON[@]}"
      [[ "$SKIP_UNINAVID" == "0" ]] && uni_arm cost_uninavid_pyramiddrop_mp16 --context-frames 64 --prune-method pyramiddrop "${UNI_PRUNE[@]}" ;;
    *) echo "unknown arm '$A' (expected full, hybrid, uniform, pyramiddrop)" >&2; exit 2 ;;
  esac
done
echo "$(date '+%F %T') done. Next: /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_cost.py --bench $OUT --name ${BENCH:-cost_bench}"
