#!/usr/bin/env bash
# Clean cost benchmark for paper/hybrid_prune/plot_cost.py: the unpruned model and the protocol at
# k = 16, for LongNav-R1 and Uni-NaVid, each arm ALONE on the GPU (the sweep ran two arms at a time,
# so its latencies are not comparable), on the 17 HM3D-v2 val episodes that run >= 250 steps under
# either LongNav-R1 arm (src/longnav/conf/episode_jsons/hm3d_v2_17_long.json). Output:
# paper_results/hybrid_prune/runs/cost_bench/cost_<arm>/. ~35 min on one RTX 5090.
#
#   conda activate longnav_vlm && bash paper/hybrid_prune/run_cost_bench.sh
#   MANIFEST=src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json BENCH=cost_bench_seed17 bash ...   # another split
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
if [[ "$SKIP_R1" == "0" ]]; then
echo "$(date '+%F %T') start cost_r1_full"
"$PY" -m longnav.scripts.eval +experiment=paper_ctx "${COMMON[@]}" task.run_name="cost_r1_full$RUN_SUFFIX" > "$OUT/cost_r1_full.log" 2>&1
echo "$(date '+%F %T') start cost_r1_hybrid_mp16"
"$PY" -m longnav.scripts.eval +experiment=paper_hybrid "${COMMON[@]}" task.run_name="cost_r1_hybrid_mp16$RUN_SUFFIX" \
  'vlm.kv_prune_stages=[0,7,14,20]' 'vlm.kv_prune_stage_importance=[attn,attn,attn,random]' vlm.kv_prune_tail_budget_frames=1 \
  vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool vlm.kv_prune_budget_frames=16 > "$OUT/cost_r1_hybrid_mp16.log" 2>&1
fi
if [[ "$SKIP_UNINAVID" == "0" ]]; then
echo "$(date '+%F %T') start cost_uninavid_full"
bash baselines/run.sh eval --labels "$MANIFEST" --expected-count "$N" --output "$OUT/cost_uninavid_full" \
  --sampling-seed 30 > "$OUT/cost_uninavid_full.log" 2>&1
echo "$(date '+%F %T') start cost_uninavid_hybrid_mp16"
bash baselines/run.sh eval --labels "$MANIFEST" --expected-count "$N" --output "$OUT/cost_uninavid_hybrid_mp16" \
  --context-frames 32 --prune-method hybrid --prune-k 16 --sampling-seed 30 --prune-budget-scope model \
  --prune-schedule-anchor pool > "$OUT/cost_uninavid_hybrid_mp16.log" 2>&1
fi
echo "$(date '+%F %T') done. Next: /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_cost.py --bench $OUT --name ${BENCH:-cost_bench}"
