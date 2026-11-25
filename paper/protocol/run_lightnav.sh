#!/usr/bin/env bash
# LightNav-0 arms of the pruning-protocol experiment (paper/protocol/README.md, "LightNav-0 arms").
#
#   bash paper/protocol/run_lightnav.sh                                  # uniform, then pyramiddrop
#   ARMS="lightnav_uniform_mp8" bash paper/protocol/run_lightnav.sh
#   PROTOCOLS="uniform" K_LIST="8 4" bash paper/protocol/run_lightnav.sh
#
# Arms: lightnav_<protocol>_mp<k>, protocol in uniform|pyramiddrop (hybrid is not run on LightNav-0),
# k in 8 4 16 2: a whole-model budget of k x 14 history visual tokens averaged over the 36 decoder
# layers, schedule anchored at the pool ("mp", as the other models' arms).
#   uniform      newest 32 frames in scope (old blocks kept), one seeded random draw at layer 0,
#                visual-blind from layer 26 (LightNav-0's information horizon), current pair included
#   pyramiddrop  whole history, attention stages from layers 0 / 9 / 18 / 27
# Each arm is `RUN_DIR=... PROTOCOL=P PRUNE_K=k bash baselines/run_lightnav.sh full`: the comparable
# 1.0 m protocol, the wrapper baselines/lightnav_hm3d/eval_protocol.py, upstream clone untouched.
# HM3D only. LightNav must have the GPU to itself (env server + vLLM at 0.65 of the memory), so the
# arms run serially, family by family; run_worker.sh can drive this script (family "lightnav").
#
# LightNav cannot resume: its results.jsonl is append-only and upstream has no episode filter, so
# a complete arm is skipped and a partial one must be moved away before it is rerun.
set -uo pipefail  # no -e: a failed arm is reported and the loop moves on (re-run to retry it)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"  # only for the completeness check
OUT_ROOT="${OUT_ROOT:-$ROOT/paper_results/protocol/runs}"
MANIFEST="$ROOT/src/longnav/conf/episode_jsons/hm3d_v2_100.json"
K_LIST="${K_LIST:-8 4 16 2}"
PROTOCOLS="${PROTOCOLS:-uniform pyramiddrop}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS=""
  for P in $PROTOCOLS; do
    for K in $K_LIST; do ARMS+=" lightnav_${P}_mp${K}"; done
  done
fi

complete() { "$PYTHON" -m longnav.analysis.results_io --complete "$1" --manifest "$MANIFEST" >/dev/null 2>&1; }

for ARM in $ARMS; do
  if [[ -d "$OUT_ROOT/$ARM" ]] && complete "$OUT_ROOT/$ARM"; then
    echo "Skipping $ARM: complete."
    continue
  fi
  if [[ "$ARM" =~ ^lightnav_(uniform|pyramiddrop)_mp([0-9]+)$ ]]; then
    P="${BASH_REMATCH[1]}"; K="${BASH_REMATCH[2]}"
  else
    echo "Unknown arm '$ARM' (expected lightnav_<uniform|pyramiddrop>_mp<k>)" >&2
    exit 2
  fi
  if [[ -f "$OUT_ROOT/$ARM/raw/results.jsonl" ]]; then
    echo "$OUT_ROOT/$ARM is partial; LightNav cannot resume (append-only results.jsonl), move it away first" >&2
    exit 1
  fi

  mkdir -p "$OUT_ROOT/$ARM"
  echo "========================================"
  echo "$(date '+%F %T')  starting $ARM  (PROTOCOL=$P PRUNE_K=$K)"
  echo "========================================"
  RUN_DIR="$OUT_ROOT/$ARM" PROTOCOL="$P" PRUNE_K="$K" RUN_NAME="$ARM" \
    bash baselines/run_lightnav.sh full 2>&1 | tee -a "$OUT_ROOT/$ARM/launch.log"
  echo "$(date '+%F %T')  finished $ARM"
done

echo "All requested LightNav-0 arms done. Next: python paper/protocol/aggregate.py"
