#!/usr/bin/env bash
# Serial chain for the influence experiment: Uni-NaVid (~30 min) then LongNav-R1 (~4 h), one GPU.
#
#   nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &
#   MEASURE=keep_one nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &   # leave-one-in variant
#
# Everything is appended to paper_results/frame_influence[_keepone]/runs/driver.log; both arm
# scripts skip finished arms and resume partial ones, so re-running the chain after an
# interruption is safe. MEASURE=keep_one runs uninavid_influence_keepone and r1_influence_keepone.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MEASURE="${MEASURE:-leave_one}"
case "$MEASURE" in
  leave_one) LOG="$ROOT/paper_results/frame_influence/runs/driver.log";         R1_ARM=r1_influence ;;
  keep_one)  LOG="$ROOT/paper_results/frame_influence_keepone/runs/driver.log"; R1_ARM=r1_influence_keepone ;;
  *) echo "Unknown MEASURE '$MEASURE' (expected leave_one or keep_one)" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$LOG")"
{
  echo "$(date '+%F %T')  chain start (measure $MEASURE)"
  MEASURE="$MEASURE" bash paper/frame_influence/run_uninavid.sh; echo "$(date '+%F %T')  uninavid exit $?"
  ARM="$R1_ARM" bash paper/frame_influence/run_r1.sh; echo "$(date '+%F %T')  r1 exit $?"
  echo "$(date '+%F %T')  chain done"
} >> "$LOG" 2>&1
