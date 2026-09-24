#!/usr/bin/env bash
# Serial chain for the pruning experiment on one GPU: the two identity arms first, the
# identity check against the ctx-ablation references, then the 40 pruning arms
# (LongNav-R1 ~10 h, Uni-NaVid ~4 h). ORDER=uninavid_first swaps the families.
#
#   nohup bash paper/token_prune/run_all.sh > /dev/null 2>&1 &
#
# Everything is appended to paper_results/token_prune/runs/driver.log; both arm scripts skip
# finished arms and resume partial ones, so re-running the chain after an interruption is safe.
# The chain stops if an identity arm does not reproduce its reference (aggregate.py
# --check-identity); pass IDENTITY_TOLERANCE=1 to accept Uni-NaVid's sampling noise.
#
# VARIANT=flat chains the flat variant (34 arms, log under paper_results/token_prune_flat/runs).
# Its identity check compares r1_identity with the ctx ablation's r1_full; Uni-NaVid's 64-frame
# identity arm has no reference run and is not gated.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/brabiei/miniconda3/envs/longnav_vlm/bin/python}"
export VARIANT="${VARIANT:-window}"
case "$VARIANT" in
  window) LOG="$ROOT/paper_results/token_prune/runs/driver.log" ;;
  flat)   LOG="$ROOT/paper_results/token_prune_flat/runs/driver.log" ;;
  *) echo "Unknown VARIANT '$VARIANT' (expected window or flat)" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$LOG")"
ORDER="${ORDER:-r1_first}"
CHECK=(--variant "$VARIANT" --check-identity)
[[ "${IDENTITY_TOLERANCE:-0}" == "1" ]] && CHECK+=(--identity-tolerance)

family() {  # <r1|uninavid> <arms...>
  local fam="$1"; shift
  ARMS="$*" bash "paper/token_prune/run_${fam}.sh"; echo "$(date '+%F %T')  ${fam} [$*] exit $?"
}

{
  echo "$(date '+%F %T')  chain start (variant=$VARIANT order=$ORDER)"
  if [[ "$ORDER" == "uninavid_first" ]]; then FAMS=(uninavid r1); else FAMS=(r1 uninavid); fi
  for fam in "${FAMS[@]}"; do family "$fam" "${fam}_identity"; done
  if ! "$PYTHON" paper/token_prune/aggregate.py "${CHECK[@]}"; then
    echo "$(date '+%F %T')  identity check FAILED; chain stopped"
    exit 1
  fi
  echo "$(date '+%F %T')  identity check passed"
  for fam in "${FAMS[@]}"; do bash "paper/token_prune/run_${fam}.sh"; echo "$(date '+%F %T')  ${fam} sweep exit $?"; done
  echo "$(date '+%F %T')  chain done"
} >> "$LOG" 2>&1
