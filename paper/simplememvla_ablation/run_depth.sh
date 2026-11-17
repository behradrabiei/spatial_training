#!/usr/bin/env bash
# Layer-depth ablation of SimpleMemVLA (default: Permanence): arms blind_l<L>, deep to shallow.
# The reference (nothing hidden, L = 32) is the `full` arm of run_ctx.sh. Same driver, same
# skip / resume / STOP rules.
#
#   bash paper/simplememvla_ablation/run_depth.sh
#   ARMS="blind_l4 blind_l0" bash paper/simplememvla_ablation/run_depth.sh
ARMS="${ARMS:-blind_l28 blind_l24 blind_l20 blind_l16 blind_l12 blind_l8}" \
  exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ctx.sh"
