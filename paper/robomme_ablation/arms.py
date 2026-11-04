#!/usr/bin/env python3
"""Arm table for the RoboMME memory ablations on perceptual-framesamp-modul (Counting suite).

Each arm is one server setting (an ablation JSON passed to serve_policy.py --ablation, see
robomme_policy_learning/src/mme_vla_suite/ablation.py) evaluated on the 4 Counting tasks x 50
test episodes with model seed 7. Families mirror the navigation studies:

  ctx      K memory frames incl. the current one: recency_k{K} = newest K decision-time frames,
           uniform_k{K} = K evenly spaced frames over the episode (old frames kept); k1 = current
           frame only; full = the native 32-frame sampler (shared reference of every family);
           recency_k{K}_keep / k1_keep = the navigation "old blocks kept" control: the native 32
           frames with their own timestamps, the content (image + state) of every frame older
           than the recency window's oldest step replaced by that step's content;
           horizon_w{W} = the native 32-frame sampler restricted to the newest W env steps (a
           window in time at the model's own frame density; added 2026-11-03 after the recency
           arms turned out to be dominated by their stride-16 frame grid); *_fill = the same
           frames duplicated cyclically to the 32 slots (count control, added after uniform_k16
           also collapsed: the model reads the slot count as elapsed time)
  horizon  blind_l{L}: from layer L on, no query reads a visual token (current image tokens
           masked, memory modulation zeroed); 18 layers, none = full
  prune    {random,divprune,dart}_b{k}: 16*k history memory tokens kept by a sticky per-episode
           ledger, current frame's 16 tokens protected; b0 = k1; recency = the ctx recency arms

    python paper/robomme_ablation/arms.py list            # name<TAB>family<TAB>json (for run_all.sh)
    python paper/robomme_ablation/arms.py json recency_k4
"""
from __future__ import annotations

import json
import sys

SEED = 7
TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]
K_CTX = [32, 16, 8, 4]
K_UNIFORM = [16, 8, 4]
W_HORIZON = [512, 256, 128, 64, 32, 16]  # env steps; recency_k{K} spans 16*K steps
K_KEEP = [32, 16, 8, 4]
K_FILL = [16, 8, 4, 1]  # uniform_k{K} with the K frames duplicated to the 32 slots; k1_fill = current frame x32
BLIND_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 17]  # 17 added 2026-11-04: blinds the last layer only
PRUNE_METHODS = ["random", "divprune", "dart"]
PRUNE_BUDGETS = [16, 8, 4, 2, 1]


def _arms() -> dict[str, dict]:
    arms: dict[str, dict] = {"full": {"family": "ctx", "x": "full", "spec": None}}
    for k in K_CTX:
        arms[f"recency_k{k}"] = {"family": "ctx", "x": k, "spec": {"window": {"mode": "recency", "k": k}}}
    for k in K_UNIFORM:
        arms[f"uniform_k{k}"] = {"family": "ctx", "x": k, "spec": {"window": {"mode": "uniform", "k": k}}}
    arms["k1"] = {"family": "ctx", "x": 1, "spec": {"window": {"mode": "recency", "k": 1}}}
    for k in K_KEEP:
        arms[f"recency_k{k}_keep"] = {"family": "ctx", "x": k,
                                      "spec": {"window": {"mode": "recency", "k": k, "keep": True}}}
    arms["k1_keep"] = {"family": "ctx", "x": 1, "spec": {"window": {"mode": "recency", "k": 1, "keep": True}}}
    for w in W_HORIZON:
        arms[f"horizon_w{w}"] = {"family": "ctx", "x": w, "spec": {"window": {"mode": "horizon", "k": w}}}
    for k in K_FILL:
        arms[f"uniform_k{k}_fill"] = {"family": "ctx", "x": k,
                                      "spec": {"window": {"mode": "uniform", "k": k, "fill": True}}}
    for layer in BLIND_LAYERS:
        arms[f"blind_l{layer}"] = {"family": "horizon", "x": layer,
                                   "spec": {"blind_from_layer": layer, "blind_scope": "visual"}}
    for m in PRUNE_METHODS:
        for b in PRUNE_BUDGETS:
            arms[f"{m}_b{b}"] = {"family": "prune", "x": b,
                                 "spec": {"prune": {"method": m, "k": b}, "seed": SEED}}
    return arms


ARMS = _arms()
# run order: the shared reference first, then the cheap frame-level arms, then layers, then pruning
RUN_ORDER = (["full"] + [f"recency_k{k}" for k in K_CTX] + ["k1"] + [f"uniform_k{k}" for k in K_UNIFORM]
             + [f"blind_l{l}" for l in BLIND_LAYERS]
             + [f"{m}_b{b}" for b in PRUNE_BUDGETS for m in PRUNE_METHODS]
             # second pass (appended after the first launch), user order: keep, horizon, fill
             + [f"recency_k{k}_keep" for k in K_KEEP] + ["k1_keep"]
             + [f"horizon_w{w}" for w in W_HORIZON]
             + [f"uniform_k{k}_fill" for k in K_FILL])
assert set(RUN_ORDER) == set(ARMS)


def spec_json(name: str) -> str:
    spec = ARMS[name]["spec"]
    return "" if spec is None else json.dumps(spec, separators=(",", ":"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for name in RUN_ORDER:
            print(f"{name}\t{ARMS[name]['family']}\t{spec_json(name)}")
    elif cmd == "json":
        print(spec_json(sys.argv[2]))
    else:
        sys.exit(f"unknown command {cmd}")
