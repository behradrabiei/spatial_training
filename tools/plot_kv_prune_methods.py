#!/usr/bin/env python3
"""KV-prune selectors vs random as the KV budget grows (kvprune_v2_full_* runs).

X axis: nominal KV budget (vlm.kv_budget). Points show mean SR / SPL over the shared
100-episode HM3D-v2 split with a 95% bootstrap CI over episodes.

Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_prune_methods.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_ROOT = Path("./dump/longnav_eval")
OUT = EVAL_ROOT / "kvprune_v2_prune_methods.png"
BUDGETS = (1187, 1799, 3022)
METHODS = (("random", "random slots", "#2a78d6", "o"),
           ("kl", "KL frame-proportional", "#eb6834", "s"),
           ("grid", "grid-stratified (within frame)", "#1baf7a", "^"),
           ("voxel_dedup_s4", "voxel dedup (0.6 m cells)", "#eda100", "D"),
           ("voxel_strat_s4", "voxel stratified (0.6 m cells)", "#e87ba4", "v"))
N_BOOT, SEED = 2000, 17


def load(run_name):
    rows = {}
    for path in sorted((EVAL_ROOT / run_name / "rollout").glob("results_*")):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["episode_label"], r)
    return rows


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    boots = rng.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    rng = np.random.default_rng(SEED)
    series = {}  # method -> list of (budget, kv_mean, {metric: (mean, lo, hi)}, n)
    for method, label, color, marker in METHODS:
        for budget in BUDGETS:
            run = f"kvprune_v2_full_{method}_w32_b{budget}"
            rows = load(run)
            if not rows:
                print(f"[skip] {run}")
                continue
            stats = {k: ci([r[k] for r in rows.values()], rng) for k in ("success", "spl")}
            kv = np.mean([r["sup/mean_kv_len"] for r in rows.values()])
            series.setdefault(method, []).append((budget, kv, stats, len(rows)))
            print(f"{run:40s} n={len(rows):3d} kv={kv:6.0f} "
                  f"SR={stats['success'][0]:.3f} [{stats['success'][1]:.3f},{stats['success'][2]:.3f}] "
                  f"SPL={stats['spl'][0]:.3f} [{stats['spl'][1]:.3f},{stats['spl'][2]:.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, key, name in ((axes[0], "success", "Success rate"), (axes[1], "spl", "SPL")):
        for method, label, color, marker in METHODS:
            pts = sorted(series.get(method, []))
            if not pts:
                continue
            x = [p[0] for p in pts]
            mean = [p[2][key][0] for p in pts]
            lo = [p[2][key][0] - p[2][key][1] for p in pts]
            hi = [p[2][key][2] - p[2][key][0] for p in pts]
            ax.errorbar(x, mean, yerr=[lo, hi], label=label, color=color, marker=marker,
                        markersize=7, linewidth=2, capsize=3, alpha=0.9)
            ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color="#444", va="center")
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("KV budget (slots)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    axes[1].set_xlim(BUDGETS[0] - 150, BUDGETS[-1] + 650)
    fig.suptitle("HM3D-v2 (100 ep), window 32: KV-prune selectors by KV budget "
                 "(95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
