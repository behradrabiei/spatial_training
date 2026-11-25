#!/usr/bin/env python3
"""Comparison figure (not a paper figure): the context-window ablation (paper/ctx_ablation) and the
two protocols of this experiment on one axis.

The shared axis is HISTORY visual tokens in frames' worth -- what both experiments ration:

  context window K   K frames visible incl. the current one, so K-1 history frames on every layer
  protocols, budget k  k frames' worth of history visual tokens (mean over layers); the current frame
                       is outside the budget and always seen in full

so a window arm sits at x = K-1 and a protocol arm at x = k. K=1 (no history) is x = 0; the axis is
symlog to hold it. The dashed gray line is the unpruned full-context model (r1_full / uninavid_full).
"Context window" is KV eviction on LongNav-R1 and the native frame window on Uni-NaVid.
What the axis does NOT equalise: the window also drops the old frames' action text, the protocols keep
all of it (compare kv_len_mean in the two summary.csv files for total resident KV). Means only, no CIs.
Writes paper_results/hybrid_prune/figures/compare_ctx.{pdf,png}; run in vln:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_vs_ctx.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CTX = ROOT / "paper_results/ctx_ablation/results/summary.csv"
HYBRID = ROOT / "paper_results/hybrid_prune/results/summary.csv"
OUT = ROOT / "paper_results/hybrid_prune/figures"
MODELS = ("LongNav-R1", "Uni-NaVid")
PANELS = {"sr": "Success rate", "spl": "SPL"}
FULL = {"LongNav-R1": "r1_full", "Uni-NaVid": "uninavid_full"}
# (source, key) -> style; key is the ctx summary's `mode` or this experiment's `protocol`
SERIES = {
    ("ctx", "evict"): dict(color="tab:blue", label="Context window", linestyle="-", linewidth=1, marker="o"),
    ("ctx", "recompute"): dict(color="tab:blue", label="Context window (leak-free recompute, LongNav-R1)", linestyle="--", linewidth=1, marker="o"),
    ("ctx", "native"): dict(color="tab:blue", label="Context window", linestyle="-", linewidth=1, marker="o"),  # Uni-NaVid
    ("protocol", "pyramiddrop"): dict(color="tab:green", label="PyramidDrop", linestyle="-", linewidth=1.8, marker="s"),
    ("protocol", "hybrid"): dict(color="tab:red", label="Protocol", linestyle="-", linewidth=1.8, marker="s"),
}
TICKS = [0, 1, 2, 4, 8, 16, 32]


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def points(table, source, model, name, metric):
    """(history frames' worth, metric) per arm: K-1 for a window arm, k for a protocol arm."""
    if source == "ctx":
        sel = [r for r in table if r["model"] == model and r["mode"] == name and r["K"] != "full"]
        return sorted((float(r["K"]) - 1, float(r[metric])) for r in sel)
    sel = [r for r in table if r["model"] == model and r["protocol"] == name]
    return sorted((float(r["k"]), float(r[metric])) for r in sel)


def main():
    ctx, hybrid = rows(CTX), rows(HYBRID)
    modes = {r["mode"] for r in ctx}
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True)
    drawn = {}
    for col, model in enumerate(MODELS):
        for row, (metric, title) in enumerate(PANELS.items()):
            ax = axes[row][col]
            ref = next((r for r in ctx if r["arm"] == FULL[model]), None)
            if ref is not None:
                h = ax.axhline(float(ref[metric]), color="gray", linestyle="--", linewidth=1, label="Full context")
                drawn.setdefault("Full context", h)
            for (source, name), style in SERIES.items():
                if source == "ctx" and name not in modes:
                    continue
                pts = points(ctx if source == "ctx" else hybrid, source, model, name, metric)
                if pts:
                    (h,) = ax.plot(*zip(*pts), markersize=4, **style)
                    drawn.setdefault(style["label"], h)
            ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.5)
            ax.set_xticks(TICKS)
            ax.set_xticklabels([str(t) for t in TICKS])
            ax.minorticks_off()
            ax.set_xlim(-0.3, 40)
            ax.set_ylim(0, 1.02 if metric == "sr" else 0.6)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(model, fontsize=9)
            if col == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel("history visual tokens (frames' worth)", fontsize=8)
    fig.legend(drawn.values(), drawn.keys(), loc="upper center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"compare_ctx.{ext}", dpi=300)
    print(f"wrote {OUT / 'compare_ctx'}.pdf/.png")


if __name__ == "__main__":
    main()
