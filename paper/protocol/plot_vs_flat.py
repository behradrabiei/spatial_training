#!/usr/bin/env python3
"""Comparison figure (not a paper figure): the flat token-prune methods (paper/token_prune, flat
variant) and the two protocols of this experiment on one budget axis.

The axis is shared because both experiments bound the SAME quantity: history visual tokens averaged
over all decoder layers, in frames' worth. The flat methods score and delete at layer 0, so every
layer holds k frames' worth; the protocols here spread k over depth (whole-model budget). What
differs between the curves is then the selection rule, the spread over depth and the window:

  flat methods    no window (LongNav-R1: whole episode; Uni-NaVid: 64 frames), uniform over depth
  pyramiddrop     no window, attention stages at 0/7/14/21 (Uni-NaVid 0/8/16/24)
  hybrid          K=32 window, attention stages + the random 1-frame tail above the horizon

Uni-NaVid caveat: the flat arms left the per-call sampler unseeded, this experiment seeds it
(--sampling-seed 30), so that panel carries the model's +-0.04 SR run-to-run floor between the two sets.
Means only, no CIs. Writes paper_results/hybrid_prune/figures/compare_flat.{pdf,png}; run in vln:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_vs_flat.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FLAT = ROOT / "paper_results/token_prune_flat/results/summary.csv"
HYBRID = ROOT / "paper_results/hybrid_prune/results/summary.csv"
OUT = ROOT / "paper_results/hybrid_prune/figures"
MODELS = ("LongNav-R1", "Uni-NaVid")
PANELS = {"sr": "Success rate", "spl": "SPL"}
# flat methods keep their colours from paper/token_prune; this experiment's protocols are drawn heavier
SERIES = {
    ("flat", "random"): dict(color="tab:blue", label="Random @0 (flat)", linestyle="--", linewidth=1, marker="o"),
    ("flat", "fastv"): dict(color="tab:orange", label="FastV @0 (flat)", linestyle="--", linewidth=1, marker="o"),
    ("flat", "divprune"): dict(color="tab:red", label="DivPrune @0 (flat)", linestyle="--", linewidth=1, marker="o"),
    ("flat", "dart"): dict(color="tab:purple", label="DART @0 (flat)", linestyle="--", linewidth=1, marker="o"),
    ("protocol", "pyramiddrop"): dict(color="tab:green", label="PyramidDrop (no window)", linestyle="-", linewidth=1.8, marker="s"),
    ("protocol", "hybrid"): dict(color="black", label="Hybrid (ours)", linestyle="-", linewidth=1.8, marker="s"),
}
REFERENCE = {"LongNav-R1": "r1_identity", "Uni-NaVid": "uninavid_identity"}  # unpruned arms of the flat run


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    flat, hybrid = rows(FLAT), rows(HYBRID)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True)
    for col, model in enumerate(MODELS):
        for row, (metric, title) in enumerate(PANELS.items()):
            ax = axes[row][col]
            ref = next((r for r in flat if r["arm"] == REFERENCE[model]), None)
            if ref is not None:
                ax.axhline(float(ref[metric]), color="gray", linestyle=":", linewidth=1, label="unpruned")
            for (source, name), style in SERIES.items():
                table, key = (flat, "method") if source == "flat" else (hybrid, "protocol")
                pts = sorted(((float(r["k"]), float(r[metric])) for r in table
                              if r["model"] == model and r[key] == name and r["k"] not in ("", None)))
                if pts:
                    ax.plot(*zip(*pts), markersize=4, **style)
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32])
            ax.set_xticklabels(["2", "4", "8", "16", "32"])
            ax.set_xlim(1.6, 40)
            ax.set_ylim(0, 1.02 if metric == "sr" else 0.6)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(model, fontsize=9)
            if col == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel("budget k (frames' worth, mean over layers)", fontsize=8)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"compare_flat.{ext}", dpi=300)
    print(f"wrote {OUT / 'compare_flat'}.pdf/.png")

    print(f"\n{'':<26}" + "".join(f"{'k=' + str(k):>14}" for k in (2, 4, 8, 16, 32)))
    for model in MODELS:
        for (source, name), style in SERIES.items():
            table, key = (flat, "method") if source == "flat" else (hybrid, "protocol")
            cells = {int(float(r["k"])): f"{float(r['sr']):.2f}/{float(r['spl']):.3f}" for r in table
                     if r["model"] == model and r[key] == name and r["k"] not in ("", None)}
            print(f"{model[:8]:<9}{style['label'][:17]:<17}" + "".join(f"{cells.get(k, '-'):>14}" for k in (2, 4, 8, 16, 32)))


if __name__ == "__main__":
    main()
