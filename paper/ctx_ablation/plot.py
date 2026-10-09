#!/usr/bin/env python3
"""Figures for the context-window ablation (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results/ctx_ablation/figures/:

  ctx_ablation.{pdf,png}         SR and SPL vs K, side by side, means only (CIs in summary.csv);
                                 large text, legend on top, shared x label
  ctx_leakage.{pdf,png}          LongNav-R1 KV eviction minus leak-free recompute at matched K, means only

Needs matplotlib, so run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/ctx_ablation/plot.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/ctx_ablation"
K_ORDER = ["1", "4", "8", "16", "32", "full"]
LARGE_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}  # ctx_ablation text sizes
PANELS = {"sr": "Success rate", "spl": "SPL", "steps": "Steps per episode"}
SERIES = {  # arm-name template, plot style (both LongNav-R1 variants share a color; recompute is dashed)
    "r1_evict": (lambda k: "r1_full" if k == "full" else f"r1_evict_k{k}",
                 dict(color="tab:red", marker="o", markersize=4, linestyle="-", label="LongNav-R1")),
    "r1_recompute": (lambda k: "r1_full" if k == "full" else f"r1_recompute_k{k}",
                     dict(color="tab:red", marker="o", markersize=4, linestyle="--",
                          label="LongNav-R1 (leak-free)")),
    "uninavid": (lambda k: "uninavid_full" if k == "full" else f"uninavid_k{k}",
                 dict(color="tab:blue", marker="o", markersize=4, linestyle="-", label="Uni-NaVid")),
}


def load_summary(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {r["arm"]: r for r in csv.DictReader(f)}


def draw(ax, summary, name, metric, prefix=""):
    """Mean line only (the 95% CIs are in summary.csv); keys are ``<prefix><metric>``."""
    arm_of, style = SERIES[name]
    xs, ys = [], []
    for i, k in enumerate(K_ORDER):
        r = summary.get(arm_of(k))
        if r is None or r.get(f"{prefix}{metric}", "") == "":
            continue
        xs.append(i)
        ys.append(float(r[f"{prefix}{metric}"]))
    if xs:
        ax.plot(xs, ys, **style)


def finish(ax, metric, n_ticks=len(K_ORDER)):
    ax.set_xticks(range(n_ticks))
    ax.set_xticklabels(K_ORDER[:n_ticks])
    ax.set_xlabel("Frames in context K")
    ax.set_title(PANELS[metric])
    ax.grid(True, alpha=0.3)
    if metric == "sr":
        ax.set_ylim(0, 1.0)
    elif metric == "spl":
        ax.set_ylim(0, 0.6)
    elif metric == "steps":
        ax.set_ylim(0, 520)


def figure_metrics(summary, metrics, figsize, out):
    """Paper figure: large text (LARGE_RC, as hybrid_prune's protocol_curves), one legend row
    across the top, one shared x label."""
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        for ax, metric in zip(axes, metrics):
            for name in SERIES:
                draw(ax, summary, name, metric)
            finish(ax, metric)
            ax.set_xlabel("")
        # One legend row above both panels: every corner of the panels has data in it.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0)
        fig.supxlabel("Frames in context K", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def figure_leakage(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, (metric, ylabel) in zip(axes, (("sr", "ΔSR"), ("spl", "ΔSPL"))):
        # leak_* columns live on the evict rows; reuse the evict template and style.
        draw(ax, summary, "r1_evict", metric, prefix="leak_")
        ax.axhline(0, color="k", linewidth=0.8)
        finish(ax, metric, n_ticks=len(K_ORDER) - 1)
        ax.set_ylim(None, None)
        ax.autoscale(axis="y")
        ax.set_title(f"Eviction leakage, {ylabel}")
        ax.set_ylabel(f"{ylabel} (evict − recompute)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=RESULTS_ROOT / "results/summary.csv")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    figure_metrics(summary, ["sr", "spl"], (7.0, 3.4), args.out / "ctx_ablation")
    figure_leakage(summary, args.out / "ctx_leakage")


if __name__ == "__main__":
    main()
