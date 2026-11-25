#!/usr/bin/env python
"""Figures for the SimpleMemVLA context-window and layer-depth ablations (needs matplotlib: the `vln` env).

    /home/brabiei/miniconda3/envs/vln/bin/python paper/simplememvla_ablation/plot.py [--suite permanence] [--suffix _partial]

Reads  paper_results/simplememvla_ablation/<suite>/results/summary.csv (aggregate.py)
Writes .../<suite>/figures/ctx_ablation{suffix}.{pdf,png}   SR | steps vs K, means only (CIs in summary.csv)
       .../<suite>/figures/ctx_by_task{suffix}.{pdf,png}    SR | steps vs K, one line per task
       .../<suite>/figures/information_perf{suffix}.{pdf,png}   SR | steps vs first hidden layer L, means only
       .../<suite>/figures/depth_by_task{suffix}.{pdf,png}      SR | steps vs L, one line per task
Only complete arms are drawn unless --allow-partial is given; the x axis lists the K values that
have an arm (K = 1 was not run on Permanence).

All four figures use the paper's two-column format (paper/figstyle.py, as
paper/ctx_ablation/plot.py): one column wide, 8 pt text, metric on the y axis, legend on top.
Include them at \\columnwidth without scaling.

The second panel is the mean number of steps per episode over all episodes of the arm (failures
and step-cap timeouts included), in the place SPL has in the navigation figures.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, COLUMN_WIDTH, top_legend  # noqa: E402  (two-column paper format)
RESULTS_ROOT = HERE.parents[1] / "paper_results/simplememvla_ablation"
K_ORDER = ["1", "4", "8", "16", "32", "full"]
LABEL, COLOR = "SimpleMemVLA", "tab:orange"
# One column of a two-column paper, drawn at final size (paper/figstyle.py).
COLUMN_FIGSIZE = (COLUMN_WIDTH, 1.5)  # SR | steps
BY_TASK_FIGSIZE = (COLUMN_WIDTH, 1.7)  # SR | steps per task: taller for the two-row legend
K_LABEL = "Frames in context K"
COLUMN_YLABEL = {"sr": "Success rate", "steps": "Episode steps"}  # must fit the height of a 1.5 in panel
TASK_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


def load_summary(suite: str, allow_partial: bool) -> dict[str, dict]:
    f = RESULTS_ROOT / suite / "results/summary.csv"
    rows = {}
    for r in csv.DictReader(open(f)):
        if allow_partial or int(r["n"]) >= int(r["n_expected"]):
            rows[r["arm"]] = r
    return rows


def arm_of(k: str) -> str:
    return "full" if k == "full" else f"k{k}"


N_LAYERS = 32
DEPTH_LABEL = f"Visual tokens hidden from layer L (of {N_LAYERS})"


def depth_axis(summary) -> list[tuple[str, str]]:
    """(tick label, arm) for the blind arms present, shallow to deep, then the unblinded model at
    L = N_LAYERS (hidden from layer 32 of 32 = nothing hidden; a numeric tick, as "none" does not
    fit next to "28" in a half-column panel)."""
    layers = sorted(int(a[len("blind_l"):]) for a in summary if a.startswith("blind_l"))
    axis = [(str(L), f"blind_l{L}") for L in layers]
    if layers and "full" in summary:
        axis.append((str(N_LAYERS), "full"))
    return axis


def k_axis(summary) -> list[str]:
    return [k for k in K_ORDER if arm_of(k) in summary]


def series(summary, arms, metric):
    """(x slots, means) of ``metric`` over the arms of an x axis; the 95% CIs stay in summary.csv."""
    xs, ys = [], []
    for i, arm in enumerate(arms):
        r = summary.get(arm)
        if r and r.get(metric, "") != "":
            xs.append(i); ys.append(float(r[metric]))
    return xs, ys


def save(fig, out_dir, name, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}{suffix}.{ext}", dpi=300)
    print("wrote", out_dir / f"{name}{suffix}.png")
    plt.close(fig)


def finish(ax, ticks, metric):
    """Categorical x ticks, metric name on the y axis (no title row), fixed SR ticks so that the
    layout pass and the final draw agree on the label widths."""
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks)
    ax.set_ylabel(COLUMN_YLABEL[metric], labelpad=2)
    ax.grid(True, alpha=0.3)
    if metric == "sr":
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    else:  # headroom above the highest point, top on a round number
        top = max((max(line.get_ydata(), default=0) for line in ax.get_lines()), default=0)
        ax.set_ylim(0, 100 * -(-top * 1.1 // 100) or None)


def finish_column(fig, axes, xlabel, out_dir, name, suffix):
    """Legend across the top, one shared x label; saved inside the rc context (pdf.fonttype)."""
    handles, labels = axes[0].get_legend_handles_labels()
    legend_height = top_legend(fig, handles, labels)
    fig.supxlabel(xlabel, fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 1 - legend_height))
    save(fig, out_dir, name, suffix)


def figure_pair(points, ticks, xlabel, out_dir, name, suffix):
    """SR | mean steps for the whole suite. ``points(metric)`` returns (x slots, means)."""
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(1, 2, figsize=COLUMN_FIGSIZE)
        for ax, metric in zip(axes, ("sr", "steps")):
            ax.plot(*points(metric), color=COLOR, marker="o", markersize=3, linestyle="-", label=LABEL)
            finish(ax, ticks, metric)
        finish_column(fig, axes, xlabel, out_dir, name, suffix)


def figure_by_task(summary, points, ticks, xlabel, out_dir, name, suffix):
    """SR | mean steps per task, one line each. ``points(metric)`` returns (x slots, means)."""
    tasks = [c[3:] for c in next(iter(summary.values())) if c.startswith("sr_") and c not in ("sr_lo", "sr_hi")]
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(1, 2, figsize=BY_TASK_FIGSIZE)
        for ax, metric in zip(axes, ("sr", "steps")):
            for task, color in zip(tasks, TASK_COLORS):
                ax.plot(*points(f"{metric}_{task}"), color=color, marker="o", markersize=3, linestyle="-", label=task)
            finish(ax, ticks, metric)
        finish_column(fig, axes, xlabel, out_dir, name, suffix)


def fig_ctx(summary, out_dir, suffix):
    ticks = k_axis(summary)
    points = lambda metric: series(summary, [arm_of(k) for k in ticks], metric)  # noqa: E731
    figure_pair(points, ticks, K_LABEL, out_dir, "ctx_ablation", suffix)
    figure_by_task(summary, points, ticks, K_LABEL, out_dir, "ctx_by_task", suffix)


def fig_depth(summary, out_dir, suffix):
    axis = depth_axis(summary)
    if not axis:
        return
    ticks = [t for t, _ in axis]
    points = lambda metric: series(summary, [a for _, a in axis], metric)  # noqa: E731
    figure_pair(points, ticks, DEPTH_LABEL, out_dir, "information_perf", suffix)
    figure_by_task(summary, points, ticks, DEPTH_LABEL, out_dir, "depth_by_task", suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="permanence")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--allow-partial", action="store_true", help="also draw arms that are still running")
    args = ap.parse_args()
    summary = load_summary(args.suite, args.allow_partial)
    if not summary:
        print("no complete arms yet"); return
    out_dir = RESULTS_ROOT / args.suite / "figures"
    if any(a.startswith("k") for a in summary):
        fig_ctx(summary, out_dir, args.suffix)
    fig_depth(summary, out_dir, args.suffix)


if __name__ == "__main__":
    main()
