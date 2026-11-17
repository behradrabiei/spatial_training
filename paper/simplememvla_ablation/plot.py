#!/usr/bin/env python
"""Figures for the SimpleMemVLA context-window and layer-depth ablations (needs matplotlib: the `vln` env).

    /home/brabiei/miniconda3/envs/vln/bin/python paper/simplememvla_ablation/plot.py [--suite permanence] [--suffix _partial]

Reads  paper_results/simplememvla_ablation/<suite>/results/summary.csv (aggregate.py)
Writes .../<suite>/figures/ctx_ablation{suffix}.{pdf,png}   SR | steps vs K, shaded 95% CI
       .../<suite>/figures/ctx_by_task{suffix}.{pdf,png}    SR vs K, one line per task
       .../<suite>/figures/information_perf{suffix}.{pdf,png}   SR | steps vs first hidden layer L
       .../<suite>/figures/depth_by_task{suffix}.{pdf,png}      SR vs L, one line per task
Only complete arms are drawn unless --allow-partial is given; the x axis lists the K values that
have an arm (K = 1 was not run on Permanence).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/simplememvla_ablation"
K_ORDER = ["1", "4", "8", "16", "32", "full"]
LABEL, COLOR = "SimpleMemVLA", "tab:orange"
LARGE_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}
PANELS = {"sr": "Success rate", "steps": "Steps per episode"}
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


def depth_axis(summary) -> list[tuple[str, str]]:
    """(tick label, arm) for the blind arms present, shallow to deep, then the unblinded model."""
    layers = sorted(int(a[len("blind_l"):]) for a in summary if a.startswith("blind_l"))
    axis = [(str(L), f"blind_l{L}") for L in layers]
    if layers and "full" in summary:
        axis.append(("none", "full"))
    return axis


def fig_depth(summary, out_dir, suffix):
    axis = depth_axis(summary)
    if not axis:
        return
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        for ax, metric in zip(axes, PANELS):
            xs = list(range(len(axis)))
            ys = [float(summary[a][metric]) for _, a in axis]
            los = [float(summary[a][f"{metric}_lo"]) for _, a in axis]
            his = [float(summary[a][f"{metric}_hi"]) for _, a in axis]
            ax.plot(xs, ys, color=COLOR, marker="o", markersize=4, linestyle="-", label=LABEL)
            ax.fill_between(xs, los, his, color=COLOR, alpha=0.2, linewidth=0)
            ax.set_xticks(xs); ax.set_xticklabels([t for t, _ in axis])
            ax.set_title(PANELS[metric]); ax.grid(True, alpha=0.3)
            ax.set_ylim(*((0, 1.0) if metric == "sr" else (0, None)))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=1, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel(f"Visual tokens hidden from layer L (of {N_LAYERS})", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_dir, "information_perf", suffix)

    tasks = [c[3:] for c in next(iter(summary.values())) if c.startswith("sr_") and c not in ("sr_lo", "sr_hi")]
    with plt.rc_context(LARGE_RC):
        fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))
        for task, color in zip(tasks, TASK_COLORS):
            pts = [(i, float(summary[a][f"sr_{task}"])) for i, (_, a) in enumerate(axis) if summary[a].get(f"sr_{task}", "") != ""]
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, marker="o", markersize=4, linestyle="-", label=task)
        ax.set_xticks(range(len(axis))); ax.set_xticklabels([t for t, _ in axis])
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.set_title("Success rate")
        ax.set_xlabel("Hidden from layer L")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        fig.tight_layout()
    save(fig, out_dir, "depth_by_task", suffix)


def k_axis(summary) -> list[str]:
    return [k for k in K_ORDER if arm_of(k) in summary]


def series(summary, metric):
    xs, ys, los, his = [], [], [], []
    for i, k in enumerate(k_axis(summary)):
        r = summary.get(arm_of(k))
        if r and r.get(metric, "") != "":
            xs.append(i); ys.append(float(r[metric]))
            los.append(float(r[f"{metric}_lo"])) if f"{metric}_lo" in r and r[f"{metric}_lo"] != "" else los.append(float(r[metric]))
            his.append(float(r[f"{metric}_hi"])) if f"{metric}_hi" in r and r[f"{metric}_hi"] != "" else his.append(float(r[metric]))
    return xs, ys, los, his


def save(fig, out_dir, name, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}{suffix}.{ext}", dpi=300)
    print("wrote", out_dir / f"{name}{suffix}.png")
    plt.close(fig)


def fig_ctx(summary, out_dir, suffix):
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        for ax, metric in zip(axes, PANELS):
            xs, ys, los, his = series(summary, metric)
            ax.plot(xs, ys, color=COLOR, marker="o", markersize=4, linestyle="-", label=LABEL)
            ax.fill_between(xs, los, his, color=COLOR, alpha=0.2, linewidth=0)
            ax.set_xticks(range(len(k_axis(summary)))); ax.set_xticklabels(k_axis(summary))
            ax.set_title(PANELS[metric]); ax.grid(True, alpha=0.3)
            ax.set_ylim(*((0, 1.0) if metric == "sr" else (0, None)))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=1, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("History frames K (incl. current)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_dir, "ctx_ablation", suffix)


def fig_by_task(summary, out_dir, suffix):
    tasks = [c[3:] for c in next(iter(summary.values())) if c.startswith("sr_") and c not in ("sr_lo", "sr_hi")]
    with plt.rc_context(LARGE_RC):
        fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))
        for task, color in zip(tasks, TASK_COLORS):
            xs, ys = [], []
            for i, k in enumerate(k_axis(summary)):
                r = summary.get(arm_of(k))
                if r and r.get(f"sr_{task}", "") != "":
                    xs.append(i); ys.append(float(r[f"sr_{task}"]))
            ax.plot(xs, ys, color=color, marker="o", markersize=4, linestyle="-", label=task)
        ax.set_xticks(range(len(k_axis(summary)))); ax.set_xticklabels(k_axis(summary))
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3); ax.set_title("Success rate")
        ax.set_xlabel("History frames K (incl. current)")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        fig.tight_layout()
    save(fig, out_dir, "ctx_by_task", suffix)


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
        fig_by_task(summary, out_dir, args.suffix)
    fig_depth(summary, out_dir, args.suffix)


if __name__ == "__main__":
    main()
