#!/usr/bin/env python3
"""Figure of the pruning-KL experiment (paper/prune_kl/README.md).

Reads ``paper_results/prune_kl/results/prune_kl_curve.csv`` (paper/prune_kl/aggregate.py) and
writes ``figures/prune_kl.{pdf,png}``: mean KL(full context || pruned context) against the frame
budget, one line per pruning method (the colours of the flat token-pruning figure), one panel per
model with its own y axis -- the curve heights are not comparable across models (4 actions on a
leaky replay vs 256 codes on a full re-prefill). Single-column format (paper/figstyle.py).

    /home/brabiei/miniconda3/envs/vln/bin/python paper/prune_kl/plot.py

--value picks the plotted statistic: kl_mean (default: pooled mean over the decisions where the
budget binds), kl_balanced (episode-balanced mean), kl_norm_balanced (each episode's mean over its
no-visual KL), kl_exec_mean (executed-action classes), flip_exec_mean, kl_all_mean (every
decision); --name sets the basename, so another view never overwrites the paper figure.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, COLUMN_WIDTH, top_legend  # noqa: E402  (two-column paper format)

RESULTS_ROOT = ROOT / "paper_results/prune_kl"
MODELS = ("LongNav-R1", "LightNav-0")
METHODS = ("random", "fastv", "divprune", "dart")
BUDGETS = (4, 8, 16, 32)
FIGSIZE = (COLUMN_WIDTH, 1.7)
YLABELS = {"kl_mean": "KL (nats)", "kl_balanced": "KL (nats)", "kl_median": "KL (nats)", "kl_all_mean": "KL (nats)",
           "kl_exec_mean": "KL (nats)", "kl_norm_balanced": "Normalized KL", "flip_mean": "Action changed",
           "flip_exec_mean": "Action changed"}


def method_styles() -> dict:
    """The method colours and labels of the flat token-pruning figure (paper/token_prune/plot.py)."""
    spec = importlib.util.spec_from_file_location("token_prune_plot", ROOT / "paper/token_prune/plot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.METHODS


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def figure(curve: list[dict], out: Path, value: str, models=MODELS) -> None:
    styles = method_styles()
    models = tuple(m for m in models if any(r["model"] == m for r in curve)) or models
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(1, len(models), figsize=FIGSIZE, squeeze=False)
        for ax, model in zip(axes[0], models):
            for method in METHODS:
                rows = sorted((r for r in curve if r["model"] == model and r["method"] == method and fl(r[value]) == fl(r[value])),
                              key=lambda r: int(r["k"]))
                if not rows:
                    print(f"[{out.name}] {model}: no data for {method}")
                    continue
                ax.plot([int(r["k"]) for r in rows], [fl(r[value]) for r in rows], marker="o", markersize=3,
                        color=styles[method]["color"], label=styles[method]["label"])
            ax.set_title(model, pad=3)
            ax.set_xticks(list(BUDGETS))
            ax.set_xlim(2, 34)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel(YLABELS.get(value, value), labelpad=2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend_height = top_legend(fig, handles, labels)
        fig.supxlabel("History tokens kept (frames' worth)", fontsize=8)
        fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 1 - legend_height))
        for ext in ("pdf", "png"):  # inside the rc context: pdf.fonttype applies at save time
            fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_ROOT / "results")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "figures")
    ap.add_argument("--value", default="kl_mean", choices=sorted(YLABELS))
    ap.add_argument("--name", default="prune_kl")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    with (args.results / "prune_kl_curve.csv").open() as f:
        curve = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)
    figure(curve, args.out / args.name, args.value, tuple(args.models))


if __name__ == "__main__":
    main()
