#!/usr/bin/env python3
"""Information-horizon figures (run aggregate.py first). Plain matplotlib defaults; writes four
figures to paper_results/info_horizon[_mp3d]/figures/, one per data source:
  information_kl.{pdf,png}    keep-one-in KL = KL(P_only_current_frame || P_no_visual) and
                              leave-one-out KL = KL(P_full || P_current_frame_hidden) of the
                              current frame, mean over every step of the 10 probe episodes
  information_perf.{pdf,png}  SR and SPL of the visual-blind rollouts, 100 episodes; dotted
                              line = unblinded reference
  information_kl_keep / information_kl_leave.{pdf,png}  the two KL panels as standalone figures
against the decoder layer L (visual tokens hidden from L on). Curves: LongNav-R1 red solid,
LongNav-R1 (leak-free re-prefill, optional arm) red dashed, Uni-NaVid blue solid; the SR/SPL
figure carries one LongNav-R1 curve (the blind rollout is leak-free by construction). Means
only; CIs are in results/horizon.csv. 1-based layer index on x (axis label $\ell_0$), each model's
"nothing hidden" point at its own depth L+1 (29 / 33, hollow marker; the last tick reads "L+1").
No horizon line unless --horizon is given.

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/info_horizon/plot.py [--dataset mp3d]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/info_horizon"
STYLE = {"LongNav-R1": dict(color="tab:red", marker="o", markersize=3, linestyle="-"),
         "LongNav-R1 (leak-free)": dict(color="tab:red", marker="o", markersize=3, linestyle="--"),
         "Uni-NaVid": dict(color="tab:blue", marker="o", markersize=3, linestyle="-")}
HORIZON_COLOR = "tab:purple"
FIGURES = {  # the KL panels come from the probe episodes, SR/SPL from the 100-episode blind runs
    "information_kl": (("keep_one_mean", "Keep-one-in KL (nats)"), ("leave_one_mean", "Leave-one-out KL (nats)")),
    "information_perf": (("sr", "SR"), ("spl", "SPL")),
    # the two KL panels again as standalone single-panel figures
    "information_kl_keep": (("keep_one_mean", "Keep-one-in KL (nats)"),),
    "information_kl_leave": (("leave_one_mean", "Leave-one-out KL (nats)"),),
}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def draw_panel(ax, rows, curves, key, title, horizon):
    for curve in curves:
        if key in ("sr", "spl") and curve == "LongNav-R1 (leak-free)":
            continue  # one SR/SPL curve per model
        rs = sorted((r for r in rows if r["curve"] == curve and r.get(key, "") != ""), key=lambda r: int(r["layer"]))
        if not rs:
            continue
        xs = [int(r["layer"]) + 1 for r in rs]  # 1-based layer index on the axis
        ys = [fl(r[key]) for r in rs]
        style = STYLE[curve]
        ax.plot(xs, ys, label=curve, **style)
        n = int(rs[0]["n_layers"])
        if xs[-1] == n + 1:  # the L+1 point (nothing hidden), hollow so the depth difference is visible
            ax.plot([xs[-1]], [ys[-1]], marker="o", markersize=4, markerfacecolor="white", color=style["color"],
                    linestyle="none")
            if key in ("sr", "spl"):
                ax.axhline(ys[-1], color=style["color"], linestyle=":", linewidth=1.6, alpha=0.85)
    if key == "sr":
        ax.set_ylim(0, 1)
    elif key == "spl":
        ax.set_ylim(0, 0.6)
    else:
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylim(bottom=-1e-4)
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.4)
    for h in horizon:
        ax.axvline(h, color=HORIZON_COLOR, linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ticks = [1, 8, 16, 24, 33]  # 1-based; the last tick is Uni-NaVid's L+1 (LongNav-R1's sits at 29)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "8", "16", "24", "L+1"])
    ax.get_xticklabels()[-1].set_horizontalalignment("left")
    ax.set_xlim(-0.5, 34.5)
    ax.grid(alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("hm3d", "mp3d"), default="hm3d")
    ap.add_argument("--results", type=Path, default=None, help="default: the dataset's results/")
    ap.add_argument("--out", type=Path, default=None, help="default: the dataset's figures/")
    ap.add_argument("--horizon", type=int, nargs="*", default=[],
                    help="1-based layer(s) to mark with a vertical dashed line (default: none)")
    args = ap.parse_args()
    root = RESULTS_ROOT if args.dataset == "hm3d" else RESULTS_ROOT.with_name("info_horizon_mp3d")
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.results / "horizon.csv").open() as f:
        rows = list(csv.DictReader(f))
    curves = [c for c in STYLE if any(r["curve"] == c for r in rows)]

    for name, panels in FIGURES.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.6), sharex=True, squeeze=False)
        axes = axes[0]
        for ax, (key, title) in zip(axes, panels):
            draw_panel(ax, rows, curves, key, title, args.horizon)
        handles, labels = [], []
        for ax in axes:  # a curve may appear in some panels only (no leak-free SR/SPL, partial data)
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel(r"First layer without visual tokens $\ell_0$", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out = args.out / name
        for ext in ("pdf", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
        print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
        plt.close(fig)
    for stale in args.out.glob("information_horizon.*"):  # the former 4-panel figure
        stale.unlink()
        print("removed stale figure", stale.name)


if __name__ == "__main__":
    main()
