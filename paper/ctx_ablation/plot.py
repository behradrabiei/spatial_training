#!/usr/bin/env python3
"""Figures for the context-window ablation (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results/ctx_ablation/figures/:

  ctx_ablation.{pdf,png}         SR and SPL vs K, side by side, means only (CIs in summary.csv):
                                 LongNav-R1 (KV eviction, and leak-free recompute dashed) and
                                 LightNav-0 (old blocks kept); legend on top, shared x label
  ctx_ablation_{sr,spl}.{pdf,png}  the same two panels as standalone single-panel figures

The SR | SPL figures are drawn at the width of one column of a two-column paper (COLUMN_WIDTH,
8 pt text): include them at \\columnwidth without scaling. Uni-NaVid and the LightNav-0 window arms
(old blocks removed) are not drawn; their numbers stay in results/summary.csv and table.tex, as
does the eviction-minus-recompute gap (leak_* columns).

--dataset navverse draws the NavVerse indoor replicate (paper_results_archive/ctx_ablation_navverse/):
the LongNav-R1 eviction arms (red) and the LightNav-0 old-blocks-kept arms (green) in
ctx_ablation.{pdf,png}, plus ctx_ablation_hm3d_vs_navverse.{pdf,png} overlaying each model's HM3D
curve (solid, circles) and its NavVerse one (dotted, squares), and failure_modes.{pdf,png} (LongNav-R1)
/ failure_modes_lightnav.{pdf,png} (LightNav-0): one stacked bar per K with the 100 episodes split
into success / stopped at the wrong place / out of steps after being near the goal / out of steps
never near the goal / fell over (results/failure_modes.csv, classes from
results_io.navverse_failure_mode). Needs matplotlib, so run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/ctx_ablation/plot.py [--dataset navverse]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, COLUMN_WIDTH  # noqa: E402  (two-column paper format)
RESULTS_ROOT = HERE.parents[1] / "paper_results/ctx_ablation"
K_ORDER = ["1", "4", "8", "16", "32", "full"]
K_ORDER_KEEP = ["1", "2", "4", "8", "16", "32", "full"]  # HM3D figure: LongNav-R1 and the LightNav-0 control have a K = 2 point
LARGE_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}  # failure-mode bars (7 in wide)
# SR | SPL figures: one column of a two-column paper, drawn at final size (paper/figstyle.py).
COLUMN_FIGSIZE = (COLUMN_WIDTH, 1.5)
SINGLE_FIGSIZE = (COLUMN_WIDTH, 1.9)  # one metric alone, still a full column wide
PANELS = {"sr": "Success rate", "spl": "SPL", "steps": "Steps per episode"}
SERIES = {  # arm-name template, plot style (both LongNav-R1 variants share a color; recompute is dashed)
    "r1_evict": (lambda k: "r1_full" if k == "full" else f"r1_evict_k{k}",
                 dict(color="tab:red", marker="o", markersize=3, linestyle="-", label="LongNav-R1")),
    "r1_recompute": (lambda k: "r1_full" if k == "full" else f"r1_recompute_k{k}",
                     dict(color="tab:red", marker="o", markersize=3, linestyle="--", markerfacecolor="white",
                          label="LongNav-R1 (leak-free)")),
    # LightNav-0 = the "old blocks kept" arms: the window's image tokens with the older segments'
    # timestamps kept (aggregate.py: the flat token-prune study's lightnav_recency_b{K-2} runs, K = 2
    # from lightnav_random_b0 (current pair, no history), K = 1 from lightnav_recency_b0_cur1); shares
    # the full-context point. The window arms with the old blocks removed (lightnav_k{K}) and
    # Uni-NaVid (uninavid_k{K}) are in summary.csv but not drawn.
    "lightnav_keep": (lambda k: "lightnav_full" if k == "full" else f"lightnav_keep_k{k}",
                      dict(color="tab:green", marker="o", markersize=3, linestyle="-", label="LightNav-0")),
}
NAVVERSE_SERIES = {  # the NavVerse indoor arms (paper_results_archive/ctx_ablation_navverse/)
    "nv_evict": (lambda k: "nv_full" if k == "full" else f"nv_evict_k{k}",
                 dict(color="tab:red", marker="o", markersize=3, linestyle="-", label="LongNav-R1")),
    "nv_recompute": (lambda k: "nv_full" if k == "full" else f"nv_recompute_k{k}",
                     dict(color="tab:red", marker="o", markersize=3, linestyle="--", markerfacecolor="white",
                          label="LongNav-R1 (leak-free)")),
    # LightNav-0 behind the same client (serve_navverse.py): the current pair plus the K-2 newest
    # history frames' image tokens kept, older blocks' timestamps and markers left in the prompt.
    # (Key deliberately not "lightnav_keep": that key switches the x axis to K_ORDER_KEEP.)
    "nvl_keep": (lambda k: "nvl_full" if k == "full" else f"nvl_keep_k{k}",
                 dict(color="tab:green", marker="o", markersize=3, linestyle="-", label="LightNav-0")),
}
OVERLAY_SERIES = {  # HM3D vs NavVerse: LongNav-R1 KV eviction and LightNav-0 old-blocks-kept, per dataset
    "r1_evict": (SERIES["r1_evict"][0],
                 dict(color="tab:red", marker="o", markersize=3, linestyle="-", label="LongNav-R1, HM3D")),
    "nv_evict": (NAVVERSE_SERIES["nv_evict"][0],
                 dict(color="tab:red", marker="s", markersize=3, linestyle=":", label="LongNav-R1, NavVerse")),
    "hm3d_lightnav_keep": (SERIES["lightnav_keep"][0],
                           dict(color="tab:green", marker="o", markersize=3, linestyle="-", label="LightNav-0, HM3D")),
    "nvl_keep": (NAVVERSE_SERIES["nvl_keep"][0],
                 dict(color="tab:green", marker="s", markersize=3, linestyle=":", label="LightNav-0, NavVerse")),
}
DATASETS = {  # dataset -> (results root, series drawn in ctx_ablation, y-limits per metric)
    "hm3d": (RESULTS_ROOT, SERIES, None),
    # NavVerse success rates sit an order of magnitude below HM3D's, so the standalone figure uses
    # a fitted y-range (widened by figure_metrics if a series exceeds it); the hm3d-vs-navverse
    # overlay keeps the full-scale axes.
    "navverse": (HERE.parents[1] / "paper_results_archive/ctx_ablation_navverse", NAVVERSE_SERIES,
                 {"sr": (0, 0.3), "spl": (0, 0.12)}),
}


def load_summary(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {r["arm"]: r for r in csv.DictReader(f)}


def draw(ax, summary, name, metric, prefix="", series=None, ks=K_ORDER):
    """Mean line only (the 95% CIs are in summary.csv); keys are ``<prefix><metric>``. ``ks`` is
    the x axis (one categorical slot per K); a series skips the Ks it has no arm at."""
    arm_of, style = (series or SERIES)[name]
    xs, ys = [], []
    for i, k in enumerate(ks):
        r = summary.get(arm_of(k))
        if r is None or r.get(f"{prefix}{metric}", "") == "":
            continue
        xs.append(i)
        ys.append(float(r[f"{prefix}{metric}"]))
    # A series whose only point is the shared full-context arm has not been run: no line, no legend entry.
    if xs and any(ks[i] != "full" for i in xs):
        ax.plot(xs, ys, **style)


def finish(ax, metric, n_ticks=None, ks=K_ORDER):
    n_ticks = len(ks) if n_ticks is None else n_ticks
    ax.set_xticks(range(n_ticks))
    ax.set_xticklabels(ks[:n_ticks])
    ax.set_xlabel("Frames in context K")
    ax.set_ylabel(PANELS[metric], labelpad=2)  # metric on the y axis: no title row
    ax.grid(True, alpha=0.3)
    if metric in ("sr", "spl"):  # fixed ticks: the layout pass and the final draw must agree on label widths
        ax.set_ylim(0, 1.0 if metric == "sr" else 0.6)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    elif metric == "steps":
        ax.set_ylim(0, 520)


def top_legend(fig, handles, labels):
    """Legend centred above the panels, in as few rows as fit the figure width (measured, so it
    holds for any label set and figure size). Returns the legend height as a figure fraction."""
    renderer = fig.canvas.get_renderer()
    for rows in range(1, len(labels) + 1):
        legend = fig.legend(handles, labels, loc="upper center", ncol=-(-len(labels) // rows), frameon=False,
                            bbox_to_anchor=(0.5, 1.0), handlelength=2.8, columnspacing=1.0,
                            handletextpad=0.5, borderaxespad=0.2)
        box = legend.get_window_extent(renderer)
        if box.width <= fig.bbox.width or rows == len(labels):
            return (fig.bbox.height - box.y0) / fig.bbox.height
        legend.remove()


def figure_metrics(summary, metrics, figsize, out, series=None, summaries=None, ylims=None):
    """Paper figure at single-column size (COLUMN_RC), legend across the top, one shared x label.
    ``series`` maps names to (arm template, style);
    ``summaries`` (name -> summary) lets an overlay draw each series from its own results;
    ``ylims`` (metric -> (lo, hi)) overrides the default full-scale axes."""
    series = series or SERIES
    summaries = summaries or {}
    ks = K_ORDER_KEEP if "lightnav_keep" in series else K_ORDER  # the HM3D figure has a K = 2 tick
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
        axes = axes[0]
        for ax, metric in zip(axes, metrics):
            for name in series:
                draw(ax, summaries.get(name, summary), name, metric, series=series, ks=ks)
            finish(ax, metric, ks=ks)
            if ylims and metric in ylims:
                lo, hi = ylims[metric]
                top = max((max(line.get_ydata(), default=0) for line in ax.get_lines()), default=0)
                if top > hi:  # a series above the fitted range: widen rather than clip it
                    hi = 0.1 * -(-top * 1.15 // 0.1) if metric == "sr" else 0.02 * -(-top * 1.15 // 0.02)
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(AutoLocator())
            ax.set_xlabel("")
        # Legend above the panels (every corner of the panels has data in it).
        handles, labels = axes[0].get_legend_handles_labels()
        legend_height = top_legend(fig, handles, labels)
        fig.supxlabel("Frames in context K", fontsize=8)
        fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 1 - legend_height))
        for ext in ("pdf", "png"):  # inside the rc context: pdf.fonttype applies at save time
            fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def metric_figures(summary, out, **kwargs):
    """The SR | SPL pair as ``out``, plus each panel on its own as ``<out>_sr`` / ``<out>_spl``."""
    figure_metrics(summary, ["sr", "spl"], COLUMN_FIGSIZE, out, **kwargs)
    for metric in ("sr", "spl"):
        figure_metrics(summary, [metric], SINGLE_FIGSIZE, out.with_name(f"{out.name}_{metric}"), **kwargs)


# Failure-mode stack, bottom to top: (csv column, legend label, fill, label ink). Hues follow a
# categorical order whose adjacent pairs were checked for colour-vision-deficiency separation
# (success / early stop / the two step-cap outcomes), with a neutral grey for the simulator's
# fall termination; counts are printed inside segments of at least MIN_LABEL episodes.
FAILURE_STACK = (
    ("success", "Success", "#2a78d6", "white"),
    ("wrong_stop", "Stopped at the wrong place", "#eb6834", "white"),
    ("cap_near_goal_no_stop", "Out of steps, was near goal", "#1baf7a", "black"),
    ("cap_never_near_goal", "Out of steps, never near goal", "#eda100", "black"),
    ("fell", "Fell over", "#8f8e8a", "white"),
    ("other", "Other", "#d9d8d4", "black"),  # drawn only if the csv has the column
)
MIN_LABEL = 6


def figure_failure_modes(csv_path, out, arm_of=NAVVERSE_SERIES["nv_evict"][0]):
    """Stacked bar per K: how the episodes of each arm ended (counts out of n, so every bar
    has the same height and success is one of the segments)."""
    with csv_path.open() as f:
        rows = {r["arm"]: r for r in csv.DictReader(f)}
    ks = [k for k in K_ORDER if arm_of(k) in rows]
    if not ks:
        print(f"[skip] {out.name}: no arms in {csv_path}")
        return
    n = {int(rows[arm_of(k)]["n"]) for k in ks}
    assert len(n) == 1, f"arms with different episode counts: {n}"
    n = n.pop()
    stack = [spec for spec in FAILURE_STACK if spec[0] in next(iter(rows.values()))]
    with plt.rc_context(LARGE_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.1))
        xs = list(range(len(ks)))
        bottom = [0] * len(ks)
        for col, label, color, ink in stack:
            vals = [int(rows[arm_of(k)][col]) for k in ks]
            ax.bar(xs, vals, 0.62, bottom=bottom, color=color, label=label,
                   edgecolor="white", linewidth=1.2)  # white seams separate touching segments
            for x, v, b in zip(xs, vals, bottom):
                if v >= MIN_LABEL:
                    ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=10, color=ink)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(xs)
        ax.set_xticklabels(ks)
        ax.set_xlabel("Frames in context K")
        ax.set_ylabel(f"Episodes (of {n})")
        ax.set_ylim(0, n)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Legend above the axis in two columns, reading row by row from the top of the stack down.
        # matplotlib fills legend columns first, so permute the entries to get row-major order.
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        ncol = 2
        order = sorted(range(len(labels)), key=lambda i: (i % ncol, i // ncol))  # column-major slots
        fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=ncol,
                   fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="hm3d")
    ap.add_argument("--summary", type=Path, default=None, help="default: <results root>/results/summary.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: <results root>/figures")
    args = ap.parse_args()
    root, series, ylims = DATASETS[args.dataset]
    summary_path = args.summary or root / "results/summary.csv"
    out = args.out or root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = load_summary(summary_path)
    metric_figures(summary, out / "ctx_ablation", series=series, ylims=ylims)
    if args.dataset == "navverse":
        hm3d = load_summary(RESULTS_ROOT / "results/summary.csv")
        metric_figures(summary, out / "ctx_ablation_hm3d_vs_navverse", series=OVERLAY_SERIES,
                       summaries={"r1_evict": hm3d, "nv_evict": summary, "hm3d_lightnav_keep": hm3d, "nvl_keep": summary})
        failure_csv = summary_path.parent / "failure_modes.csv"
        if failure_csv.is_file():
            figure_failure_modes(failure_csv, out / "failure_modes")
            figure_failure_modes(failure_csv, out / "failure_modes_lightnav", arm_of=NAVVERSE_SERIES["nvl_keep"][0])
        else:
            print(f"[skip] {failure_csv} not found (run aggregate.py --dataset navverse first)")


if __name__ == "__main__":
    main()
