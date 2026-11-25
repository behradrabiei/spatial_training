#!/usr/bin/env python3
"""Figures for the visual-token pruning experiment (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results_archive/token_prune/figures/:

  budget_curves.{pdf,png}     SR (top) and SPL (bottom) vs history tokens kept in frames' worth,
                              one column per model, one line per method, the ctx-ablation recency
                              arms in black, the 32-frame reference as a dashed line with its CI band
  retention_by_age.{pdf,png}  fraction of a history frame's tokens retained vs its age at one
                              budget (--k, default 8), one panel per model stacked vertically, one
                              line per method

--variant flat draws the flat variant (README "Flat variant": full context, every method at
layer 0) from paper_results/token_prune_flat/ as the paper figure: LongNav-R1 and LightNav-0
columns, random pruning and the published selectors FastV, DivPrune and DART as plain lines with
markers at k = 4, 8, 16, 32 and the unpruned identity arm as a dotted reference line. It is drawn
at the width of one column of a two-column paper (paper/figstyle.py, 8 pt text, legend across the
top): include it at \\columnwidth without scaling. Uni-NaVid, the whole-frame arms and the arms
below the k grid are not drawn; their numbers stay in results/summary.csv and table.tex. A model
column with no series for the requested methods is left out.

--recency draws the reference view instead (not the paper figure; give it a --name): x axis from
0, the context-window ablation's recency curve as black squares -- the series its own figure draws
(LongNav-R1 KV eviction, LightNav-0 old blocks kept), read from
paper_results/ctx_ablation/results/summary.csv and placed at the history a K-frame window holds
(RECENCY_SOURCE) -- and LightNav-0's Random line continued through its k = 2 / 1 / 0 arms (FLAT_EXTRA).

Series whose arms have not been aggregated yet are skipped (and listed on the console); partial
arms are drawn with hollow markers and annotated with their episode count. Needs matplotlib, so
run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py [--k 8] [--variant flat]

--methods a,b (flat only) draws another set of method lines (unpruned reference kept) and --name
sets the budget figure's basename, e.g. the keep-one comparison (with random, FastV, whole-frame recency):

    ... plot.py --variant flat --methods random,fastv,kl_keep_one,entropy_keep_one,recency --name budget_curves_keep_one
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, COLUMN_WIDTH  # noqa: E402  (two-column paper format)
RESULTS_ROOTS = {"window": HERE.parents[1] / "paper_results_archive/token_prune",
                 "flat": HERE.parents[1] / "paper_results/token_prune_flat"}
MODELS = ("LongNav-R1", "Uni-NaVid")
FLAT_MODELS = ("LongNav-R1", "LightNav-0")  # Uni-NaVid is not a baseline of the paper (2026-11-21): csv/tex only
FLAT_FIGSIZE = (COLUMN_WIDTH, 2.5)  # SR over SPL, one column per model, drawn at final size
METHODS = {  # method -> style; fixed so colours mean the same thing in every figure
    "random": dict(color="tab:blue", label="Random"),
    "fastv": dict(color="tab:orange", label="FastV"),
    "pyramiddrop": dict(color="tab:green", label="PyramidDrop"),
    "divprune": dict(color="tab:red", label="DivPrune"),
    "dart": dict(color="tab:purple", label="DART"),
}
RECENCY = dict(color="black", marker="s", markersize=4, linestyle="-", label="recency window (Exp 1)")
# Flat figure: the recency curve of the context-window ablation, as its figure draws it
# (paper/ctx_ablation/plot.py SERIES). model -> (arm template, history frames' worth a K-frame window
# holds): LongNav-R1's window is the current frame plus K - 1 history frames; LightNav-0's old-blocks-kept
# arm is the current 2-frame pair plus K - 2 history frames at 14 tokens each (lightnav_keep_k{K} is the
# run lightnav_recency_b{K-2}, K = 2 the no-history run). K = 1 of LightNav-0 (current frame doubled,
# less than the pair every prune arm keeps) has no place on this axis.
CTX_SUMMARY = HERE.parents[1] / "paper_results/ctx_ablation/results/summary.csv"
RECENCY_SOURCE = {"LongNav-R1": ("r1_evict_k{k}", 1), "LightNav-0": ("lightnav_keep_k{k}", 2)}
RECENCY_K = (1, 2, 4, 8, 16, 32)
RECENCY_FLAT = dict(color="black", marker="s", markersize=3, linestyle="-", label="Recency window")
PANELS = {"sr": "Success rate", "spl": "SPL"}
DODGE = 0.15  # x offset between methods so error bars at the same budget stay legible
WINDOW = 31
# Flat variant only (METHODS above also sets the windowed figure's dodge, so it stays as published):
# the whole-frame arms added 2026-10-09 -- keep-one KL ranking and plain frame recency -- and the
# minimum-entropy variant of 2026-10-12.
FLAT_STYLES = METHODS | {
    "kl_keep_one": dict(color="tab:brown", label="Keep-one KL"),
    "entropy_keep_one": dict(color="tab:pink", label="Keep-one entropy"),
    "recency": dict(color="tab:olive", label="Recency (frames)"),
}
FLAT_METHODS = ("random", "fastv", "divprune", "dart")  # the lines of the paper figure; the other FLAT_STYLES via --methods
# Extra arms drawn as part of a method's line (flat variant): the LightNav-0 layout-check arms below the
# k grid -- random at k = 2 / 1 (method label random_lo) and k = 0, no visual history (nohistory) -- extend
# the Random line to the left (2026-10-28). The window-content controls (recency_lo, nohistory_cur1) and
# the K = 1 window (window_lo) stay out of the figure.
FLAT_EXTRA = {"random": ("random_lo", "nohistory")}


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"[skip] {path} not found")
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def series(summary, model, method):
    rows = sorted((r for r in summary if r["model"] == model and r["method"] == method), key=lambda r: fl(r["frames_worth"]))
    return rows


def draw_budget_panel(ax, summary, model, metric, show_legend):
    missing = []
    ref = series(summary, model, "reference")
    if ref:
        r = ref[0]
        ax.axhline(fl(r[metric]), color="grey", linestyle="--", linewidth=1, label="32-frame reference (Exp 1)")
        ax.axhspan(fl(r[f"{metric}_lo"]), fl(r[f"{metric}_hi"]), color="grey", alpha=0.15, linewidth=0)
    else:
        missing.append("reference")
    rec = series(summary, model, "recency")
    if rec:
        x = [fl(r["frames_worth"]) for r in rec]
        y = [fl(r[metric]) for r in rec]
        ax.errorbar(x, y, yerr=[[y_ - fl(r[f"{metric}_lo"]) for y_, r in zip(y, rec)],
                                [fl(r[f"{metric}_hi"]) - y_ for y_, r in zip(y, rec)]],
                    capsize=2, linewidth=1, **RECENCY)
    else:
        missing.append("recency")
    for i, (method, style) in enumerate(METHODS.items()):
        rows = series(summary, model, method)
        if not rows:
            missing.append(method)
            continue
        dx = (i - (len(METHODS) - 1) / 2) * DODGE
        full = [r for r in rows if r["partial"] == "0"]
        part = [r for r in rows if r["partial"] != "0"]
        x = [fl(r["frames_worth"]) + dx for r in rows]
        y = [fl(r[metric]) for r in rows]
        ax.plot(x, y, color=style["color"], linestyle="-", linewidth=1, label=style["label"])
        for subset, face in ((full, style["color"]), (part, "white")):
            if not subset:
                continue
            xs = [fl(r["frames_worth"]) + dx for r in subset]
            ys = [fl(r[metric]) for r in subset]
            ax.errorbar(xs, ys, yerr=[[y_ - fl(r[f"{metric}_lo"]) for y_, r in zip(ys, subset)],
                                       [fl(r[f"{metric}_hi"]) - y_ for y_, r in zip(ys, subset)]],
                        fmt="o", markersize=4, color=style["color"], markerfacecolor=face, capsize=2, linewidth=1)
        for r in part:
            ax.annotate(f"n={r['n']}", (fl(r["frames_worth"]) + dx, fl(r[metric])), fontsize=5,
                        xytext=(2, 2), textcoords="offset points", color=style["color"])
    ax.set_xticks([0, 1, 4, 8, 16])
    ax.set_xlim(-1, 17.5)
    ax.set_ylim(0, 1.0 if metric == "sr" else 0.6)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=6)
    return missing


def recency_points(ctx, model, metric):
    """(frames' worth of history, value) of the ctx-ablation recency curve for one model, from its
    summary.csv rows keyed by arm (RECENCY_SOURCE); windows holding less than no history are skipped."""
    template, current = RECENCY_SOURCE.get(model, (None, 0))
    points = []
    for k in RECENCY_K:
        r = ctx.get(template.format(k=k)) if template else None
        if r is not None and k - current >= 0 and r.get(metric, "") != "":
            points.append((k - current, fl(r[metric])))
    return points


def draw_flat_panel(ax, summary, model, metric, methods=FLAT_METHODS, ctx=None):
    """Flat variant: one plain line per method on the k grid 4 ... 32, no error bars; the unpruned
    identity arm (full context) is the dotted reference. `methods` sets the lines (--methods); `ctx`
    (the ctx-ablation summary by arm, --recency) draws the reference view instead: x axis from 0, its
    recency curve as black squares and the arms below the k grid (FLAT_EXTRA)."""
    missing = []
    ident = series(summary, model, "identity")
    if ident:
        ax.axhline(fl(ident[0][metric]), color="gray", linestyle=":", label="Unpruned")
    else:
        missing.append("identity")
    rec = recency_points(ctx, model, metric) if ctx is not None else []
    if rec:
        ax.plot([x for x, _ in rec], [y for _, y in rec], **RECENCY_FLAT)
    elif ctx is not None:
        missing.append("ctx-ablation recency")
    low = ctx is not None  # the reference view: x axis from 0, with the recency curve and the arms below the k grid
    for method in methods:
        style = FLAT_STYLES[method]
        rows = series(summary, model, method)
        for extra in FLAT_EXTRA.get(method, ()) if low else ():
            rows += series(summary, model, extra)
        rows.sort(key=lambda r: fl(r["frames_worth"]))
        if not rows:
            missing.append(method)
            continue
        x = [fl(r["frames_worth"]) for r in rows]
        y = [fl(r[metric]) for r in rows]
        ax.plot(x, y, color=style["color"], linestyle="-", marker="o", markersize=3, label=style["label"])
        for r in rows:
            if r["partial"] != "0":  # interim look: hollow marker + episode count
                ax.plot([fl(r["frames_worth"])], [fl(r[metric])], marker="o", markersize=3,
                        color=style["color"], markerfacecolor="white", linestyle="none")
                ax.annotate(f"n={r['n']}", (fl(r["frames_worth"]), fl(r[metric])), fontsize=5,
                            xytext=(2, 2), textcoords="offset points", color=style["color"])
    ax.set_xticks([0, 4, 8, 16, 32] if low else [4, 8, 16, 32])
    ax.set_xlim(-1 if low else 2, 34)
    ax.set_ylim(0, 1.0 if metric == "sr" else 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # fixed ticks: layout pass and final draw agree
    ax.grid(True, alpha=0.3)
    return missing


def top_legend(fig, handles, labels):
    """Legend centred above the panels, in as few rows as fit the figure width (the helper of
    paper/ctx_ablation/plot.py). Returns the legend height as a figure fraction."""
    renderer = fig.canvas.get_renderer()
    for rows in range(1, len(labels) + 1):
        legend = fig.legend(handles, labels, loc="upper center", ncol=-(-len(labels) // rows), frameon=False,
                            bbox_to_anchor=(0.5, 1.0), handlelength=2.8, columnspacing=1.0,
                            handletextpad=0.5, borderaxespad=0.2)
        box = legend.get_window_extent(renderer)
        if box.width <= fig.bbox.width or rows == len(labels):
            return (fig.bbox.height - box.y0) / fig.bbox.height
        legend.remove()


def figure_flat(summary, out: Path, methods=FLAT_METHODS, ctx=None, models=FLAT_MODELS):
    """The flat variant's paper figure at single-column size (COLUMN_RC): SR over SPL, one column
    per model (one without any series for `methods` is dropped), legend across the top, one
    shared x label."""
    models = tuple(m for m in models if any(series(summary, m, meth) for meth in methods)) or models
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(2, len(models), figsize=FLAT_FIGSIZE, sharex=True, sharey="row", squeeze=False)
        for col, model in enumerate(models):
            for row, metric in enumerate(PANELS):
                ax = axes[row][col]
                missing = draw_flat_panel(ax, summary, model, metric, methods=methods, ctx=ctx)
                if row == 0:
                    ax.set_title(model, pad=3)
                    if missing:
                        print(f"[{out.name}] {model}: no data yet for {', '.join(missing)}")
                if col == 0:
                    ax.set_ylabel(PANELS[metric], labelpad=2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        legend_height = top_legend(fig, handles, labels)
        fig.supxlabel("History tokens kept (frames' worth)", fontsize=8)
        fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.8, rect=(0, 0, 1, 1 - legend_height))
        for ext in ("pdf", "png"):  # inside the rc context: pdf.fonttype applies at save time
            fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def figure_budget(summary, out: Path):
    """The windowed variant's figure: in-panel legend, error bars, one column per model."""
    fig, axes = plt.subplots(2, len(MODELS), figsize=(7.5, 5.5), sharex=True, squeeze=False)
    missing = {}
    for col, model in enumerate(MODELS):
        for row, metric in enumerate(PANELS):
            ax = axes[row][col]
            m = draw_budget_panel(ax, summary, model, metric, show_legend=(row == 0 and col == 0))
            missing.setdefault(model, m)
            if row == 0:
                ax.set_title(model)
            if col == 0:
                ax.set_ylabel(PANELS[metric])
            if row == 1:
                ax.set_xlabel("history tokens kept (frames' worth)")
    for model, m in missing.items():
        if m:
            print(f"[budget_curves] {model}: no data yet for {', '.join(m)}")
    save(fig, out)


def figure_retention(retention, k: int, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.6), sharex=True)  # models stacked vertically
    for ax, model in zip(axes, MODELS):
        drawn = False
        for method, style in METHODS.items():
            rows = sorted((r for r in retention if r["model"] == model and r["method"] == method and r["k"] == str(k)),
                          key=lambda r: int(r["age"]))
            rows = [r for r in rows if 1 <= int(r["age"]) <= WINDOW]
            if not rows:
                print(f"[retention_by_age] {model}: no data yet for {method} at k={k}")
                continue
            age = [int(r["age"]) for r in rows]
            ax.plot(age, [fl(r["frac_mean"]) for r in rows], color=style["color"], marker="o", markersize=2,
                    linewidth=1, label=style["label"])
            ax.fill_between(age, [fl(r["lo"]) for r in rows], [fl(r["hi"]) for r in rows],
                            color=style["color"], alpha=0.2, linewidth=0)
            drawn = True
        ax.set_title(f"{model}, budget k={k}")
        ax.set_ylabel("fraction of tokens retained")
        ax.set_xlim(0.5, WINDOW + 0.5)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if drawn and ax is axes[0]:
            ax.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel("frame age (steps before the decision)")
    save(fig, out)


def save(fig, out: Path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(RESULTS_ROOTS), default="window",
                    help="window = Exp 4 as published; flat = full context / layer 0 (token_prune_flat)")
    ap.add_argument("--results", type=Path, default=None, help="default: <variant root>/results")
    ap.add_argument("--out", type=Path, default=None, help="default: <variant root>/figures")
    ap.add_argument("--k", type=int, default=8, help="budget (frames' worth) for the retention figure")
    ap.add_argument("--methods", default=None,
                    help=f"flat only: comma-separated methods to draw (default: {','.join(FLAT_METHODS)}); "
                         "the unpruned line is always kept")
    ap.add_argument("--recency", action="store_true",
                    help="flat only: the reference view -- x axis from 0, the ctx-ablation recency curve "
                         "(black squares) and the arms below the k grid")
    ap.add_argument("--name", default="budget_curves", help="output basename of the budget figure")
    args = ap.parse_args()
    root = RESULTS_ROOTS[args.variant]
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_rows(args.results / "summary.csv")
    if args.variant != "flat" and (args.methods or args.recency):
        ap.error("--methods / --recency apply to --variant flat only")
    if summary and args.variant == "flat":
        methods = FLAT_METHODS
        if args.methods:
            methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
            unknown = [m for m in methods if m not in FLAT_STYLES]
            if unknown:
                ap.error(f"unknown --methods {unknown}; known: {', '.join(FLAT_STYLES)}")
        ctx = {r["arm"]: r for r in load_rows(CTX_SUMMARY)} if args.recency else None
        figure_flat(summary, args.out / args.name, methods=methods, ctx=ctx)
    elif summary:
        figure_budget(summary, args.out / args.name)
    # The flat variant logs no LongNav-R1 keep masks (full context), so it has no retention figure.
    retention = load_rows(args.results / "retention.csv") if args.variant == "window" else []
    if retention:
        figure_retention(retention, args.k, args.out / "retention_by_age")


if __name__ == "__main__":
    main()
