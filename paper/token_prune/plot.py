#!/usr/bin/env python3
"""Figures for the visual-token pruning experiment (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results_archive/token_prune/figures/:

  budget_curves.{pdf,png}     SR (top) and SPL (bottom) vs history tokens kept in frames' worth,
                              one column per model, one line per method, the ctx-ablation recency
                              arms in black, the 32-frame reference as a dashed line with its CI band
  retention_by_age.{pdf,png}  fraction of a history frame's tokens retained vs its age at one
                              budget (--k, default 8), one panel per model stacked vertically, one
                              line per method

--variant flat draws the flat variant (README "Flat variant": full context / 64 frames, every
method at layer 0) from paper_results/token_prune_flat/: the four methods only, plain lines with
markers -- no recency curve, no error bars -- at k = 4, 8, 16, 32, plus the unpruned identity arm
(full context / 64 frames) as a dotted reference line. Its budget_curves figure is drawn in the
hybrid_prune protocol_curves style: smaller canvas so the text reads larger, legend across the top.
The flat variant has a third column, LightNav-0 (its four token-budget arms, identity = the
information-horizon reference, the ctx-ablation lightnav_k windows as the black squares); a model
column with no series for the requested methods is left out (so --methods subsets stay two-column).

Series whose arms have not been aggregated yet are skipped (and listed on the console); partial
arms are drawn with hollow markers and annotated with their episode count. Needs matplotlib, so
run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py [--k 8] [--variant flat]

--methods a,b (flat only) draws a subset of the method lines (unpruned reference kept, ctx-window
squares dropped) and --name sets the budget figure's basename, e.g. the keep-one figure (with random, FastV, recency):

    ... plot.py --variant flat --methods random,fastv,kl_keep_one,entropy_keep_one,recency --name budget_curves_keep_one
"""
from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOTS = {"window": HERE.parents[1] / "paper_results_archive/token_prune",
                 "flat": HERE.parents[1] / "paper_results/token_prune_flat"}
MODELS = ("LongNav-R1", "Uni-NaVid")
FLAT_MODELS = ("LongNav-R1", "Uni-NaVid", "LightNav-0")
METHODS = {  # method -> style; fixed so colours mean the same thing in every figure
    "random": dict(color="tab:blue", label="Random"),
    "fastv": dict(color="tab:orange", label="FastV"),
    "pyramiddrop": dict(color="tab:green", label="PyramidDrop"),
    "divprune": dict(color="tab:red", label="DivPrune"),
    "dart": dict(color="tab:purple", label="DART"),
}
RECENCY = dict(color="black", marker="s", markersize=4, linestyle="-", label="recency window (Exp 1)")
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
FLAT_METHODS = ("random", "fastv", "divprune", "dart", "kl_keep_one", "entropy_keep_one", "recency")  # no PyramidDrop
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


def draw_flat_panel(ax, summary, model, metric, show_legend, methods=FLAT_METHODS, window=True):
    """Flat variant: one plain line per method, no error bars; the unpruned identity arm (full
    context / 64 frames) is the dotted reference and the ctx-ablation recency windows (merged by
    aggregate.py as method 'window', at k-1 history frames) the black squares. `methods` restricts
    the lines (--methods); `window` drops the black squares for such a subset figure."""
    missing = []
    ident = series(summary, model, "identity")
    if ident:
        ax.axhline(fl(ident[0][metric]), color="gray", linestyle=":", linewidth=1.6, label="Unpruned")
    else:
        missing.append("identity")
    win = series(summary, model, "window") if window else []
    if win:
        ax.plot([fl(r["frames_worth"]) for r in win], [fl(r[metric]) for r in win], linewidth=1, **RECENCY)
    elif window:
        missing.append("window")
    low = False  # any point below the k grid (LightNav-0 random at k = 2 / 1 / 0)
    for method in methods:
        style = FLAT_STYLES[method]
        rows = series(summary, model, method)
        for extra in FLAT_EXTRA.get(method, ()):
            rows += series(summary, model, extra)
        rows.sort(key=lambda r: fl(r["frames_worth"]))
        if not rows:
            missing.append(method)
            continue
        x = [fl(r["frames_worth"]) for r in rows]
        low = low or min(x) < 4
        y = [fl(r[metric]) for r in rows]
        ax.plot(x, y, color=style["color"], linestyle="-", linewidth=1, marker="o", markersize=4,
                label=style["label"])
        for r in rows:
            if r["partial"] != "0":  # interim look: hollow marker + episode count
                ax.plot([fl(r["frames_worth"])], [fl(r[metric])], marker="o", markersize=4,
                        color=style["color"], markerfacecolor="white", linestyle="none")
                ax.annotate(f"n={r['n']}", (fl(r["frames_worth"]), fl(r[metric])), fontsize=5,
                            xytext=(2, 2), textcoords="offset points", color=style["color"])
    # Axes are shared across the columns: the k = 0 point of one model extends every panel to 0.
    any_low = low or any(r["method"] in sum(FLAT_EXTRA.values(), ()) and fl(r["frames_worth"]) < 4
                         for r in summary if any(m in FLAT_EXTRA for m in methods))
    ax.set_xticks([0, 4, 8, 16, 32] if any_low else [4, 8, 16, 32])
    ax.set_xlim(-1 if any_low else 2, 34)
    ax.set_ylim(0, 1.0 if metric == "sr" else 0.6)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=6)
    return missing


def figure_budget(summary, out: Path, draw_panel=None, legend_top=False, legend_ncol=None, models=MODELS,
                  methods=None):
    """legend_top: the protocol_curves layout (hybrid_prune) -- 7.0x4.8 canvas, large text, one
    legend row across the top instead of the in-panel legend, one shared x label (two per-column
    labels collide at this font size). `models`: the columns; one without any series for `methods`
    (the drawn method set) is dropped. The canvas widens by 3.0 in per column beyond two."""
    draw_panel = draw_panel or draw_budget_panel
    if methods is not None:
        models = tuple(m for m in models if any(series(summary, m, meth) for meth in methods)) or models[:2]
    rc = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14} if legend_top else {}
    with plt.rc_context(rc):
        extra = 3.0 * (len(models) - 2)
        fig, axes = plt.subplots(2, len(models), figsize=(7.0 + extra, 4.8) if legend_top else (7.5 + extra, 5.5),
                                 sharex=True, squeeze=False)
        missing = {}
        for col, model in enumerate(models):
            for row, metric in enumerate(PANELS):
                ax = axes[row][col]
                m = draw_panel(ax, summary, model, metric, show_legend=(row == 0 and col == 0 and not legend_top))
                missing.setdefault(model, m)
                if row == 0:
                    ax.set_title(model)
                if col == 0:
                    ax.set_ylabel(PANELS[metric])
                if row == 1 and not legend_top:
                    ax.set_xlabel("history tokens kept (frames' worth)")
        rect = None
        if legend_top:
            handles, names = axes[0][0].get_legend_handles_labels()
            # Up to eight entries: four columns (two rows, as published). Nine or more: three
            # columns, so the long whole-frame labels fit the canvas width.
            ncol = min(len(handles), 4) if len(handles) <= 8 else 3
            ncol = legend_ncol or ncol  # --methods subsets: three columns, balanced rows
            fig.legend(handles, names, loc="upper center", ncol=ncol, fontsize=12, frameon=False,
                       handlelength=1.6, columnspacing=1.0)
            fig.supxlabel("history tokens kept (frames' worth)", fontsize=14)
            rows = -(-len(handles) // ncol)  # legend rows; 0.93 / 0.87 for one / two rows as published
            rect = (0, 0, 1, 0.93 - 0.06 * (rows - 1))
        for model, m in missing.items():
            if m:
                print(f"[budget_curves] {model}: no data yet for {', '.join(m)}")
        save(fig, out, rect=rect)


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


def save(fig, out: Path, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
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
                    help="flat only: comma-separated subset of methods to draw (default: all); the "
                         "ctx-window squares are omitted for a subset, the unpruned line is kept")
    ap.add_argument("--name", default="budget_curves", help="output basename of the budget figure")
    args = ap.parse_args()
    root = RESULTS_ROOTS[args.variant]
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_rows(args.results / "summary.csv")
    if summary:
        panel, ncol, models, methods = None, None, MODELS, None
        if args.variant == "flat":
            panel, models, methods = draw_flat_panel, FLAT_MODELS, FLAT_METHODS
            if args.methods:
                methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
                unknown = [m for m in methods if m not in FLAT_STYLES]
                if unknown:
                    ap.error(f"unknown --methods {unknown}; known: {', '.join(FLAT_STYLES)}")
                panel = functools.partial(draw_flat_panel, methods=methods, window=False)
                ncol = 3
        elif args.methods:
            ap.error("--methods applies to --variant flat only")
        figure_budget(summary, args.out / args.name, draw_panel=panel, legend_top=args.variant == "flat",
                      legend_ncol=ncol, models=models, methods=methods)
    # The flat variant logs no LongNav-R1 keep masks (full context), so it has no retention figure.
    retention = load_rows(args.results / "retention.csv") if args.variant == "window" else []
    if retention:
        figure_retention(retention, args.k, args.out / "retention_by_age")


if __name__ == "__main__":
    main()
