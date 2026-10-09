#!/usr/bin/env python3
"""Figures for the visual-token pruning experiment (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results/token_prune/figures/:

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

Series whose arms have not been aggregated yet are skipped (and listed on the console); partial
arms are drawn with hollow markers and annotated with their episode count. Needs matplotlib, so
run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py [--k 8] [--variant flat]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOTS = {"window": HERE.parents[1] / "paper_results/token_prune",
                 "flat": HERE.parents[1] / "paper_results/token_prune_flat"}
MODELS = ("LongNav-R1", "Uni-NaVid")
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
FLAT_METHODS = ("random", "fastv", "divprune", "dart")  # PyramidDrop is layer-staged by construction


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


def draw_flat_panel(ax, summary, model, metric, show_legend):
    """Flat variant: one plain line per method -- no recency curve, no error bars; the unpruned
    identity arm (full context / 64 frames) is the dotted reference."""
    missing = []
    ident = series(summary, model, "identity")
    if ident:
        ax.axhline(fl(ident[0][metric]), color="gray", linestyle=":", linewidth=1.6, label="Unpruned")
    else:
        missing.append("identity")
    for method in FLAT_METHODS:
        style = METHODS[method]
        rows = series(summary, model, method)
        if not rows:
            missing.append(method)
            continue
        x = [fl(r["frames_worth"]) for r in rows]
        y = [fl(r[metric]) for r in rows]
        ax.plot(x, y, color=style["color"], linestyle="-", linewidth=1, marker="o", markersize=4,
                label=style["label"])
        for r in rows:
            if r["partial"] != "0":  # interim look: hollow marker + episode count
                ax.plot([fl(r["frames_worth"])], [fl(r[metric])], marker="o", markersize=4,
                        color=style["color"], markerfacecolor="white", linestyle="none")
                ax.annotate(f"n={r['n']}", (fl(r["frames_worth"]), fl(r[metric])), fontsize=5,
                            xytext=(2, 2), textcoords="offset points", color=style["color"])
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xlim(2, 34)
    ax.set_ylim(0, 1.0 if metric == "sr" else 0.6)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=6)
    return missing


def figure_budget(summary, out: Path, draw_panel=None, legend_top=False):
    """legend_top: the protocol_curves layout (hybrid_prune) -- 7.0x4.8 canvas, large text, one
    legend row across the top instead of the in-panel legend, one shared x label (two per-column
    labels collide at this font size)."""
    draw_panel = draw_panel or draw_budget_panel
    rc = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14} if legend_top else {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8) if legend_top else (7.5, 5.5), sharex=True)
        missing = {}
        for col, model in enumerate(MODELS):
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
            fig.legend(handles, names, loc="upper center", ncol=len(handles), fontsize=12, frameon=False,
                       handlelength=1.6, columnspacing=1.0)
            fig.supxlabel("history tokens kept (frames' worth)", fontsize=14)
            rect = (0, 0, 1, 0.93)
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
    args = ap.parse_args()
    root = RESULTS_ROOTS[args.variant]
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_rows(args.results / "summary.csv")
    if summary:
        figure_budget(summary, args.out / "budget_curves",
                      draw_panel=draw_flat_panel if args.variant == "flat" else None,
                      legend_top=args.variant == "flat")
    # The flat variant logs no LongNav-R1 keep masks (full context), so it has no retention figure.
    retention = load_rows(args.results / "retention.csv") if args.variant == "window" else []
    if retention:
        figure_retention(retention, args.k, args.out / "retention_by_age")


if __name__ == "__main__":
    main()
