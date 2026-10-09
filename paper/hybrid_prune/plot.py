#!/usr/bin/env python3
"""Figures for the hybrid-pruning comparison (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results/hybrid_prune/figures/:

  protocol_curves.{pdf,png}  (large text, legend on top, shared x label) SR (top) and SPL (bottom) vs the budget k -- history visual tokens
                             averaged over all decoder layers, in frames' worth -- one column per
                             model, one plain line with markers per protocol (pyramiddrop, hybrid,
                             uniform = uniform random below the cut + visual-blind above it; means only; the CIs and
                             paired tests are in results/summary.csv and results/paired.csv); the
                             dotted line is the model unpruned with its full context (Exp 1's
                             r1_full / uninavid_full, from paper_results/ctx_ablation)
  budget_met.{pdf,png}       what each arm actually held by that measure vs k (on the diagonal =
                             the budget is met), one panel per model
  memory_tradeoff.{pdf,png}  LongNav-R1 only: SR and SPL vs ALL resident KV slots (layer mean, text
                             included), points labelled by k

  <figure>_<protocol>.{pdf,png}  the same three figures once per protocol other than pyramiddrop
                             (_hybrid, _uniform), each showing ONLY PyramidDrop and that protocol, which
                             is labelled "Protocol" (red) -- one standalone figure set per protocol.
                             Written for every protocol that has aggregated arms; --no-split skips them.

--budget last_stage draws the first, superseded run (k bounded the last scheduled stage only) from
results_last_stage/ into figures_last_stage/; its budget_met panel shows how far off k those arms sat.

--dataset mp3d draws the MP3D replicate (paper_results/hybrid_prune_mp3d/); its dotted reference is
the unpruned full-context arm of the MP3D information-horizon run (r1_ref / uninavid_ref in
paper_results/info_horizon_mp3d/results/horizon.csv).

Arms not aggregated yet are skipped (and listed on the console); partial arms are drawn with
hollow markers and annotated with their episode count. Needs matplotlib, so run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot.py [--dataset mp3d] [--protocols ...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/hybrid_prune"
MODELS = ("LongNav-R1", "Uni-NaVid")
ALL_PROTOCOLS = {  # pyramiddrop keeps its colour from paper/token_prune
    "pyramiddrop": dict(color="tab:green", label="PyramidDrop"),
    "hybrid": dict(color="tab:red", label="Protocol"),
    "uniform": dict(color="tab:blue", label="Uniform + blind"),
}
PROTOCOLS = dict(ALL_PROTOCOLS)  # narrowed by --protocols
BASELINE = "pyramiddrop"  # drawn in every per-protocol figure set
# unpruned full-context arms of the context ablation (Exp 1), drawn as the dashed reference
FULL_CONTEXT = {"LongNav-R1": "r1_full", "Uni-NaVid": "uninavid_full"}
FULL_SUMMARY = HERE.parents[1] / "paper_results/ctx_ablation/results/summary.csv"
# MP3D: the same protocol's arms live in the information-horizon replicate (rows keyed by blind_arm)
FULL_CONTEXT_MP3D = {"LongNav-R1": "r1_ref", "Uni-NaVid": "uninavid_ref"}
FULL_SUMMARY_MP3D = HERE.parents[1] / "paper_results/info_horizon_mp3d/results/horizon.csv"
PANELS = {"sr": "Success rate", "spl": "SPL"}
CURVES_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}  # protocol_curves text sizes
Y_TOP = {"hm3d": {"sr": 1.02, "spl": 0.6}, "mp3d": {"sr": 0.75, "spl": 0.4}}  # protocol_curves y-limits per dataset
K_TICKS = {"model": [2, 4, 8, 16], "last_stage": [1, 4, 8, 16]}
DIRS = {"model": ("results", "figures"), "last_stage": ("results_last_stage", "figures_last_stage")}


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


def series(summary, model, protocol, x_col):
    rows = [r for r in summary if r["model"] == model and r["protocol"] == protocol]
    return sorted(rows, key=lambda r: fl(r[x_col]))


def mark_partial(ax, rows, x_col, metric, color):
    for r in rows:
        if r["partial"] != "0":  # interim look: hollow marker + episode count
            ax.plot([fl(r[x_col])], [fl(r[metric])], marker="o", markersize=4, color=color,
                    markerfacecolor="white", linestyle="none")
            ax.annotate(f"n={r['n']}", (fl(r[x_col]), fl(r[metric])), fontsize=5, xytext=(2, 2),
                        textcoords="offset points", color=color)


def figure_curves(summary, full, out: Path, ticks, full_arms=FULL_CONTEXT, arm_col="arm", y_top=Y_TOP["hm3d"],
                  styles=None, suffix=""):
    with plt.rc_context(CURVES_RC):  # large text (paper figure); one shared x label, legend on top
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
        missing = []
        for col, model in enumerate(MODELS):
            ref = next((r for r in full if r[arm_col] == full_arms[model]), None)
            for row, (metric, title) in enumerate(PANELS.items()):
                ax = axes[row][col]
                if ref is not None:
                    ax.axhline(fl(ref[metric]), color="gray", linestyle=":", linewidth=1.6, label="Unpruned")
                for protocol, style in (styles or PROTOCOLS).items():
                    rows = series(summary, model, protocol, "k")
                    if not rows:
                        missing.append(f"{model}/{protocol}")
                        continue
                    x = [fl(r["k"]) for r in rows]
                    ax.plot(x, [fl(r[metric]) for r in rows], color=style["color"], linewidth=1, marker="o",
                            markersize=4, label=style["label"])
                    mark_partial(ax, rows, "k", metric, style["color"])
                ax.set_xticks(ticks)
                ax.set_xlim(0, 17)
                ax.set_ylim(0, y_top[metric])
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(model)
                if col == 0:
                    ax.set_ylabel(title)
        handles, names = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, names, loc="upper center", ncol=len(handles), fontsize=12, frameon=False,
                       handlelength=1.6, columnspacing=1.0)
        fig.supxlabel("budget k (frames' worth)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out / f"protocol_curves{suffix}")
    return sorted(set(missing))


def figure_budget_met(summary, out: Path, ticks, styles=None, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, model in zip(axes, MODELS):
        top = max(ticks)
        for protocol, style in (styles or PROTOCOLS).items():
            rows = [r for r in series(summary, model, protocol, "k") if fl(r["held_frames"]) == fl(r["held_frames"])]
            if not rows:
                continue
            ax.plot([fl(r["k"]) for r in rows], [fl(r["held_frames"]) for r in rows], color=style["color"],
                    linewidth=1, marker="o", markersize=4, label=style["label"])
            top = max(top, max(fl(r["held_frames"]) for r in rows))
        ax.plot([0, top], [0, top], color="gray", linestyle="--", linewidth=0.8, label="budget")
        ax.set_xticks(ticks)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 1.05 * top)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("budget k (frames' worth)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("held, mean over layers\n(frames' worth)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / f"budget_met{suffix}")


def figure_memory(summary, out: Path, styles=None, suffix=""):
    model = MODELS[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    drawn = False
    for ax, (metric, title) in zip(axes, PANELS.items()):
        for protocol, style in (styles or PROTOCOLS).items():
            rows = [r for r in series(summary, model, protocol, "kv_len_mean") if fl(r["kv_len_mean"]) == fl(r["kv_len_mean"])]
            if not rows:
                continue
            drawn = True
            x = [fl(r["kv_len_mean"]) for r in rows]
            ax.plot(x, [fl(r[metric]) for r in rows], color=style["color"], linewidth=1, marker="o",
                    markersize=4, label=style["label"])
            for r in rows:
                ax.annotate(f"k={r['k']}", (fl(r["kv_len_mean"]), fl(r[metric])), fontsize=5, xytext=(2, -7),
                            textcoords="offset points", color=style["color"])
            mark_partial(ax, rows, "kv_len_mean", metric, style["color"])
        ax.set_ylim(0, 1.02 if metric == "sr" else 0.6)
        ax.set_xlabel("all resident KV slots,\ntext incl. (layer mean)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    if not drawn:
        plt.close(fig)
        return
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / f"memory_tradeoff{suffix}")


def save(fig, stem: Path):
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", choices=list(DIRS), default="model")
    ap.add_argument("--dataset", choices=("hm3d", "mp3d"), default="hm3d")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--protocols", nargs="+", choices=list(ALL_PROTOCOLS), default=list(ALL_PROTOCOLS),
                    help="protocols to draw (default: all; an arm not aggregated is skipped anyway)")
    ap.add_argument("--no-split", action="store_true",
                    help="skip the per-protocol figure sets (<figure>_<protocol>: PyramidDrop vs that protocol)")
    args = ap.parse_args()
    PROTOCOLS.clear()
    PROTOCOLS.update({p: ALL_PROTOCOLS[p] for p in ALL_PROTOCOLS if p in args.protocols})
    root = RESULTS_ROOT if args.dataset == "hm3d" else RESULTS_ROOT.with_name("hybrid_prune_mp3d")
    results = args.results or root / DIRS[args.budget][0]
    out = args.out or root / DIRS[args.budget][1]
    summary = load_rows(results / "summary.csv")
    if not summary:
        return
    out.mkdir(parents=True, exist_ok=True)
    ticks = K_TICKS[args.budget]
    if args.dataset == "mp3d":
        full, curve_kw = load_rows(FULL_SUMMARY_MP3D), dict(full_arms=FULL_CONTEXT_MP3D, arm_col="blind_arm",
                                                             y_top=Y_TOP["mp3d"])
    else:
        full, curve_kw = load_rows(FULL_SUMMARY), {}

    def draw(styles=None, suffix=""):
        missing = figure_curves(summary, full, out, ticks, styles=styles, suffix=suffix, **curve_kw)
        figure_budget_met(summary, out, ticks, styles=styles, suffix=suffix)
        figure_memory(summary, out, styles=styles, suffix=suffix)
        return missing

    missing = draw()  # every selected protocol in one figure
    if not args.no_split:
        # One standalone set per protocol: PyramidDrop vs that protocol, which is called "Protocol".
        present = {r["protocol"] for r in summary}
        for protocol in PROTOCOLS:
            if protocol == BASELINE or protocol not in present:
                continue
            styles = {BASELINE: ALL_PROTOCOLS[BASELINE], protocol: dict(color="tab:red", label="Protocol")}
            draw(styles, suffix=f"_{protocol}")
    if missing:
        print("[missing] " + ", ".join(missing))


if __name__ == "__main__":
    main()
