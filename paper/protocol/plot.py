#!/usr/bin/env python3
"""Figures for the hybrid-pruning comparison (run aggregate.py first). Plain matplotlib
defaults; writes to paper_results/protocol/figures/:

  protocol_curves.{pdf,png}  the paper figure, the ONLY one written by default (since 2026-11-22), at
                             single-column size (paper/figstyle.py; legend on top, shared x label), models
                             LongNav-R1 and LightNav-0 (--models adds Uni-NaVid back; its arms stay in
                             results/), lines PyramidDrop and "Protocol" = the uniform protocol, the
                             paper's (--protocols adds the retired hybrid back as "Hybrid").
                             SR (top) and SPL (bottom) vs the budget k -- history visual tokens
                             averaged over all decoder layers, in frames' worth -- one column per
                             model, one plain line with markers per protocol (pyramiddrop, hybrid,
                             uniform = uniform random below the cut + visual-blind above it; means only; the CIs and
                             paired tests are in results/summary.csv and results/paired.csv); the
                             dotted line is the model unpruned with its full context (Exp 1's
                             r1_full / uninavid_full, from paper_results/ctx_ablation; LightNav-0:
                             lightnav_ref of paper_results/info_horizon, the same eager kernel as its
                             arms). A model column is drawn when the summary holds arms of that model
                             for a drawn protocol (LightNav-0 runs pyramiddrop and uniform, HM3D only)

With --diagnostics also (not kept in figures/ since 2026-11-22, rebuilt on demand):

  budget_met.{pdf,png}       what each arm actually held by that measure vs k (on the diagonal =
                             the budget is met), one panel per model
  memory_tradeoff.{pdf,png}  LongNav-R1 only: SR and SPL vs ALL resident KV slots (layer mean, text
                             included), points labelled by k

  <figure>_<protocol>.{pdf,png}  the same three figures once per protocol other than pyramiddrop
                             (_hybrid, _uniform), each showing ONLY PyramidDrop and that protocol, which
                             is labelled "Protocol" (red) -- one standalone figure set per protocol.
                             Written for every protocol that has aggregated arms; --no-split skips them.
                             These include protocol_curves_<protocol>.

Superseded unmatched-budget figures are archived.

--dataset mp3d draws the MP3D replicate (paper_results_archive/protocol_mp3d/, archived 2026-11-22); its dotted reference is
the unpruned full-context arm of the MP3D information-horizon run (r1_ref / uninavid_ref in
paper_results_archive/info_horizon_mp3d/results/horizon.csv, archived 2026-11-23).

Arms not aggregated yet are skipped (and listed on the console); partial arms are drawn with
hollow markers and annotated with their episode count. Needs matplotlib, so run in the vln env:

    /home/brabiei/miniconda3/envs/vln/bin/python paper/protocol/plot.py [--dataset mp3d] [--protocols ...]
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
from figstyle import COLUMN_RC, COLUMN_WIDTH, TEXT_WIDTH, top_legend  # noqa: E402

RESULTS_ROOT = HERE.parents[1] / "paper_results/protocol"
MODELS = ("LongNav-R1", "Uni-NaVid", "LightNav-0")  # column order; a model without arms is left out
FIGURE_MODELS = ("LongNav-R1", "LightNav-0")  # drawn by default: Uni-NaVid is not a paper baseline (user, 2026-11-21)
DRAWN = list(FIGURE_MODELS)  # narrowed or widened by --models
# The paper's protocol is `uniform` (user, 2026-11-22: "keep the most recent 32, remove everything after the
# information horizon, and if we still don't meet budget do random pruning"); the hybrid is no longer used
# and is drawn only on request (--protocols ... hybrid). pyramiddrop keeps its colour from paper/token_prune.
ALL_PROTOCOLS = {
    "pyramiddrop": dict(color="tab:green", label="PyramidDrop"),
    "uniform": dict(color="tab:red", label="Protocol"),
    "hybrid": dict(color="tab:blue", label="Hybrid"),
}
FIGURE_PROTOCOLS = ("pyramiddrop", "uniform")  # drawn by default
PROTOCOLS = {p: ALL_PROTOCOLS[p] for p in FIGURE_PROTOCOLS}  # replaced by --protocols
BASELINE = "pyramiddrop"  # drawn in every per-protocol figure set
# unpruned full-context arms of the context ablation (Exp 1), drawn as the dashed reference
FULL_CONTEXT = {"LongNav-R1": "r1_full", "Uni-NaVid": "uninavid_full"}
FULL_SUMMARY = HERE.parents[1] / "paper_results/ctx_ablation/results/summary.csv"
# LightNav-0's unpruned reference on the eager masked kernel its arms run on (rows keyed by blind_arm)
LIGHTNAV_REF = ("LightNav-0", "lightnav_ref", HERE.parents[1] / "paper_results/info_horizon/results/horizon.csv")
# MP3D: the same protocol's arms live in the information-horizon replicate (rows keyed by blind_arm)
FULL_CONTEXT_MP3D = {"LongNav-R1": "r1_ref", "Uni-NaVid": "uninavid_ref"}
FULL_SUMMARY_MP3D = HERE.parents[1] / "paper_results_archive/info_horizon_mp3d/results/horizon.csv"
PANELS = {"sr": "Success rate", "spl": "SPL"}
CURVES_HEIGHT = 2.5  # protocol_curves, inches; COLUMN_WIDTH wide (TEXT_WIDTH with more than two models)
Y_TOP = {"hm3d": {"sr": 1.02, "spl": 0.6}, "mp3d": {"sr": 0.75, "spl": 0.4}}  # protocol_curves y-limits per dataset
K_TICKS = {"model": [2, 4, 8, 16]}
DIRS = {"model": ("results", "figures")}


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


def models_drawn(summary, styles=None):
    """The drawn models (--models) with at least one aggregated arm of a drawn protocol, in MODELS order."""
    protocols = set(styles or PROTOCOLS)
    return [m for m in MODELS if m in DRAWN and any(r["model"] == m and r["protocol"] in protocols for r in summary)]


def figure_curves(summary, refs, out: Path, ticks, y_top=Y_TOP["hm3d"], styles=None, suffix=""):
    """`refs`: model -> its unpruned reference row (or None). Single-column format (COLUMN_RC), as the
    flat token-prune figure: SR over SPL, one column per drawn model, legend across the top."""
    models = models_drawn(summary, styles)
    width = COLUMN_WIDTH if len(models) <= 2 else TEXT_WIDTH
    with plt.rc_context(COLUMN_RC):
        fig, axes = plt.subplots(2, len(models), figsize=(width, CURVES_HEIGHT), sharex=True, sharey="row",
                                 squeeze=False)
        missing = []
        for col, model in enumerate(models):
            ref = refs.get(model)
            for row, (metric, title) in enumerate(PANELS.items()):
                ax = axes[row][col]
                if ref is not None:
                    ax.axhline(fl(ref[metric]), color="gray", linestyle=":", label="Unpruned")
                for protocol, style in (styles or PROTOCOLS).items():
                    rows = series(summary, model, protocol, "k")
                    if not rows:
                        missing.append(f"{model}/{protocol}")
                        continue
                    x = [fl(r["k"]) for r in rows]
                    ax.plot(x, [fl(r[metric]) for r in rows], color=style["color"], marker="o",
                            markersize=3, label=style["label"])
                    mark_partial(ax, rows, "k", metric, style["color"])
                ax.set_xticks(ticks)
                ax.set_xlim(0, 17)
                ax.set_ylim(0, y_top[metric])
                ax.yaxis.set_major_locator(MultipleLocator(0.2))  # fixed ticks: layout pass and final draw agree
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(model, pad=3)
                if col == 0:
                    ax.set_ylabel(title, labelpad=2)
        # the legend lists every protocol drawn in any column (LightNav-0 has no hybrid arm)
        entries = {}
        for ax in axes[0]:
            for handle, name in zip(*ax.get_legend_handles_labels()):
                entries.setdefault(name, handle)
        order = ["Unpruned"] + [st["label"] for st in (styles or PROTOCOLS).values()]
        names = [n for n in order if n in entries]
        legend_height = top_legend(fig, [entries[n] for n in names], names) if names else 0.0
        fig.supxlabel("Budget k (frames' worth)", fontsize=8)
        fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.8, rect=(0, 0, 1, 1 - legend_height))
        stem = out / f"protocol_curves{suffix}"
        for ext in ("pdf", "png"):  # inside the rc context: pdf.fonttype applies at save time
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")
    return sorted(set(missing))


def figure_budget_met(summary, out: Path, ticks, styles=None, suffix=""):
    models = models_drawn(summary, styles)
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 + 3.25 * (len(models) - 2), 2.8), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
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
    ap.add_argument("--protocols", nargs="+", choices=list(ALL_PROTOCOLS), default=list(FIGURE_PROTOCOLS),
                    help="protocols to draw (default: pyramiddrop and uniform, the paper's protocol; hybrid on request)")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(FIGURE_MODELS),
                    help="models drawn (default: LongNav-R1 and LightNav-0; Uni-NaVid on request)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also write budget_met, memory_tradeoff and the per-protocol figure sets")
    ap.add_argument("--no-split", action="store_true",
                    help="with --diagnostics: skip the per-protocol sets (<figure>_<protocol>: PyramidDrop vs that protocol)")
    args = ap.parse_args()
    DRAWN[:] = args.models
    PROTOCOLS.clear()
    PROTOCOLS.update({p: ALL_PROTOCOLS[p] for p in ALL_PROTOCOLS if p in args.protocols})
    root = RESULTS_ROOT if args.dataset == "hm3d" else RESULTS_ROOT.parents[1] / "paper_results_archive/protocol_mp3d"
    results = args.results or root / DIRS[args.budget][0]
    out = args.out or root / DIRS[args.budget][1]
    summary = load_rows(results / "summary.csv")
    if not summary:
        return
    out.mkdir(parents=True, exist_ok=True)
    ticks = K_TICKS[args.budget]
    if args.dataset == "mp3d":
        full, full_arms, arm_col, curve_kw = load_rows(FULL_SUMMARY_MP3D), FULL_CONTEXT_MP3D, "blind_arm", dict(y_top=Y_TOP["mp3d"])
    else:
        full, full_arms, arm_col, curve_kw = load_rows(FULL_SUMMARY), FULL_CONTEXT, "arm", {}
    refs = {model: next((r for r in full if r[arm_col] == arm), None) for model, arm in full_arms.items()}
    if args.dataset == "hm3d" and any(r["model"] == LIGHTNAV_REF[0] for r in summary):
        refs[LIGHTNAV_REF[0]] = next((r for r in load_rows(LIGHTNAV_REF[2]) if r["blind_arm"] == LIGHTNAV_REF[1]), None)

    def draw(styles=None, suffix=""):
        missing = figure_curves(summary, refs, out, ticks, styles=styles, suffix=suffix, **curve_kw)
        if args.diagnostics:
            figure_budget_met(summary, out, ticks, styles=styles, suffix=suffix)
            figure_memory(summary, out, styles=styles, suffix=suffix)
        return missing

    missing = draw()  # every selected protocol in one figure
    if args.diagnostics and not args.no_split:
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
