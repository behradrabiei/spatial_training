#!/usr/bin/env python3
"""Information-horizon figures (run aggregate.py first). Writes to paper_results/info_horizon/figures/
(--dataset mp3d: the archived MP3D replicate, paper_results_archive/info_horizon_mp3d/figures/):
  information_kl_blind.{pdf,png}  blind-from-layer KL = KL(P_full || P_every visual token hidden
                              from layer L on), mean over every decision of the 10 probe episodes
                              (results/blind_curve.csv, aggregate_blind.py); LongNav-R1, leaky probe
  information_kl_blind_lightnav.{pdf,png}  the same measure for LightNav-0 (256 <act_l0> codes, full
                              re-prefill); its own figure: the two models' KL scales differ by 300x
  information_kl_blind_norm.{pdf,png}  both models in one figure, each curve divided by its own value
                              at the first layer (KL(full || no visual input)), so both run from 1 to 0
  information_perf.{pdf,png}  SR and SPL of the visual-blind rollouts, 100 episodes; dotted
                              line = unblinded reference
  information_perf_sr / information_perf_spl.{pdf,png}  the two SR/SPL panels as standalone figures
against the decoder layer L (visual tokens hidden from L on).

The former current-frame KL figures (removed from figures/ on 2026-11-23) are drawn only on request,
--only information_kl information_kl_keep information_kl_leave: keep-one-in KL =
KL(P_only_current_frame || P_no_visual) and leave-one-out KL = KL(P_full || P_current_frame_hidden).

Drawn at the width of one column of a two-column paper (paper/figstyle.py: 8 pt text, the metric
on the y axis instead of a panel title, legend across the top, one shared x label), plain
matplotlib defaults otherwise. Models: LongNav-R1 (red) and LightNav-0 (green; SR/SPL only, no KL
probe for that model). Uni-NaVid is not a paper baseline: its rows stay in results/horizon.csv
and --models "LongNav-R1" "Uni-NaVid" "LightNav-0" draws it again (blue). The optional leak-free
LongNav-R1 probe arm is dashed red with hollow markers; the SR/SPL figure carries one LongNav-R1
curve (the blind rollout is leak-free by construction). Means only; CIs are in
results/horizon.csv. 1-based layer index on x (axis label $\\ell_0$), each model's "nothing hidden"
point at its own depth L+1 (LongNav-R1 29, Uni-NaVid 33, LightNav-0 37; larger hollow marker); the
last tick, at the deepest drawn model's L+1, reads "L+1". No horizon line unless --horizon is given.

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/info_horizon/plot.py [--dataset mp3d]
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
from figstyle import COLUMN_RC, COLUMN_WIDTH, top_legend  # noqa: E402  (two-column paper format)

RESULTS_ROOT = HERE.parents[1] / "paper_results/info_horizon"
RESULTS_ROOT_MP3D = HERE.parents[1] / "paper_results_archive/info_horizon_mp3d"  # archived 2026-11-23
MODELS = ("LongNav-R1", "LightNav-0")  # the paper's models; Uni-NaVid only with --models
STYLE = {"LongNav-R1": dict(color="tab:red", marker="o", markersize=3, linestyle="-"),
         "LongNav-R1 (leak-free)": dict(color="tab:red", marker="o", markersize=3, linestyle="--",
                                        markerfacecolor="white"),
         "Uni-NaVid": dict(color="tab:blue", marker="o", markersize=3, linestyle="-"),
         "LightNav-0": dict(color="tab:green", marker="o", markersize=3, linestyle="-")}
HORIZON_COLOR = "tab:purple"
PAIR_FIGSIZE = (COLUMN_WIDTH, 1.5)    # two panels in one column, drawn at final size
SINGLE_FIGSIZE = (COLUMN_WIDTH, 1.9)  # one panel alone, still a full column wide
# y label per metric: (two-panel figure, standalone figure); the KL labels wrap in the narrow panels
PANELS = {"keep_one_mean": ("Keep-one-in\nKL (nats)", "Keep-one-in KL (nats)"),
          "leave_one_mean": ("Leave-one-out\nKL (nats)", "Leave-one-out KL (nats)"),
          "sr": ("Success rate", "Success rate"), "spl": ("SPL", "SPL")}
FIGURES = {  # SR/SPL of the 100-episode blind runs (results/horizon.csv)
    "information_perf": ("sr", "spl"),
    # each panel again as a standalone figure
    "information_perf_sr": ("sr",),
    "information_perf_spl": ("spl",),
}
LEGACY_FIGURES = {  # current-frame KL of the probe episodes: only with --only
    "information_kl": ("keep_one_mean", "leave_one_mean"),
    "information_kl_keep": ("keep_one_mean",),
    "information_kl_leave": ("leave_one_mean",),
}
BLIND_FIGURES = {  # blind-from-layer KL (results/blind_curve.csv): name -> (models, normalised).
    # One figure per model in nats (the scales differ: 4 actions vs 256 codes), and both models
    # together with each curve divided by its own first-layer (no visual input) value.
    "information_kl_blind": (("LongNav-R1",), False),
    "information_kl_blind_lightnav": (("LightNav-0",), False),
    "information_kl_blind_norm": (("LongNav-R1", "LightNav-0"), True),
}
BLIND_YLABEL_NORM = "Normalized KL"
BLIND_YLABEL = "KL to full context (nats)"
REF_DASH, REF_GAP = 2.0, 1.0  # unblinded reference lines: dash and gap between models' dashes, in line widths


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def curve_rows(rows, curve, key):
    return sorted((r for r in rows if r["curve"] == curve and r.get(key, "") != ""), key=lambda r: int(r["layer"]))


def curves_for(rows, curves, key):
    """The curves with data for ``key``; one SR/SPL curve per model."""
    return [c for c in curves if not (key in ("sr", "spl") and c == "LongNav-R1 (leak-free)")
            and curve_rows(rows, c, key)]


def draw_panel(ax, rows, curves, key, ylabel, end, horizon):
    """``end`` = the figure's last x position, the deepest drawn model's L+1 (1-based)."""
    drawn = curves_for(rows, curves, key)
    for i, curve in enumerate(drawn):
        rs = curve_rows(rows, curve, key)
        xs = [int(r["layer"]) + 1 for r in rs]  # 1-based layer index on the axis
        ys = [fl(r[key]) for r in rs]
        style = STYLE[curve]
        ax.plot(xs, ys, label=curve, **style)
        n = int(rs[0]["n_layers"])
        if xs[-1] == n + 1:  # the L+1 point (nothing hidden), hollow so the depth difference is visible
            ax.plot([xs[-1]], [ys[-1]], marker="o", markersize=4, markerfacecolor="white", color=style["color"],
                    linestyle="none")
            if key in ("sr", "spl"):
                # Unblinded reference. The models' dashes interleave (one dash each per period), so two
                # references at the same height (LongNav-R1 and LightNav-0 SR) both stay visible.
                period = len(drawn) * (REF_DASH + REF_GAP)
                ax.axhline(ys[-1], color=style["color"], linewidth=1.0, alpha=0.85,
                           linestyle=(i * (REF_DASH + REF_GAP), (REF_DASH, period - REF_DASH)))
    if key == "sr":  # fixed ticks: the layout pass and the final draw must agree on label widths
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    elif key == "spl":
        ax.set_ylim(0, 0.6)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    else:
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylim(bottom=-1e-4)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    for h in horizon:
        ax.axvline(h, color=HORIZON_COLOR, linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel, labelpad=2)  # metric on the y axis: no title row
    layer_axis(ax, end)


def layer_axis(ax, end):
    ticks = [1] + [t for t in range(8, end - 6, 8)] + [end]  # 1-based; the last tick is the deepest L+1
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks[:-1]] + ["L+1"])
    ax.set_xlim(-0.5, end + 1.5)
    ax.grid(alpha=0.3)


def save(fig, axes, out):
    """Top legend, the shared x label, and both file formats (call inside the rc context:
    pdf.fonttype applies at save time)."""
    handles, labels = [], []
    for ax in axes:  # a curve may appear in some panels only (no leak-free SR/SPL, partial data)
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    legend_height = top_legend(fig, handles, labels)
    fig.supxlabel(r"First layer without visual tokens $\ell_0$", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 1 - legend_height))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def draw_blind(results: Path, out: Path, name: str, models, normalised: bool = False) -> None:
    """Blind-from-layer KL against the first blinded layer: pooled mean over every decision of the
    probe episodes, the nothing-hidden endpoint (0) at the model's L+1. ``normalised`` divides each
    model's curve by its value at its shallowest probed layer (layer 0: no visual input at all)."""
    path = results / "blind_curve.csv"
    if not path.is_file():
        print(f"[skip] {name}: no {path.name} (run aggregate_blind.py)")
        return
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["model"] in models and r.get("kl_mean", "") != ""]
    if not rows:
        print(f"[skip] {name}: no rows for {', '.join(models)}")
        return
    end = max(int(r["n_layers"]) for r in rows) + 1
    with plt.rc_context(COLUMN_RC):
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
        for model in models:
            rs = sorted((r for r in rows if r["model"] == model), key=lambda r: int(r["layer"]))
            if not rs:
                continue
            xs = [int(r["layer"]) + 1 for r in rs]  # 1-based layer index on the axis
            ys = [fl(r["kl_mean"]) for r in rs]
            if normalised:
                ys = [y / ys[0] for y in ys]
            style = STYLE[model]
            ax.plot(xs, ys, label=model, **style)
            if xs[-1] == int(rs[0]["n_layers"]) + 1:  # the L+1 point (nothing hidden), hollow
                ax.plot([xs[-1]], [ys[-1]], marker="o", markersize=4, markerfacecolor="white",
                        color=style["color"], linestyle="none")
        ax.set_ylim(bottom=0)
        ax.set_ylabel(BLIND_YLABEL_NORM if normalised else BLIND_YLABEL, labelpad=2)
        layer_axis(ax, end)
        save(fig, [ax], out / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("hm3d", "mp3d"), default="hm3d")
    ap.add_argument("--results", type=Path, default=None, help="default: the dataset's results/")
    ap.add_argument("--out", type=Path, default=None, help="default: the dataset's figures/")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=("LongNav-R1", "Uni-NaVid", "LightNav-0"),
                    help="models to draw (default: the paper's, LongNav-R1 and LightNav-0)")
    ap.add_argument("--horizon", type=int, nargs="*", default=[],
                    help="1-based layer(s) to mark with a vertical dashed line (default: none)")
    ap.add_argument("--only", nargs="+", default=None, choices=[*FIGURES, *BLIND_FIGURES, *LEGACY_FIGURES],
                    help="figures to draw (default: all but the former current-frame KL figures)")
    args = ap.parse_args()
    root = RESULTS_ROOT if args.dataset == "hm3d" else RESULTS_ROOT_MP3D
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.results / "horizon.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["model"] in args.models]
    curves = [c for c in STYLE if any(r["curve"] == c for r in rows)]

    names = args.only or [*FIGURES, *BLIND_FIGURES]
    for name, (models, normalised) in BLIND_FIGURES.items():
        if name in names and any(m in args.models for m in models):
            draw_blind(args.results, args.out, name, [m for m in models if m in args.models], normalised)
    for name, keys in {**FIGURES, **LEGACY_FIGURES}.items():
        if name not in names:
            continue
        drawn = [c for c in curves if any(c in curves_for(rows, curves, key) for key in keys)]
        if not drawn:
            print(f"[skip] {name}: no data for {', '.join(args.models)}")
            continue
        end = max(int(r["n_layers"]) for r in rows if r["curve"] in drawn) + 1
        pair = len(keys) > 1
        with plt.rc_context(COLUMN_RC):
            fig, axes = plt.subplots(1, len(keys), figsize=PAIR_FIGSIZE if pair else SINGLE_FIGSIZE, squeeze=False)
            axes = axes[0]
            for ax, key in zip(axes, keys):
                draw_panel(ax, rows, curves, key, PANELS[key][0 if pair else 1], end, args.horizon)
            save(fig, axes, args.out / name)
    for stale in args.out.glob("information_horizon.*"):  # the former 4-panel figure
        stale.unlink()
        print("removed stale figure", stale.name)


if __name__ == "__main__":
    main()
