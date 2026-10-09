#!/usr/bin/env python3
"""Figures of the selector-criterion vs KL probe (run aggregate.py and showcase_frames.py first).
Plain matplotlib defaults; writes to paper_results/selector_kl/figures/ (the budgeted mass-kept statistics stay in results/summary.csv, no figure):

  showcase.{pdf,png}   one decision, one row: the observation, then the per-token keep-one-in KL,
                       FastV's layer-0 attention, DivPrune's nearest-neighbour distance and DART's
                       pivot dissimilarity, each smoothed onto the greyscale frame as in
                       paper/layer_token_info (own colour scale per panel), with the criterion's
                       Spearman rho vs the KL in its title. The episode, decision, action and
                       frame total are printed for the caption.
  spearman.{pdf,png}   one panel per model (LongNav-R1 | Uni-NaVid): per method, the distribution
                       over decisions of the Spearman rho between its criterion and the keep-one-in
                       KL; box = quartiles, whiskers = 5-95 %, dot = pooled mean.

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/selector_kl/plot.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from longnav.analysis import selector as sel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/selector_kl"
ARM = "r1_selector_probe"
COLORS = {"random": "tab:blue", "attn": "tab:orange", "divprune": "tab:red", "dart": "tab:purple"}  # paper/token_prune


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TI_PLOT = load_by_path("lti_plot", ROOT / "paper/layer_token_info/plot.py")  # to_grid, overlay, greyscale, CMAP


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def save(fig, out: Path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf/.png")


# --------------------------------------------------------------------------- showcase


def action_name(rec):
    """The action word of the decision, from the run's saved config (falls back to the id)."""
    try:
        import yaml
        cfg = yaml.safe_load((RESULTS_ROOT / "runs" / ARM / "config.yaml").read_text())
        return str(cfg["rollout"]["action_space"][int(rec["action"])])
    except Exception:
        return str(rec["action"])


CRITERIA = ((sel.MEASURE, "keep-one-in KL", None),
            ("attn", "FastV: attention", "attn"),
            ("divprune_nn", "DivPrune: NN distance", "divprune"),
            ("dart_dis", "DART: pivot dissimilarity", "dart"))


def showcase(rec, rgb, out):
    """One row: the observation, then the keep-one-in KL and each criterion as a per-token map on
    the greyscale frame (own colour scale each, 99.5th percentile), rho vs the KL in the title."""
    grid = lambda v: TI_PLOT.to_grid(np.asarray(v, dtype=np.float64), rec["raster"], rec["grid_thw"])
    crit = sel.criteria(rec)
    y = rec[sel.MEASURE]
    aspect = rgb.shape[0] / rgb.shape[1]
    n_panels = 1 + len(CRITERIA)
    fig = plt.figure(figsize=(2.3 * n_panels + 0.4, 2.3 * aspect + 0.45))
    widths = [1] + [1, 0.32] * len(CRITERIA)  # room for an inset colorbar, its two labels and a clear gap to the next panel
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("observation", fontsize=9)
    for j, (key, label, method) in enumerate(CRITERIA):
        ax = fig.add_subplot(gs[0, 1 + 2 * j])
        g = grid(rec[key])
        finite = g[np.isfinite(g)]
        lo = 0.0 if method is None else float(finite.min())  # KL from zero, a criterion from its own floor
        vmax = max(float(np.percentile(finite, TI_PLOT.VMAX_PCT)), lo + 1e-12)
        im = TI_PLOT.overlay(ax, rgb, g, lo, vmax)
        title = label if method is None else f"{label}\nρ = {sel.spearman(crit[method], y):.2f}"
        ax.set_title(title, fontsize=9)
        # the colorbar as an inset so it follows the image axes' (aspect-adjusted) height;
        # the gridspec's slim spacer column just reserves the room for it
        cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
        cb = fig.colorbar(im, cax=cax, ticks=[lo, vmax])
        lo_txt = "0" if lo < 1e-3 * vmax else f"{lo:.2g}"
        cb.ax.set_yticklabels([lo_txt, f"{vmax:.2g}"], fontsize=5)
        cb.ax.tick_params(length=2, pad=1)
        labels = cb.ax.get_yticklabels()
        labels[0].set_va("bottom"); labels[1].set_va("top")  # keep the end labels inside the bar's height
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf/.png  ({rec['episode']} decision {rec['step'] + 1}, action {action_name(rec)}, "
          f"{rec['n_frames']} frames, keep-one-in total {float(np.sum(y)):.2g} nats)")


# --------------------------------------------------------------------------- statistics


MODELS = ("LongNav-R1", "Uni-NaVid")


def spearman_figure(dec_rows, summary, out):
    """One panel per model present in the data: per method the per-decision rho distribution."""
    measure = sel.MEASURE
    models = [m for m in MODELS if any(r.get("model", "LongNav-R1") == m for r in dec_rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 2.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        rows = [r for r in dec_rows if r.get("model", "LongNav-R1") == model and r["measure"] == measure and r["valid"] == "1"]
        data = []
        for i, m in enumerate(sel.METHODS):
            v = np.array([fl(r[f"rho_{m}"]) for r in rows])
            v = v[np.isfinite(v)]
            data.append(v)
            s = next((r for r in summary if (r.get("model", "LongNav-R1"), r["measure"], r["method"], r["stat"])
                      == (model, measure, m, "rho")), None)
            if s is not None:
                ax.plot([i + 1], [fl(s["mean"])], "o", color=COLORS[m], markersize=4, zorder=3)  # pooled mean
        bp = ax.boxplot(data, positions=range(1, len(data) + 1), widths=0.55, whis=(5, 95), showfliers=False,
                        medianprops=dict(color="0.3"), patch_artist=True)
        for patch, m in zip(bp["boxes"], sel.METHODS):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.25)
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([sel.LABELS[m] for m in sel.METHODS], fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Spearman ρ vs keep-one-in KL")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--showcase", help="LABEL:STEP (0-based); default results/showcase.json")
    args = ap.parse_args()
    res, figs = RESULTS_ROOT / "results", RESULTS_ROOT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summary = read_csv(res / "summary.csv")
    dec_rows = read_csv(res / "per_decision.csv")
    spearman_figure(dec_rows, summary, figs / "spearman")

    if args.showcase:
        label, step = args.showcase.rsplit(":", 1); step = int(step)
    else:
        sc = json.loads((res / "showcase.json").read_text())
        label, step = sc["episode_label"], int(sc["step"])
    frame = RESULTS_ROOT / "runs/showcase" / label / "frames" / f"step_{step:03d}.png"
    if not frame.exists():
        print(f"skip showcase: {frame} missing (run showcase_frames.py)")
        return
    recs = [r for r in sel.load_r1_selector(RESULTS_ROOT / "runs" / ARM) if r["episode"] == label and r["step"] == step]
    if not recs:
        raise SystemExit(f"no probe row for {label} step {step}")
    rgb = np.asarray(Image.open(frame).convert("RGB"))
    showcase(recs[0], rgb, figs / "showcase")


if __name__ == "__main__":
    main()
