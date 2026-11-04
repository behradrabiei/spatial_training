#!/usr/bin/env python3
"""Figures for the RoboMME memory ablations (run aggregate.py first). Plain matplotlib, the
navigation figures' conventions (large text, legend on top, means only; CIs in summary.csv).

Per model (paper_results/robomme_ablation/<model>/figures/) and combined over the models that
have data (paper_results/robomme_ablation/figures/, one colour per model):

  ctx_ablation.{pdf,png}      SR | steps vs memory frames K (categorical 1,4,8,16,32,full):
                              recency window solid, old frames' timestamps kept dashed (the nav
                              control); the modul uniform / fill side arms are not drawn
  ctx_horizon.{pdf,png}       SR | steps vs window length W in env steps (16..512, full): the
                              native 32-frame sampler inside the window (solid, the nav window),
                              its "old frames' timestamps kept" control (dashed, W/16 real
                              frames) and the decision-frame recency arms (grey dotted)
  information_perf.{pdf,png}  SR | steps vs the first blinded layer (1-based), "none" at 19 as a
                              hollow marker with a dotted reference line
  budget_curves.{pdf,png}     SR | steps vs history tokens kept (frames' worth): one line per
                              selector in the navigation colours, the recency window as black
                              squares at K-1, the unpruned model as a grey dotted line; the
                              combined figure has one row per model

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/robomme_ablation/plot.py [--model all|modul|context] [--suffix _partial]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/robomme_ablation"
MODELS = {  # short name -> (label, colour); results live in RESULTS_ROOT/<short name>/
    "modul": ("FrameSamp+Modul", "tab:purple"),      # both are pi0.5 + the 512-token perceptual memory
    "context": ("FrameSamp+Context", "tab:green"),
}
LARGE_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}
PANELS = {"sr": "Success rate", "steps": "Steps per episode"}
YLIM = {"sr": (0, 1.0), "steps": (0, 1320)}
K_ORDER = ["1", "4", "8", "16", "32", "full"]
W_ORDER = ["16", "32", "64", "128", "256", "512", "full"]
N_LAYERS = 18
PRUNE_STYLE = {
    "random": dict(color="tab:blue", label="Random"),
    "fastv": dict(color="tab:orange", label="FastV"),
    "divprune": dict(color="tab:red", label="DivPrune"),
    "dart": dict(color="tab:purple", label="DART"),
}
RECENCY = dict(color="black", marker="s", markersize=4, linestyle="-", label="recency window (ctx)")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary(model):
    """Arms of <model>/results/summary.csv; an arm still running (n < n_expected) is left out: its
    task mix is biased (tasks run in sequence and differ a lot in SR). None if no summary yet."""
    path = RESULTS_ROOT / model / "results/summary.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if fl(r.get("n")) < fl(r.get("n_expected")):
            print(f"skip {model}/{r['arm']}: {r['n']}/{r['n_expected']} episodes")
            continue
        out[r["arm"]] = r
    return out


def val(summary, arm, metric):
    r = summary.get(arm)
    return fl(r[metric]) if r and r.get(metric, "") != "" else None


def finish_legend(fig, axes, rows=1, ncol=None):
    """Top legend from the first panel; a label used by several models is shown once."""
    handles, labels = [], []
    for h, l in zip(*axes[0].get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
    if not handles:
        return
    # model-specific entries first, then the shared style entries (dashed = kept, grey = side arms)
    order = sorted(range(len(labels)), key=lambda i: 0 if any(m in labels[i] for m, _ in MODELS.values()) else 1)
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    ncol = ncol or -(-len(labels) // rows)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0)


def save(fig, out_dir, name, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}{suffix}.{ext}", dpi=300)
    print("wrote", out_dir / f"{name}{suffix}.png")
    plt.close(fig)


def plot_series(ax, summary, xs_labels, arm_of, metric, style):
    """One line over a categorical x axis; drawn only if one of the series' own arms exists
    (full / k1 are shared by every series)."""
    xs, ys = [], []
    for i, x in enumerate(xs_labels):
        arm = arm_of(x)
        v = val(summary, arm, metric) if arm else None
        if v is not None:
            xs.append(i); ys.append(v)
    if xs and any(arm_of(xs_labels[i]) not in ("full", "k1") for i in xs):
        ax.plot(xs, ys, **style)


# ---------------------------------------------------------------- ctx window
def fig_ctx(models, out_dir, suffix):
    def series(label, color):
        shared = "k1"
        return [(lambda k: "full" if k == "full" else (shared if k == "1" else f"recency_k{k}"),
                 dict(color=color, marker="o", markersize=4, linestyle="-", label=f"{label}, recency window")),
                (lambda k: "full" if k == "full" else ("k1_keep" if k == "1" else f"recency_k{k}_keep"),
                 dict(color=color, marker="o", markersize=4, linestyle="--", label="old frames' timestamps kept"))]
    # the modul-only uniform_k / uniform_k_fill side arms are in summary.csv but not drawn
    # (dropped from the figure 2026-11-04 at the user's request)
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        for ax, metric in zip(axes, PANELS):
            for summary, label, color in models:
                for arm_of, style in series(label, color):
                    plot_series(ax, summary, K_ORDER, arm_of, metric, style)
            ax.set_xticks(range(len(K_ORDER))); ax.set_xticklabels(K_ORDER)
            ax.set_title(PANELS[metric]); ax.grid(True, alpha=0.3); ax.set_ylim(*YLIM[metric])
        finish_legend(fig, axes, rows=2)
        fig.supxlabel("Memory frames K (incl. current)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, out_dir, "ctx_ablation", suffix)


# ---------------------------------------------------------------- window in time
def fig_ctx_horizon(models, out_dir, suffix):
    def decision_arm(w):
        return {"16": "k1", "64": "recency_k4", "128": "recency_k8", "256": "recency_k16",
                "512": "recency_k32", "full": "full"}.get(w)

    def keep_arm(w):  # control: native 32 slots + timestamps, content older than W = cutoff content
        return {"16": "k1_keep", "64": "recency_k4_keep", "128": "recency_k8_keep", "256": "recency_k16_keep",
                "512": "recency_k32_keep", "full": "full"}.get(w)
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        drawn = False
        for ax, metric in zip(axes, PANELS):
            for summary, label, color in models:
                if not any(a.startswith("horizon_w") for a in summary):
                    continue  # side arms exist for the modulation model only
                drawn = True
                plot_series(ax, summary, W_ORDER, lambda w: "full" if w == "full" else f"horizon_w{w}", metric,
                            dict(color=color, marker="o", markersize=4, linestyle="-", label=f"{label}, 32 frames in window"))
                plot_series(ax, summary, W_ORDER, keep_arm, metric,
                            dict(color=color, marker="s", markersize=4, linestyle="--",
                                 label="old frames' timestamps kept (W/16 real frames)"))
                plot_series(ax, summary, W_ORDER, decision_arm, metric,
                            dict(color="grey", marker="^", markersize=4, linestyle=":",
                                 label="decision frames only (W/16 slots)"))
            ax.set_xticks(range(len(W_ORDER))); ax.set_xticklabels(W_ORDER)
            ax.set_title(PANELS[metric]); ax.grid(True, alpha=0.3); ax.set_ylim(*YLIM[metric])
        if not drawn:
            plt.close(fig)
            return
        finish_legend(fig, axes, rows=3)
        fig.supxlabel("Memory window W (env steps, 16 per policy call)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, out_dir, "ctx_horizon", suffix)


# ---------------------------------------------------------------- layer horizon
def fig_horizon(models, out_dir, suffix):
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True)
        for ax, metric in zip(axes, PANELS):
            for summary, label, color in models:
                pts = []
                for arm, meta in sorted(((a, m) for a, m in summary.items() if a.startswith("blind_l")),
                                        key=lambda t: int(t[0][7:])):
                    v = fl(meta[metric])
                    if v == v:
                        pts.append((int(arm[7:]) + 1, v))  # 1-based first blinded layer
                ref = val(summary, "full", metric)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", markersize=3,
                            linestyle="-", label=label)
                if ref is not None:
                    ax.plot([N_LAYERS + 1], [ref], marker="o", markersize=4, markerfacecolor="white", color=color,
                            linestyle="none")
                    ax.axhline(ref, color=color, linestyle=":", linewidth=1.6, alpha=0.85)
            ax.set_xticks([1, 4, 8, 12, 16, N_LAYERS + 1]); ax.set_xticklabels(["1", "4", "8", "12", "16", "L+1"])
            ax.set_xlim(0, N_LAYERS + 2)
            ax.set_title(PANELS[metric]); ax.grid(True, alpha=0.3); ax.set_ylim(*YLIM[metric])
        finish_legend(fig, axes)
        fig.supxlabel(r"First layer without visual tokens $\ell_0$", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out_dir, "information_perf", suffix)


# ---------------------------------------------------------------- token pruning
def fig_budget(models, out_dir, suffix):
    budgets = [0, 1, 2, 4, 8, 16]
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(len(models), 2, figsize=(7.0, 3.0 * len(models) + 0.4), sharex=True, squeeze=False)
        for row, (summary, label, color) in zip(axes, models):
            for ax, metric in zip(row, PANELS):
                for method, st in PRUNE_STYLE.items():
                    xs, ys = [], []
                    for b in budgets:
                        v = val(summary, "k1" if b == 0 else f"{method}_b{b}", metric)
                        if v is not None:
                            xs.append(b); ys.append(v)
                    if xs and any(b > 0 for b in xs):
                        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1, **st)
                # recency window arms at K-1 history frames (as the navigation budget curves)
                xs, ys = [], []
                for k in (1, 4, 8, 16, 32):
                    v = val(summary, "k1" if k == 1 else f"recency_k{k}", metric)
                    if v is not None:
                        xs.append(k - 1); ys.append(v)
                if xs:
                    ax.plot(xs, ys, **RECENCY)
                ref = val(summary, "full", metric)
                if ref is not None:
                    ax.axhline(ref, color="gray", linestyle=":", linewidth=1.6, label="Unpruned (32 frames)")
                ax.set_xticks([0, 1, 2, 4, 8, 16, 31]); ax.set_xlim(-1, 33)
                if ax in axes[0]:
                    ax.set_title(PANELS[metric])
                if len(models) > 1 and metric == "sr":
                    ax.set_ylabel(label, fontsize=12)
                ax.grid(True, alpha=0.3); ax.set_ylim(*YLIM[metric])
        finish_legend(fig, axes[0], rows=2)
        fig.supxlabel("History tokens kept (frames' worth, 16 tokens each)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.16 / len(models)))
    save(fig, out_dir, "budget_curves", suffix)


def draw_all(models, out_dir, suffix):
    fig_ctx(models, out_dir, suffix)
    fig_ctx_horizon(models, out_dir, suffix)
    fig_horizon(models, out_dir, suffix)
    fig_budget(models, out_dir, suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="e.g. _partial while the sweep is running")
    ap.add_argument("--model", choices=sorted(MODELS) + ["all"], default="all",
                    help="one model's figures, or (default) every model with data plus the combined figures")
    args = ap.parse_args()
    names = list(MODELS) if args.model == "all" else [args.model]
    loaded = [(load_summary(m), *MODELS[m], m) for m in names]
    loaded = [t for t in loaded if t[0]]
    for summary, label, color, name in loaded:
        draw_all([(summary, label, color)], RESULTS_ROOT / name / "figures", args.suffix)
    if args.model == "all" and loaded:
        draw_all([t[:3] for t in loaded], RESULTS_ROOT / "figures", args.suffix)


if __name__ == "__main__":
    main()
