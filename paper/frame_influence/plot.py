#!/usr/bin/env python3
"""Figures for the leave-one-frame-out influence experiment (run aggregate.py first). Plain
matplotlib defaults; writes pdf+png to paper_results/frame_influence/figures/. One figure per
question the experiment answers (the first in two renderings):

  influence_vs_age        How fast does a single frame's influence decay with age? Episode-
                          balanced mean KL vs age (log y, age 0 = the current frame) with +-1 SEM
                          across episodes.
  influence_vs_age_box    The same data as box plots: at every other age, the distribution across
                          episodes of each episode's mean influence (the values the curve above
                          averages), one box per model.
  influence_vs_age_box_split  The same boxes, one panel per model side by side (shared y).
  influence_rank_heatmap  Is a frame's importance stable over time? For one episode per model,
                          rows are decisions, columns frames, colour the frame's rank by raw
                          influence within that decision among frames above the model's noise
                          level (white at or below it). Recency is the red band on the diagonal;
                          persistence would be vertical stripes.

Everything else the experiment measures (joint horizon removal, cumulative share, flip rates,
covariates, lag correlations) is a number in results/ and the README table, not a figure.

--measure keep_one draws the same figures for the leave-one-in variant from
paper_results/frame_influence_keepone/ (influence = KL(only frame i || no visual)).

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/frame_influence/plot.py [--measure keep_one]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_ROOTS = {"leave_one": HERE.parents[1] / "paper_results/frame_influence",
                 "keep_one": HERE.parents[1] / "paper_results/frame_influence_keepone"}
STYLE = {  # one color per model; the leak-free LongNav-R1 variant is dashed (same convention as ctx_ablation)
         "LongNav-R1": dict(color="tab:red", marker="o", markersize=4, linestyle="-"),
         "LongNav-R1 (leak-free)": dict(color="tab:red", marker="o", markersize=4, linestyle="--"),
         "Uni-NaVid": dict(color="tab:blue", marker="o", markersize=4, linestyle="-")}
MAX_AGE_PLOT = 30  # the curves have plateaued by then
MARK_EVERY = 2  # markers on every other age keep the curves legible
NOISE_FACTOR = 2.0  # rank heatmap: a frame is above a decision's noise if > this multiple of the row median
FIGURES = ("influence_vs_age", "influence_vs_age_box", "influence_vs_age_box_split", "influence_rank_heatmap")


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def save(fig, out: Path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def average_rank(x):
    """0-based ranks with ties given their average rank. Ties are common here (bf16 quantisation
    makes many ablations identical), and breaking them by index would paint a spurious gradient."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=ranks)[inv] / counts[inv]


def models_in(rows):
    return [m for m in STYLE if any(r["model"] == m for r in rows)]


def age0_episode_means(steps, m):
    """Each episode's mean influence of the current frame (age 0) for model m, from steps.csv."""
    per_ep: dict[str, list[float]] = {}
    for s_ in steps:
        if s_["model"] == m and math.isfinite(fl(s_["age0"])):
            per_ep.setdefault(s_["episode"], []).append(fl(s_["age0"]))
    return np.array([np.mean(v) for v in per_ep.values()])


def figure_age(age_rows, steps, out):
    """Episode-balanced mean influence of removing a single frame vs its age (mean over episodes
    of each episode's mean at that age), with a capped vertical +-1 SEM bar across episodes at every
    marked age: the uncertainty of the mean (figure_age_box shows the spread of the episodes
    themselves). Age 0 is the current frame, balanced the same way from steps.csv. No title, and
    the models' numerical floors are left to the caption."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for m in models_in(age_rows):
        rs = sorted((r for r in age_rows if r["model"] == m and 1 <= int(r["age"]) <= MAX_AGE_PLOT), key=lambda r: int(r["age"]))
        age = [int(r["age"]) for r in rs]; mean = [fl(r["balanced_mean"]) for r in rs]; sem = [fl(r["balanced_sem"]) for r in rs]
        # age 0: per-episode mean of the current frame's influence, then mean and SEM over episodes
        ep_means = age0_episode_means(steps, m)
        if len(ep_means):
            age, mean = [0] + age, [float(ep_means.mean())] + mean
            sem = [float(ep_means.std(ddof=1) / math.sqrt(len(ep_means))) if len(ep_means) > 1 else 0.0] + sem
        mean, sem = np.array(mean), np.nan_to_num(np.array(sem))
        lower = np.minimum(sem, mean - 1e-7)  # keep the lower end positive on the log axis
        style = dict(STYLE[m], markersize=3)  # smaller than the shared default so the error bars read
        ax.errorbar(age, mean, yerr=[lower, sem], label=m, markevery=MARK_EVERY, errorevery=MARK_EVERY,
                    elinewidth=1.2, capsize=2.5, capthick=1.2, **style)
    ax.set_yscale("log")
    ax.set_xlim(-1, MAX_AGE_PLOT + 0.5)
    ax.set_xlabel("Frame age (steps)")
    ax.set_ylabel("Influence (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, out)


def age_box_data(rows, steps, m):
    """Per age, the episodes' mean influence for model m (age 0 from steps.csv)."""
    per_age: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["model"] == m and math.isfinite(fl(r["infl"])) and 1 <= int(r["age"]) <= MAX_AGE_PLOT:
            per_age.setdefault(int(r["age"]), {}).setdefault(r["episode"], []).append(fl(r["infl"]))
    data = {a: np.array([np.mean(v) for v in eps.values()]) for a, eps in per_age.items()}
    data[0] = age0_episode_means(steps, m)
    return data


def draw_age_box(ax, data, m, offset=0.0, width=0.8 * MARK_EVERY):
    """One model's boxes on ax: quartiles with 1.5 IQR whiskers, median line, no fliers. The
    leak-free LongNav-R1 boxes are unfilled and dashed, matching its dashed curve. Returns the
    first box (legend handle) or None."""
    ages = list(range(0, MAX_AGE_PLOT + 1, MARK_EVERY))
    shown = [a for a in ages if len(data.get(a, ()))]
    if not shown:
        return None
    color, dashed = STYLE[m]["color"], STYLE[m]["linestyle"] == "--"
    line = dict(color=color, linewidth=1.0, linestyle="--" if dashed else "-")
    bp = ax.boxplot([data[a] for a in shown], positions=[a + offset for a in shown],
                    widths=0.85 * width, patch_artist=True, manage_ticks=False, showfliers=False,
                    boxprops=dict(facecolor="none" if dashed else color, alpha=None, edgecolor=color,
                                  linewidth=1.0, linestyle="--" if dashed else "-"),
                    whiskerprops=line, capprops=dict(color=color, linewidth=1.0),
                    medianprops=dict(color="black", linewidth=1.2))
    if not dashed:
        for box in bp["boxes"]:
            box.set_facecolor((*plt.matplotlib.colors.to_rgb(color), 0.35))
    return bp["boxes"][0]


def finish_age_box(ax, xlabel=True, ylabel=True):
    ax.set_yscale("log")
    ax.set_xlim(-1.2, MAX_AGE_PLOT + 1.2)
    ax.set_xticks(range(0, MAX_AGE_PLOT + 1, 5))
    if xlabel:
        ax.set_xlabel("Frame age (steps)")
    if ylabel:
        ax.set_ylabel("Influence (nats)")
    ax.grid(True, axis="y", alpha=0.3)


def figure_age_box(rows, steps, out):
    """The data of figure_age as box plots: at every MARK_EVERY-th age, one box per model over the
    episodes' mean influence at that age (the values whose mean and SEM figure_age draws), so the
    box shows how much the episodes disagree rather than the uncertainty of their mean. All models
    share one axis, boxes side by side at each age."""
    models = models_in(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.8 * MARK_EVERY / max(len(models), 1)
    handles, labels = [], []
    for k, m in enumerate(models):
        h = draw_age_box(ax, age_box_data(rows, steps, m), m, offset=(k - (len(models) - 1) / 2) * width, width=width)
        if h is not None:
            handles.append(h); labels.append(m)
    finish_age_box(ax)
    ax.legend(handles, labels, fontsize=7, loc="upper right")
    save(fig, out)


def figure_age_box_split(rows, steps, out):
    """The same boxes as figure_age_box, one panel per model stacked horizontally with a shared
    log y-axis, so each model's spread is read on its own without interleaving."""
    models = models_in(rows)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models) + 0.6, 3.4), sharey=True, squeeze=False)
    for i, (ax, m) in enumerate(zip(axes[0], models)):
        draw_age_box(ax, age_box_data(rows, steps, m), m)
        finish_age_box(ax, ylabel=(i == 0))
        ax.set_title(m, fontsize=9)
    save(fig, out)


def figure_rank_heatmap(rows, steps, summary, episode, out):
    """One episode per model: rows are decisions, columns frames (the diagonal is the current
    frame), colour is the frame's rank of raw influence within that decision among the frames above
    that decision's noise level; cells at or below it are white. Most frames at a decision are noise, so the decision's noise level is
    NOISE_FACTOR times the row's median influence, floored at the model's typical (median) probe
    noise. Recency is the red band on the diagonal; persistence would be vertical stripes."""
    models = models_in(rows)
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 0.8, 3.8), squeeze=False)
    im = None
    for ax, m in zip(axes[0], models):
        d = summary.get(m, {})
        # Typical (median) noise of the model's probe: the mean is inflated by a few very uncertain
        # decisions and would blank out rows whose recent frames clearly stand out within the row.
        noise = float(d.get("replay_floor_kl_median", d.get("merged_control_kl_mean", 0.0)) or 0.0)
        rs = [r for r in rows if r["model"] == m and r["episode"] == episode and math.isfinite(fl(r["infl"]))]
        age0 = {int(s_["step"]): fl(s_["age0"]) for s_ in steps if s_["model"] == m and s_["episode"] == episode}
        ts = sorted({int(r["step"]) for r in rs} | {t for t, v in age0.items() if math.isfinite(v)})
        row_of = {t: k for k, t in enumerate(ts)}
        T = (max(ts) if ts else 0) + 1
        M = np.full((len(ts), T), np.nan)
        for r in rs:
            M[row_of[int(r["step"])], int(r["frame"])] = fl(r["infl"])
        for t, v in age0.items():  # the current frame sits on the diagonal
            if math.isfinite(v):
                M[row_of[t], t] = v
        R = np.full_like(M, np.nan)  # NaN (white) = unseen frame or influence at/below the noise level
        for k in range(M.shape[0]):
            v = M[k]; fin = np.isfinite(v)
            # Most frames in a row are noise, so the row median is the decision's noise level; the
            # model's typical probe noise is a lower bound.
            level = max(noise, NOISE_FACTOR * float(np.median(v[fin])) if fin.sum() >= 4 else 0.0)
            ok = fin & (v > level)
            if ok.sum() >= 2:
                R[k, ok] = average_rank(v[ok]) / (ok.sum() - 1)
            elif ok.sum() == 1:
                R[k, ok] = 1.0
        im = ax.imshow(R, cmap="RdBu_r", origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ticks = np.linspace(0, len(ts) - 1, min(6, len(ts))).astype(int) if ts else []
        ax.set_yticks(ticks); ax.set_yticklabels([str(ts[k]) for k in ticks])
        ax.set_title(m, fontsize=9)
    axes[0][0].set_ylabel("decision (env step t)")
    axes[0][len(models) // 2].set_xlabel("frame i")
    fig.suptitle("Diagonal = current frame; recency is the red band along it; persistence would be vertical "
                 "stripes; white = at or below the noise level", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label="rank of influence within the decision")
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", choices=sorted(RESULTS_ROOTS), default="leave_one",
                    help="leave_one (default) or keep_one (leave-one-in results root)")
    ap.add_argument("--results", type=Path, default=None, help="default <results root>/results")
    ap.add_argument("--out", type=Path, default=None, help="default <results root>/figures")
    ap.add_argument("--episode", help="episode label for the heatmap (default: summary.json's representative episode)")
    args = ap.parse_args()
    root = RESULTS_ROOTS[args.measure]
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    R = args.results
    summary = json.loads((R / "summary.json").read_text())
    steps = read_csv(R / "steps.csv")
    figure_age(read_csv(R / "age_curve.csv"), steps, args.out / "influence_vs_age")
    rows = read_csv(R / "influence.csv.gz")
    figure_age_box(rows, steps, args.out / "influence_vs_age_box")
    figure_age_box_split(rows, steps, args.out / "influence_vs_age_box_split")
    episode = args.episode or summary.get("representative_episode")
    figure_rank_heatmap(rows, steps, summary, episode, args.out / "influence_rank_heatmap")
    stale = [p for p in args.out.glob("influence_*") if p.stem not in FIGURES]
    for p in stale:
        p.unlink()
        print("removed stale figure", p.name)


if __name__ == "__main__":
    main()
