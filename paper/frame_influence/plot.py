#!/usr/bin/env python3
"""Figures for the leave-one-frame-out influence experiment (run aggregate.py first). Plain
matplotlib defaults; writes pdf+png to paper_results/frame_influence/figures/. One figure per
question the experiment answers (the first in two renderings):

  influence_vs_age        How fast does a single frame's influence decay with age? Episode-
                          balanced mean KL vs age (log y, age 0 = the current frame) with +-1 SEM
                          across episodes. Large text, legend row on top.
  influence_vs_age_box    The same data as box plots: at every other age, the distribution across
                          episodes of each episode's mean influence (the values the curve above
                          averages), one box per model.
  influence_vs_age_box_split  The same boxes, one panel per model side by side (shared y); large text.
  influence_rank_heatmap  Is a frame's importance stable over time? For one episode per model,
                          rows are decisions, columns frames, colour the frame's rank by raw
                          influence within that decision among frames above the model's noise
                          level (white at or below it). Recency is the red band on the diagonal;
                          persistence would be vertical stripes.

Everything else the experiment measures (joint horizon removal, cumulative share, flip rates,
covariates, lag correlations) is a number in results/ and the README table, not a figure.

--measure keep_one draws the same figures for the leave-one-in variant from
paper_results/frame_influence_keepone/ (influence = KL(only frame i || no visual)).

--measure recent_k draws one figure from paper_results/frame_influence_recent/ (user decision
2026-11-20), x = frames in context K instead of frame age:

  recent_frames_normalized_box_split  The share of each model's no-frame KL (K = 0) still open with
                          only the K most recent frames, per episode, on the decisions with at least
                          32 history frames: one panel per model, a box per K across episodes (outlier
                          points not drawn) and the median across episodes as a line through them.
                          Default models: LongNav-R1 (leaky) and LightNav-0 (old blocks kept).

The other views are written only when named with --only (--recent-models picks the models,
default all five there; --tag suffixes the file names):

  recent_frames           How far is the decision made with only the K most recent frames from
                          the full-context one? Episode-balanced mean KL(full || K most recent
                          frames) vs K (log y; K = 0 no frame, K = 1 the current frame only), each
                          K over the decisions with at least K history frames, +-1 SEM across episodes.
  recent_frames_fixed     The same curve on one fixed decision set (the decisions with at least 32
                          history frames), where the shape cannot come from short episodes leaving.
  recent_frames_normalized  The fixed-set curve on a common linear scale: the share of each model's
                          no-frame KL (K = 0) still open with K frames, per episode, +-1 SEM.
  recent_frames_flip      Share of decisions whose top action changes with only the K most recent
                          frames, vs K (episode-balanced mean +-1 SEM; LightNav-0 on its six
                          executed-action classes). Does not depend on the softmax temperature.
  recent_frames_box       The data of recent_frames as box plots: at every other K, the distribution
                          across episodes of each episode's mean, one box per model.
  recent_frames_box_split The same boxes, one panel per model side by side (shared y).

LightNav-0 has two arms there, as in the context-window ablation figure: solid = the K-frame
window (old blocks gone), dashed = old blocks kept (their visual tokens hidden).

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
                 "keep_one": HERE.parents[1] / "paper_results/frame_influence_keepone",
                 "keep_one_full": HERE.parents[1] / "paper_results/frame_influence_keepone_full",
                 "recent_k": HERE.parents[1] / "paper_results/frame_influence_recent"}
# y-axis label per measure; the module-level value is set from --measure in main()
YLABELS = {"leave_one": "Influence (nats)", "keep_one": "Influence (nats)",
           "keep_one_full": "KL(full ‖ only frame) (nats)",
           "recent_k": "KL(full ‖ K recent frames) (nats)"}
YLABEL = YLABELS["leave_one"]
STYLE = {  # one color per model; the leak-free LongNav-R1 variant is dashed (same convention as ctx_ablation)
         "LongNav-R1": dict(color="tab:red", marker="o", markersize=4, linestyle="-"),
         "LongNav-R1 (leak-free)": dict(color="tab:red", marker="o", markersize=4, linestyle="--"),
         "Uni-NaVid": dict(color="tab:blue", marker="o", markersize=4, linestyle="-"),
         # LightNav-0 (keep-one only): KL over the 256 coarse RVQ codes; dashed = the same probes folded
         # onto the six executed-action classes
         "LightNav-0": dict(color="tab:green", marker="o", markersize=4, linestyle="-"),
         "LightNav-0 (6-bin)": dict(color="tab:green", marker="o", markersize=4, linestyle="--"),
         # recent_k: the window arm is "LightNav-0"; dashed = old blocks kept (ctx_ablation's convention)
         "LightNav-0 (old blocks kept)": dict(color="tab:green", marker="o", markersize=4, linestyle="--")}
RECENT_MODELS = ("LongNav-R1", "LongNav-R1 (leak-free)", "Uni-NaVid", "LightNav-0", "LightNav-0 (old blocks kept)")  # the 6-bin folds stay in the CSV
RECENT_MAX_K = 32
# the models of the default recent_k figure (user decision 2026-11-20)
RECENT_PAPER_MODELS = ("LongNav-R1", "LightNav-0 (old blocks kept)")
NORM_YLABEL = "Normalized KL (%)"  # KL(full || K recent frames) as a share of KL(full || no frame): 100 at K = 0
# figure_recent_flip: LightNav-0's argmax flips are read on the six executed-action classes
RECENT_FLIP_SOURCE = {"LightNav-0": "LightNav-0 (6-bin)", "LightNav-0 (old blocks kept)": "LightNav-0 (old blocks kept, 6-bin)"}
MAX_AGE_PLOT = 30  # the curves have plateaued by then
MARK_EVERY = 2  # markers on every other age keep the curves legible
NOISE_FACTOR = 2.0  # rank heatmap: a frame is above a decision's noise if > this multiple of the row median
LARGE_RC = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14}  # paper-figure text sizes
FIGURES = ("influence_vs_age", "influence_vs_age_box", "influence_vs_age_box_split", "influence_rank_heatmap",
           "lightnav_agreement")


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def save(fig, out: Path, layout=True):
    if layout:
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
    with plt.rc_context(LARGE_RC):  # paper figure: large text, legend row on top
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
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
        ax.set_ylabel(YLABEL)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=12, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, layout=False)


def legend_rows(n):
    """Rows of the top legend: one up to three labels, then two columns per row (the long
    LightNav-0 labels do not fit three across)."""
    return 1 if n <= 3 else math.ceil(n / 2)


def recent_figsize(n):
    """figure_age's 6.0 x 3.8, 0.3 in taller per extra legend row."""
    return (6.0, 3.8 + 0.3 * (legend_rows(n) - 1))


def top_legend(fig, ax, top=0.9):
    """The legend above the axes (figure_age's row); more rows when the labels would not fit one."""
    handles, labels = ax.get_legend_handles_labels()
    rows = legend_rows(len(labels))
    fig.legend(handles, labels, loc="upper center", ncol=math.ceil(len(labels) / rows), fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, top - 0.07 * (rows - 1)))


def recent_models(rows):
    return [m for m in RECENT_MODELS if any(r["model"] == m for r in rows)]


def figure_recent(curve_rows, out, max_k=RECENT_MAX_K, fixed=False):
    """Episode-balanced mean KL(full || only the K most recent frames) vs K, +-1 SEM across
    episodes at every marked K (each K over the decisions with at least K history frames; K = 0 is
    the no-visual decision, K = 1 the current frame alone). Same look as figure_age. ``fixed``
    draws the fixed-set columns instead: one decision set at every K (the decisions with at least
    32 history frames), so the shape is within-decision and free of the short episodes leaving."""
    mean_key, sem_key = ("fixed_balanced_mean", "fixed_balanced_sem") if fixed else ("balanced_mean", "balanced_sem")
    with plt.rc_context(LARGE_RC):
        models = recent_models(curve_rows)
        fig, ax = plt.subplots(figsize=recent_figsize(len(models)))
        for m in models:
            rs = sorted((r for r in curve_rows if r["model"] == m and int(r["K"]) <= max_k
                         and math.isfinite(fl(r[mean_key]))), key=lambda r: int(r["K"]))
            k = [int(r["K"]) for r in rs]
            mean = np.array([fl(r[mean_key]) for r in rs])
            sem = np.nan_to_num(np.array([fl(r[sem_key]) for r in rs]))
            lower = np.minimum(sem, mean - 1e-7)  # keep the lower end positive on the log axis
            marked = [i for i, x in enumerate(k) if x % MARK_EVERY == 0]  # markers and bars at every other K
            off = np.array([i not in marked for i in range(len(k))])
            style = dict(STYLE[m], markersize=3)
            ax.errorbar(k, mean, yerr=[np.where(off, np.nan, lower), np.where(off, np.nan, sem)], label=m, markevery=marked,
                        elinewidth=1.2, capsize=2.5, capthick=1.2, **style)
        ax.set_yscale("log")
        ax.set_xlim(-1, max_k + 0.5)
        ax.set_xticks(range(0, max_k + 1, 5))
        ax.set_xlabel("Frames in context K")
        ax.set_ylabel(YLABEL)
        ax.grid(True, alpha=0.3)
        top_legend(fig, ax)
    save(fig, out, layout=False)


def figure_recent_norm(curve_rows, out, max_k=RECENT_MAX_K):
    """figure_recent_fixed on a common linear scale: the share of each model's no-frame KL that is
    still open with the K most recent frames (per episode, mean KL at K over the mean KL(full || no
    visual) of the same decisions -- those with at least 32 history frames --, then mean +-1 SEM
    over episodes). 100 % at K = 0 by construction; LightNav-0's window arm has no K = 0 prompt
    and starts at K = 1, against the same decisions' no-visual KL."""
    with plt.rc_context(LARGE_RC):
        models = recent_models(curve_rows)
        fig, ax = plt.subplots(figsize=recent_figsize(len(models)))
        for m in models:
            rs = sorted((r for r in curve_rows if r["model"] == m and int(r["K"]) <= max_k
                         and math.isfinite(fl(r.get("norm_balanced_mean")))), key=lambda r: int(r["K"]))
            k = [int(r["K"]) for r in rs]
            mean = 100 * np.array([fl(r["norm_balanced_mean"]) for r in rs])
            sem = 100 * np.nan_to_num(np.array([fl(r["norm_balanced_sem"]) for r in rs]))
            marked = [i for i, x in enumerate(k) if x % MARK_EVERY == 0]
            off = np.array([i not in marked for i in range(len(k))])
            ax.errorbar(k, mean, yerr=np.where(off, np.nan, sem), label=m, markevery=marked,
                        elinewidth=1.2, capsize=2.5, capthick=1.2, **dict(STYLE[m], markersize=3))
        ax.set_xlim(-1, max_k + 0.5)
        ax.set_ylim(0, 105)
        ax.set_xticks(range(0, max_k + 1, 5))
        ax.set_xlabel("Frames in context K")
        ax.set_ylabel(NORM_YLABEL)
        ax.grid(True, alpha=0.3)
        top_legend(fig, ax)
    save(fig, out, layout=False)


def figure_recent_flip(curve_rows, out, max_k=RECENT_MAX_K):
    """Share of decisions whose top action changes with only the K most recent frames in context,
    vs K: episode-balanced mean +-1 SEM, same look and decision sets as figure_recent. Free of the
    softmax temperature (the argmax is). LongNav-R1 and Uni-NaVid: argmax over their four actions;
    LightNav-0: over its six executed-action classes (RECENT_FLIP_SOURCE), so that a move between
    two of the 256 codes that decode to the same action does not count as a change."""
    with plt.rc_context(LARGE_RC):
        models = [m for m in RECENT_MODELS if any(r["model"] == RECENT_FLIP_SOURCE.get(m, m) for r in curve_rows)]
        fig, ax = plt.subplots(figsize=recent_figsize(len(models)))
        for m in models:
            src = RECENT_FLIP_SOURCE.get(m, m)
            rs = sorted((r for r in curve_rows if r["model"] == src and int(r["K"]) <= max_k
                         and math.isfinite(fl(r.get("flip_balanced")))), key=lambda r: int(r["K"]))
            k = [int(r["K"]) for r in rs]
            mean = 100 * np.array([fl(r["flip_balanced"]) for r in rs])
            sem = 100 * np.nan_to_num(np.array([fl(r["flip_balanced_sem"]) for r in rs]))
            marked = [i for i, x in enumerate(k) if x % MARK_EVERY == 0]
            off = np.array([i not in marked for i in range(len(k))])
            ax.errorbar(k, mean, yerr=np.where(off, np.nan, sem), label=m, markevery=marked,
                        elinewidth=1.2, capsize=2.5, capthick=1.2, **dict(STYLE[m], markersize=3))
        ax.set_xlim(-1, max_k + 0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, max_k + 1, 5))
        ax.set_xlabel("Frames in context K")
        ax.set_ylabel("Top action changes (% of decisions)")
        ax.grid(True, alpha=0.3)
        top_legend(fig, ax)
    save(fig, out, layout=False)


def recent_box_data(steps, m, max_k=RECENT_MAX_K):
    """Per K, the episodes' mean KL(full || K most recent frames) for model m, from steps.csv
    (each K over the decisions with at least K history frames, as the curve)."""
    data = {}
    for k in range(0, max_k + 1):
        per_ep: dict[str, list[float]] = {}
        for s_ in steps:
            v = fl(s_.get(f"horizon_{k}"))
            if s_["model"] == m and math.isfinite(v) and int(s_["n_history"]) >= k:
                per_ep.setdefault(s_["episode"], []).append(v)
        data[k] = np.array([np.mean(v) for v in per_ep.values()])
    return data


def figure_recent_box(steps, out, max_k=RECENT_MAX_K):
    """figure_age_box for the recent-window measure: at every other K, one box per model over
    the episodes' mean (the values whose mean and SEM figure_recent draws)."""
    models = recent_models(steps)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.8 * MARK_EVERY / max(len(models), 1)
    handles, labels = [], []
    for i, m in enumerate(models):
        h = draw_age_box(ax, recent_box_data(steps, m, max_k), m, offset=(i - (len(models) - 1) / 2) * width, width=width, max_x=max_k)
        if h is not None:
            handles.append(h); labels.append(m)
    finish_age_box(ax, max_x=max_k, xlabel_text="Frames in context K")
    ax.legend(handles, labels, fontsize=7, loc="upper right")
    save(fig, out)


def recent_norm_box_data(steps, m, max_k=RECENT_MAX_K, fixed_min=32):
    """Per K, each episode's share (%) of its no-frame KL still open with K frames: mean KL(full ||
    K most recent frames) over mean KL(full || no visual) on the same decisions, those with at
    least ``fixed_min`` history frames -- the values whose mean and SEM figure_recent_norm draws.
    LightNav-0's window rows carry the no-visual KL in ``visual_all`` (they have no K = 0 band)."""
    data = {}
    for k in range(0, max_k + 1):
        per_ep: dict[str, list[tuple[float, float]]] = {}
        for s_ in steps:
            if s_["model"] != m or int(s_["n_history"]) < fixed_min:
                continue
            v, v0 = fl(s_.get(f"horizon_{k}")), fl(s_.get("visual_all"))
            if math.isfinite(v) and math.isfinite(v0):
                per_ep.setdefault(s_["episode"], []).append((v, v0))
        data[k] = np.array([100 * np.mean([a for a, _ in v]) / np.mean([b for _, b in v]) for v in per_ep.values()])
    return data


def figure_recent_box_split(steps, out, max_k=RECENT_MAX_K, normalized=False):
    """figure_age_box_split for the recent-window measure: one panel per model, shared log y.
    ``normalized`` draws the boxes of figure_recent_norm instead (each episode's share of its
    no-frame KL still open with K frames, linear %, fixed set), with the median across episodes
    (the bar of each box, here at every K) drawn through them. The median, not figure_recent_norm's
    mean: one episode far above the rest (leak-free LongNav-R1's 500-step one) lifts the mean out of
    the boxes, whose outlier points are not drawn."""
    models = recent_models(steps)
    with plt.rc_context(LARGE_RC):
        fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models) + 0.6, 3.6), sharey=True, squeeze=False)
        for i, (ax, m) in enumerate(zip(axes[0], models)):
            if normalized:
                data = recent_norm_box_data(steps, m, max_k)
                draw_age_box(ax, data, m, max_x=max_k)
                # the episode median at every K through the boxes: every episode is exactly 100 % at
                # K = 0, so that box has no height and only the line shows the drop
                ks = [k for k in sorted(data) if len(data[k])]
                ax.plot(ks, [float(np.median(data[k])) for k in ks], color=STYLE[m]["color"], linestyle=STYLE[m]["linestyle"],
                        linewidth=1.2, marker="o", markersize=4, markevery=[0], zorder=1)
                finish_age_box(ax, xlabel=False, ylabel=False, max_x=max_k)
                ax.set_yscale("linear")
                def reach(v):  # what is drawn at one K: up to the upper whisker (1.5 IQR, fliers hidden)
                    q1, q3 = np.percentile(v, [25, 75])
                    return float(v[v <= q3 + 1.5 * (q3 - q1)].max())
                top = max([105.0] + [1.05 * reach(v) for v in data.values() if len(v)])
                ax.set_ylim(0, max(top, ax.get_ylim()[1] if i else 0))  # shared y: an episode can exceed 100 %
                if i == 0:
                    ax.set_ylabel(NORM_YLABEL)
            else:
                draw_age_box(ax, recent_box_data(steps, m, max_k), m, max_x=max_k)
                finish_age_box(ax, xlabel=False, ylabel=(i == 0), max_x=max_k)
            ax.set_title(m)
        fig.supxlabel("Frames in context K", fontsize=14)
        fig.tight_layout()
    save(fig, out, layout=False)


def age_box_data(rows, steps, m):
    """Per age, the episodes' mean influence for model m (age 0 from steps.csv)."""
    per_age: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["model"] == m and math.isfinite(fl(r["infl"])) and 1 <= int(r["age"]) <= MAX_AGE_PLOT:
            per_age.setdefault(int(r["age"]), {}).setdefault(r["episode"], []).append(fl(r["infl"]))
    data = {a: np.array([np.mean(v) for v in eps.values()]) for a, eps in per_age.items()}
    data[0] = age0_episode_means(steps, m)
    return data


def draw_age_box(ax, data, m, offset=0.0, width=0.8 * MARK_EVERY, max_x=MAX_AGE_PLOT):
    """One model's boxes on ax: quartiles with 1.5 IQR whiskers, median line, no fliers. The
    leak-free LongNav-R1 boxes are unfilled and dashed, matching its dashed curve. Returns the
    first box (legend handle) or None."""
    ages = list(range(0, max_x + 1, MARK_EVERY))
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


def finish_age_box(ax, xlabel=True, ylabel=True, max_x=MAX_AGE_PLOT, xlabel_text="Frame age (steps)"):
    ax.set_yscale("log")
    ax.set_xlim(-1.2, max_x + 1.2)
    ax.set_xticks(range(0, max_x + 1, 5))
    if xlabel:
        ax.set_xlabel(xlabel_text)
    if ylabel:
        ax.set_ylabel(YLABEL)
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
    with plt.rc_context(LARGE_RC):  # paper figure: large text, one shared x label
        fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models) + 0.6, 3.6), sharey=True, squeeze=False)
        for i, (ax, m) in enumerate(zip(axes[0], models)):
            draw_age_box(ax, age_box_data(rows, steps, m), m)
            finish_age_box(ax, xlabel=False, ylabel=(i == 0))
            ax.set_title(m)
        fig.supxlabel("Frame age (steps)", fontsize=14)
        fig.tight_layout()
    save(fig, out, layout=False)


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


def figure_lightnav_agreement(episode_rows, out):
    """LightNav-0 (keep-one): a lone tubelet scored against the FULL-context decision instead of
    the no-visual prior, vs its age. One box per age over the 10 episodes' agreement rates (how
    often the only-tubelet argmax over the six executed-action classes equals the decision's
    argmax): quartiles, 1.5 IQR whiskers, median line, no fliers. Steady-state ages only (even:
    the current tubelet holds frames t-1 and t); the forward/turn split, 256-code agreement and
    KL(full || only tubelet) stay in lightnav_agreement.csv."""
    data: dict[int, list[float]] = {}
    for r in episode_rows:
        a = int(r["age"])
        if a % 2 == 0 and a <= MAX_AGE_PLOT + 2 and math.isfinite(fl(r["agree_bin"])):
            data.setdefault(a, []).append(fl(r["agree_bin"]))
    if not data:
        return
    ages = sorted(data)
    color = STYLE["LightNav-0"]["color"]
    with plt.rc_context(LARGE_RC):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        line = dict(color=color, linewidth=1.0)
        ax.boxplot([data[a] for a in ages], positions=ages, widths=1.4, patch_artist=True, manage_ticks=False,
                   showfliers=False, boxprops=dict(facecolor=(*plt.matplotlib.colors.to_rgb(color), 0.35),
                                                   edgecolor=color, linewidth=1.0),
                   whiskerprops=line, capprops=line, medianprops=dict(color="black", linewidth=1.2))
        ax.set_ylim(0, 1); ax.set_xlim(-1.5, MAX_AGE_PLOT + 3.5)
        ax.set_xticks(range(0, MAX_AGE_PLOT + 3, 5))
        ax.set_xlabel("Tubelet age (steps)"); ax.set_ylabel("Agreement with full decision")
        ax.grid(True, axis="y", alpha=0.3)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", choices=sorted(RESULTS_ROOTS), default="leave_one",
                    help="leave_one (default), keep_one (leave-one-in results root), keep_one_full (KL(full || only frame)) "
                         "or recent_k (KL(full || K most recent frames); recent_frames_normalized_box_split)")
    ap.add_argument("--results", type=Path, default=None, help="default <results root>/results")
    ap.add_argument("--out", type=Path, default=None, help="default <results root>/figures")
    ap.add_argument("--recent-models", help="recent_k: comma-separated subset of the models to draw "
                                            "(default: RECENT_PAPER_MODELS for the default figure, all with --only)")
    ap.add_argument("--only", help="recent_k: comma-separated figure names to write "
                                   "(default: recent_frames_normalized_box_split only)")
    ap.add_argument("--tag", help="recent_k: suffix for the file names, so a subset does not overwrite the standard set")
    ap.add_argument("--episode", help="episode label for the heatmap (default: summary.json's representative episode)")
    args = ap.parse_args()
    global YLABEL
    YLABEL = YLABELS[args.measure]
    root = RESULTS_ROOTS[args.measure]
    args.results = args.results or root / "results"
    args.out = args.out or root / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    R = args.results
    if args.measure == "recent_k":
        global RECENT_MODELS
        if args.recent_models:
            chosen = [m.strip() for m in args.recent_models.split(",")]
            unknown = [m for m in chosen if m not in RECENT_MODELS]
            if unknown:
                ap.error(f"--recent-models: unknown {unknown}; choose from {list(RECENT_MODELS)}")
            RECENT_MODELS = tuple(m for m in RECENT_MODELS if m in chosen)
        elif not args.only:  # the paper figure
            RECENT_MODELS = RECENT_PAPER_MODELS
        curve, steps = read_csv(R / "recent_curve.csv"), read_csv(R / "steps.csv")
        figures = {  # name -> draw(out); only the last is written by default (user decision 2026-11-20)
            "recent_frames": lambda o: figure_recent(curve, o),
            "recent_frames_fixed": lambda o: figure_recent(curve, o, fixed=True),
            "recent_frames_normalized": lambda o: figure_recent_norm(curve, o),
            "recent_frames_flip": lambda o: figure_recent_flip(curve, o),
            "recent_frames_box": lambda o: figure_recent_box(steps, o),
            "recent_frames_box_split": lambda o: figure_recent_box_split(steps, o),
            "recent_frames_normalized_box_split": lambda o: figure_recent_box_split(steps, o, normalized=True),
        }
        names = [n.strip() for n in args.only.split(",")] if args.only else ["recent_frames_normalized_box_split"]
        unknown = [n for n in names if n not in figures]
        if unknown:
            ap.error(f"--only: unknown {unknown}; choose from {list(figures)}")
        for n in names:
            figures[n](args.out / (f"{n}_{args.tag}" if args.tag else n))
        return
    summary = json.loads((R / "summary.json").read_text())
    steps = read_csv(R / "steps.csv")
    figure_age(read_csv(R / "age_curve.csv"), steps, args.out / "influence_vs_age")
    rows = read_csv(R / "influence.csv.gz")
    figure_age_box(rows, steps, args.out / "influence_vs_age_box")
    figure_age_box_split(rows, steps, args.out / "influence_vs_age_box_split")
    if args.measure != "keep_one_full":  # no heatmap for the full-reference measure (user decision 2026-11-07)
        episode = args.episode or summary.get("representative_episode")
        figure_rank_heatmap(rows, steps, summary, episode, args.out / "influence_rank_heatmap")
    else:
        for p in args.out.glob("influence_rank_heatmap.*"):
            p.unlink()
            print("removed", p.name)
    if (R / "lightnav_agreement_episodes.csv").exists():  # keep_one with the LightNav-0 arm
        figure_lightnav_agreement(read_csv(R / "lightnav_agreement_episodes.csv"), args.out / "lightnav_agreement")
    stale = [p for p in args.out.glob("influence_*") if p.stem not in FIGURES]
    for p in stale:
        p.unlink()
        print("removed stale figure", p.name)


if __name__ == "__main__":
    main()
