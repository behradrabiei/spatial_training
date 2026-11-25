#!/usr/bin/env python3
"""Figures for the layer-wise visual-token information experiment (run probe_episode.py first).

Per arm (leaky = mask, leak-free = recompute) and measure (keep-one-in with the current frame
as reference, leave-one-out, keep-one-in with every visual token hidden as reference) one
two-part figure in the layout of Fig. 7 of arXiv 2512.07580:

  (a) box plot of the per-token KL (nats) of the current frame's visual tokens at every decoder
      layer -- the spread across tokens is the figure's statistic (information becoming uniform
      before it vanishes);
  (b) the raw current observation, then the same frame in greyscale with the smoothed per-token
      map of three layers overlaid on a shared colour scale (greyscale so the colour is the map).

Layers for (b): --layers a b c, else two of the most mutually dissimilar maps before the
information horizon (the largest consecutive drop of the layers' top-5% token mean; greedy
farthest-point over unit-max-normalised maps, seeded at the layer with the largest spatial
coefficient of variation, picks at least 6 layers apart) plus the first layer past the horizon,
so the panel shows the structure and its vanishing. Output
goes to figures/<episode>/ and results/<episode>/. Plain matplotlib defaults. Layer indices are
0-based everywhere in the data, --layers and layer_summary.csv (the probe's convention); only the
figure shows them 1-based (box-plot axis and map titles), so --layers 1 12 18 is titled 2 / 13 / 19.

--study NAME plots a second protocol kept under paper_results/layer_token_info/NAME/ (see
probe_episode.py --study): every runs/<episode>/token_info_<arm>_t<step>.npz found there (or the
--episode / --steps subset), whatever measures each file holds, to
figures/<episode>/t<step>_<arm>_<measure>.{pdf,png} and results/<episode>/layer_summary_t<step>.csv.
The `nohist` study holds the two measures of the current frame WITHOUT visual history (every
history frame's visual tokens hidden; the past actions stay): keep_all and leave_nohist, drawn
at the fixed layers 1 / 12 / 18 (--layers 1 12 18).
The `keepfull` study (2026-11-23) holds keep_full = KL(P_full || P_only token k visible, every other
visual token hidden), the per-token counterpart of the blind-from-layer KL of paper/info_horizon.
Its figure differs from the others: (a) is the box plot of keep_full with the "no token kept" value
KL(P_full || P_no visual from this layer on) -- the depth curve at this decision -- as a line, and
(b) maps how much of that gap each token closes alone, no-token-kept minus keep_full (nats; a token
that moves the decision further from the full one is negative and drawn as zero). That figure (and
the block one below) is drawn in the paper's two-column format (paper/figstyle.py): the full text
width of a figure*, 7.16 x 1.42 in at final size, 8 pt labels and 7 pt ticks, no header (the goal and
the chosen action sit under the observation), each map titled by its first hidden layer, TrueType
fonts in the PDF; the box plot has one box per layer and no reference line. The September figures keep their original format. With --overview
it also writes figures/<episode>/overview_<arm>_keepone_full.{pdf,png}: the gap and
the best token's share of it at every decision, and one tile per decision with the share of the gap
each token closes at one layer (--overview-layer, default the first of --layers or 1); it needs
at least five probed decisions and also reads runs/<episode>/sparse_layers/ (every-step probes at
a few layers, which get no per-decision figure).
The `leaveblock` study (2026-11-23) holds leave_b<n> = KL(P_full || P_the n x n block of patches
centred on the token hidden), everything else visible: leave-one-out for a region instead of one
redundant patch. Panel (a) carries the whole-current-frame-hidden KL as a line (the limit of a
block that covers the frame); the maps are the block KL in nats, placed at the block's centre.

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/layer_token_info/plot.py [--episode L] [--layers a b c]
    /home/brabiei/miniconda3/envs/vln/bin/python paper/layer_token_info/plot.py --study nohist --layers 1 12 18
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, TEXT_WIDTH  # noqa: E402  (two-column paper format)

RESULTS_ROOT = HERE.parents[1] / "paper_results/layer_token_info"
PAPER_HEIGHT = 1.42  # in: the keep_full / block figures, drawn at full text width (figure*)
ARMS = (("mask", "leaky"), ("recompute", "leakfree"))
MEASURES = (("keep_cur", "keepone", "keep-one-in KL (current frame hidden as reference)"),
            ("leave", "leaveone", "leave-one-out KL"),
            ("keep_all", "keepone_allvisual", "keep-one-in KL (every visual token hidden as reference)"),
            ("leave_nohist", "leaveone_nohist", "leave-one-out KL (history visual tokens hidden)"),
            ("keep_full", "keepone_full", "keep-one-in KL to the full context (every other visual token hidden)"))
# names inside a --study: the same "allvisual" naming as the headline figures; leave_nohist is the
# leave-one-out counterpart of keep_all (the history's visual tokens hidden in both forwards)
STUDY_MEASURES = {"keep_all": ("keepone_allvisual", "keep-one-in KL (every visual token hidden as reference)"),
                  "leave_nohist": ("leaveone_allvisual", "leave-one-out KL (every history visual token hidden)")}
ARM_LABEL = {"mask": "leaky (decision-replay mask)", "recompute": "leak-free (full re-prefill)"}
FLOOR = 1e-6  # log-axis floor for zero KLs
MIN_GAP = 6  # layers between the two signal-layer picks (maps within the signal range are alike)
VMAX_PCT = 99.5  # shared colour scale of panel (b): the top ~4 of 900 cells saturate
# upper half of jet: zero information is green, as on the paper's diverging scale, then yellow, red
CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "jet_upper", plt.get_cmap("jet")(np.linspace(0.5, 1.0, 256)))
ALPHA = 0.75  # overlay opacity at the top of the colour scale
ALPHA_MIN = 0.0  # opacity at zero information: the greyscale frame stays grey where there is none


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def to_grid(values: np.ndarray, raster: np.ndarray, grid_thw) -> np.ndarray:
    """Scatter per-token values into the merged (h//2, w//2) grid; missing tokens are NaN."""
    gh, gw = int(grid_thw[1]) // 2, int(grid_thw[2]) // 2
    g = np.full(gh * gw, np.nan, dtype=np.float64)
    g[raster] = values
    return g.reshape(gh, gw)


def layer_strength(maps: np.ndarray) -> np.ndarray:
    """Per-layer strength: mean of the top 5 % of tokens (keep-one maps are dense, leave-one maps
    are a few hot tokens over near-zeros; a floor of equal small values must not count)."""
    finite = np.nan_to_num(maps, nan=0.0)
    k = max(1, int(round(0.05 * finite.shape[1])))
    return np.sort(finite, axis=1)[:, -k:].mean(axis=1)


def horizon_index(maps: np.ndarray) -> int:
    """Index of the last layer before the information horizon: the largest drop of the layer
    strength between consecutive layers (log ratio, floored)."""
    s = np.maximum(layer_strength(maps), FLOOR)
    drops = np.log(s[:-1]) - np.log(s[1:])
    # The last layer feeds the head directly, so a drop into it is not a horizon.
    return int(np.argmax(drops[:-1])) if len(drops) > 1 else 0


def signal_layers(maps: np.ndarray) -> np.ndarray:
    """Indices of the layers up to and including the horizon."""
    return np.arange(horizon_index(maps) + 1)


def pick_layers(maps: np.ndarray, layers: np.ndarray, k: int = 3) -> list[int]:
    """Two of the most dissimilar maps before the information horizon (greedy farthest-point
    over unit-max-normalised maps, seeded at the largest spatial CV, >= MIN_GAP apart) plus the
    first layer past the horizon, so the panel shows where the information sits and its
    vanishing. Falls back to farthest-point over all layers when there is no layer past it."""
    finite = np.nan_to_num(maps, nan=0.0)
    peak = finite.max(axis=1, keepdims=True)
    norm = np.where(peak > 0, finite / np.maximum(peak, 1e-30), 0.0)
    mean = finite.mean(axis=1)
    cv = np.where(mean > 0, finite.std(axis=1) / np.maximum(mean, 1e-30), 0.0)
    dist = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=-1)
    sig = list(signal_layers(maps))
    horizon = max(sig)
    post = [i for i in range(len(layers)) if i > horizon]
    pool = sig if post else list(range(len(layers)))
    n_pool = k - 1 if post else k
    order = [max(pool, key=lambda i: cv[i])]
    while len(order) < n_pool:
        cands = [i for i in pool if i not in order
                 and all(abs(int(layers[i]) - int(layers[j])) >= MIN_GAP for j in order)]
        if not cands:
            cands = [i for i in pool if i not in order]
        if not cands:
            break
        order.append(max(cands, key=lambda i: min(dist[i, j] for j in order)))
    if post:
        order.append(post[0])
    return sorted(int(layers[i]) for i in order)


def upsample_smooth(grid: np.ndarray, H: int, W: int, vmax: float) -> np.ndarray:
    """Bilinear upsampling followed by a Gaussian blur of half a cell (the look of the paper's
    maps). Values are clipped to the colour scale BEFORE blurring, so a token far above the
    scale cannot bleed into a larger blob than a normal hot cell; missing tokens count as 0."""
    from scipy.ndimage import gaussian_filter, zoom
    gh, gw = grid.shape
    g = np.clip(np.nan_to_num(grid, nan=0.0), 0.0, vmax)
    up = zoom(g, (H / gh, W / gw), order=1)[:H, :W]
    return gaussian_filter(up, sigma=0.5 * min(H / gh, W / gw))


def greyscale(rgb: np.ndarray) -> np.ndarray:
    """Luminance of the frame, replicated to three channels."""
    y = np.asarray(Image.fromarray(rgb).convert("L"))
    return np.repeat(y[..., None], 3, axis=-1)


def overlay(ax, rgb: np.ndarray, grid: np.ndarray, vmin: float, vmax: float):
    """Heat overlay on the greyscale frame, whose opacity follows the value: a faint tint
    (ALPHA_MIN) at vmin, ALPHA at vmax, so the scene shows through wherever a token carries no
    information and every colour in the panel belongs to the map."""
    ax.imshow(greyscale(rgb))
    up = upsample_smooth(grid, rgb.shape[0], rgb.shape[1], vmax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(CMAP)(norm(np.nan_to_num(up, nan=vmin)))
    frac = np.clip((np.nan_to_num(up, nan=vmin) - vmin) / max(vmax - vmin, 1e-30), 0, 1)
    rgba[..., 3] = ALPHA_MIN + (ALPHA - ALPHA_MIN) * frac  # a faint tint at zero, ALPHA at vmax
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    # an invisible mappable carrying the colour scale for the colorbar
    return plt.cm.ScalarMappable(norm=norm, cmap=CMAP)


def figure(res: dict, measure: str, ylabel: str, rgb: np.ndarray, layers_b: list[int], out: Path,
           maps: np.ndarray | None = None, ref: np.ndarray | None = None, ref_label: str = "",
           cb_label: str = "KL (nats)", box_label: str | None = None, concise: bool = False,
           vmax: float | None = None):
    """``maps`` [L, N]: values drawn in panel (b) when they are not the box-plot values (keep_full:
    the gap closed by each token); ``ref`` [L]: a per-layer reference line in panel (a).
    ``concise``: the paper format (see figure_paper)."""
    layers = res["layers"].astype(int)
    X = res[measure].astype(np.float64)  # [L, N]
    M = X if maps is None else np.asarray(maps, dtype=np.float64)
    if concise:
        figure_paper(res, X, M, rgb, layers_b, out, ref, ref_label, cb_label, box_label or "KL (nats)", vmax)
        return
    n_img = 1 + len(layers_b)
    # every panel, the box plot included, has the frame's aspect and the same width
    aspect = rgb.shape[0] / rgb.shape[1]
    fig = plt.figure(figsize=(2.8 * (1 + n_img), 2.8 * aspect + 0.6))
    gs = fig.add_gridspec(1, 1 + n_img, wspace=0.08)
    # (a) box plot per layer
    ax = fig.add_subplot(gs[0, 0])
    ax.set_box_aspect(aspect)
    data = [np.maximum(X[i], FLOOR) for i in range(len(layers))]
    ax.boxplot(data, positions=layers, widths=0.6, whis=(0, 100), showfliers=False,
               medianprops=dict(color="tab:red"))  # whiskers = full range: the few hot tokens matter
    if ref is not None:
        ax.plot(layers, np.maximum(ref, FLOOR), color="k", linewidth=1.0, marker=".", markersize=3,
                label=ref_label, zorder=3)
        ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_yscale("log")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(box_label or f"per-token {ylabel.split(' (')[0]} (nats)")
    ticks = layers if len(layers) <= 8 else layers[::3]  # a sparse probe: label every probed layer
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(l + 1) for l in ticks])  # 1-based on the axis (data stays 0-based)
    for l in layers_b:
        ax.axvline(l, color="0.6", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.3)
    if not concise:
        ax.set_title("(a) information across layers", fontsize=10, loc="left")
    # (b) the raw observation, then the smoothed map of each chosen layer on the greyscale frame
    grids = [to_grid(M[list(layers).index(l)], res["raster"], res["grid_thw"]) for l in layers_b]
    vmax = float(np.nanpercentile(np.concatenate([g.ravel() for g in grids]), VMAX_PCT))
    vmax = max(vmax, FLOOR)
    axr = fig.add_subplot(gs[0, 1])
    axr.imshow(rgb)
    axr.set_xticks([])
    axr.set_yticks([])
    axr.set_title("Observation" if concise else "(b) observation", fontsize=9)
    im = None
    heat_axes = []
    for j, (l, g) in enumerate(zip(layers_b, grids)):
        axb = fig.add_subplot(gs[0, 2 + j])
        im = overlay(axb, rgb, g, 0.0, vmax)
        heat_axes.append(axb)
        if concise:  # only the layer: lone-token gaps and overlapping blocks do not add up to anything
            axb.set_title(f"Layer {l + 1}", fontsize=9)
            continue
        total = float(np.nansum(g))  # leave-one-out sums sit at 1e-4..1e-2: keep two significant digits
        axb.set_title(f"Layer {l + 1}   (sum {total:.2f} nats)" if total >= 0.1 or total == 0
                      else f"Layer {l + 1}   (sum {total:.1e} nats)", fontsize=9)  # 1-based label
    cb = fig.colorbar(im, ax=heat_axes, fraction=0.04, pad=0.02)
    cb.set_label(cb_label)
    goal = str(res["goal"]) if "goal" in res else ""
    action = str(res["action"]) if "action" in res else ""
    step = f"decision {int(res['steps'])}    |    " if "steps" in res else ""
    if concise:
        fig.suptitle(f"Goal: {goal}    |    step {int(res['steps'])}    |    action: {action}", fontsize=10, y=1.02)
    else:
        fig.suptitle(f"Goal: {goal}    |    {step}chosen action: {action}    |    {ARM_LABEL[str(res['arm'])]}, {ylabel}",
                     fontsize=10, y=1.04)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure_paper(res: dict, X: np.ndarray, M: np.ndarray, rgb: np.ndarray, layers_b: list[int], out: Path,
                 ref, ref_label: str, cb_label: str, box_label: str, vmax: float | None = None):
    """The same two parts at the full text width of a two-column paper, drawn at final size
    (paper/figstyle.py): box plot of ``X`` per first hidden layer with the reference line, the
    observation, and the maps of ``M`` at ``layers_b`` on one colour scale. Layers are 1-based on
    the figure (the data stays 0-based)."""
    layers = res["layers"].astype(int)
    aspect = rgb.shape[0] / rgb.shape[1]
    n_img = 1 + len(layers_b)
    thin = dict(linewidth=0.5)
    with plt.rc_context(COLUMN_RC):
        fig = plt.figure(figsize=(TEXT_WIDTH, PAPER_HEIGHT), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
        gs = fig.add_gridspec(1, 2 + n_img, width_ratios=[1.0] * (1 + n_img) + [0.2])  # last: the colour bar
        ax = fig.add_subplot(gs[0, 0])
        ax.set_box_aspect(aspect)
        # one box per probed layer and no reference line (user, 2026-11-23); ``ref`` stays in the
        # layer summary CSV (text_all_kl / text_cur_kl)
        dense = len(layers) > 8
        ax.boxplot([np.maximum(X[i], FLOOR) for i in range(len(layers))], positions=layers, widths=0.6,
                   whis=(0, 100), showfliers=False, boxprops=thin, whiskerprops=thin, capprops=thin,
                   medianprops=dict(color="tab:red", linewidth=0.9))
        ax.set_yscale("log")
        ticks = np.array([l for l in (0, 6, 12, 18, 24) if l in layers]) if dense else layers
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(l + 1) for l in ticks])
        ax.set_xlim(layers.min() - 1, layers.max() + 1)
        ax.set_xlabel(r"First hidden layer $\ell_0$", labelpad=1)
        ax.set_ylabel(box_label, labelpad=1)
        ax.grid(alpha=0.3)
        grids = [to_grid(M[list(layers).index(l)], res["raster"], res["grid_thw"]) for l in layers_b]
        if vmax is None:
            vmax = max(float(np.nanpercentile(np.concatenate([g.ravel() for g in grids]), VMAX_PCT)), FLOOR)
        axr = fig.add_subplot(gs[0, 1])
        axr.imshow(rgb)
        axr.set_xticks([])
        axr.set_yticks([])
        axr.set_title("Observation", pad=2)
        if "goal" in res:
            axr.set_xlabel(f"Goal: {str(res['goal']).replace('_', ' ')}\nAction: {res['action']}", fontsize=7, labelpad=2)
        im, heat_axes = None, []
        for j, (l, g) in enumerate(zip(layers_b, grids)):
            axb = fig.add_subplot(gs[0, 2 + j])
            im = overlay(axb, rgb, g, 0.0, vmax)
            axb.set_title(rf"$\ell_0 = {l + 1}$", pad=2)
            heat_axes.append(axb)
        pad = fig.add_subplot(gs[0, 2 + len(layers_b)])  # room for the bar, which hangs off the last map
        pad.axis("off")
        cb = fig.colorbar(im, cax=heat_axes[-1].inset_axes([1.05, 0.0, 0.06, 1.0]))  # as tall as the maps
        cb.set_label(cb_label, labelpad=2)
        cb.outline.set_linewidth(0.5)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_file(path: Path, rgb: np.ndarray, out_stem: str, fig_dir: Path, fixed_layers, study: bool) -> list[dict]:
    """Every measure held by one probe file -> one figure each; returns the layer-summary rows."""
    res = load_npz(path)
    arm = str(res["arm"])
    arm_tag = dict(ARMS)[arm]
    layers = res["layers"].astype(int)
    rows = []
    blocks = sorted((int(k[len("leave_b"):]), k) for k in res if k.startswith("leave_b") and k[len("leave_b"):].isdigit())
    block_measures = tuple((k, f"leaveblock{n}", f"leave-block-out KL ({n}x{n} patches hidden)") for n, k in blocks)
    for measure, tag, ylabel in MEASURES + block_measures:
        if measure not in res:
            continue
        if study:
            tag, ylabel = STUDY_MEASURES.get(measure, (tag, ylabel))
        X = res[measure].astype(np.float64)
        chosen = [l for l in fixed_layers if l in layers] if fixed_layers else pick_layers(X, layers)
        if not chosen:
            print(f"skip {path.name}/{measure}: none of the layers {fixed_layers} was probed")
            continue
        out = fig_dir / out_stem.format(arm=arm_tag, tag=tag)
        extra = {}
        if measure == "keep_full":  # panel (b): the part of the no-token-kept gap each token closes alone
            blind = res["text_all_kl"].astype(np.float64)
            extra = dict(maps=blind[:, None] - X, ref=blind, ref_label="no token kept",
                         cb_label="KL reduction", box_label="KL$(P_{\\mathrm{full}}\\,\\|\\,P_k)$ (nats)", concise=True)
        if measure.startswith("leave_b"):  # the limit of a growing block: the whole current frame hidden
            extra = dict(ref=res["text_cur_kl"].astype(np.float64), ref_label="whole frame hidden",
                         box_label="KL to full context (nats)", concise=True)
        figure(res, measure, ylabel, rgb, chosen, out, **extra)
        horizon = set(signal_layers(X))
        for i, l in enumerate(layers):
            mean = float(X[i].mean())
            row = {"arm": arm, "measure": measure, "layer": int(l), "mean": mean, "sum": float(X[i].sum()),
                   "median": float(np.median(X[i])), "max": float(X[i].max()),
                   "cv": float(X[i].std() / mean) if mean > 0 else float("nan"),
                   "text_cur_kl": float(res["text_cur_kl"][i]), "text_all_kl": float(res["text_all_kl"][i]),
                   "text_hist_kl": float(res["text_hist_kl"][i]) if "text_hist_kl" in res else float("nan"),
                   "before_horizon": int(i in horizon), "chosen": int(int(l) in chosen)}
            if measure == "keep_full":  # the gap closed, nothing-kept minus keep_full, per token
                gain = float(res["text_all_kl"][i]) - X[i]
                row |= {"gain_max": float(gain.max()), "gain_median": float(np.median(gain)),
                        "gain_min": float(gain.min()), "gain_positive_frac": float((gain > 0).mean())}
            rows.append(row)
        print(f"{path.parent.name}/{path.stem}/{measure}: layers {chosen} -> {out.name}.pdf/.png")
    return rows


GAP_FLOOR = 1e-3  # nats: below this the no-token-kept gap is noise and a share of it means nothing
OVERVIEW_COLS = 8


def overview(ep: Path, fig_dir: Path, layer: int, arm: str = "mask") -> None:
    """keepfull study, the whole episode on one page: per decision the no-token-kept KL at ``layer``
    (the gap) and the best token's share of it, then one tile per decision with the share of the gap
    each token closes alone (0..1, negatives drawn as zero; blank where the gap is below GAP_FLOOR)."""
    items = []
    # every-step probes at a few layers live in sparse_layers/; a dense file of the same step wins
    paths = {p.name: p for d in (ep / "sparse_layers", ep) for p in sorted(d.glob(f"token_info_{arm}_t[0-9]*.npz"))}
    for path in paths.values():
        res = load_npz(path)
        layers = list(res["layers"].astype(int))
        if "keep_full" not in res or layer not in layers:
            continue
        li = layers.index(layer)
        step = int(res["steps"])
        gap = float(res["text_all_kl"][li])
        share = (gap - res["keep_full"][li].astype(np.float64)) / max(gap, 1e-30)
        rgb = np.asarray(Image.open(ep / "frames" / f"step_{step - 1:02d}.png").convert("RGB"))
        items.append(dict(step=step, action=str(res["action"]), gap=gap, share=share, rgb=rgb,
                          grid=to_grid(share, res["raster"], res["grid_thw"]), goal=str(res["goal"])))
    if len(items) < 5:  # a couple of probed decisions are not an episode
        return
    items.sort(key=lambda d: d["step"])
    n = len(items)
    cols = min(OVERVIEW_COLS, n)
    rows = -(-n // cols)
    aspect = items[0]["rgb"].shape[0] / items[0]["rgb"].shape[1]
    tile = 1.7
    fig = plt.figure(figsize=(tile * cols + 0.6, 3.4 + rows * (tile * aspect + 0.3)))
    # two series rows, a spacer for their x labels, then the tiles
    gs = fig.add_gridspec(3 + rows, cols, height_ratios=[1.0, 1.0, 0.55] + [tile * aspect + 0.3] * rows,
                          hspace=0.3, wspace=0.05)
    steps = np.array([d["step"] for d in items])
    gaps = np.array([d["gap"] for d in items])
    ok = gaps >= GAP_FLOOR
    best = np.array([float(d["share"].max()) for d in items])
    half = np.array([float((d["share"] > 0.5).mean()) for d in items])
    worse = np.array([float((d["share"] < 0).mean()) for d in items])
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(steps, gaps, color="0.45", width=0.7)
    ax1.set_ylabel("gap (nats)")
    ax1.set_xticks(steps)
    ax1.set_xticklabels([])
    ax1.set_xlim(steps[0] - 0.6, steps[-1] + 0.6)
    ax1.grid(alpha=0.3, axis="y")
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(steps[ok], best[ok], color="tab:red", marker="o", markersize=3, label="best token")
    ax2.plot(steps[ok], half[ok], color="tab:blue", marker="o", markersize=3, label="tokens closing > 50 %")
    ax2.plot(steps[ok], worse[ok], color="0.5", marker="o", markersize=3, label="tokens making it worse")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("share")
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"{d['step']}\n{d['action'][0].upper()}" for d in items], fontsize=7)
    ax2.set_xlim(steps[0] - 0.6, steps[-1] + 0.6)
    ax2.set_xlabel("step and action (L left, R right, F forward, S stop)", fontsize=8)
    ax2.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, borderaxespad=0.1)
    ax2.grid(alpha=0.3)
    im, tiles = None, []
    for i, d in enumerate(items):
        ax = fig.add_subplot(gs[3 + i // cols, i % cols])
        if d["gap"] >= GAP_FLOOR:
            im = overlay(ax, d["rgb"], d["grid"], 0.0, 1.0)
        else:  # nothing to close: the frame alone
            ax.imshow(greyscale(d["rgb"]))
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"{d['step']}  {d['action']}", fontsize=8, pad=2)
        tiles.append(ax)
    if im is not None:
        cb = fig.colorbar(im, ax=tiles, fraction=0.015, pad=0.01)
        cb.set_label("share of the gap closed")
    fig.suptitle(f"Goal: {items[0]['goal']}    |    layer {layer + 1}", fontsize=10, y=0.995)
    out = fig_dir / f"overview_{dict(ARMS)[arm]}_keepone_full"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"{ep.name}: overview of {n} decisions at layer {layer} -> {out.name}.pdf/.png")


def write_summary(rows: list[dict], path: Path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", help="default: runs/selected.json (every episode with --study)")
    ap.add_argument("--layers", nargs=3, type=int, help="the three layers of panel (b), all figures")
    ap.add_argument("--study", help="plot paper_results/layer_token_info/<study>/ (per-step probe files)")
    ap.add_argument("--steps", nargs="*", type=int, help="with --study: only these decisions")
    ap.add_argument("--overview-layer", type=int, help="with --study: layer (0-based) of the whole-episode "
                                                       "keep_full overview (default: the first of --layers, else 1)")
    ap.add_argument("--overview", action="store_true", help="with --study: also draw the whole-episode overview")
    ap.add_argument("--overview-only", action="store_true", help="with --study: the overview without the per-decision figures")
    args = ap.parse_args()
    root = RESULTS_ROOT / args.study if args.study else RESULTS_ROOT
    runs = root / "runs"
    if args.study:
        episodes = [runs / args.episode] if args.episode else sorted(p for p in runs.iterdir() if p.is_dir())
        for ep in episodes:
            ov_layer = args.overview_layer if args.overview_layer is not None else (args.layers[0] if args.layers else 1)
            (root / "figures" / ep.name).mkdir(parents=True, exist_ok=True)
            for arm, _ in ARMS if args.overview or args.overview_only else ():
                overview(ep, root / "figures" / ep.name, ov_layer, arm)
            if args.overview_only:
                continue
            by_step = {}
            for path in sorted(ep.glob("token_info_*_t[0-9]*.npz")):
                step = int(path.stem.rsplit("_t", 1)[1])
                if not args.steps or step in args.steps:
                    by_step.setdefault(step, []).append(path)
            for step, paths in sorted(by_step.items()):
                fig_dir = root / "figures" / ep.name
                res_dir = root / "results" / ep.name
                fig_dir.mkdir(parents=True, exist_ok=True)
                res_dir.mkdir(parents=True, exist_ok=True)
                rgb = np.asarray(Image.open(ep / "frames" / f"step_{step - 1:02d}.png").convert("RGB"))
                rows = []
                for path in paths:
                    rows += plot_file(path, rgb, f"t{step:03d}_{{arm}}_{{tag}}", fig_dir, args.layers, True)
                write_summary(rows, res_dir / f"layer_summary_t{step:03d}.csv")
        return
    label = args.episode or json.loads((runs / "selected.json").read_text())["episode_label"]
    ep = runs / label
    fig_dir = root / "figures" / label
    res_dir = root / "results" / label
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    traj = json.loads((ep / "trajectory.json").read_text())
    rgb = np.asarray(Image.open(ep / "frames" / f"step_{traj['steps'] - 1:02d}.png").convert("RGB"))
    rows = []
    for arm, _ in ARMS:
        path = ep / f"token_info_{arm}.npz"
        if not path.exists():
            print(f"skip {arm}: {path} missing")
            continue
        rows += plot_file(path, rgb, "token_info_{arm}_{tag}", fig_dir, args.layers, False)
    write_summary(rows, res_dir / "layer_summary.csv")


if __name__ == "__main__":
    main()
