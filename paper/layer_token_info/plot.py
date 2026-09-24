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

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/layer_token_info/plot.py [--episode L] [--layers a b c]
    /home/brabiei/miniconda3/envs/vln/bin/python paper/layer_token_info/plot.py --study nohist --layers 1 12 18
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parents[1] / "paper_results/layer_token_info"
ARMS = (("mask", "leaky"), ("recompute", "leakfree"))
MEASURES = (("keep_cur", "keepone", "keep-one-in KL (current frame hidden as reference)"),
            ("leave", "leaveone", "leave-one-out KL"),
            ("keep_all", "keepone_allvisual", "keep-one-in KL (every visual token hidden as reference)"),
            ("leave_nohist", "leaveone_nohist", "leave-one-out KL (history visual tokens hidden)"))
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


def figure(res: dict, measure: str, ylabel: str, rgb: np.ndarray, layers_b: list[int], out: Path):
    layers = res["layers"].astype(int)
    X = res[measure].astype(np.float64)  # [L, N]
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
    ax.set_yscale("log")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(f"per-token {ylabel.split(' (')[0]} (nats)")
    ax.set_xticks(layers[::3])
    ax.set_xticklabels([str(l + 1) for l in layers[::3]])  # 1-based on the axis (data stays 0-based)
    for l in layers_b:
        ax.axvline(l, color="0.6", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.3)
    ax.set_title("(a) information across layers", fontsize=10, loc="left")
    # (b) the raw observation, then the smoothed map of each chosen layer on the greyscale frame
    grids = [to_grid(X[list(layers).index(l)], res["raster"], res["grid_thw"]) for l in layers_b]
    vmax = float(np.nanpercentile(np.concatenate([g.ravel() for g in grids]), VMAX_PCT))
    vmax = max(vmax, FLOOR)
    axr = fig.add_subplot(gs[0, 1])
    axr.imshow(rgb)
    axr.set_xticks([])
    axr.set_yticks([])
    axr.set_title("(b) observation", fontsize=9)
    im = None
    heat_axes = []
    for j, (l, g) in enumerate(zip(layers_b, grids)):
        axb = fig.add_subplot(gs[0, 2 + j])
        im = overlay(axb, rgb, g, 0.0, vmax)
        heat_axes.append(axb)
        total = float(np.nansum(g))  # leave-one-out sums sit at 1e-4..1e-2: keep two significant digits
        axb.set_title(f"Layer {l + 1}   (sum {total:.2f} nats)" if total >= 0.1 or total == 0
                      else f"Layer {l + 1}   (sum {total:.1e} nats)", fontsize=9)  # 1-based label
    cb = fig.colorbar(im, ax=heat_axes, fraction=0.04, pad=0.02)
    cb.set_label("KL (nats)")
    goal = str(res["goal"]) if "goal" in res else ""
    action = str(res["action"]) if "action" in res else ""
    step = f"decision {int(res['steps'])}    |    " if "steps" in res else ""
    fig.suptitle(f"Goal: {goal}    |    {step}chosen action: {action}    |    {ARM_LABEL[str(res['arm'])]}, {ylabel}",
                 fontsize=10, y=1.04)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_file(path: Path, rgb: np.ndarray, out_stem: str, fig_dir: Path, fixed_layers, study: bool) -> list[dict]:
    """Every measure held by one probe file -> one figure each; returns the layer-summary rows."""
    res = load_npz(path)
    arm = str(res["arm"])
    arm_tag = dict(ARMS)[arm]
    layers = res["layers"].astype(int)
    rows = []
    for measure, tag, ylabel in MEASURES:
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
        figure(res, measure, ylabel, rgb, chosen, out)
        horizon = set(signal_layers(X))
        for i, l in enumerate(layers):
            mean = float(X[i].mean())
            row = {"arm": arm, "measure": measure, "layer": int(l), "mean": mean, "sum": float(X[i].sum()),
                   "median": float(np.median(X[i])), "max": float(X[i].max()),
                   "cv": float(X[i].std() / mean) if mean > 0 else float("nan"),
                   "text_cur_kl": float(res["text_cur_kl"][i]), "text_all_kl": float(res["text_all_kl"][i]),
                   "text_hist_kl": float(res["text_hist_kl"][i]) if "text_hist_kl" in res else float("nan"),
                   "before_horizon": int(i in horizon), "chosen": int(int(l) in chosen)}
            rows.append(row)
        print(f"{path.parent.name}/{path.stem}/{measure}: layers {chosen} -> {out.name}.pdf/.png")
    return rows


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
    args = ap.parse_args()
    root = RESULTS_ROOT / args.study if args.study else RESULTS_ROOT
    runs = root / "runs"
    if args.study:
        episodes = [runs / args.episode] if args.episode else sorted(p for p in runs.iterdir() if p.is_dir())
        for ep in episodes:
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
