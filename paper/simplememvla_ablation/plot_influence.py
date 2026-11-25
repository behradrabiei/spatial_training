#!/usr/bin/env python
"""Figure for the SimpleMemVLA velocity-field influence run (needs matplotlib: the `vln` env).

    /home/brabiei/miniconda3/envs/vln/bin/python paper/simplememvla_ablation/plot_influence.py [--suite permanence] [--suffix _partial]

Reads  paper_results/simplememvla_influence/<suite>/results/episode_norm.csv (aggregate_influence.py)
Writes .../<suite>/figures/influence_recent_k{suffix}.{pdf,png}

The counterpart of the navigation `recent_k` figure (paper/frame_influence/plot.py): per K, the
box of the episodes' velocity-field influence with only the newest K frames, as a percentage of
the same episode's influence with the current frame only (K = 1, 100 % by construction), and the
median across episodes drawn through the boxes. Quartile boxes, 1.5 IQR whiskers, outlier points
not drawn. `--set` picks the decisions an episode is averaged over (aggregate_influence.py):
`all` (default; every decision, a window that covers the whole history counts as 0), `cut`
(decisions where the window removes a frame) or `fixed` (more than 32 history frames).
One column of a two-column paper (paper/figstyle.py); include at \\columnwidth without scaling.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figstyle import COLUMN_RC, COLUMN_WIDTH  # noqa: E402  (two-column paper format)

RESULTS_ROOT = HERE.parents[1] / "paper_results/simplememvla_influence"
LABEL, COLOR = "SimpleMemVLA", "tab:orange"
K_LABEL = "Frames in context K"
Y_LABEL = "Normalized L2 (%)"  # must fit the height of a 1.6 in panel
MIN_EPISODES = 5  # a K with fewer episodes in the chosen set is not drawn


def load(root: Path, which: str) -> dict[int, np.ndarray]:
    per_k: dict[int, list[float]] = defaultdict(list)
    for r in csv.DictReader(open(root / "results/episode_norm.csv")):
        if r["set"] == which:
            per_k[int(r["K"])].append(float(r["norm"]))
    return {k: np.array(v) for k, v in sorted(per_k.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="permanence")
    ap.add_argument("--set", default="all", choices=["all", "cut", "fixed"], dest="which")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--root", type=Path, default=None,
                    help="study directory holding results/ (default: paper_results/.../<suite>)")
    ap.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    args = ap.parse_args()
    root = args.root or RESULTS_ROOT / args.suite
    data = {k: v for k, v in load(root, args.which).items() if len(v) >= args.min_episodes}
    ks = list(data)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(COLUMN_RC):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.6))
        pos = list(range(len(ks)))  # K doubles from tick to tick
        line = dict(color=COLOR, linewidth=0.7)
        bp = ax.boxplot([data[k] for k in ks], positions=pos, widths=0.55, patch_artist=True,
                        manage_ticks=False, showfliers=False,
                        boxprops=dict(edgecolor=COLOR, linewidth=0.7), whiskerprops=line, capprops=line,
                        medianprops=dict(color="black", linewidth=0.85))
        for box in bp["boxes"]:
            box.set_facecolor((*matplotlib.colors.to_rgb(COLOR), 0.35))
        ax.plot(pos, [float(np.median(data[k])) for k in ks], color=COLOR, linewidth=1.0,
                marker="o", markersize=3, markevery=[0], zorder=1, label=LABEL)

        def reach(v):  # what is drawn at one K: up to the upper whisker
            q1, q3 = np.percentile(v, [25, 75])
            return float(v[v <= q3 + 1.5 * (q3 - q1)].max())
        ax.set_ylim(0, max([105.0] + [1.05 * reach(v) for v in data.values()]))
        ax.set_xlim(-0.6, len(ks) - 0.4)
        ax.set_xticks(pos, [str(k) for k in ks])
        ax.set_xlabel(K_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_title(LABEL)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            f = out / f"influence_recent_k{args.suffix}.{ext}"
            fig.savefig(f, dpi=300)
            print("wrote", f)
        plt.close(fig)
    for k in ks:
        print(f"K={k:3d}  n_ep={len(data[k]):3d}  median={np.median(data[k]):6.1f} %  mean={data[k].mean():6.1f} %")


if __name__ == "__main__":
    main()
