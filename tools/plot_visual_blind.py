#!/usr/bin/env python3
"""Visual-blind arms (vlm.kv_prune_visual_blind) by first blind layer: the horizon test.

For each run the model could not attend to ANY visual token -- history or current frame --
on decoder layers >= L. Plots SR / SPL / mean steps against L with 95% bootstrap CIs over
episodes, the unpruned window as a dashed reference, and the layer-influence horizon curve
(tools/layer_influence.py) alongside when a diagnostic run is given.

Run names follow tools/run_kv_prune_ablation.sh with VISUAL_BLIND=true:
    <prefix>_w<window>_vblind[_l<L>]         (no _l = blind from layer 0)
Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/plot_visual_blind.py --prefix hm3d_v2_100_w32
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SERIES, REF_COLOR, TEXT, MUTED, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#52514e", "#d9d8d3"
METRICS = (("success", "Success rate"), ("spl", "SPL"), ("n_steps", "steps per episode"))
N_BOOT, SEED = 2000, 17


def load_rows(run_dir: Path) -> dict:
    rows = {}
    for path in sorted((run_dir / "rollout").glob("results_*")):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["episode_label"], r)
    return rows


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    boots = rng.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def style(ax):
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="hm3d_v2_100_w32", help="run-name prefix before '_vblind'")
    ap.add_argument("--eval-root", type=Path, default=Path("./dump/longnav_eval"))
    ap.add_argument("--baseline", default="hm3d_v2_100_win32")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    pattern = re.compile(rf"^{re.escape(args.prefix)}_vblind(?:_l(?P<l>\d+))?$")
    arms = []
    for run_dir in sorted(args.eval_root.glob(f"{args.prefix}_vblind*")):
        m = pattern.match(run_dir.name)
        if m and (rows := load_rows(run_dir)):
            arms.append((int(m["l"] or 0), run_dir.name, rows))
    if not arms:
        raise SystemExit(f"no {args.prefix}_vblind* runs under {args.eval_root}")
    arms.sort()
    base = load_rows(args.eval_root / args.baseline)

    print(f"{'first blind layer':>17s} {'n':>4s} {'SR':>6s} {'SPL':>6s} {'steps':>6s}  run")
    stats = []
    for layer, run, rows in arms:
        s = {k: ci([r[k] for r in rows.values()], rng) for k, _ in METRICS}
        stats.append((layer, len(rows), s))
        print(f"{layer:17d} {len(rows):4d} {s['success'][0]:6.3f} {s['spl'][0]:6.3f} "
              f"{s['n_steps'][0]:6.1f}  {run}")
    layers = [t[0] for t in stats] + [args.n_layers]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.9))
    for ax, (key, name) in zip(axes, METRICS):
        mean = [t[2][key][0] for t in stats]
        lo = [t[2][key][0] - t[2][key][1] for t in stats]
        hi = [t[2][key][2] - t[2][key][0] for t in stats]
        ax.errorbar([t[0] for t in stats], mean, yerr=[lo, hi], color=SERIES, marker="o",
                    markersize=6, linewidth=2, capsize=3, label="visual-blind from layer L")
        if base:
            b = ci([r[key] for r in base.values()], rng)
            ax.axhline(b[0], color=REF_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.plot([args.n_layers], [b[0]], marker="o", color=REF_COLOR, markersize=6)
            ax.annotate(f"{args.baseline} (n={len(base)})", (0.01, b[0]),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xticks(layers)
        ax.set_xlabel(f"first visual-blind decoder layer L ({args.n_layers} = none)", color=TEXT)
        ax.set_ylabel(name, color=TEXT)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    n_ep = sorted({t[1] for t in stats})
    fig.suptitle(f"No visual tokens visible from layer L on -- {args.prefix} "
                 f"(n={'/'.join(map(str, n_ep))} episodes, 95% bootstrap CI)", color=TEXT)
    fig.tight_layout()
    out = args.out or args.eval_root / f"{args.prefix}_visual_blind.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
