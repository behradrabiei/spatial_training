#!/usr/bin/env python3
"""Standalone vs unique visual information by layer: reads `vlm.kv_prune_log_keep_one=true` runs.

For each scored step t, probe layer i and cached frame f the worker logged, with every hiding
applied on decoder layers >= i:
    keep_one[t][i][f]  = KL( P with only frame f's visual slots || P with no visual slots )
    leave_one[t][i][f] = KL( P full || P without frame f's visual slots )
The first is the CVPR'26 "token information" measure (arXiv 2512.07580, Eq. 6) read as a
distribution change -- how much frame f says ON ITS OWN; the second is what the `kl` selector
ranks -- how much frame f says that NO OTHER frame says. Their ratio is the frame's
uniqueness; the spread of keep_one across frames at a layer is the paper's Figure-7 statistic
(information becoming uniform before it vanishes).

Reports per probe layer and frame-age bin (0 = current frame): mean keep-one, mean
leave-one, uniqueness = mean leave / mean keep, and the across-frame coefficient of variation
of keep-one. Frames whose visual tokens were all sparse-filtered (kv_visual_new == 0) are
excluded. Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/keep_one_influence.py dump/longnav_eval/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGE_BINS = (("current (age 0)", 0, 0), ("age 1", 1, 1), ("age 2-7", 2, 7), ("age 8+", 8, 10 ** 9))
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # one per age bin, fixed order
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
N_BOOT, SEED, EPS = 2000, 17, 1e-9


def probe_layers(run_dir: Path, override):
    if override:
        return [int(x) for x in override.split(",")]
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return [int(x) for x in cfg["vlm"]["kv_prune_layer_influence_starts"]]


def load_pairs(run_dir: Path, n_probes: int):
    """Flat records (episode, t, age, probe_idx, keep, leave) over scored steps and live frames."""
    records = []
    n_episodes = 0
    for seq in sorted(run_dir.glob("rollout/*/*/sequence.json")):
        d = json.load(seq.open())
        keep_rows, leave_rows = d.get("sup/keep_one_influence"), d.get("sup/leave_one_influence")
        if not keep_rows or not leave_rows:
            continue
        n_episodes += 1
        vis_new = d.get("kv_visual_new") or []
        for t, (keep, leave) in enumerate(zip(keep_rows, leave_rows)):
            if len(keep) != n_probes or len(leave) != n_probes:
                continue  # skipped step (stride) or malformed
            for f in range(t + 1):
                if f < len(vis_new) and vis_new[f] == 0:
                    continue
                for j in range(n_probes):
                    if f < len(keep[j]) and f < len(leave[j]):
                        records.append((n_episodes - 1, t, t - f, j, keep[j][f], leave[j][f]))
    return np.array(records, dtype=float), n_episodes


def boot_mean(values, rng):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def style(ax, xlabel, ylabel, probes):
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_xticks(probes)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--probes", default=None, help="comma-separated probe layers (default: config.yaml)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    probes = probe_layers(args.run_dir, args.probes)
    rec, n_episodes = load_pairs(args.run_dir, len(probes))
    if rec.size == 0:
        raise SystemExit(f"no sup/keep_one_influence rows under {args.run_dir}/rollout")
    rng = np.random.default_rng(SEED)
    ep, t, age, probe, keep, leave = rec.T
    n_steps = len({(int(e), int(s)) for e, s in zip(ep, t)})
    print(f"{n_episodes} episodes, {n_steps} scored steps, {rec.shape[0] // len(probes)} (step, frame) "
          f"pairs, probes {probes}")

    curves = {}  # (bin label, stat) -> per-probe list
    print(f"\n{'age bin':>16s} {'probe':>5s} {'keep-one':>9s} {'leave-one':>10s} {'unique':>7s} {'CV keep':>8s} {'n':>6s}")
    for label, lo, hi in AGE_BINS:
        sel_bin = (age >= lo) & (age <= hi)
        for j, layer in enumerate(probes):
            sel = sel_bin & (probe == j)
            k_mean = boot_mean(keep[sel], rng)
            l_mean = boot_mean(leave[sel], rng)
            unique = l_mean[0] / max(k_mean[0], EPS) if np.isfinite(k_mean[0]) else np.nan
            # across-frame spread of keep-one within a step (paper Fig. 7): CV per step, averaged
            cvs = []
            for key in {(int(e), int(s)) for e, s in zip(ep[sel], t[sel])}:
                m = sel & (ep == key[0]) & (t == key[1])
                if m.sum() >= 2 and keep[m].mean() > EPS:
                    cvs.append(keep[m].std() / keep[m].mean())
            cv = float(np.mean(cvs)) if cvs else np.nan
            curves[(label, "keep")] = curves.get((label, "keep"), []) + [k_mean]
            curves[(label, "leave")] = curves.get((label, "leave"), []) + [l_mean]
            curves[(label, "unique")] = curves.get((label, "unique"), []) + [unique]
            curves[(label, "cv")] = curves.get((label, "cv"), []) + [cv]
            print(f"{label:>16s} {layer:5d} {k_mean[0]:9.5f} {l_mean[0]:10.5f} {unique:7.3f} {cv:8.3f} {int(sel.sum()):6d}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = ((axes[0][0], "keep", "keep-one-in KL (standalone information)", True),
              (axes[0][1], "leave", "leave-one-out KL (unique information)", True),
              (axes[1][0], "unique", "uniqueness = leave-one / keep-one", False),
              (axes[1][1], "cv", "across-frame CV of keep-one (paper Fig. 7)", False))
    for ax, stat, title, has_ci in panels:
        for (label, _, _), color in zip(AGE_BINS, SERIES_COLORS):
            vals = curves[(label, stat)]
            if has_ci:
                mean = [v[0] for v in vals]
                lo = [v[0] - v[1] for v in vals]
                hi = [v[2] - v[0] for v in vals]
                ax.errorbar(probes, mean, yerr=[lo, hi], color=color, marker="o", markersize=6,
                            linewidth=2, capsize=3, label=label)
            else:
                ax.plot(probes, vals, color=color, marker="o", markersize=6, linewidth=2, label=label)
        if stat == "unique":
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylim(bottom=0)
        if stat in ("keep", "leave"):
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        style(ax, "first hidden decoder layer i", "", probes)
    fig.suptitle(f"Standalone vs unique visual information by layer -- {args.run_dir.name} "
                 f"({n_episodes} episodes, {n_steps} scored steps)", color=TEXT)
    fig.tight_layout()
    out = args.out or args.run_dir / "keep_one_influence.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
