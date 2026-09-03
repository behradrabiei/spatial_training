#!/usr/bin/env python3
"""KV-prune selectors vs the layer they prune at (tools/run_kv_prune_layer_sweep.sh runs).

One column per KV budget; rows are success rate, SPL and the layer-mean resident cache
length (the memory-equivalent kv_len). X axis: vlm.kv_prune_layer_start (0 = every layer,
i.e. the uniform arms). Points are means over the shared episode split with a 95%
bootstrap CI over episodes; dashed lines are unpruned references.

Run names are discovered under --eval-root as
    <prefix>_<method>_w<window>[_l<start>[-<end>]]_b<budget>
(the naming of tools/run_kv_prune_ablation.sh). Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_prune_layers.py --prefix kvprune_layer
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Categorical slots in fixed order (validated adjacent-pair palette); color follows the
# method, never its rank in the current figure.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
METHOD_ORDER = ("random", "attn", "kl", "keep_one", "fisher", "fisher_diversity", "stratified",
                "grid", "diversity", "voxel_dedup", "voxel_strat")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
METRICS = (("success", "Success rate"), ("spl", "SPL"), ("sup/mean_kv_len", "kv_len (layer mean, slots)"))
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
    if values.size == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def discover(eval_root: Path, prefix: str, methods=None):
    """Arms under `prefix`; `methods` (default METHOD_ORDER) keeps sibling prefixes such as
    <prefix>_screen_* from parsing as a method called 'screen_random'."""
    allowed = set(METHOD_ORDER if methods is None else methods)
    pattern = re.compile(rf"^{re.escape(prefix)}_(?P<method>.+?)_(?:w(?P<w>\d+)|nowin)"
                         rf"(?:_l(?P<l>\d+)(?:-(?P<le>\d+))?)?_b(?P<b>\d+)$")
    arms = []
    for run_dir in sorted(eval_root.glob(f"{prefix}_*")):
        m = pattern.match(run_dir.name)
        if not m or not run_dir.is_dir() or m["method"] not in allowed:
            continue
        rows = load_rows(run_dir)
        if not rows:
            print(f"[skip] {run_dir.name}: no results")
            continue
        arms.append({"run": run_dir.name, "method": m["method"], "layer": int(m["l"] or 0),
                     "layer_end": None if m["le"] is None else int(m["le"]),
                     "budget": int(m["b"]), "rows": rows})
    return arms


def method_slot(method: str, seen: list) -> int:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    if method not in seen:
        seen.append(method)
    return len(METHOD_ORDER) + seen.index(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="kvprune_layer")
    ap.add_argument("--eval-root", type=Path, default=Path("./dump/longnav_eval"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--baselines", default="win32=hm3d_v2_100_win32,full=hm3d_v2_100_winfull",
                    help="comma-separated label=run_name references drawn as dashed lines")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--methods", default=",".join(METHOD_ORDER),
                    help="comma-separated selector names to include (run-name <method> field)")
    ap.add_argument("--uniform", default="kvprune_v2_full_random_w32_b1187,kvprune_v2_full_random_w32_b1799,"
                                         "kvprune_v2_full_random_w32_b3022,hm3d_v2_100_win32@5474",
                    help="comma-separated uniform-random / window runs forming the memory reference curve "
                         "of the second figure (<out>_memory.png); 'run@kv' pins the x position for runs "
                         "that predate kv_len logging (win32 measured ~5474 slots); '' disables it")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    arms = discover(args.eval_root, args.prefix, [m for m in args.methods.split(",") if m])
    if not arms:
        raise SystemExit(f"no runs matching {args.prefix}_* under {args.eval_root}")
    arms = [a for a in arms if a["layer_end"] is None]  # bands are a separate figure
    budgets = sorted({a["budget"] for a in arms})
    methods = sorted({a["method"] for a in arms}, key=lambda m: method_slot(m, []))
    extra = []
    slots = {m: method_slot(m, extra) for m in methods}

    baselines = {}
    for item in filter(None, args.baselines.split(",")):
        label, run = item.split("=", 1)
        rows = load_rows(args.eval_root / run)
        if rows:
            baselines[label] = {k: ci([r.get(k, np.nan) for r in rows.values()], rng)[0] for k, _ in METRICS}
            baselines[label]["n"] = len(rows)

    stats = {}  # (method, budget) -> list of (layer, n, {metric: (mean, lo, hi)})
    print(f"{'run':52s} {'n':>3s} {'SR':>6s} {'SPL':>6s} {'kv':>7s}")
    for a in sorted(arms, key=lambda a: (slots[a["method"]], a["budget"], a["layer"])):
        s = {k: ci([r.get(k, np.nan) for r in a["rows"].values()], rng) for k, _ in METRICS}
        stats.setdefault((a["method"], a["budget"]), []).append((a["layer"], len(a["rows"]), s))
        print(f"{a['run']:52s} {len(a['rows']):3d} {s['success'][0]:6.3f} {s['spl'][0]:6.3f} "
              f"{s['sup/mean_kv_len'][0]:7.0f}")

    fig, axes = plt.subplots(len(METRICS), len(budgets), figsize=(max(7.5, 4.6 * len(budgets) + 1), 9.5),
                             squeeze=False, sharex=True)
    for col, budget in enumerate(budgets):
        for row, (key, name) in enumerate(METRICS):
            ax = axes[row][col]
            for method in methods:
                pts = sorted(stats.get((method, budget), []))
                if not pts:
                    continue
                x = [p[0] for p in pts]
                mean = np.array([p[2][key][0] for p in pts])
                lo = mean - np.array([p[2][key][1] for p in pts])
                hi = np.array([p[2][key][2] for p in pts]) - mean
                color = SERIES_COLORS[slots[method] % len(SERIES_COLORS)]
                marker = MARKERS[slots[method] % len(MARKERS)]
                # Small x-dodge so methods at the same layer do not stack their error bars.
                dodge = (methods.index(method) - (len(methods) - 1) / 2) * 0.35
                xd = [xi + dodge for xi in x]
                ax.errorbar(xd, mean, yerr=[lo, hi], label=method, color=color, marker=marker,
                            markersize=6, linewidth=2, capsize=3, alpha=0.95)
                if len(methods) <= 4:
                    ax.annotate(method, (xd[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                                fontsize=8, color=MUTED, va="center")
            for label, ref in baselines.items():
                if np.isfinite(ref[key]) and not (key == "sup/mean_kv_len" and label == "full"):
                    ax.axhline(ref[key], color=MUTED, linestyle="--", linewidth=1, alpha=0.7)
                    ax.annotate(f"{label} (n={ref['n']})", (0.01, ref[key]),
                                xycoords=("axes fraction", "data"), xytext=(0, 3),
                                textcoords="offset points", fontsize=7, color=MUTED)
            if key == "sup/mean_kv_len":
                ax.set_ylim(bottom=0)
            ax.set_xticks(sorted({p[0] for pts in stats.values() for p in pts}))
            ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            if col == 0:
                ax.set_ylabel(name, color=TEXT)
            if row == 0:
                ax.set_title(f"budget {budget} slots / pruned layer", color=TEXT, fontsize=10)
            if row == len(METRICS) - 1:
                ax.set_xlabel(f"first pruned decoder layer (0 = all {args.n_layers})", color=TEXT)
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower left")
    n_ep = sorted({len(a["rows"]) for a in arms})
    fig.suptitle(f"KV-prune selectors by prune layer, {args.prefix} "
                 f"(n={'/'.join(map(str, n_ep))} episodes, 95% bootstrap CI)", color=TEXT)
    fig.tight_layout()
    out = args.out or args.eval_root / f"{args.prefix}_by_layer.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    if args.uniform:
        plot_memory_matched(args, arms, methods, slots, stats, rng, out)


def plot_memory_matched(args, arms, methods, slots, stats, rng, out):
    """SR / SPL against layer-mean kv_len: does WHERE you prune matter at matched memory?

    The reference curve is uniform random pruning at increasing budgets (plus the unpruned
    window); each layer-selective arm is one point, labelled with its prune layer.
    """
    ref = []
    for item in filter(None, args.uniform.split(",")):
        run, _, pinned = item.partition("@")
        rows = load_rows(args.eval_root / run)
        if not rows:
            continue
        entry = {k: ci([r.get(k, np.nan) for r in rows.values()], rng) for k, _ in METRICS}
        x = float(pinned) if pinned else entry["sup/mean_kv_len"][0]
        if np.isfinite(x):
            ref.append((x, run, entry))
        else:
            print(f"[memory view] {run} logs no kv_len; pin it with {run}@<slots>")
    ref.sort(key=lambda t: t[0])
    xs = [t[0] for t in ref]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (key, name) in zip(axes, METRICS[:2]):
        if ref:
            ref_e = [t[2] for t in ref]
            ax.annotate(ref[-1][1], (xs[-1], ref_e[-1][key][0]), xytext=(0, 8),
                        textcoords="offset points", fontsize=7, color=MUTED, ha="center")
            ax.errorbar(xs, [r[key][0] for r in ref_e],
                        yerr=[[r[key][0] - r[key][1] for r in ref_e], [r[key][2] - r[key][0] for r in ref_e]],
                        color=MUTED, marker="o", markersize=5, linewidth=1.5, capsize=3,
                        linestyle="--", label="uniform random, by budget")
        for method in methods:
            pts = [(p[0], p[2]) for b in sorted({a["budget"] for a in arms})
                   for p in stats.get((method, b), [])]
            if not pts:
                continue
            color = SERIES_COLORS[slots[method] % len(SERIES_COLORS)]
            marker = MARKERS[slots[method] % len(MARKERS)]
            x = [p[1]["sup/mean_kv_len"][0] for p in pts]
            y = [p[1][key][0] for p in pts]
            ax.errorbar(x, y, yerr=[[p[1][key][0] - p[1][key][1] for p in pts],
                                    [p[1][key][2] - p[1][key][0] for p in pts]],
                        color=color, marker=marker, markersize=6, linewidth=0, elinewidth=1.5,
                        capsize=3, label=f"{method}, by prune layer")
            for xi, yi, (layer, _) in zip(x, y, pts):
                ax.annotate(f"L{layer}", (xi, yi), xytext=(5, -3), textcoords="offset points",
                            fontsize=7, color=color)
        ax.set_xlabel("resident KV slots (layer mean)", color=TEXT)
        ax.set_ylabel(name, color=TEXT)
        ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Memory-matched view: layer-selective arms vs the uniform random budget curve "
                 f"({args.prefix})", color=TEXT)
    fig.tight_layout()
    out_mem = out.with_name(out.stem + "_memory.png")
    fig.savefig(out_mem, dpi=150, facecolor="white")
    print(f"wrote {out_mem}")


if __name__ == "__main__":
    main()
