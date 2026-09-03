#!/usr/bin/env python3
"""Where is the visual information horizon? Reads `vlm.kv_prune_log_layer_influence=true` runs.

For each step t and probe layer i the worker logged
    KL_i = KL( P_t || P_t with visual slots hidden from decoder layers >= i )
twice: for the history's visual slots only (sup/layer_influence_hist -- everything a KV
budget could prune) and for every visual slot including the current frame
(sup/layer_influence_all). Hiding from layer i onward leaves what the slots contributed to
layers < i in the residual stream, so KL_i falls with i and reaches 0 at i = n_layers; the
layer where it is already ~0 is the horizon of "When Token Pruning is Worse than Random"
(CVPR'26, arXiv 2512.07580): pruning at or beyond it should be free, and any selector can
only matter below it.

Reports per probe layer: mean / median KL over steps, the retained fraction
mean(KL_i) / mean(KL_0), and the same split by episode-step tercile (early / mid / late).
Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/layer_influence.py dump/longnav_eval/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # early / mid / late (fixed order)
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
KINDS = (("hist", "history visual slots hidden (what a budget can prune)"),
         ("all", "every visual slot hidden (incl. the current frame)"))
N_BOOT, SEED = 2000, 17
EPS = 1e-6


def probe_layers(run_dir: Path, override):
    if override:
        return [int(x) for x in override.split(",")]
    cfg = run_dir / "config.yaml"
    try:
        import yaml
        starts = yaml.safe_load(cfg.read_text())["vlm"]["kv_prune_layer_influence_starts"]
        return [int(x) for x in starts]
    except Exception as exc:  # noqa: BLE001 -- fall back to an explicit list
        raise SystemExit(f"could not read vlm.kv_prune_layer_influence_starts from {cfg} ({exc}); "
                         "pass --probes 0,4,8,...") from exc


def load_episodes(run_dir: Path, n_probes: int):
    episodes = []
    for seq in sorted(run_dir.glob("rollout/*/*/sequence.json")):
        d = json.load(seq.open())
        per_kind = {}
        for kind, _ in KINDS:
            rows = d.get(f"sup/layer_influence_{kind}")
            if rows is None:
                continue
            T = len(rows)
            M = np.full((T, n_probes), np.nan)
            for t, row in enumerate(rows):
                if len(row) == n_probes:
                    M[t] = row
            per_kind[kind] = M
        if per_kind:
            episodes.append({"label": seq.parent.name.split(".")[0], **per_kind})
    return episodes


def tercile_masks(T):
    edges = np.linspace(0, T, 4).astype(int)
    idx = np.arange(T)
    return [(idx >= edges[k]) & (idx < edges[k + 1]) for k in range(3)]


def summarize(episodes, kind, probes):
    """Per-episode curves (mean over steps) and the pooled per-tercile means."""
    ep_mean, ep_frac, terc = [], [], {k: [] for k in range(3)}
    for ep in episodes:
        M = ep.get(kind)
        if M is None or not np.isfinite(M).any():
            continue
        valid = np.isfinite(M).all(axis=1)
        if not valid.any():
            continue
        curve = np.nanmean(M[valid], axis=0)
        ep_mean.append(curve)
        # Ratio of means, not mean of per-step ratios: single steps with KL_0 ~ 0 (a
        # saturated action distribution) would otherwise dominate the fraction.
        ep_frac.append(curve / curve[0] if curve[0] > EPS else np.full_like(curve, np.nan))
        for k, mask in enumerate(tercile_masks(M.shape[0])):
            sel = valid & mask
            if sel.any():
                terc[k].append(np.nanmean(M[sel], axis=0))
    return np.array(ep_mean), np.array(ep_frac), {k: np.array(v) for k, v in terc.items()}


def boot_ci(curves, rng):
    """Mean curve over episodes with a 95% bootstrap CI over episodes."""
    if curves.size == 0:
        return None
    mean = np.nanmean(curves, axis=0)
    idx = rng.integers(0, curves.shape[0], size=(N_BOOT, curves.shape[0]))
    boots = np.nanmean(curves[idx], axis=1)
    return mean, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def style(ax, xlabel=None, ylabel=None):
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--probes", default=None, help="comma-separated probe layers (default: config.yaml)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    probes = probe_layers(args.run_dir, args.probes)
    episodes = load_episodes(args.run_dir, len(probes))
    if not episodes:
        raise SystemExit(f"no sup/layer_influence_* rows under {args.run_dir}/rollout")
    rng = np.random.default_rng(SEED)
    n_steps = sum(int(np.isfinite(ep[k]).all(axis=1).sum()) for ep in episodes for k, _ in KINDS if k in ep)
    print(f"{len(episodes)} episodes, {n_steps} scored (step, kind) rows, probes {probes}")

    fig, axes = plt.subplots(2, len(KINDS), figsize=(5.2 * len(KINDS), 7.5), squeeze=False, sharex=True)
    for col, (kind, title) in enumerate(KINDS):
        ep_mean, ep_frac, terc = summarize(episodes, kind, probes)
        if ep_mean.size == 0:
            print(f"[{kind}] no rows")
            continue
        mean_ci = boot_ci(ep_mean, rng)
        pooled_frac = mean_ci[0] / max(mean_ci[0][0], EPS)  # ratio of pooled means
        print(f"\n[{kind}] {title}")
        print(f"  {'probe':>5s} {'mean KL':>9s} {'median':>9s} {'CI lo':>8s} {'CI hi':>8s} {'kept frac':>9s}")
        for j, layer in enumerate(probes):
            med = np.nanmedian(np.concatenate([ep[kind][:, j] for ep in episodes if kind in ep]))
            print(f"  {layer:5d} {mean_ci[0][j]:9.5f} {med:9.5f} {mean_ci[1][j]:8.5f} {mean_ci[2][j]:8.5f} "
                  f"{pooled_frac[j]:9.3f}")
        horizon = next((layer for j, layer in enumerate(probes) if pooled_frac[j] < 0.05), None)
        print(f"  first probe with < 5% of the layer-0 effect left: {horizon}")

        ax = axes[0][col]
        for curve in ep_mean:
            ax.plot(probes, curve, color=MUTED, linewidth=0.8, alpha=0.25)
        ax.fill_between(probes, mean_ci[1], mean_ci[2], color=SERIES_COLORS[0], alpha=0.15, linewidth=0)
        ax.plot(probes, mean_ci[0], color=SERIES_COLORS[0], linewidth=2, marker="o", markersize=6,
                label=f"mean over {ep_mean.shape[0]} episodes (95% CI)")
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        style(ax, ylabel="KL(P || P hidden from layer i on)" if col == 0 else None)

        ax = axes[1][col]
        for k, name in enumerate(("early third", "middle third", "late third")):
            ci = boot_ci(terc[k], rng) if terc[k].size else None
            if ci is None:
                continue
            base = np.maximum(ci[0][:1], EPS)
            ax.plot(probes, ci[0] / base, color=SERIES_COLORS[k], linewidth=2, marker="o",
                    markersize=6, label=name)
        ax.plot(probes, pooled_frac, color=TEXT, linewidth=1.2, linestyle=":", label="all steps")
        ax.axhline(0.05, color=MUTED, linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate("5%", (probes[0], 0.05), xytext=(2, 3), textcoords="offset points", fontsize=7, color=MUTED)
        top = max(1.05, 1.05 * max(float(np.nanmax(l.get_ydata())) for l in ax.get_lines()))
        ax.set_ylim(-0.02, min(top, 2.0))
        ax.legend(fontsize=8, frameon=False)
        style(ax, xlabel="first hidden decoder layer i", ylabel="fraction of the layer-0 effect" if col == 0 else None)
        ax.set_xticks(probes)

    fig.suptitle(f"Visual information horizon -- {args.run_dir.name}", color=TEXT)
    fig.tight_layout()
    out = args.out or args.run_dir / "layer_influence.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
