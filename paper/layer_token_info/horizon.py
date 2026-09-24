#!/usr/bin/env python3
"""Information horizon figure (the Fig. 8 analogue of arXiv 2512.07580).

Against the decoder layer L, four single-axis panels (means only; the 95% episode-bootstrap
CIs go to results/horizon.csv):
  * keep-one-in KL and leave-one-out KL of the CURRENT frame's visual tokens when hidden from
    layer L on (keep-one-in: "only the current frame visible" vs "no visual token visible";
    leave-one-out: "everything visible" vs "everything but the current frame"), averaged over
    the scored steps of the 10-episode keep-one diagnostic (`vlm.kv_prune_log_keep_one`, probe
    layers 0/8/16/20/24/28). Frame-level by design; --with-token-curve adds the dense 28-layer
    per-token mean of this experiment's leaky arm (2 episodes) as a dotted line.
  * SR and SPL of the visual-blind arms, where no query may read any visual token (history or
    current frame) on layers >= L (`vlm.kv_prune_visual_blind`, 100 episodes); the dashed line
    is the unblinded reference run, also drawn at L = 28 ("none": nothing hidden).
The purple dashed line marks the horizon (--horizon, default 18: the layer at which the per-token
maps of this experiment collapse; the frame-level probes only bracket it between 16 and 20).

Reads runs/<blind arm>/rollout/results_* (copied from dump/longnav_eval), the keep-one diagnostic
from dump/longnav_eval/keep_one_influence_w32_b10000000 (aggregated into results/horizon.csv),
and runs/<episode>/token_info_mask.npz. Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python paper/layer_token_info/horizon.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot import load_npz  # noqa: E402

ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/layer_token_info"
KEEP_ONE_RUN = ROOT / "dump/longnav_eval/keep_one_influence_w32_b10000000"
BLIND_PREFIX = "hm3d_v2_100_w32_vblind"
REFERENCE = "hm3d_v2_100_win32"
N_LAYERS = 28
N_BOOT, SEED = 2000, 17
CURVE_COLOR, HORIZON_COLOR = "tab:blue", "tab:purple"


def load_rows(run_dir: Path) -> dict:
    rows = {}
    for path in sorted((run_dir / "rollout").glob("results_*")):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["episode_label"], r)
    return rows


def boot_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float("nan"), float("nan")
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(N_BOOT)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def blind_arms(rng):
    """{L: {metric: (mean, lo, hi)}} for the visual-blind arms plus the reference at L = N_LAYERS."""
    out = {}
    for run_dir in sorted((RESULTS_ROOT / "runs").glob(f"{BLIND_PREFIX}*")):
        m = re.fullmatch(rf"{re.escape(BLIND_PREFIX)}(?:_l(\d+))?", run_dir.name)
        if not m:
            continue
        L = int(m.group(1)) if m.group(1) else 0
        rows = load_rows(run_dir)
        out[L] = {k: boot_ci([r[k] for r in rows.values()], rng) for k in ("success", "spl")}
        out[L]["n"] = len(rows)
    ref = load_rows(RESULTS_ROOT / "runs" / REFERENCE)
    out[N_LAYERS] = {k: boot_ci([r[k] for r in ref.values()], rng) for k in ("success", "spl")}
    out[N_LAYERS]["n"] = len(ref)
    return out


def keep_one_diagnostic(rng):
    """Per probe layer: current-frame keep-one-in and leave-one-out KL, mean over scored steps,
    episode-bootstrap CI over the per-episode means."""
    import yaml
    cfg = yaml.safe_load((KEEP_ONE_RUN / "config.yaml").read_text())
    probes = [int(x) for x in cfg["vlm"]["kv_prune_layer_influence_starts"]]
    per_episode = {"keep": [], "leave": []}
    n_steps = 0
    for seq_path in sorted(KEEP_ONE_RUN.glob("rollout/*/*/sequence.json")):
        s = json.loads(seq_path.read_text())
        keep, leave = s["sup/keep_one_influence"], s["sup/leave_one_influence"]
        ep = {"keep": [], "leave": []}
        for t, (k, l) in enumerate(zip(keep, leave)):
            if not k:
                continue  # stride-skipped step
            ep["keep"].append([row[t] for row in k])  # frame t = the current frame
            ep["leave"].append([row[t] for row in l])
            n_steps += 1
        for key in ep:
            per_episode[key].append(np.mean(ep[key], axis=0))
    out = {}
    for key in per_episode:
        arr = np.asarray(per_episode[key])  # [episodes, probes]
        out[key] = [boot_ci(arr[:, j], rng) for j in range(len(probes))]
    return probes, out, len(per_episode["keep"]), n_steps


def dense_curve(episodes):
    """Mean over tokens of the leaky keep-one-in per layer, averaged over episodes -> (layers, mean)."""
    curves = []
    for label in episodes:
        path = RESULTS_ROOT / "runs" / label / "token_info_mask.npz"
        if path.exists():
            res = load_npz(path)
            curves.append((res["layers"].astype(int), res["keep_cur"].astype(float)))
    if not curves:
        return None, None, None
    layers = curves[0][0]
    maps = np.mean([c[1] for c in curves], axis=0)
    return layers, maps.mean(axis=1), maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="token-info episodes for the dense curve (default: every runs/*/token_info_mask.npz)")
    ap.add_argument("--horizon", type=int, default=18,
                    help="horizon layer to mark (default 18: the cliff of the per-token maps of this experiment)")
    ap.add_argument("--with-token-curve", action="store_true",
                    help="also draw the 2-episode per-token mean (leaky arm) as a dotted line")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    fig_dir, res_dir = RESULTS_ROOT / "figures", RESULTS_ROOT / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    perf = blind_arms(rng)
    probes, info, n_ep, n_steps = keep_one_diagnostic(rng)
    episodes = args.episodes or sorted(p.parent.name for p in (RESULTS_ROOT / "runs").glob("*/token_info_mask.npz"))
    d_layers, d_mean, d_maps = dense_curve(episodes) if args.with_token_curve else (None, None, None)
    horizon = args.horizon

    # results table
    rows = []
    for j, L in enumerate(probes):
        rows.append({"layer": L, "source": f"keep_one_diagnostic ({n_ep} episodes, {n_steps} steps)",
                     "keep_one_cur_mean": info["keep"][j][0], "keep_one_cur_lo": info["keep"][j][1], "keep_one_cur_hi": info["keep"][j][2],
                     "leave_one_cur_mean": info["leave"][j][0], "leave_one_cur_lo": info["leave"][j][1], "leave_one_cur_hi": info["leave"][j][2]})
    if d_mean is not None:
        for L, v in zip(d_layers, d_mean):
            rows.append({"layer": int(L), "source": f"token_info per-token mean, leaky ({', '.join(episodes)})", "keep_one_cur_mean": float(v)})
    for L in sorted(perf):
        rows.append({"layer": L, "source": "visual_blind" if L < N_LAYERS else "reference (no blinding)", "n": perf[L]["n"],
                     "sr": perf[L]["success"][0], "sr_lo": perf[L]["success"][1], "sr_hi": perf[L]["success"][2],
                     "spl": perf[L]["spl"][0], "spl_lo": perf[L]["spl"][1], "spl_hi": perf[L]["spl"][2]})
    fields = ["layer", "source", "n", "keep_one_cur_mean", "keep_one_cur_lo", "keep_one_cur_hi",
              "leave_one_cur_mean", "leave_one_cur_lo", "leave_one_cur_hi", "sr", "sr_lo", "sr_hi", "spl", "spl_lo", "spl_hi"]
    with open(res_dir / "horizon.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # figure: four single-axis panels sharing L -- keep-one-in KL, leave-one-out KL, SR, SPL.
    # Means only (the CIs are in horizon.csv); one vertical line marks the horizon in every panel.
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharex=True)
    Ls = sorted(perf)
    xs = [L for L in Ls if L < N_LAYERS] + [N_LAYERS]
    ticks = list(range(0, N_LAYERS, 8)) + [N_LAYERS]  # 0 8 16 24 none
    for ax, key, title in zip(axes[:2], ("keep", "leave"), ("Keep-one-in KL (nats)", "Leave-one-out KL (nats)")):
        y = np.array([c[0] for c in info[key]])
        ax.plot(probes, y, color=CURVE_COLOR, marker="o", markersize=4, linewidth=1.2)
        if key == "keep" and d_mean is not None:
            ax.plot(d_layers, d_mean, color=CURVE_COLOR, linestyle=":", linewidth=1.2)
        ax.axhline(0, color="k", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylim(bottom=-1e-4)
        ax.set_title(title, fontsize=10)
    for ax, metric, title in zip(axes[2:], ("success", "spl"), ("SR", "SPL")):
        ax.axhline(perf[N_LAYERS][metric][0], color=CURVE_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)  # no blinding
        ax.plot(xs, [perf[L][metric][0] for L in xs], color=CURVE_COLOR, marker="o", markersize=4, linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.axvline(horizon, color=HORIZON_COLOR, linestyle="--", linewidth=1.2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) if x < N_LAYERS else "none" for x in ticks])
        ax.get_xticklabels()[-1].set_horizontalalignment("left")  # "none" grows outward, away from "24"
        ax.grid(alpha=0.3)
    fig.supxlabel("First layer without visual tokens", fontsize=10)
    fig.tight_layout()
    out = fig_dir / "information_horizon"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"horizon layer {horizon}; blind arms at L = {[L for L in Ls if L < N_LAYERS]}; wrote {out}.pdf/.png and {res_dir / 'horizon.csv'}")


if __name__ == "__main__":
    main()
