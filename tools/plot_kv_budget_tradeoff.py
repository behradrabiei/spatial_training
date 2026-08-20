#!/usr/bin/env python3
"""Memory-performance tradeoff for the KV-budget pruning experiments.

X axis: mean resident KV length per step (measured via sup/mean_kv_len where the run
logged it; evict baselines predate that metric, so their kv_len is the fitted retirement
schedule prefix + N*tokens_per_turn, whose two free parameters were fitted on the
measured reindex win4/win32 relogs -- the schedule is identical across modes).

Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_budget_tradeoff.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_ROOT = Path("./dump/longnav_eval")
OUT = EVAL_ROOT / "hm3d_v2_100_kv_budget_tradeoff.png"
PREFIX_FIT, TOK_PER_TURN = 574.2, 153.1  # fitted on reindexv2win4 / reindexv2win32

# (run_name, series, fallback window size for fitted kv_len)
RUNS = [
    ("hm3d_v2_100_win4", "evict window (recency)", 4),
    ("hm3d_v2_100_win8", "evict window (recency)", 8),
    ("hm3d_v2_100_win16", "evict window (recency)", 16),
    ("hm3d_v2_100_win32", "evict window (recency)", 32),
    ("hm3d_v2_100_prune_w32_b1187", "prune: attn slots", None),
    ("hm3d_v2_100_prune_w32_b1799", "prune: attn slots", None),
    ("hm3d_v2_100_pruneturn_w32_b1187", "prune: attn keyframes", None),
    ("hm3d_v2_100_pruneturn_w32_b1799", "prune: attn keyframes", None),
    ("hm3d_v2_100_prunemerge_w32_b1187", "prune: attn slots + merge", None),
    ("hm3d_v2_100_prunerand_w32_b1187", "prune: random slots", None),
    ("hm3d_v2_100_prunerand_w32_b1799", "prune: random slots", None),
    ("hm3d_v2_100_prunerand_w32_b3022", "prune: random slots", None),
]
STYLE = {
    "evict window (recency)": dict(color="tab:gray", marker="o"),
    "prune: attn slots": dict(color="tab:orange", marker="s"),
    "prune: attn keyframes": dict(color="tab:red", marker="^"),
    "prune: attn slots + merge": dict(color="tab:brown", marker="D"),
    "prune: random slots": dict(color="tab:blue", marker="*"),
}
REFERENCE = ("hm3d_v2_100_win32", "win32")


def load(run_name):
    rows = []
    for path in sorted((EVAL_ROOT / run_name / "rollout").glob("results_*")):
        for line in path.open():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return None
    seen = {}
    for r in rows:
        seen.setdefault(r["episode_label"], r)
    rows = list(seen.values())
    kv = [r["sup/mean_kv_len"] for r in rows if "sup/mean_kv_len" in r]
    return {
        "n": len(rows),
        "sr": sum(r.get("success", 0.0) for r in rows) / len(rows),
        "spl": sum(r.get("spl", 0.0) for r in rows) / len(rows),
        "kv": sum(kv) / len(kv) if kv else None,
    }


def main():
    series = {}
    for run, label, window in RUNS:
        m = load(run)
        if m is None:
            print(f"[skip] {run}")
            continue
        kv = m["kv"] if m["kv"] is not None else PREFIX_FIT + TOK_PER_TURN * window
        series.setdefault(label, []).append((kv, m["sr"], m["spl"], run))
        print(f"{run:45s} n={m['n']:3d} kv={kv:6.0f} SR={m['sr']:.3f} SPL={m['spl']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, name in ((axes[0], 1, "Success rate"), (axes[1], 2, "SPL")):
        for label, pts in series.items():
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[key] for p in pts],
                    label=label, alpha=0.85, markersize=9 if "random" in label else 6,
                    **STYLE[label])
        ref = load(REFERENCE[0])
        if ref:
            ax.axhline(ref["sr"] if key == 1 else ref["spl"], color="tab:green",
                       linestyle=":", alpha=0.7,
                       label=f"{REFERENCE[1]} reference")
        ax.set_xlabel("mean resident KV length (slots/step)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("HM3D-v2 (100 ep): navigation vs KV-cache budget "
                 "(evict baselines use fitted kv_len)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
