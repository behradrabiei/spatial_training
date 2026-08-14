#!/usr/bin/env python3
"""Aggregate success rate across context-window ablation runs and plot a curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Plot order: small → large → full
WINDOW_TAGS = ["1", "4", "8", "16", "32", "64", "full"]


def load_run_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    results_dir = run_dir / "rollout"
    if not results_dir.is_dir():
        return None
    rows: List[dict] = []
    for path in sorted(results_dir.glob("results_*")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        return None
    n = len(rows)
    return {
        "n": float(n),
        "success": sum(float(r.get("success", 0.0)) for r in rows) / n,
        "spl": sum(float(r.get("spl", 0.0)) for r in rows) / n,
    }


def collect_metrics(eval_root: Path, prefix: str) -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    for tag in WINDOW_TAGS:
        run_dir = eval_root / f"{prefix}{tag}"
        metrics = load_run_metrics(run_dir)
        if metrics is None:
            print(f"[skip] missing or empty results: {run_dir}")
            continue
        out.append((tag, metrics))
        print(
            f"{tag:>4}: n={int(metrics['n']):3d}  "
            f"SR={metrics['success']:.4f}  SPL={metrics['spl']:.4f}  ({run_dir.name})"
        )
    return out


def plot_curve(
    arms: List[Tuple[str, List[Tuple[str, Dict[str, float]]]]],
    out_path: Path,
    title: str,
) -> None:
    # One shared categorical axis over every tag any arm produced, so a comparison arm
    # that skips "full" still lines up with the arm that has it.
    tags = [t for t in WINDOW_TAGS if any(t in dict(points) for _, points in arms)]
    x_of = {t: i for i, t in enumerate(tags)}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, (label, points) in enumerate(arms):
        by_tag = dict(points)
        xs = [x_of[t] for t, _ in points]
        suffix = f" ({label})" if len(arms) > 1 else ""
        ax.plot(xs, [by_tag[t]["success"] for t, _ in points], marker="o", linewidth=2,
                linestyle="-" if idx == 0 else "--", color=f"C{idx}",
                label=f"Success rate{suffix}")
        ax.plot(xs, [by_tag[t]["spl"] for t, _ in points], marker="s", linewidth=1.5,
                linestyle=":" if idx == 0 else "-.", color=f"C{idx}", alpha=0.7,
                label=f"SPL{suffix}")
    x = list(range(len(tags)))
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_xlabel("Context window (frames; full = entire episode)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("dump/longnav_eval"),
        help="Directory containing hm3d_v2_100_win* run folders",
    )
    parser.add_argument(
        "--prefix",
        default="hm3d_v2_100_win",
        help="Run-name prefix before the window tag",
    )
    parser.add_argument(
        "--label",
        default="evict",
        help="Legend label for the --prefix arm",
    )
    parser.add_argument(
        "--compare-prefix",
        default=None,
        help="Optional second arm to overlay, e.g. hm3d_v2_100_recompwin",
    )
    parser.add_argument(
        "--compare-label",
        default="recompute",
        help="Legend label for the --compare-prefix arm",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dump/longnav_eval/hm3d_v2_100_context_window_sr.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    arms = [(args.label, collect_metrics(args.eval_root, args.prefix))]
    if args.compare_prefix:
        print()
        arms.append((args.compare_label, collect_metrics(args.eval_root, args.compare_prefix)))
    arms = [(label, points) for label, points in arms if points]
    if not arms:
        raise SystemExit("No completed ablation runs found.")
    plot_curve(arms, args.out, title="HM3D V2 (~100 eps): SR vs context window")


if __name__ == "__main__":
    main()
