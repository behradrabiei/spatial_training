#!/usr/bin/env python3
"""Validate and compare Uni-NaVid and LongNav-R1 context-window ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


WINDOWS = [1, 4, 8, 16, 32]
LONGNAV_TAGS = ["1", "4", "8", "16", "32", "64", "full"]
MATCHED_INFERENCE_SETTINGS = (
    "seed",
    "temperature",
    "max_new_tokens",
    "actions_per_inference",
    "max_episode_steps",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(
    run_dir: Path,
    expected_window: int | None,
    expect_visual_tokens: bool = True,
    expected_current_history_tokens: int = 4,
) -> tuple[float, float, dict[str, Any]]:
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "manifest.json")
    if summary.get("completed_episodes") != summary.get("expected_episodes"):
        raise ValueError(f"Incomplete evaluation: {run_dir}")
    if summary.get("expected_episodes") != 100 or summary.get("missing_episodes"):
        raise ValueError(f"Expected 100 episodes with none missing: {run_dir}")

    inference = manifest["inference"]
    visual_tokens_enabled = inference.get("visual_tokens_enabled", True)
    if visual_tokens_enabled != expect_visual_tokens:
        raise ValueError(f"Visual-token setting mismatch for {run_dir}")
    recorded_current_history = inference.get(
        "current_frame_history_tokens", 4 if expect_visual_tokens else 0
    )
    if recorded_current_history != expected_current_history_tokens:
        raise ValueError(f"Current-frame history-token mismatch for {run_dir}")
    recorded_window = inference.get("context_window_previous_frames")
    if recorded_window != expected_window:
        raise ValueError(
            f"Context-window mismatch for {run_dir}: "
            f"{recorded_window} != {expected_window}"
        )
    expected_total = (
        0
        if not expect_visual_tokens
        or (expected_window == 0 and expected_current_history_tokens == 0)
        else expected_window + 1
        if expected_window is not None
        else None
    )
    if inference.get("max_visual_history_frames") != expected_total:
        # The reused full-history manifest predates these explicit null fields.
        if expected_window is not None or "max_visual_history_frames" in inference:
            raise ValueError(f"Maximum-history mismatch for {run_dir}")

    return float(summary["success_rate"]), float(summary["spl"]), manifest


def load_longnav_metrics(
    eval_root: Path, requested_labels: list[str]
) -> list[tuple[str, float, float]]:
    expected_labels = set(requested_labels)
    points: list[tuple[str, float, float]] = []
    for tag in LONGNAV_TAGS:
        run_dir = eval_root / f"hm3d_v2_100_win{tag}" / "rollout"
        rows: list[dict[str, Any]] = []
        for path in sorted(run_dir.glob("results_*")):
            with path.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        labels = [row.get("episode_label") for row in rows]
        if len(rows) != 100 or len(set(labels)) != 100:
            raise ValueError(f"Expected 100 unique LongNav-R1 episodes: {run_dir}")
        if set(labels) != expected_labels:
            raise ValueError(f"LongNav-R1 episode-label mismatch: {run_dir}")
        success = sum(float(row.get("success", 0.0)) for row in rows) / len(rows)
        spl = sum(float(row.get("spl", 0.0)) for row in rows) / len(rows)
        points.append((tag, success, spl))
    return points


def main() -> None:
    baseline_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=baseline_root / "results")
    parser.add_argument(
        "--longnav-root",
        type=Path,
        default=baseline_root.parent / "dump/longnav_eval",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=baseline_root / "results/hm3d_v2_100_context_window_sr_spl.png",
    )
    args = parser.parse_args()

    points: list[tuple[str, float, float, dict[str, Any]]] = []
    sr, spl, manifest = validate_run(
        args.results_root / "hm3d_v2_100_no_visual", None, False, 0
    )
    points.append(("no vision", sr, spl, manifest))
    sr, spl, manifest = validate_run(
        args.results_root / "hm3d_v2_100_current64", 0, True, 0
    )
    points.append(("0 (64)", sr, spl, manifest))
    sr, spl, manifest = validate_run(
        args.results_root / "hm3d_v2_100_win0", 0, True, 4
    )
    points.append(("0 (68)", sr, spl, manifest))
    for window in WINDOWS:
        sr, spl, manifest = validate_run(
            args.results_root / f"hm3d_v2_100_win{window}", window
        )
        points.append((str(window), sr, spl, manifest))
    sr, spl, manifest = validate_run(args.results_root / "hm3d_v2_100", None)
    points.append(("full", sr, spl, manifest))

    reference = points[0][3]
    for tag, _, _, candidate in points[1:]:
        if candidate["labels_sha256"] != reference["labels_sha256"]:
            raise ValueError(f"Label checksum mismatch for {tag}")
        for setting in MATCHED_INFERENCE_SETTINGS:
            if candidate["inference"].get(setting) != reference["inference"].get(setting):
                raise ValueError(f"Inference setting {setting!r} differs for {tag}")

    longnav_points = load_longnav_metrics(
        args.longnav_root, reference["requested_labels"]
    )

    print("Uni-NaVid")
    for tag, sr_value, spl_value, _ in points:
        print(f"{tag:>4}: SR={sr_value:.4f}  SPL={spl_value:.4f}")
    print("LongNav-R1")
    for tag, sr_value, spl_value in longnav_points:
        print(f"{tag:>4}: SR={sr_value:.4f}  SPL={spl_value:.4f}")

    axis_tags = [
        "no vision", "0 (64)", "0 (68)", "1", "4", "8", "16", "32", "64", "full"
    ]
    x_of = {tag: index for index, tag in enumerate(axis_tags)}
    controls = [point for point in points if point[0] in {"no vision", "0 (64)"}]
    uninavid_curve = [
        point for point in points if point[0] not in {"no vision", "0 (64)", "full"}
    ]
    uninavid_full = next(point for point in points if point[0] == "full")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for metric_index, (ax, metric_name) in enumerate(
        zip(axes, ("Success rate", "SPL")), start=1
    ):
        uninavid_line = ax.plot(
            [x_of[point[0]] for point in uninavid_curve],
            [point[metric_index] for point in uninavid_curve],
            marker="o",
            linewidth=2,
            label="Uni-NaVid",
        )[0]
        ax.scatter(
            [x_of["full"]],
            [uninavid_full[metric_index]],
            marker="o",
            s=45,
            color=uninavid_line.get_color(),
        )
        ax.scatter(
            [x_of[point[0]] for point in controls],
            [point[metric_index] for point in controls],
            marker="X",
            s=65,
            color=uninavid_line.get_color(),
            label="Uni-NaVid controls",
        )
        ax.plot(
            [x_of[point[0]] for point in longnav_points],
            [point[metric_index] for point in longnav_points],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="LongNav-R1",
        )
        ax.set_xticks(range(len(axis_tags)), axis_tags, rotation=25, ha="right")
        ax.set_xlabel("Previous-frame context (current frame is additional)")
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Metric")
    axes[0].set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "HM3D-v2 context-window ablation (same 100 episodes)",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
