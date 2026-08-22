#!/usr/bin/env python3
"""Validate and plot Uni-NaVid's HM3D-v2 context-window ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


WINDOWS = [1, 4, 8, 16, 32]
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


def main() -> None:
    baseline_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=baseline_root / "results")
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

    tags = [point[0] for point in points]
    success = [point[1] for point in points]
    spl_values = [point[2] for point in points]
    for tag, sr_value, spl_value, _ in points:
        print(f"{tag:>4}: SR={sr_value:.4f}  SPL={spl_value:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(tags)))
    ax.plot(x, success, marker="o", linewidth=2, label="Success rate")
    ax.plot(x, spl_values, marker="s", linewidth=2, label="SPL")
    ax.set_xticks(x, tags)
    ax.set_xlabel("Ablation / previous-frame context (current frame is additional)")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Uni-NaVid on HM3D-v2 (100 episodes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
