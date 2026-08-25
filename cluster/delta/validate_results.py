#!/usr/bin/env python3
"""Validate that a smoke evaluation produced exactly the requested episodes."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("episode_json", type=Path)
    args = parser.parse_args()

    expected = json.loads(args.episode_json.read_text(encoding="utf-8"))
    rows = []
    for result_file in sorted((args.run_dir / "rollout").glob("results_*")):
        with result_file.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    labels = [row.get("episode_label") for row in rows]
    if len(rows) != len(expected) or sorted(labels) != sorted(expected):
        raise RuntimeError(
            f"Expected {expected}, but found {labels} in {args.run_dir / 'rollout'}"
        )
    config_path = args.run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved configuration is missing: {config_path}")
    print(f"Evaluation results: OK ({len(rows)} episodes in {args.run_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
