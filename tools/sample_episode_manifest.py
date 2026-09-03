#!/usr/bin/env python3
"""Create a fixed, reproducible subset from a JSON episode-label manifest."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    labels = json.loads(args.source.read_text())
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("episode manifest must be a JSON list of string labels")
    if not 0 <= args.count <= len(labels):
        raise ValueError(f"count must be between 0 and {len(labels)}, got {args.count}")
    selected = random.Random(args.seed).sample(labels, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"wrote {len(selected)} labels to {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
