#!/usr/bin/env python3
"""Apply the KV-pruning screen/full Pareto promotion rules to completed runs."""
import argparse
import json
from pathlib import Path


def load(root, run):
    rows = {}
    for path in sorted((root / run / "rollout").glob("results_*")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.setdefault(row["episode_label"], row)
    if not rows:
        raise FileNotFoundError(f"no result rows for {run}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="METHOD=RUN")
    parser.add_argument("--eval-root", type=Path, default=Path("dump/longnav_eval"))
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    parser.add_argument("--print-methods", action="store_true")
    args = parser.parse_args()
    base = load(args.eval_root, args.baseline)
    sr_floor, spl_floor = ((-0.04, -0.02) if args.stage == "screen" else (-0.02, -0.01))
    promoted = []
    for spec in args.candidate:
        method, run = spec.split("=", 1)
        rows = load(args.eval_root, run)
        labels = sorted(set(base) & set(rows))
        if len(labels) != len(base) or len(labels) != len(rows):
            raise RuntimeError(f"{run} is not exactly paired with {args.baseline}")
        dsr = sum(float(rows[x]["success"]) - float(base[x]["success"]) for x in labels) / len(labels)
        dspl = sum(float(rows[x]["spl"]) - float(base[x]["spl"]) for x in labels) / len(labels)
        passes = (dsr > 0 or dspl > 0) and dsr >= sr_floor and dspl >= spl_floor
        if passes:
            promoted.append(method)
        if not args.print_methods:
            print(f"{method}: dSR={dsr:+.4f} dSPL={dspl:+.4f} {'PROMOTE' if passes else 'stop'}")
    if args.print_methods:
        print(" ".join(promoted))


if __name__ == "__main__":
    main()
