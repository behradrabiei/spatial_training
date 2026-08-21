"""Aggregate a multi-object eval run into OneMap-style benchmark metrics.

Usage: python3 tools/onemap_multi_report.py dump/longnav_eval/<run_name> [repair_dir ...]

Reads rollout/results_* JSONL (deduped by episode_label, valid/newest row wins) and
reports, following OneMap's read_results naming:
  PR  (Progress) = mean fraction of sub-goals reached
  PPL            = mean of sum(spl_leg)/n_goals
  SR             = fraction of episodes with all sub-goals reached
  SPL            = mean of (all_success * sum(spl_leg)/n_goals)
plus per-leg-index, per-category, and failure-mode breakdowns. Stdlib only.
"""

import argparse
import glob
import json
import os


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def load_rows(run_dirs):
    rows = {}
    paths = [
        path
        for run_dir in run_dirs
        for path in sorted(glob.glob(os.path.join(run_dir, "rollout", "results_*")))
    ]
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    label = row["episode_label"]
                    previous = rows.get(label)
                    rank = ("progress" in row, row.get("timestamp", 0))
                    previous_rank = (
                        "progress" in previous,
                        previous.get("timestamp", 0),
                    ) if previous else None
                    if previous_rank is None or rank > previous_rank:
                        rows[label] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels-json", default="src/longnav/conf/episode_jsons/onemap_multi.json",
                        help="expected episode labels; missing/broken episodes count as full failures")
    args = parser.parse_args()

    rows = load_rows(args.run_dirs)
    if not rows:
        raise SystemExit(f"no rollout/results_* rows found under {args.run_dirs}")

    with open(args.labels_json) as f:
        expected = json.load(f)
    missing = [l for l in expected if l not in rows]
    broken = [l for l, r in rows.items() if "progress" not in r]
    if missing:
        print(f"WARNING: {len(missing)} of {len(expected)} episodes missing (counted as full failures): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if broken:
        print(f"WARNING: {len(broken)} rows lack multi metrics (crashed/truncated; counted as full failures): {broken[:10]}")

    valid = [r for r in rows.values() if "progress" in r]
    n_total = len(expected) + len(set(rows) - set(expected))  # tolerate extra labels
    n_failed_pad = n_total - len(valid)

    progress = [r["progress"] for r in valid] + [0.0] * n_failed_pad
    ppl = [r["ppl"] for r in valid] + [0.0] * n_failed_pad
    sr = [r["all_success"] for r in valid] + [0.0] * n_failed_pad
    spl = [r["all_success"] * r["ppl"] for r in valid] + [0.0] * n_failed_pad

    run_names = " + ".join(os.path.basename(os.path.normpath(d)) for d in args.run_dirs)
    print(f"\n=== OneMap multi-object benchmark: {run_names} ===")
    print(f"episodes: {len(valid)} evaluated / {n_total} total\n")
    print(f"  PR  (progress)          : {mean(progress):.4f}")
    print(f"  PPL (progress-wtd. SPL) : {mean(ppl):.4f}")
    print(f"  SR  (all goals reached) : {mean(sr):.4f}")
    print(f"  SPL                     : {mean(spl):.4f}")

    # Per-leg records: (index, category, result, spl). Legs never attempted
    # (episode ended earlier) count as "not_attempted" failures.
    legs = []
    for r in valid:
        cats = r["goal_sequence"].split(",")
        results = r["leg_results"].split(",") if r["leg_results"] else []
        spls = [float(x) for x in r["leg_spls"].split(",")] if r["leg_spls"] else []
        for i, cat in enumerate(cats):
            if i < len(results):
                legs.append((i, cat, results[i], spls[i]))
            else:
                legs.append((i, cat, "not_attempted", 0.0))

    print("\n  per leg index:        SR      SPL   (n)")
    for i in sorted({l[0] for l in legs}):
        sub = [l for l in legs if l[0] == i]
        print(f"    leg {i + 1}:            {mean([l[2] == 'success' for l in sub]):.3f}   {mean([l[3] for l in sub]):.3f}  ({len(sub)})")

    print("\n  per category:         SR      SPL   (n legs)")
    for cat in sorted({l[1] for l in legs}):
        sub = [l for l in legs if l[1] == cat]
        print(f"    {cat:<18}  {mean([l[2] == 'success' for l in sub]):.3f}   {mean([l[3] for l in sub]):.3f}  ({len(sub)})")

    print("\n  episode outcomes:")
    reasons = {}
    for r in valid:
        reasons[r["done_reason"]] = reasons.get(r["done_reason"], 0) + 1
    if n_failed_pad:
        reasons["missing/crashed"] = n_failed_pad
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<18} {count:>4}  ({count / n_total:.1%})")

    wrong_stop_by_leg = {i: 0 for i in range(3)}
    immediate_post_switch = 0
    for r in valid:
        results = r["leg_results"].split(",") if r["leg_results"] else []
        steps = [int(x) for x in r["leg_steps_list"].split(",")] if r["leg_steps_list"] else []
        for i, result in enumerate(results):
            if result != "wrong_stop":
                continue
            wrong_stop_by_leg[i] += 1
            if i > 0 and i < len(steps) and steps[i] == 1:
                immediate_post_switch += 1

    wrong_stop_total = sum(wrong_stop_by_leg.values())
    print("\n  wrong-stop location:")
    for i, count in wrong_stop_by_leg.items():
        print(f"    leg {i + 1}:            {count:>4}  ({count / n_total:.1%} of episodes)")
    print(f"    immediate after switch: {immediate_post_switch:>4}  "
          f"({immediate_post_switch / n_total:.1%} of episodes; "
          f"{immediate_post_switch / wrong_stop_total:.1%} of wrong stops)"
          if wrong_stop_total else
          "    immediate after switch:    0")
    print()


if __name__ == "__main__":
    main()
