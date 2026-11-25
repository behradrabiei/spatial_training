#!/usr/bin/env python
"""Aggregate the SimpleMemVLA context-window and layer-depth ablations (one episodes.jsonl per arm).

    python paper/simplememvla_ablation/aggregate.py                  # Permanence suite
    python paper/simplememvla_ablation/aggregate.py --suite counting

Reads  /mnt/vault/robomme/runs/simplememvla_ablation/<suite>/<arm>/episodes.jsonl
Writes paper_results/simplememvla_ablation/<suite>/results/{per_episode,summary}.csv and copies
each arm's small run files to .../<suite>/runs/<arm>/. Arms still running are included with
n < n_expected (plot.py draws complete arms only). Statistics as in the navigation studies:
95% percentile-bootstrap CIs (2000 resamples, seed 17), differences vs `full` paired on
(task, episode) with a two-sided sign-flip test (10 000 resamples).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robomme_ablation"))
from aggregate import bootstrap_mean_ci, fmt, load_arm, paired_delta  # noqa: E402  (stats helpers)

SUITES = {
    "permanence": ["VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "reference": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "imitation": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
N_LAYERS = 32
ARM_ORDER = (["full", "k32", "k16", "k8", "k4", "k2", "k1"]
             + [f"blind_l{L}" for L in range(N_LAYERS - 1, -1, -1)])
RAW_ROOT = Path("/mnt/vault/robomme/runs/simplememvla_ablation")
RESULTS_ROOT = HERE.parents[1] / "paper_results/simplememvla_ablation"
EPISODES_PER_TASK = 50
COPY = ("episodes.jsonl", "results.json", "run_config.json")


def arm_x(arm: str):
    """k<K> -> K, blind_l<L> -> L (first hidden decoder layer, zero-based), full -> "full"."""
    if arm == "full":
        return "full"
    return int(arm[len("blind_l"):]) if arm.startswith("blind_l") else int(arm[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="permanence", choices=sorted(SUITES))
    ap.add_argument("--runs", type=Path, default=None, help="raw runs dir (default: vault/<suite>)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: paper_results/.../<suite>)")
    args = ap.parse_args()
    tasks = SUITES[args.suite]
    runs = args.runs or RAW_ROOT / args.suite
    out = args.out or RESULTS_ROOT / args.suite
    (out / "results").mkdir(parents=True, exist_ok=True)

    arms = {}
    for arm in ARM_ORDER:
        rows = {k: r for k, r in load_arm(runs / arm).items() if k[0] in tasks and k[1] < EPISODES_PER_TASK}
        if not rows:
            continue
        arms[arm] = rows
        dst = out / "runs" / arm
        dst.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            if (runs / arm / name).exists():
                shutil.copy2(runs / arm / name, dst / name)

    with open(out / "results/per_episode.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "x", "task", "episode", "status", "success", "steps", "n_calls",
                    "max_history_age_steps", "max_history_frames", "blind_layer"])
        for arm, rows in arms.items():
            for (task, ep), r in sorted(rows.items(), key=lambda kv: (tasks.index(kv[0][0]), kv[0][1])):
                w.writerow([arm, arm_x(arm), task, ep, r["status"], int(bool(r["success"])), r["steps"],
                            r["n_calls"], r.get("max_history_age_steps", ""), r.get("max_history_frames", ""),
                            "" if r.get("blind_layer") is None else r["blind_layer"]])

    full = arms.get("full")
    n_expected = EPISODES_PER_TASK * len(tasks)
    cols = (["arm", "x", "n", "n_expected", "sr", "sr_lo", "sr_hi", "steps", "steps_lo", "steps_hi",
             "steps_success", "fail_rate", "timeout_rate", "n_calls"]
            + [f"sr_{t}" for t in tasks] + [f"n_{t}" for t in tasks]
            + ["dsr_vs_full", "dsr_lo", "dsr_hi", "p_sr_vs_full", "dsteps_vs_full", "p_steps_vs_full", "n_paired"]
            + [f"dsr_{t}" for t in tasks] + [f"p_sr_{t}" for t in tasks]
            + ["max_history_age_steps", "max_history_frames"]
            + [f"steps_{t}" for t in tasks])  # mean steps per task, all episodes (as `steps`)
    summary = []
    for arm, rows in arms.items():
        rs = list(rows.values())
        succ = np.array([float(bool(r["success"])) for r in rs])
        steps = np.array([float(r["steps"]) for r in rs])
        sr, sr_lo, sr_hi = bootstrap_mean_ci(succ)
        st, st_lo, st_hi = bootstrap_mean_ci(steps)
        row = {"arm": arm, "x": arm_x(arm), "n": len(rs), "n_expected": n_expected,
               "sr": sr, "sr_lo": sr_lo, "sr_hi": sr_hi, "steps": st, "steps_lo": st_lo, "steps_hi": st_hi,
               "steps_success": float(steps[succ > 0].mean()) if (succ > 0).any() else float("nan"),
               "fail_rate": float(np.mean([r["status"] == "fail" for r in rs])),
               "timeout_rate": float(np.mean([r["status"] not in ("success", "fail") for r in rs])),
               "n_calls": float(np.mean([r["n_calls"] for r in rs])),
               "max_history_age_steps": max(int(r.get("max_history_age_steps", 0) or 0) for r in rs),
               "max_history_frames": max(int(r.get("max_history_frames", 0) or 0) for r in rs)}
        for t in tasks:
            tr = [r for (task, _), r in rows.items() if task == t]
            row[f"n_{t}"] = len(tr)
            row[f"sr_{t}"] = float(np.mean([bool(r["success"]) for r in tr])) if tr else float("nan")
            row[f"steps_{t}"] = float(np.mean([float(r["steps"]) for r in tr])) if tr else float("nan")
        if full is not None and arm != "full":
            d = paired_delta(rows, full, "success")
            ds = paired_delta(rows, full, "steps")
            row.update({"dsr_vs_full": d["delta"], "dsr_lo": d["lo"], "dsr_hi": d["hi"], "p_sr_vs_full": d["p"],
                        "dsteps_vs_full": ds["delta"], "p_steps_vs_full": ds["p"], "n_paired": d["n"]})
            for t in tasks:
                a = {k: r for k, r in rows.items() if k[0] == t}
                b = {k: r for k, r in full.items() if k[0] == t}
                if set(a) & set(b):
                    dt = paired_delta(a, b, "success")
                    row[f"dsr_{t}"], row[f"p_sr_{t}"] = dt["delta"], dt["p"]
        summary.append(row)

    with open(out / "results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in summary:
            w.writerow([fmt(row.get(c, "")) for c in cols])

    print(f"{args.suite}: {len(summary)} arm(s) -> {out / 'results/summary.csv'}")
    for row in summary:
        per_task = "  ".join(f"{t} {row[f'sr_{t}']:.2f}({row[f'n_{t}']})" for t in tasks)
        delta = (f"  dSR {row['dsr_vs_full']:+.3f} p={row['p_sr_vs_full']:.4f} (n={row['n_paired']})"
                 if "dsr_vs_full" in row else "")
        print(f"  {row['arm']:5s} n={row['n']:3d}/{n_expected}  SR {row['sr']:.3f} [{row['sr_lo']:.3f},{row['sr_hi']:.3f}]"
              f"  steps {row['steps']:.0f}  fail {row['fail_rate']:.2f} timeout {row['timeout_rate']:.2f}"
              f"  hist<= {row['max_history_age_steps']} steps/{row['max_history_frames']} frames{delta}\n        {per_task}")


if __name__ == "__main__":
    main()
