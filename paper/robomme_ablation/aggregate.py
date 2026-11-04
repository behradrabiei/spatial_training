#!/usr/bin/env python3
"""Aggregate the RoboMME memory-ablation runs (paper/robomme_ablation/arms.py).

Reads <runs>/<arm>/ckpt79999/seed7/episodes.jsonl (one line per finished episode written by the
patched examples/robomme/eval.py: task, episode, difficulty, outcome, steps, n_calls, wall_s) and
writes paper_results/robomme_ablation/<model>/results/
  per_episode.csv   arm, family, x, task, episode, difficulty, outcome, success, steps, n_calls
  summary.csv       one row per arm: n, sr (+95% bootstrap CI), steps (+CI), steps on successes,
                    timeout rate, per-task SR, dsr/dsteps vs full with paired sign-flip p, n_calls
and copies each arm's episodes.jsonl / log.json / policy_calls.jsonl into
paper_results/robomme_ablation/<model>/runs/<arm>/ (compact inputs; videos stay on the vault).
`--model all` (the default) aggregates every model with runs and also writes the combined
paper_results/robomme_ablation/results/summary.csv (same columns, with a leading `model`).

Statistics follow the navigation aggregators (longnav.analysis.stats): percentile bootstrap of the
mean (2000 resamples, seed 17) and a two-sided paired sign-flip test (10 000 resamples) on episodes
paired by (task, episode). Pure numpy, so it runs with any python:

    python paper/robomme_ablation/aggregate.py [--model modul|context|all] [--runs <raw runs dir>]
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
sys.path.insert(0, str(HERE))
from arms import ARMS, RUN_ORDER, TASKS  # noqa: E402

MODELS = {  # short name -> (checkpoint, raw runs dir); results go to RESULTS_ROOT/<short name>/
    "modul": ("perceptual-framesamp-modul", Path("/mnt/vault/robomme/runs/ablation")),
    "context": ("perceptual-framesamp-context", Path("/mnt/vault/robomme/runs/ablation_context")),
}
RESULTS_ROOT = HERE.parents[1] / "paper_results/robomme_ablation"
N_BOOT, N_PERM, SEED = 2000, 10000, 17


# ---- stats (copied from src/longnav/analysis/stats.py so this runs outside the nav envs) ----
def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def sign_flip_p(deltas, n_perm=N_PERM, seed=SEED):
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    return float(((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def paired_delta(rows_a, rows_b, key):
    common = sorted(set(rows_a) & set(rows_b))
    a = np.array([float(rows_a[k][key]) for k in common])
    b = np.array([float(rows_b[k][key]) for k in common])
    d = a - b
    mean, lo, hi = bootstrap_mean_ci(d)
    return {"delta": mean, "lo": lo, "hi": hi, "p": sign_flip_p(d), "n": len(common)}


# ---- loading ----
def load_arm(run_dir: Path) -> dict[tuple[str, int], dict]:
    f = run_dir / "episodes.jsonl"
    if not f.exists():
        return {}
    rows = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[(r["task"], int(r["episode"]))] = r  # a rerun of an episode overrides the older line
    return rows


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS) + ["all"], default="all")
    ap.add_argument("--runs", type=Path, default=None, help="raw runs dir (default: the model's; single model only)")
    args = ap.parse_args()
    models = sorted(MODELS) if args.model == "all" else [args.model]
    if args.runs is not None and len(models) > 1:
        ap.error("--runs needs a single --model")
    combined = []
    for model in models:
        runs = args.runs or MODELS[model][1]
        if not runs.exists():
            continue
        combined += aggregate_model(model, runs, RESULTS_ROOT / model)
    if args.model == "all" and combined:
        out = RESULTS_ROOT / "results" / "summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            w.writeheader(); w.writerows(combined)
        print(f"combined summary: {len(combined)} arm rows -> {out}")


def aggregate_model(model: str, runs: Path, root: Path) -> list[dict]:
    """Aggregate one model's raw runs into root/{results,runs}; returns the summary rows."""
    res = root / "results"
    res.mkdir(parents=True, exist_ok=True)
    print(f"== {model} ({MODELS[model][0]}), runs {runs}")

    per_arm = {}
    for arm in RUN_ORDER:
        run_dir = runs / arm / "ckpt79999/seed7"
        rows = load_arm(run_dir)
        if not rows:
            continue
        per_arm[arm] = rows
        dst = root / "runs" / arm
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("episodes.jsonl", "log.json", "progress.json", "policy_calls.jsonl"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, dst / name)

    # per-episode table
    with (res / "per_episode.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "family", "x", "task", "episode", "difficulty", "outcome", "success", "steps", "n_calls"])
        for arm, rows in per_arm.items():
            for (task, ep), r in sorted(rows.items()):
                w.writerow([arm, ARMS[arm]["family"], ARMS[arm]["x"], task, ep, r.get("difficulty", ""),
                            r["outcome"], int(bool(r["success"])), r.get("steps", ""), r.get("n_calls", "")])

    # summary
    full = per_arm.get("full")
    cols = (["arm", "family", "x", "n", "n_expected", "sr", "sr_lo", "sr_hi", "steps", "steps_lo", "steps_hi",
             "steps_success", "timeout_rate", "n_calls"]
            + [f"sr_{t}" for t in TASKS]
            + ["dsr_vs_full", "dsr_lo", "dsr_hi", "p_sr_vs_full", "dsteps_vs_full", "p_steps_vs_full", "n_paired"])
    summary_rows = []
    with (res / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, rows in per_arm.items():
            succ = np.array([float(bool(r["success"])) for r in rows.values()])
            steps = np.array([float(r.get("steps", np.nan)) for r in rows.values()])
            sr, sr_lo, sr_hi = bootstrap_mean_ci(succ)
            st, st_lo, st_hi = bootstrap_mean_ci(steps)
            row = {"arm": arm, "family": ARMS[arm]["family"], "x": ARMS[arm]["x"], "n": len(rows),
                   "n_expected": 50 * len(TASKS), "sr": fmt(sr), "sr_lo": fmt(sr_lo), "sr_hi": fmt(sr_hi),
                   "steps": fmt(st, 1), "steps_lo": fmt(st_lo, 1), "steps_hi": fmt(st_hi, 1),
                   "steps_success": fmt(float(np.nanmean(steps[succ > 0])) if (succ > 0).any() else float("nan"), 1),
                   "timeout_rate": fmt(float(np.mean([r["outcome"] == "timeout" for r in rows.values()]))),
                   "n_calls": fmt(float(np.nanmean([float(r.get("n_calls", np.nan)) for r in rows.values()])), 1)}
            for t in TASKS:
                v = [float(bool(r["success"])) for (task, _), r in rows.items() if task == t]
                row[f"sr_{t}"] = fmt(float(np.mean(v))) if v else ""
            if full is not None and arm != "full":
                for r in rows.values():
                    r["_s"] = float(bool(r["success"])); r["_st"] = float(r.get("steps", np.nan))
                for r in full.values():
                    r["_s"] = float(bool(r["success"])); r["_st"] = float(r.get("steps", np.nan))
                d = paired_delta(rows, full, "_s")
                ds = paired_delta(rows, full, "_st")
                row.update({"dsr_vs_full": fmt(d["delta"]), "dsr_lo": fmt(d["lo"]), "dsr_hi": fmt(d["hi"]),
                            "p_sr_vs_full": fmt(d["p"]), "dsteps_vs_full": fmt(ds["delta"], 1),
                            "p_steps_vs_full": fmt(ds["p"]), "n_paired": d["n"]})
            w.writerow(row)
            summary_rows.append({"model": model, **{c: row.get(c, "") for c in cols}})
    print(f"{len(per_arm)} arms -> {res / 'summary.csv'}")
    for arm, rows in per_arm.items():
        s = np.mean([float(bool(r['success'])) for r in rows.values()])
        print(f"  {arm:14s} n={len(rows):3d} SR={s:.3f}")
    return summary_rows


if __name__ == "__main__":
    main()
