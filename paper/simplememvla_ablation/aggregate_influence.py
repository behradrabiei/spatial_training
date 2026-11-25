#!/usr/bin/env python
"""Aggregate the SimpleMemVLA velocity-field influence run (run_influence.sh).

    robomme_benchmark/.venv-simplememvla/bin/python paper/simplememvla_ablation/aggregate_influence.py
    ... aggregate_influence.py --partial        # while the run is going: finished episodes only

Reads  /mnt/vault/robomme/runs/simplememvla_ablation/<suite>/influence_recent/{influence,episodes}.jsonl
Writes paper_results/simplememvla_influence/<suite>/results/
         steps.csv                one row per decision x K
         recent_curve.csv         one row per K (and the `replay` floor), all tasks
         recent_curve_by_task.csv the same per task
         recent_by_t.csv          per K and flow timestep, the mean influence
         episode_norm.csv         per episode x K, the values the figure's boxes are made of
         summary.json
       and copies the run's small files to .../<suite>/runs/influence_recent/.

Infl(K) of a decision = mean over the full-context sampler's states of || v_full - v_K ||^2 (whole
action chunk, normalised action units); `rel` divides it by the same mean of || v_full ||^2.

Three decision sets, because a window that covers the whole history changes nothing (the input is
identical and the influence is exactly 0):
  cut    decisions where the K window removes at least one frame (n_history > K)
  all    every decision, the zeros included: what a K window does to the episode as it ran
  fixed  decisions with more than FIXED_MIN history frames (the same decisions for every K <= 32)
Means are episode-balanced (mean over episodes of the episode's mean), +- SEM across episodes.
`norm_*` is each episode's mean Infl(K) as a percentage of its mean Infl(1) on the same decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robomme_ablation"))
from aggregate import fmt, load_arm  # noqa: E402

SUITES = {
    "permanence": ["VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "reference": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "imitation": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
RAW_ROOT = Path("/mnt/vault/robomme/runs/simplememvla_ablation")
RESULTS_ROOT = HERE.parents[1] / "paper_results/simplememvla_influence"
ARM = "influence_recent"
REPLAY = "replay"
FIXED_MIN = 32
COPY = ("influence.jsonl", "episodes.jsonl", "results.json", "run_config.json")
SETS = ("cut", "all", "fixed")


def load_decisions(run: Path, episodes: dict) -> list[dict]:
    """Decisions of the recorded episodes; of an episode written twice, the last block only."""
    by_ep: dict[tuple[str, int], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    f = run / "influence.jsonl"
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip():
            r = json.loads(line)
            by_ep[(r["task"], int(r["episode"]))][int(r["block"])].append(r)
    out = []
    for key in sorted(by_ep):
        if key in episodes:
            out.extend(sorted(by_ep[key][max(by_ep[key])], key=lambda r: r["call"]))
    return out


def arm_k(arm: str):
    return REPLAY if arm == REPLAY else int(arm[1:])


def in_set(row: dict, name: str) -> bool:
    if name == "cut":
        return not row["same_input"]
    if name == "fixed":
        return row["n_history"] > FIXED_MIN
    return True


def balanced(per_ep: dict) -> tuple[float, float, int]:
    """Mean over episodes of the episode means, its SEM, number of episodes."""
    v = np.array([np.mean(x) for x in per_ep.values()], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"), int(v.size)


def episode_norm(rows_k: list[dict], rows_1: dict, name: str) -> dict:
    """Per episode: 100 * mean Infl(K) / mean Infl(1) over the decisions of set ``name``."""
    acc: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows_k:
        if in_set(r, name):
            acc[(r["task"], r["episode"])].append((r["infl"], rows_1[(r["task"], r["episode"], r["call"])]))
    return {ep: 100 * np.mean([a for a, _ in v]) / np.mean([b for _, b in v])
            for ep, v in acc.items() if np.mean([b for _, b in v]) > 0}


def curve_row(k, rows_k: list[dict], rows_1: dict | None) -> dict:
    row = {"K": k}
    for name in SETS:
        sel = [r for r in rows_k if in_set(r, name)]
        per_ep, per_ep_rel, per_ep_exec, per_ep_flip = (defaultdict(list) for _ in range(4))
        for r in sel:
            ep = (r["task"], r["episode"])
            per_ep[ep].append(r["infl"])
            per_ep_rel[ep].append(r["rel"])
            per_ep_exec[ep].append(r["infl_exec"])
            per_ep_flip[ep].append(0.0 if r["subtask_same"] else 1.0)
        mean, sem, n_ep = balanced(per_ep)
        infl = np.array([r["infl"] for r in sel], dtype=float)
        same = np.array([r["subtask_same"] for r in sel], dtype=bool)
        row.update({
            f"{name}_n_steps": len(sel), f"{name}_n_episodes": n_ep,
            f"{name}_mean": mean, f"{name}_sem": sem,
            f"{name}_pooled_mean": float(infl.mean()) if infl.size else float("nan"),
            f"{name}_pooled_median": float(np.median(infl)) if infl.size else float("nan"),
            f"{name}_rel_mean": balanced(per_ep_rel)[0], f"{name}_rel_sem": balanced(per_ep_rel)[1],
            f"{name}_exec_mean": balanced(per_ep_exec)[0],
            f"{name}_flip_rate": balanced(per_ep_flip)[0], f"{name}_flip_sem": balanced(per_ep_flip)[1],
            f"{name}_mean_subtask_same": float(infl[same].mean()) if same.any() else float("nan"),
            f"{name}_mean_subtask_changed": float(infl[~same].mean()) if (~same).any() else float("nan"),
        })
        if rows_1 is not None:
            norm = np.array(list(episode_norm(rows_k, rows_1, name).values()), dtype=float)
            row.update({
                f"{name}_norm_mean": float(norm.mean()) if norm.size else float("nan"),
                f"{name}_norm_median": float(np.median(norm)) if norm.size else float("nan"),
                f"{name}_norm_sem": float(norm.std(ddof=1) / np.sqrt(norm.size)) if norm.size > 1 else float("nan"),
            })
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, ""), 6) for c in cols})
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="permanence", choices=sorted(SUITES))
    ap.add_argument("--run", type=Path, default=None, help="raw run dir (default: vault/<suite>/influence_recent)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: paper_results/.../<suite>)")
    ap.add_argument("--partial", action="store_true", help="aggregate a run that has not finished")
    args = ap.parse_args()
    run = args.run or RAW_ROOT / args.suite / ARM
    out = args.out or RESULTS_ROOT / args.suite
    if not (run / "results.json").exists() and not args.partial:
        raise SystemExit(f"{run} has no results.json (still running?); use --partial for an interim look")
    episodes = load_arm(run)
    decisions = load_decisions(run, episodes)
    if not decisions:
        raise SystemExit(f"no finished episode with influence records under {run}")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "runs" / ARM).mkdir(parents=True, exist_ok=True)
    for name in COPY:
        if (run / name).exists():
            shutil.copy2(run / name, out / "runs" / ARM / name)

    arms = list(decisions[0]["arms"])
    steps, by_t = [], defaultdict(list)
    for d in decisions:
        ep = episodes[(d["task"], int(d["episode"]))]
        for arm in arms:
            a = d["arms"][arm]
            steps.append({
                "task": d["task"], "episode": int(d["episode"]), "call": int(d["call"]), "step": int(d["step"]),
                "n_history": int(d["n_history"]), "success": int(bool(ep["success"])),
                "K": arm_k(arm), "infl": a["infl"], "infl_exec": a["infl_exec"],
                "rel": a["infl"] / d["v_sq"], "v_sq": d["v_sq"], "v_sq_exec": d["v_sq_exec"],
                "same_input": bool(a["same_input"]), "subtask_same": bool(a["subtask_same"]),
                "subtask_full": d["subtask"], "subtask_k": a["subtask"],
            })
            if not a["same_input"]:
                by_t[arm_k(arm)].append(a["infl_by_t"])
    write_csv(out / "results/steps.csv", [{**s, "same_input": int(s["same_input"]),
                                           "subtask_same": int(s["subtask_same"])} for s in steps])

    ks = [arm_k(a) for a in arms]
    by_k = {k: [s for s in steps if s["K"] == k] for k in ks}
    ref = {(s["task"], s["episode"], s["call"]): s["infl"] for s in by_k.get(1, [])} or None
    write_csv(out / "results/recent_curve.csv", [curve_row(k, by_k[k], ref) for k in ks])
    tasks = [t for t in SUITES[args.suite] if any(s["task"] == t for s in steps)]
    write_csv(out / "results/recent_curve_by_task.csv", [
        {"task": t, **curve_row(k, [s for s in by_k[k] if s["task"] == t], ref)} for t in tasks for k in ks])
    n_t = len(decisions[0]["v_sq_by_t"])
    t_grid = np.linspace(1.0, 0.0, n_t + 1)[:-1]
    write_csv(out / "results/recent_by_t.csv", [
        {"K": k, "t": float(t_grid[i]), "n_steps": len(v), "mean": float(np.mean([x[i] for x in v]))}
        for k, v in by_t.items() for i in range(n_t)])
    if ref is not None:
        write_csv(out / "results/episode_norm.csv", [
            {"set": name, "K": k, "task": ep[0], "episode": ep[1], "norm": v}
            for name in SETS for k in ks if k != REPLAY
            for ep, v in sorted(episode_norm(by_k[k], ref, name).items())])

    replay = np.array([s["infl"] for s in by_k.get(REPLAY, [])], dtype=float)
    n_ep = len({(d["task"], d["episode"]) for d in decisions})
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    summary = {
        "suite": args.suite, "complete": (run / "results.json").exists(),
        "n_episodes": n_ep, "n_decisions": len(decisions),
        "episodes_per_task": {t: len({d["episode"] for d in decisions if d["task"] == t}) for t in tasks},
        "success_rate": float(np.mean([episodes[k]["success"] for k in {(d["task"], int(d["episode"])) for d in decisions}])),
        "ks": [k for k in ks if k != REPLAY],
        "noise_draws": cfg.get("influence_noise"), "probe_seed": cfg.get("influence_seed"),
        "replay_floor_max": float(replay.max()) if replay.size else None,
        "replay_floor_mean": float(replay.mean()) if replay.size else None,
        "v_sq_mean": float(np.mean([d["v_sq"] for d in decisions])),
        "n_history_max": int(max(d["n_history"] for d in decisions)),
        "probe_seconds_total": float(sum(d["probe_s"] for d in decisions)),
        "fixed_min_history": FIXED_MIN,
    }
    (out / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out / "results/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
