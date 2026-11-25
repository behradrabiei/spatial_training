#!/usr/bin/env python3
"""Aggregate the blind-from-layer KL probes into paper_results/info_horizon/results/.

    Blind_t(L) = KL(P(action | full context) || P(action | every visual token hidden from layer L on))

the depth analogue of the recent-window measure of paper/frame_influence (expectation under the
full-context decision, T = 1, nats). L = 0 is KL(full || no visual), that study's K = 0 point;
L = n_layers hides nothing and is 0. Arms (the 10 seed-17 HM3D episodes, every decision):

  runs/r1_probe_leaky_dense   LongNav-R1, leaky decision replay, every layer 0..27
                              (vlm.horizon_probe_blind; 4 actions). Also carries the keep-one-in /
                              leave-one-out probes at every layer and KL(full || history hidden).
  runs/lightnav_probe         LightNav-0, full re-prefill (leak-free by construction), every second
                              layer 0..34 (eval_influence.py --measure blind_layer; 256 <act_l0>
                              codes and their 6 executed-action bins).

Writes:
  blind_steps.csv.gz   every (model, episode, step, layer): kl, kl_exec, flips, argmaxes, entropies,
                       the LongNav-R1 side measures, token counts (horizon.BLIND_FIELDS)
  blind_episode.csv    per (model, episode, layer): the episode's means, median, and its curve
                       normalised by its own value at the shallowest layer
  blind_curve.csv      per (model, layer): pooled mean + episode-bootstrap CI, episode-balanced mean
                       +- SEM, median, flip rates, normalised curve (balanced, SEM, quartiles); the
                       nothing-hidden endpoint L = n_layers is appended (0 by construction for
                       LongNav-R1, the measured replay floor for LightNav-0)
  blind_summary.json   arm completeness, noise floors, and reference checks against the
                       recent-window runs (L = 0 vs their K = 0) and the sparse leaky probe

Runs in longnav_vlm (numpy only):  python paper/info_horizon/aggregate_blind.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import horizon as hz
from longnav.analysis import influence as inf
from longnav.analysis.results_io import load_lightnav_run, load_manifest, load_r1_run, split_status

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/info_horizon"
MANIFEST = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
RECENT_ROOT = ROOT / "paper_results/frame_influence_recent/runs"
# model -> (arm, row loader, run loader, n_layers, recent-window run for the L = 0 check, its loader kwargs)
ARMS = {
    "LongNav-R1": ("r1_probe_leaky_dense", hz.load_r1_blind, load_r1_run, 28,
                   "r1_influence_recent", inf.load_r1_influence, {}),
    "LightNav-0": ("lightnav_probe", hz.load_lightnav_blind, load_lightnav_run, 36,
                   "lightnav_influence_recent", inf.load_lightnav_influence, {"recent": "keep"}),
}
SPARSE_PROBE = "r1_probe_leaky"  # the 2026-09 leaky probe on a 10-layer grid: keep/leave must agree


def write_csv(path: Path, rows: list[dict], fields: list[str], gz: bool = False) -> None:
    opener = (lambda: gzip.open(path, "wt", newline="")) if gz else (lambda: path.open("w", newline=""))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items() if k in fields})


def reference_check(rows: list[dict], steps: list[dict], first_layer: int) -> dict:
    """The probe's shallowest layer against the recent-window run's K = 0 (KL(full || no visual))
    on the decisions both runs scored; the executed actions must agree for the two to be the
    same decisions."""
    ref = {(s["episode"], s["step"]): s for s in steps}
    diffs, same_action, n = [], 0, 0
    for r in rows:
        if r["layer"] != first_layer or not np.isfinite(r["kl"]):
            continue
        s = ref.get((r["episode"], r["step"]))
        if s is None or not np.isfinite(s["visual_all"]):
            continue
        n += 1
        diffs.append(abs(r["kl"] - s["visual_all"]))
        same_action += int(str(r["chosen_action"]) == str(s["chosen_action"]))
    if not n:
        return {"n_common": 0}
    d = np.asarray(diffs)
    return {"n_common": n, "same_chosen_action": same_action, "abs_diff_median": float(np.median(d)),
            "abs_diff_p95": float(np.percentile(d, 95)), "abs_diff_max": float(d.max()),
            "mean_here": float(np.mean([r["kl"] for r in rows if r["layer"] == first_layer and np.isfinite(r["kl"])])),
            "mean_reference": float(np.mean([s["visual_all"] for s in steps if np.isfinite(s["visual_all"])]))}


def sparse_probe_check(rows: list[dict], run_dir: Path) -> dict:
    """keep-one-in / leave-one-out of the dense run against the sparse leaky probe at the layers
    both probed (same deterministic trajectories, same replay: expected to agree to float noise)."""
    old, _ = hz.load_r1_horizon(run_dir)
    ref = {(r["episode"], r["step"], r["layer"]): r for r in old}
    out = {}
    for key in ("keep_one", "leave_one"):
        d = [abs(r[key] - ref[(r["episode"], r["step"], r["layer"])][key]) for r in rows
             if (r["episode"], r["step"], r["layer"]) in ref and r["n_cur_tokens"] != 0]
        out[key] = {"n_common": len(d), "abs_diff_max": float(max(d)) if d else None,
                    "abs_diff_median": float(np.median(d)) if d else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULTS_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "results")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--models", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--allow-partial", action="store_true", help="aggregate incomplete arms (interim looks only)")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, summary, incomplete = [], {"arms": {}, "checks": {}}, []
    for model in args.models:
        arm, loader, run_loader, n_layers, recent_arm, recent_loader, recent_kw = ARMS[model]
        run_dir = args.runs / arm
        if not run_dir.is_dir():
            incomplete.append(f"{arm}: missing")
            continue
        missing, extra = split_status(run_loader(run_dir), manifest)
        if missing or extra:
            incomplete.append(f"{arm}: {len(missing)} of {len(manifest)} episodes missing, {len(extra)} extra")
        arm_rows, layers = loader(run_dir, model) if model != "LongNav-R1" else loader(run_dir, model, n_layers)
        probed = sorted({r["episode"] for r in arm_rows})
        if set(probed) != set(manifest) - set(missing):
            incomplete.append(f"{arm}: probe rows for {len(probed)} episodes, results for {len(manifest) - len(missing)}")
        steps = {(r["episode"], r["step"]) for r in arm_rows}
        floors = np.array([r["kl_floor"] for r in arm_rows if r["layer"] == layers[0] and np.isfinite(r["kl_floor"])]) if layers else np.array([])
        summary["arms"][arm] = {
            "model": model, "n_layers": n_layers, "layers": layers, "episodes": len(probed), "decisions": len(steps),
            "rows": len(arm_rows), "rows_without_value": int(sum(1 for r in arm_rows if not np.isfinite(r["kl"]))),
            "kl_floor_mean": float(floors.mean()) if floors.size else None,
            "kl_floor_max": float(floors.max()) if floors.size else None,
        }
        if (RECENT_ROOT / recent_arm).is_dir() and layers:
            _, recent_steps = recent_loader(RECENT_ROOT / recent_arm, model, measure="recent_k", **recent_kw)
            summary["checks"][f"{arm}: L={layers[0]} vs {recent_arm} K=0"] = reference_check(arm_rows, recent_steps, layers[0])
        if model == "LongNav-R1" and (args.runs / SPARSE_PROBE).is_dir():
            summary["checks"][f"{arm} vs {SPARSE_PROBE}"] = sparse_probe_check(arm_rows, args.runs / SPARSE_PROBE)
        rows += arm_rows
    summary["incomplete"] = incomplete
    if incomplete and not args.allow_partial:
        sys.exit("incomplete arms (pass --allow-partial for an interim look):\n  " + "\n  ".join(incomplete))
    if not rows:
        sys.exit("no probe rows found")

    curve = hz.blind_curve(rows)
    # the nothing-hidden endpoint: 0 by construction for the replay probe, the measured replay floor for LightNav-0
    for model in args.models:
        mine = [r for r in rows if r["model"] == model]
        if not mine:
            continue
        n_layers = mine[0]["n_layers"]
        per_ep: dict[str, dict[int, float]] = {}
        for r in mine:
            if np.isfinite(r["kl_floor"]):
                per_ep.setdefault(r["episode"], {})[r["step"]] = r["kl_floor"]
        measured = model == "LightNav-0"
        ep_means = np.array([np.mean(list(v.values())) for v in per_ep.values()]) if measured and per_ep else np.zeros(1)
        end = {"model": model, "layer": n_layers, "n_layers": n_layers, "layer_frac": 1.0,
               "n_steps": len({(r["episode"], r["step"]) for r in mine}) if measured else 0,
               "n_episodes": len(per_ep) if measured else 0,
               "kl_mean": float(np.mean([x for v in per_ep.values() for x in v.values()])) if measured and per_ep else 0.0,
               "kl_balanced": float(ep_means.mean()), "kl_norm_balanced": float("nan"),
               "flip_mean": 0.0 if not measured else float("nan"), "flip_exec_mean": 0.0 if not measured else float("nan")}
        if not measured:
            end |= {"kl_exec_mean": 0.0, "kl_exec_balanced": 0.0, "kl_norm_balanced": 0.0, "kl_exec_norm_balanced": 0.0}
        curve.append(end)
    curve.sort(key=lambda r: (list(ARMS).index(r["model"]), r["layer"]))
    episodes = hz.blind_episode_table(rows)

    write_csv(args.out / "blind_steps.csv.gz", rows, hz.BLIND_FIELDS, gz=True)
    write_csv(args.out / "blind_episode.csv", episodes, list(episodes[0]))
    fields = list(dict.fromkeys(k for r in curve for k in r))
    write_csv(args.out / "blind_curve.csv", curve, fields)
    (args.out / "blind_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"{'model':<12} {'L':>3} {'KL bal':>9} {'+-SEM':>8} {'KL pooled':>10} {'median':>9} {'norm':>6} "
          f"{'KL exec':>9} {'flip':>6} {'flip ex':>7} {'steps':>6}")
    for r in curve:
        def g(key, spec):
            v = r.get(key, float("nan"))
            return format(v, spec) if isinstance(v, (int, float)) and np.isfinite(v) else "-"
        print(f"{r['model']:<12} {r['layer']:>3} {g('kl_balanced', '9.4f')} {g('kl_sem', '8.4f')} {g('kl_mean', '10.4f')} "
              f"{g('kl_median', '9.4f')} {g('kl_norm_balanced', '6.3f')} {g('kl_exec_balanced', '9.4f')} "
              f"{g('flip_mean', '6.3f')} {g('flip_exec_mean', '7.3f')} {r['n_steps']:>6}")
    for name, check in summary["checks"].items():
        print("check:", name, json.dumps(check))
    for line in incomplete:
        print("INCOMPLETE:", line)
    print(f"wrote {args.out / 'blind_curve.csv'}, blind_episode.csv, blind_steps.csv.gz, blind_summary.json")


if __name__ == "__main__":
    main()
