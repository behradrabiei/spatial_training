#!/usr/bin/env python3
"""Aggregate the pruning-KL probes into paper_results/prune_kl/results/.

    Prune_t(M, k) = KL(P(action | full context) || P(action | the visual tokens method M holds at
                                                                 k frames' worth))

on the full-context trajectory (expectation under the full-context decision, T = 1, nats): the
pruning analogue of the recent-window measure of paper/frame_influence and the blind-from-layer
measure of paper/info_horizon. One shadow pruner per (method, k) -- random / FastV / DivPrune /
DART at k = 4, 8, 16, 32 frames' worth -- is advanced along the trajectory with the flat
token-pruning arm's rule (permanent deletion). Arms (the 10 seed-17 HM3D episodes, every decision):

  runs/r1_prune_kl         LongNav-R1, leaky decision replay (vlm.prune_kl_probe; 4 actions)
  runs/lightnav_prune_kl   LightNav-0, full re-prefill, leak-free by construction
                           (eval_influence.py --measure prune_kl; 256 <act_l0> codes and their
                           6 executed-action bins)

The headline average is over the decisions where the budget binds (some method's shadow hides a
token: the same decisions for every method at a given k); the all-decisions mean is kept beside it.

Writes:
  prune_kl_steps.csv.gz   every (model, episode, step, method, k) (prune_kl.PRUNE_KL_FIELDS)
  prune_kl_episode.csv    per (model, episode, method, k): the episode's means over its binding
                          decisions, the all-decisions mean, KL normalised by the no-visual KL
  prune_kl_curve.csv      per (model, method, k): pooled mean + episode-bootstrap CI,
                          episode-balanced mean +- SEM, median, flip rates, normalised KL, the
                          no-visual ceiling, and the all-decisions means
  prune_kl_summary.json   arm completeness, noise floors and the reference checks: both runs
                          against the recent-window runs (same decisions, same full-context
                          distribution, same no-visual KL), the LightNav-0 random shadow against the
                          masks of the live lightnav_random_b<k> pruning runs, and, when the run
                          carries recency check arms, those against the recent-window run's
                          old-blocks-kept probes

Runs in longnav_vlm (numpy only):  python paper/prune_kl/aggregate.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import influence as inf
from longnav.analysis import prune_kl as pk
from longnav.analysis.results_io import load_lightnav_run, load_manifest, load_r1_run, split_status

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/prune_kl"
MANIFEST = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
RECENT_ROOT = ROOT / "paper_results/frame_influence_recent/runs"
FLAT_ROOT = ROOT / "paper_results/token_prune_flat/runs"
# model -> (arm, row loader, run loader, recent-window run of the same decisions)
ARMS = {
    "LongNav-R1": ("r1_prune_kl", pk.load_r1_prune_kl, load_r1_run, "r1_influence_recent"),
    "LightNav-0": ("lightnav_prune_kl", pk.load_lightnav_prune_kl, load_lightnav_run, "lightnav_influence_recent"),
}


def write_csv(path: Path, rows: list[dict], fields: list[str], gz: bool = False) -> None:
    opener = (lambda: gzip.open(path, "wt", newline="")) if gz else (lambda: path.open("w", newline=""))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items() if k in fields})


def _diff_stats(d: list[float]) -> dict:
    a = np.asarray(d, dtype=float)
    return {"n": int(a.size), "abs_diff_max": float(a.max()) if a.size else None,
            "abs_diff_median": float(np.median(a)) if a.size else None}


def r1_reference_check(run_dir: Path, ref_dir: Path) -> dict:
    """The probe run against the recent-window run on the same deterministic trajectories: equal
    episode lengths, executed actions and live probabilities; the replayed full decision and the
    no-visual KL logged by this run's joint probes and by its shadow probe must be the recent
    run's (same replay on the same cache)."""
    ref = {label: d for label, d in inf._r1_sequences(ref_dir)}
    out = {"episodes": 0, "same_length": 0, "same_actions": 0, "probs_max_abs_diff": 0.0, "base_logp_equal": 0,
           "visual_all_equal": 0, "history_all_equal": 0, "probe_base_equals_joint_base": 0, "probe_novis_equals_joint": 0}
    for label, d in inf._r1_sequences(run_dir):
        r = ref.get(label)
        if r is None or not d.get("sup/prune_kl_kl"):
            continue
        out["episodes"] += 1
        same_len = len(d["sup/prune_kl_kl"]) == len(r["sup/kl_influence_base_logp"])
        out["same_length"] += int(same_len)
        out["same_actions"] += int(d.get("action_history") == r.get("action_history"))
        if same_len:
            gap = np.abs(np.asarray(d["sup/action_probs"], dtype=float) - np.asarray(r["sup/action_probs"], dtype=float))
            out["probs_max_abs_diff"] = max(out["probs_max_abs_diff"], float(gap.max()))
        out["base_logp_equal"] += int(d.get("sup/kl_influence_base_logp") == r["sup/kl_influence_base_logp"])
        out["visual_all_equal"] += int(d.get("sup/kl_influence_visual_all") == r["sup/kl_influence_visual_all"])
        out["history_all_equal"] += int(d.get("sup/kl_influence_history_all") == r["sup/kl_influence_history_all"])
        out["probe_base_equals_joint_base"] += int(d["sup/prune_kl_base_logp"] == d.get("sup/kl_influence_base_logp"))
        out["probe_novis_equals_joint"] += int(d["sup/prune_kl_visual_all"] == d.get("sup/kl_influence_visual_all"))
    return out


def lightnav_reference_check(run_dir: Path, ref_dir: Path) -> dict:
    """The probe run against the recent-window run: equal decisions (generated tokens) at every
    step, and the no-visual KL of the two runs on those decisions."""
    ref = {(label, int(rec["step"])): rec for label, rec in pk.lightnav_records(ref_dir)}
    n = same_tokens = 0
    diffs, steps_here, steps_ref = [], {}, {}
    for label, rec in pk.lightnav_records(run_dir):
        steps_here[label] = steps_here.get(label, 0) + 1
        other = ref.get((label, int(rec["step"])))
        if other is None:
            continue
        n += 1
        same_tokens += int(rec["full"].get("tokens") == other["full"].get("tokens"))
        a, b = (rec.get("visual_all_mask") or {}).get("kl"), (other.get("visual_all_mask") or {}).get("kl")
        if a is not None and b is not None:
            diffs.append(abs(a - b))
    for (label, _) in ref:
        steps_ref[label] = steps_ref.get(label, 0) + 1
    return {"n_common": n, "same_decision_tokens": same_tokens,
            "episodes_same_length": sum(1 for e, c in steps_here.items() if steps_ref.get(e) == c),
            "episodes": len(steps_here), "novis_kl": _diff_stats(diffs)}


def lightnav_random_mask_check(run_dir: Path, flat_root: Path) -> dict:
    """The random shadow's keep bits against the masks the live ``lightnav_random_b<k>`` pruning
    run logged for the same episode and step. The tubelet layout depends on the step alone and the
    random draw on (seed, episode counter, step), so the two must agree on every step both runs
    reached, whatever their trajectories."""
    out = {}
    live_cache: dict[tuple, dict] = {}
    for label, rec in pk.lightnav_records(run_dir):
        for pr in rec["prune_kl"]:
            if pr["method"] != "random":
                continue
            k = int(pr["k"])
            path = flat_root / f"lightnav_random_b{k}" / "prune" / f"{label}.jsonl"
            if (k, label) not in live_cache:
                live_cache[(k, label)] = ({int(r["step"]): r for r in map(json.loads, path.read_text().splitlines())}
                                          if path.is_file() else {})
            live = live_cache[(k, label)].get(int(rec["step"]))
            s = out.setdefault(f"k={k}", {"n_common": 0, "same_layout": 0, "same_bits": 0, "same_pool": 0, "steps_dropping": 0})
            if live is None:
                continue
            s["n_common"] += 1
            s["same_layout"] += int(live["tubelets"] == rec["tubelets"] and live["pools"] == rec["pools"])
            s["same_bits"] += int(live["bits"] == pr["bits"])
            s["same_pool"] += int(live["n_pool"] == pr["n_pool"] and live["n_dropped"] == pr["n_dropped"])
            s["steps_dropping"] += int(pr["n_dropped"] > 0)
    return out


def lightnav_recency_check(run_dir: Path, ref_dir: Path) -> dict:
    """Recency check arms (``recency:k``, k frames' worth = the newest k / 2 history tubelets)
    against the recent-window run's old-blocks-kept probe at K = k + 2 frames, on the steps where
    that probe exists (the newest history tubelets are exactly the K - 2 frames before the
    current pair)."""
    ref = {(label, int(rec["step"])): {int(p["k"]): p for p in rec.get("recent_keep", [])}
           for label, rec in pk.lightnav_records(ref_dir)}
    out = {}
    for label, rec in pk.lightnav_records(run_dir):
        for pr in rec["prune_kl"]:
            if pr["method"] != "recency" or pr["n_hidden"] == 0:
                continue
            k = int(pr["k"])
            other = ref.get((label, int(rec["step"])), {}).get(k + 2)
            s = out.setdefault(f"recency:{k} vs K={k + 2}", {"n_binding": 0, "diffs": [], "same_code": 0})
            s["n_binding"] += 1
            if other is None or other.get("kl") is None or pr.get("kl") is None:
                continue
            s["diffs"].append(abs(pr["kl"] - other["kl"]))
            s["same_code"] += int(pr.get("code") == other.get("code"))
    return {name: {"n_binding": s["n_binding"], "same_code": s["same_code"], "kl": _diff_stats(s["diffs"])}
            for name, s in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULTS_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "results")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--models", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--arm", action="append", default=[], metavar="MODEL=ARM",
                    help="read another run directory for a model (smoke runs), e.g. LongNav-R1=r1_prune_kl_smoke")
    ap.add_argument("--allow-partial", action="store_true", help="aggregate incomplete arms (interim looks only)")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    arm_names = {m: a[0] for m, a in ARMS.items()} | dict(x.split("=", 1) for x in args.arm)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, summary, incomplete = [], {"arms": {}, "checks": {}}, []
    for model in args.models:
        _, loader, run_loader, recent_arm = ARMS[model]
        arm = arm_names[model]
        run_dir = args.runs / arm
        if not run_dir.is_dir():
            incomplete.append(f"{arm}: missing")
            continue
        missing, extra = split_status(run_loader(run_dir), manifest)
        if missing or extra:
            incomplete.append(f"{arm}: {len(missing)} of {len(manifest)} episodes missing, {len(extra)} extra")
        arm_rows = loader(run_dir, model)
        probed = sorted({r["episode"] for r in arm_rows})
        if set(probed) != set(manifest) - set(missing):
            incomplete.append(f"{arm}: probe rows for {len(probed)} episodes, results for {len(manifest) - len(missing)}")
        steps = {(r["episode"], r["step"]) for r in arm_rows}
        floors = {(r["episode"], r["step"]): r["kl_floor"] for r in arm_rows if np.isfinite(r["kl_floor"])}
        fl = np.array(list(floors.values()))
        summary["arms"][arm] = {
            "model": model, "methods": sorted({r["method"] for r in arm_rows}), "budgets": sorted({r["k"] for r in arm_rows}),
            "episodes": len(probed), "decisions": len(steps), "rows": len(arm_rows),
            "rows_without_value": int(sum(1 for r in arm_rows if not np.isfinite(r["kl"]))),
            "binding_decisions": {str(k): len({(r["episode"], r["step"]) for r in arm_rows if r["k"] == k and r["binding"]})
                                  for k in sorted({r["k"] for r in arm_rows})},
            "kl_floor_mean": float(fl.mean()) if fl.size else None, "kl_floor_max": float(fl.max()) if fl.size else None,
        }
        recent = RECENT_ROOT / recent_arm
        if recent.is_dir():
            check = r1_reference_check if model == "LongNav-R1" else lightnav_reference_check
            summary["checks"][f"{arm} vs {recent_arm}"] = check(run_dir, recent)
        if model == "LightNav-0":
            summary["checks"][f"{arm}: random shadow vs live lightnav_random_b<k> masks"] = lightnav_random_mask_check(run_dir, FLAT_ROOT)
            recency = lightnav_recency_check(run_dir, recent) if recent.is_dir() else {}
            if recency:
                summary["checks"][f"{arm}: recency shadow vs {recent_arm} old-blocks-kept"] = recency
        rows += arm_rows
    summary["incomplete"] = incomplete
    if incomplete and not args.allow_partial:
        sys.exit("incomplete arms (pass --allow-partial for an interim look):\n  " + "\n  ".join(incomplete))
    if not rows:
        sys.exit("no probe rows found")

    order = {m: i for i, m in enumerate(pk.METHODS)}
    curve = sorted(pk.prune_kl_curve(rows), key=lambda r: (list(ARMS).index(r["model"]), order.get(r["method"], 99), r["k"]))
    episodes = pk.prune_kl_episode_table(rows)
    write_csv(args.out / "prune_kl_steps.csv.gz", rows, pk.PRUNE_KL_FIELDS, gz=True)
    write_csv(args.out / "prune_kl_episode.csv", episodes, list(episodes[0]))
    write_csv(args.out / "prune_kl_curve.csv", curve, list(dict.fromkeys(k for r in curve for k in r)))
    (args.out / "prune_kl_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"{'model':<12} {'method':<9} {'k':>3} {'KL pooled':>10} {'KL bal':>9} {'+-SEM':>8} {'median':>9} {'norm':>6} "
          f"{'no-vis':>8} {'KL exec':>9} {'flip':>6} {'flip ex':>7} {'KL all':>9} {'steps':>6} {'eps':>4}")
    for r in curve:
        def g(key, spec):
            v = r.get(key, float("nan"))
            return format(v, spec) if isinstance(v, (int, float)) and np.isfinite(v) else "-".rjust(int(spec.split(".")[0]))
        print(f"{r['model']:<12} {r['method']:<9} {r['k']:>3} {g('kl_mean', '10.5f')} {g('kl_balanced', '9.5f')} "
              f"{g('kl_sem', '8.5f')} {g('kl_median', '9.5f')} {g('kl_norm_balanced', '6.3f')} {g('novis_mean', '8.4f')} "
              f"{g('kl_exec_mean', '9.5f')} {g('flip_mean', '6.3f')} {g('flip_exec_mean', '7.3f')} {g('kl_all_mean', '9.5f')} "
              f"{r['n_steps']:>6} {r['n_episodes']:>4}")
    for name, check in summary["checks"].items():
        print("check:", name, json.dumps(check))
    for line in incomplete:
        print("INCOMPLETE:", line)
    print(f"wrote {args.out / 'prune_kl_curve.csv'}, prune_kl_episode.csv, prune_kl_steps.csv.gz, prune_kl_summary.json")


if __name__ == "__main__":
    main()
