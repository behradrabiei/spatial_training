#!/usr/bin/env python3
"""Aggregate the leave-one-frame-out influence experiment into paper_results/frame_influence/results/.

Reads the LongNav-R1 arm (runs/r1_influence: rollout/*/*/sequence.json + rollout/results_*) and
the Uni-NaVid arm (runs/uninavid_influence: influence/*.jsonl + episodes/*.json + traces/), checks
both against the 10-episode manifest, and writes:

  influence.csv.gz     one row per (model, episode, step t, history frame i): age t-i,
                       Infl_t(i) = KL(pi_full || pi_without_i), the two log-prob vectors, the
                       chosen/argmax actions, whether removal flips the argmax, episode success,
                       and the covariates (action taken at frame i, decision entropy, distance to
                       goal). This is the join table for captured-influence analyses.
  steps.csv            one row per decision: full distribution, joint probes (history_all,
                       visual_all, age0, horizon_K), sum of single-frame influences, covariates.
  lightnav_agreement.csv  (keep_one, LightNav-0 only) per age: a lone tubelet's agreement with the
                       full-context decision (6-bin / 256-code argmax), p of the full decision's bin, and
                       KL(full || only tubelet); pooled + episode-balanced means with SEMs
  lightnav_agreement_episodes.csv  the per-(age, episode) 6-bin agreement behind those means (box plot)
  age_curve.csv        mean/std/median/n per age with episode-cluster bootstrap CIs and the
                       episode-balanced mean, per model.
  age_bins.csv         mean/median/mass share per age bin, per model.
  cumulative_share.csv share of single-frame influence in the k most recent frames, k in
                       {1,4,8,16,32}: pooled over steps with >= 32 history frames and mean per step.
  horizon.csv          joint influence of the frames of age >= K (what the last K frames do not
                       explain) with CIs, next to the additive sum of their single-frame influences.
  persistence.csv      age-detrended (and raw) Spearman rho of frame influence across lags,
                       top-quartile retention, plus revival counts (summary.json).
  covariates.csv       mean influence by age bin x covariate level; ols.json: OLS of log influence.
  flips.csv            argmax-flip rate by age bin.
  summary.csv/json     headline numbers per model; table.tex: booktabs (cumulative share, horizon).

--measure keep_one aggregates the leave-one-IN variant instead (arms r1_influence_keepone and
uninavid_influence_keepone under paper_results/frame_influence_keepone/): infl = KL(only frame
i's visual tokens || no visual tokens), age0 = the current frame alone vs no visual; logp_full then
holds the reference (no-visual) log-probs and flip compares the only-frame argmax against it. The
joint probes (history_all, visual_all, horizon_K) keep their full-context meaning.

--measure keep_one_full is derived offline from the SAME keep-one runs (no arm of its own; LongNav-R1
leaky and LightNav-0 only): infl = KL(full decision || only frame i's visual tokens), i.e. how far
the lone frame is from reproducing the decision actually taken; age0 = the current frame alone vs
the full decision; logp_full = the full decision; visual_all = KL(full || no visual), the ceiling an
uninformative frame sits at (summary: visual_all_mean / visual_all_balanced). Results go to
paper_results/frame_influence_keepone_full/.

--measure recent_k is the recent-window variant (runs under paper_results/frame_influence_recent/,
arms r1_influence_recent / uninavid_influence_recent, logged with no per-frame probe and the joint
age bands at every K = 2..32): KL(full || only the K most recent frames), K = 0 (no frame), 1 (the
current frame), ..., 32. It writes steps.csv, recent_curve.csv (per K over the decisions with >= K
history frames: pooled mean + episode-bootstrap CI, episode-balanced mean +- SEM, flip rate, and the
same means on the fixed set of decisions with >= 32 history frames), recent_reference.csv (the
shared K re-measured against the leave-one-out runs, per decision) and summary.json; none of the
per-frame statistics exist for this measure.

Runs in longnav_vlm (numpy only):  python paper/frame_influence/aggregate.py [--measure keep_one|keep_one_full|recent_k]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from functools import partial
import math
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import influence as inf
from longnav.analysis.results_io import assert_split, load_lightnav_run, load_manifest, load_r1_run, load_uninavid_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_10 = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
MANIFEST_9 = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_9_seed17_short.json"
# measure -> (results root, {model label -> (arm dir, influence loader, episode-results loader,
# manifest the arm must complete)})
MEASURES = {
    "leave_one": (ROOT / "paper_results/frame_influence", {
        "LongNav-R1": ("r1_influence", inf.load_r1_influence, load_r1_run, MANIFEST_10),
        "LongNav-R1 (leak-free)": ("r1_influence_recompute", inf.load_r1_influence, load_r1_run, MANIFEST_9),
        "Uni-NaVid": ("uninavid_influence", inf.load_uninavid_influence, load_uninavid_run, MANIFEST_10)}),
    "keep_one": (ROOT / "paper_results/frame_influence_keepone", {
        "LongNav-R1": ("r1_influence_keepone", inf.load_r1_influence, load_r1_run, MANIFEST_10),
        "Uni-NaVid": ("uninavid_influence_keepone", inf.load_uninavid_influence, load_uninavid_run, MANIFEST_10),
        # the same LightNav arm read twice: KL over the 256 coarse RVQ codes, and folded onto 6 executed-action bins
        "LightNav-0": ("lightnav_influence_keepone", inf.load_lightnav_influence, load_lightnav_run, MANIFEST_10),
        "LightNav-0 (6-bin)": ("lightnav_influence_keepone", partial(inf.load_lightnav_influence, space="bins"),
                               load_lightnav_run, MANIFEST_10)}),
    # derived from the keep-one runs (see RUNS_ROOT): the lone frame scored against the full decision
    # (256-code LightNav only: the 6-bin fold was dropped from this measure at the user's request, 2026-11-07;
    # Uni-NaVid added the same day from its masked keep-one run)
    "keep_one_full": (ROOT / "paper_results/frame_influence_keepone_full", {
        "LongNav-R1": ("r1_influence_keepone", inf.load_r1_influence, load_r1_run, MANIFEST_10),
        "Uni-NaVid": ("uninavid_influence_keepone", inf.load_uninavid_influence, load_uninavid_run, MANIFEST_10),
        "LightNav-0": ("lightnav_influence_keepone", inf.load_lightnav_influence, load_lightnav_run, MANIFEST_10)}),
    # joint probes only: KL(full || only the K most recent frames), K = 0..32 (no per-frame rows)
    "recent_k": (ROOT / "paper_results/frame_influence_recent", {
        "LongNav-R1": ("r1_influence_recent", inf.load_r1_influence, load_r1_run, MANIFEST_10),
        # the leak-free twin (full re-prefill probes); all 10 episodes, unlike the leave-one-out leak-free arm
        # (9 short ones), whose run it is checked against on those nine
        "LongNav-R1 (leak-free)": ("r1_influence_recent_recompute", inf.load_r1_influence, load_r1_run, MANIFEST_10),
        "Uni-NaVid": ("uninavid_influence_recent", inf.load_uninavid_influence, load_uninavid_run, MANIFEST_10),
        # one LightNav run, read four times: the K-frame window (old blocks gone) and the old-blocks-kept
        # arm (older tubelets' visual columns hidden), each over the 256 codes and folded onto the 6 bins
        "LightNav-0": ("lightnav_influence_recent", partial(inf.load_lightnav_influence, recent="window"),
                       load_lightnav_run, MANIFEST_10),
        "LightNav-0 (old blocks kept)": ("lightnav_influence_recent", partial(inf.load_lightnav_influence, recent="keep"),
                                         load_lightnav_run, MANIFEST_10),
        "LightNav-0 (6-bin)": ("lightnav_influence_recent", partial(inf.load_lightnav_influence, recent="window", space="bins"),
                               load_lightnav_run, MANIFEST_10),
        "LightNav-0 (old blocks kept, 6-bin)": ("lightnav_influence_recent",
                                                partial(inf.load_lightnav_influence, recent="keep", space="bins"),
                                                load_lightnav_run, MANIFEST_10)}),
}
# measures without runs of their own read another measure's runs directory (default: <results root>/runs)
RUNS_ROOT = {"keep_one_full": ROOT / "paper_results/frame_influence_keepone/runs"}
KS = inf.KS
LAGS = (1, 2, 4, 8, 16)


def fmt(v, nd=4):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.{nd}g}" if abs(v) < 1e-3 else f"{v:.{nd}f}"
    return v


def write_csv(path: Path, rows: list[dict], fields=None, gz=False):
    if not rows:
        return
    fields = list(fields or rows[0].keys())
    opener = (lambda p: gzip.open(p, "wt", newline="")) if gz else (lambda p: p.open("w", newline=""))
    with opener(path) as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps([round(x, 6) for x in v]) if isinstance(v, list) else v) for k, v in r.items()})


def trajectory_check(rows_r1: dict, labels: list[str]) -> list[str]:
    """The prune-substrate full-context run must reproduce the ctx-ablation full arm exactly."""
    ref_dir = ROOT / "paper_results/ctx_ablation/runs/r1_full"
    if not ref_dir.is_dir():
        return ["r1_full reference not found; trajectory check skipped"]
    ref = load_r1_run(ref_dir)
    msgs = []
    for label in labels:
        a, b = rows_r1.get(label), ref.get(label)
        if a is None or b is None:
            msgs.append(f"{label}: missing in {'influence' if a is None else 'r1_full'}")
        elif (a["success"], a["n_steps"]) != (b["success"], b["n_steps"]):
            msgs.append(f"{label}: influence run success={a['success']} steps={a['n_steps']} vs "
                        f"r1_full success={b['success']} steps={b['n_steps']}")
    return msgs


def lightnav_reference_steps(ref_dir: Path, model: str) -> dict:
    """LightNav-0's kept-blocks arm against its keep-one run (2026-10-26): that run stored the
    no-visual and the only-current-tubelet distributions of every decision, i.e. K = 0 and K = 2 of
    this measure once scored against the full decision (``keep_one_full``)."""
    space = "bins" if "6-bin" in model else "codes"
    ref = {}
    for s in inf.load_lightnav_influence(ref_dir, model, measure="keep_one_full", space=space)[1]:
        ref[(s["episode"], s["step"])] = {"horizon": {0: s["visual_all"], 2: s["age0"]}, "argmax_action": s["argmax_action"]}
    return ref


def recent_reference_check(model: str, steps: list[dict], ref_dir: Path, loader) -> list[dict]:
    """The recent-window run repeats trajectories the leave-one-out run already probed at K = 1, 4,
    8, 16, 32 (and K = 0), so those values must come out again: per K, the paired difference over
    the decisions both runs scored (``same_decision`` = share of the paired decisions with the same
    full-context argmax, 1.0 when the trajectories are the same). Empty when the reference run is
    absent. ``loader`` None = LightNav-0's kept-blocks arm against its keep-one run (K = 0, 2)."""
    if not ref_dir.is_dir():
        return []
    if loader is None:
        ref, ks = lightnav_reference_steps(ref_dir, model), (0, 2)
    else:
        ref, ks = {(s["episode"], s["step"]): s for s in loader(ref_dir, model, measure="recent_k")[1]}, (0, *KS)
    out = []
    for k in ks:
        diffs, same, n_new = [], [], 0
        for s in steps:
            if k not in s["horizon"] or s["n_history"] < k:
                continue
            n_new += 1
            o = ref.get((s["episode"], s["step"]))
            if o is not None and k in o["horizon"] and math.isfinite(o["horizon"][k]):
                diffs.append(s["horizon"][k] - o["horizon"][k])
                same.append(s["argmax_action"] == o["argmax_action"])
        d = np.abs(np.asarray(diffs, dtype=float))
        out.append({"model": model, "K": k, "n_steps": n_new, "n_paired": int(d.size),
                    "same_decision": float(np.mean(same)) if same else float("nan"),
                    "max_abs_diff": float(d.max()) if d.size else float("nan"),
                    "mean_abs_diff": float(d.mean()) if d.size else float("nan"),
                    "median_abs_diff": float(np.median(d)) if d.size else float("nan")})
    return out


def aggregate_recent(args, arms: dict, per_model: dict, labels_of: dict, summary: dict) -> None:
    """--measure recent_k: the K curve and its checks (no per-frame statistics exist)."""
    ks = inf.RECENT_KS
    step_rows, curve_rows, ref_rows = [], [], []
    for model, (_, steps, episodes) in per_model.items():
        for s in steps:
            r = {k: v for k, v in s.items() if k not in ("horizon", "horizon_flip", "sum_loo", "n_scored", "age0")}
            for k in ks:
                r[f"horizon_{k}"] = s["horizon"].get(k, float("nan"))
            step_rows.append(r)
        curve = inf.recent_window_curve(steps, ks)
        curve_rows += [{"model": model, **r} for r in curve]
        arm, loader = arms[model][0], arms[model][1]
        ref_arm = {"LongNav-R1": "r1_influence", "LongNav-R1 (leak-free)": "r1_influence_recompute",
                   "Uni-NaVid": "uninavid_influence"}.get(model)
        ref = recent_reference_check(model, steps, MEASURES["leave_one"][0] / "runs" / ref_arm, loader) if ref_arm else []
        if "old blocks kept" in model:  # LightNav-0: K = 0 and 2 exist in its keep-one run
            ref = recent_reference_check(model, steps, MEASURES["keep_one"][0] / "runs" / "lightnav_influence_keepone", None)
        ref_rows += ref
        labels = labels_of[model]
        summ = summary.setdefault(model, {})
        summ.update({
            "arm": arm, "n_episodes": len({s["episode"] for s in steps}), "n_decisions": len(steps),
            "success_rate": float(np.mean([episodes[l]["success"] for l in labels if l in episodes])),
            "mean_steps": float(np.mean([episodes[l]["n_steps"] for l in labels if l in episodes])),
            "max_K": max((k for s in steps for k in s["horizon"]), default=None),
            "curve_balanced": {str(r["K"]): r["balanced_mean"] for r in curve},
            "reference_max_abs_diff": {str(r["K"]): r["max_abs_diff"] for r in ref},
        })
        floor = np.array([s["kl_live_vs_replay"] for s in steps if math.isfinite(s["kl_live_vs_replay"])])
        if floor.size:  # LongNav-R1: live vs replayed base; LightNav-0: decision vs an identical re-run
            summ.update({"replay_floor_kl_mean": float(floor.mean()), "replay_floor_kl_median": float(np.median(floor)),
                         "replay_floor_kl_p90": float(np.percentile(floor, 90))})
        if model == "Uni-NaVid":
            merged = []
            for path in sorted((args.runs / arm / "influence").glob("*.jsonl")):
                for line in path.read_text().splitlines():
                    if line.strip():
                        merged += [c["kl"] for c in json.loads(line).get("merged_control", [])]
            summ.update({"p_mass_mean": float(np.mean([s["p_mass"] for s in steps])),
                         "merged_control_kl_mean": float(np.mean(merged)) if merged else float("nan"),
                         "merged_control_n": len(merged),
                         "probe_seconds_total": float(np.nansum([s["probe_seconds"] for s in steps]))})
    summary["measure"] = args.measure
    summary["fixed_set_min_history"] = inf.RECENT_FIXED_MIN
    write_csv(args.out / "steps.csv", step_rows)
    write_csv(args.out / "recent_curve.csv", curve_rows)
    write_csv(args.out / "recent_reference.csv", ref_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    show = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    print(f"measure: {args.measure} -- KL(full || only the K most recent frames), nats; K = 0 no frame, 1 = current only")
    for model in per_model:
        d = summary[model]
        rows = {r["K"]: r for r in curve_rows if r["model"] == model}
        print(f"\n== {model}: {d['n_episodes']} eps, {d['n_decisions']} decisions, SR {d['success_rate']:.2f}, "
              f"{d['mean_steps']:.0f} steps, bands up to K={d['max_K']}")
        print("   K        : " + "  ".join(f"{k:>7d}" for k in show))
        for key, tag in (("balanced_mean", "balanced "), ("balanced_sem", "  +- SEM "), ("mean", "pooled   "),
                         ("fixed_balanced_mean", "fixed set"), ("flip_rate", "flip rate")):
            print(f"   {tag}: " + "  ".join(f"{rows[k][key]:7.4f}" if k in rows and math.isfinite(rows[k][key]) else "     --" for k in show))
        print("   episodes : " + "  ".join(f"{rows[k]['n_episodes']:>7d}" if k in rows else "     --" for k in show)
              + f"   (fixed set: {max(r['fixed_n_episodes'] for r in rows.values())} eps, "
                f"{max(r['fixed_n_steps'] for r in rows.values())} decisions)")
        if "replay_floor_kl_mean" in d:
            print(f"   replay floor KL(live || replayed base): mean {d['replay_floor_kl_mean']:.2e}, median {d['replay_floor_kl_median']:.2e}")
        if model == "Uni-NaVid":
            print(f"   sanity: p_mass {d['p_mass_mean']:.3f}, merged_control KL {d['merged_control_kl_mean']:.2e} "
                  f"(n={d['merged_control_n']}), probes {d['probe_seconds_total']:.0f} s")
        ref = [r for r in ref_rows if r["model"] == model]
        if ref:
            print("   vs the earlier run on the same episodes, |diff| max / median per K: "
                  + "  ".join(f"K{r['K']}={r['max_abs_diff']:.1e}/{r['median_abs_diff']:.1e} (n={r['n_paired']}/{r['n_steps']}, "
                              f"same decision {r['same_decision']:.3f})" for r in ref))
    print(f"\nwrote {args.out}/{{steps.csv, recent_curve.csv, recent_reference.csv, summary.json}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", choices=list(MEASURES), default="leave_one",
                    help="leave_one (default), keep_one (leave-one-in; picks the _keepone results root and arms), "
                         "keep_one_full (KL(full || only frame), derived from the keep_one runs into _keepone_full) or "
                         "recent_k (KL(full || only the K most recent frames), K = 0..32, the _recent root)")
    ap.add_argument("--runs", type=Path, default=None, help="default <results root>/runs (keep_one_full: the keep_one runs)")
    ap.add_argument("--out", type=Path, default=None, help="default <results root>/results")
    ap.add_argument("--allow-partial", action="store_true", help="aggregate incomplete arms (interim looks only)")
    ap.add_argument("--skip-partial", action="store_true",
                    help="leave out arms that are still running instead of failing (interim looks only)")
    args = ap.parse_args()
    results_root, ARMS = MEASURES[args.measure]
    args.runs = args.runs or RUNS_ROOT.get(args.measure, results_root / "runs")
    args.out = args.out or results_root / "results"
    args.out.mkdir(parents=True, exist_ok=True)

    rows_all, steps_all, summary, missing = [], [], {}, []
    per_model, labels_of = {}, {}
    for model, (arm, loader, run_loader, manifest) in ARMS.items():
        run_dir = args.runs / arm
        if not run_dir.is_dir():
            missing.append(arm)
            continue
        labels = load_manifest(manifest)
        labels_of[model] = labels
        episodes = run_loader(run_dir)
        got = [l for l in labels if l in episodes]
        if args.skip_partial and len(got) < len(labels):
            print(f"[partial] {arm}: {len(got)}/{len(labels)} episodes, skipped")
            missing.append(arm)
            continue
        if args.allow_partial:
            print(f"[partial ok] {arm}: {len(got)}/{len(labels)} episodes")
        else:
            assert_split(episodes, labels, arm)
        rows, steps = loader(run_dir, model, measure=args.measure)
        rows = [r for r in rows if r["episode"] in labels]
        steps = [s for s in steps if s["episode"] in labels]
        per_model[model] = (rows, steps, episodes)
        rows_all += rows
        steps_all += steps
        if model.startswith("LongNav-R1") and not args.allow_partial:
            problems = trajectory_check(episodes, labels)
            print(f"{model}: trajectory check vs ctx_ablation/r1_full: " + ("OK, identical" if not problems else "; ".join(problems)))
            summary.setdefault(model, {})["trajectory_matches_r1_full"] = not problems
    if missing:
        print(f"[missing] {' '.join(missing)}")
    if not per_model:
        sys.exit("no arms found")
    if args.measure == "recent_k":
        aggregate_recent(args, ARMS, per_model, labels_of, summary)
        return

    # ---- long tables -------------------------------------------------------------------
    write_csv(args.out / "influence.csv.gz", rows_all, inf.ROW_FIELDS, gz=True)
    step_rows = []
    for s in steps_all:
        r = {k: v for k, v in s.items() if k != "horizon"}
        for k in KS:
            r[f"horizon_{k}"] = s["horizon"].get(k, float("nan"))
        step_rows.append(r)
    write_csv(args.out / "steps.csv", step_rows)

    # ---- per-model statistics ----------------------------------------------------------
    age_rows, bin_rows, share_rows, hor_rows, pers_rows, cov_rows, flip_rows = [], [], [], [], [], [], []
    ols_all, revival_all = {}, {}
    for model, (rows, steps, episodes) in per_model.items():
        hist_rows = [r for r in rows if r["age"] >= 1]
        for r in inf.age_curve(hist_rows):
            age_rows.append({"model": model, **r})
        for r in inf.age_bin_table(hist_rows):
            bin_rows.append({"model": model, **r})
        for r in inf.cumulative_share(hist_rows, KS):
            share_rows.append({"model": model, **r})
        loo = inf.loo_sum_by_horizon(hist_rows, steps, KS)
        for r in inf.horizon_table(steps, KS):
            hor_rows.append({"model": model, **r, "loo_sum_mean": loo[r["K"]]})
        mats = [inf.influence_matrix(rows, steps, ep) for ep in sorted({s["episode"] for s in steps})]
        for r in inf.persistence(mats, LAGS):
            pers_rows.append({"model": model, **{k: v for k, v in r.items()}})
        revival_all[model] = inf.revival(mats)
        for r in inf.covariate_bins(hist_rows):
            cov_rows.append({"model": model, **r})
        ols_all[model] = inf.ols_log_influence(hist_rows)
        for r in inf.flip_rate_by_age(hist_rows):
            flip_rows.append({"model": model, **r})

        fin = [r["infl"] for r in hist_rows if math.isfinite(r["infl"])]
        hist_all = [s["history_all"] for s in steps if math.isfinite(s["history_all"]) and s["n_history"] > 0]
        vis_all = [s["visual_all"] for s in steps if math.isfinite(s["visual_all"])]
        age0 = [s["age0"] for s in steps if math.isfinite(s["age0"])]
        sum_loo = [s["sum_loo"] for s in steps if math.isfinite(s["sum_loo"]) and s["n_history"] > 0]
        labels = labels_of[model]
        summ = summary.setdefault(model, {})
        summ.update({
            "n_episodes": len({s["episode"] for s in steps}), "n_decisions": len(steps),
            "n_pairs": len(hist_rows), "n_pairs_scored": len(fin),
            "success_rate": float(np.mean([episodes[l]["success"] for l in labels if l in episodes])),
            "mean_steps": float(np.mean([episodes[l]["n_steps"] for l in labels if l in episodes])),
            "infl_mean": float(np.mean(fin)) if fin else float("nan"),
            "infl_median": float(np.median(fin)) if fin else float("nan"),
            "age0_mean": float(np.mean(age0)) if age0 else float("nan"),
            "history_all_mean": float(np.mean(hist_all)) if hist_all else float("nan"),
            "sum_loo_mean": float(np.mean(sum_loo)) if sum_loo else float("nan"),
            "visual_all_mean": float(np.mean(vis_all)) if vis_all else float("nan"),
            "flip_rate": float(np.mean([int(r["flip"]) for r in hist_rows if r["flip"] != ""])) if hist_rows else float("nan"),
            "revival": revival_all[model], "ols": ols_all[model],
        })
        if args.measure == "keep_one_full":  # the ceiling KL(full || no visual), balanced like the age curve
            per_ep: dict[str, list[float]] = {}
            for s in steps:
                if math.isfinite(s["visual_all"]):
                    per_ep.setdefault(s["episode"], []).append(s["visual_all"])
            ep_means = np.array([np.mean(v) for v in per_ep.values()])
            summ["visual_all_balanced"] = float(ep_means.mean()) if ep_means.size else float("nan")
            summ["visual_all_balanced_sem"] = (float(ep_means.std(ddof=1) / math.sqrt(len(ep_means)))
                                               if len(ep_means) > 1 else float("nan"))
        floor = np.array([s["kl_live_vs_replay"] for s in steps if math.isfinite(s["kl_live_vs_replay"])])
        if floor.size:  # LongNav-R1 (live vs replayed base) and LightNav-0 (decision vs an identical re-run)
            summ.update({"replay_floor_kl_mean": float(floor.mean()), "replay_floor_kl_median": float(np.median(floor)),
                         "replay_floor_kl_p90": float(np.percentile(floor, 90))})
        if model.startswith("LightNav-0"):
            summ.update({"p_mass_mean": float(np.nanmean([s["p_mass"] for s in steps])),
                         "p_apos_mean": float(np.nanmean([s["p_apos"] for s in steps])),
                         "p_opos_mean": float(np.nanmean([s["p_opos"] for s in steps])),
                         "n_tubelets_unprobed_mean": float(np.mean([s["n_unprobed"] for s in steps])),
                         "probe_seconds_total": float(np.nansum([s["probe_seconds"] for s in steps]))})
        if model == "Uni-NaVid":
            in_set = [s["sampled_in_set"] for s in steps if s["sampled_in_set"] is not None]
            agree = [s["chosen_action"] == s["argmax_action"] for s in steps]
            merged = []
            for path in sorted((args.runs / ARMS[model][0] / "influence").glob("*.jsonl")):
                for line in path.read_text().splitlines():
                    if line.strip():
                        merged += [c["kl"] for c in json.loads(line).get("merged_control", [])]
            summ.update({"sampled_first_token_in_set_rate": float(np.mean(in_set)) if in_set else float("nan"),
                         "argmax_vs_sampled_agreement": float(np.mean(agree)) if agree else float("nan"),
                         "p_mass_mean": float(np.mean([s["p_mass"] for s in steps])),
                         "merged_control_kl_mean": float(np.mean(merged)) if merged else float("nan"),
                         "merged_control_n": len(merged),
                         "probe_seconds_total": float(np.nansum([s["probe_seconds"] for s in steps]))})

    # LightNav-0: the keep-one KL against the no-visual prior saturates (near one-hot codes), so also
    # score every lone tubelet against the full-context decision from the stored 256-way log-probs.
    agree_rows = []
    if "LightNav-0" in per_model and args.measure == "keep_one":  # keep_one_full IS this analysis, as a measure
        agree_rows, agree_summ, agree_eps = inf.lightnav_full_agreement(args.runs / ARMS["LightNav-0"][0])
        summary["LightNav-0"]["full_agreement"] = agree_summ
        write_csv(args.out / "lightnav_agreement.csv", agree_rows)
        write_csv(args.out / "lightnav_agreement_episodes.csv", agree_eps)
    summary["representative_episode"] = inf.pick_representative_episode(*(v[1] for v in per_model.values()))
    summary["measure"] = args.measure
    # Eviction leakage at the decision level: evict-style minus leak-free horizon probes, paired
    # on (episode, step) over the episodes both LongNav-R1 arms ran.
    leak_rows = []
    if "LongNav-R1" in per_model and "LongNav-R1 (leak-free)" in per_model:
        leak_rows = inf.paired_horizon_gap(per_model["LongNav-R1"][1], per_model["LongNav-R1 (leak-free)"][1], KS)
        write_csv(args.out / "leak.csv", leak_rows)
    write_csv(args.out / "age_curve.csv", age_rows)
    write_csv(args.out / "age_bins.csv", bin_rows)
    write_csv(args.out / "cumulative_share.csv", share_rows)
    write_csv(args.out / "horizon.csv", hor_rows)
    write_csv(args.out / "persistence.csv", [{k: v for k, v in r.items() if k != "rho_per_episode"} for r in pers_rows])
    write_csv(args.out / "covariates.csv", cov_rows)
    write_csv(args.out / "flips.csv", flip_rows)
    (args.out / "ols.json").write_text(json.dumps(ols_all, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_csv(args.out / "summary.csv", [{"model": m, **{k: v for k, v in d.items() if not isinstance(v, dict)}}
                                          for m, d in summary.items() if isinstance(d, dict)])

    # ---- LaTeX --------------------------------------------------------------------------
    models = list(per_model)
    lines = [r"% Generated by paper/frame_influence/aggregate.py -- do not edit by hand.",
             r"\begin{tabular}{l " + " ".join("cc" for _ in models) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models))),
             r"$k$ & " + " & ".join(r"share$_{\le k}$ & KL$_{\ge k}$" for _ in models) + r" \\", r"\midrule"]
    share_by = {(r["model"], r["k"]): r for r in share_rows}
    hor_by = {(r["model"], r["K"]): r for r in hor_rows}
    for k in KS:
        cells = []
        for m in models:
            s, h = share_by.get((m, k)), hor_by.get((m, k))
            cells += [f"{s['pooled_share']:.2f}" if s else "--", f"{h['mean']:.4f}" if h else "--"]
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")
    single = {"leave_one": "single-frame (leave-one-out) influence",
              "keep_one": "single-frame leave-one-in influence, KL(only frame || no visual),",
              "keep_one_full": "single-frame distance from the full decision, KL(full || only frame),"}[args.measure]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"% share$_{{\le k}}$: share of {single} carried by the k most recent",
              r"%   history frames, pooled over decisions with >= 32 history frames.",
              r"% KL$_{\ge k}$: mean KL(full || all frames of age >= k removed), i.e. what the last k frames",
              r"%   (current included) do not explain. Influence = KL in nats over the 4 actions, T = 1.",
              "% n = " + "; ".join(f"{m}: {len(labels_of[m])} episodes" for m in models)
              + " (HM3D-v2 val, seed-17 sample of the 100-episode split; the leak-free arm omits the 500-step episode)."]
    (args.out / "table.tex").write_text("\n".join(lines) + "\n")

    # ---- console ------------------------------------------------------------------------
    print(f"measure: {args.measure}; representative episode for the heatmap: {summary['representative_episode']}")
    for m in models:
        d = summary[m]
        print(f"\n== {m}: {d['n_episodes']} eps, {d['n_decisions']} decisions, {d['n_pairs_scored']}/{d['n_pairs']} pairs scored, "
              f"SR {d['success_rate']:.2f}, {d['mean_steps']:.0f} steps")
        tag = {"leave_one": "LOO", "keep_one": "keep-one", "keep_one_full": "KL(full||only)"}[args.measure]
        print(f"   mean {tag} {d['infl_mean']:.5f} (median {d['infl_median']:.5f}); age0 {d['age0_mean']:.4f}; "
              f"history_all {d['history_all_mean']:.4f} vs sum {tag} {d['sum_loo_mean']:.4f}; visual_all {d['visual_all_mean']:.4f}; "
              f"flip rate {d['flip_rate']:.3f}")
        if "visual_all_balanced" in d:
            print(f"   ceiling KL(full || no visual): pooled {d['visual_all_mean']:.4f}, balanced {d['visual_all_balanced']:.4f} "
                  f"+- {d['visual_all_balanced_sem']:.4f} SEM")
        print("   age bins: " + "  ".join(f"{r['age_bin']}:{r['mean']:.4f}({r['mass_share']:.2f})" for r in bin_rows if r["model"] == m))
        print("   share<=k: " + "  ".join(f"k{r['k']}={r['pooled_share']:.2f}/{r['per_step_share']:.2f}" for r in share_rows if r["model"] == m))
        print("   horizon : " + "  ".join(f"K{r['K']}={r['mean']:.4f}(sumLOO {r['loo_sum_mean']:.4f})" for r in hor_rows if r["model"] == m))
        print("   persist : " + "  ".join(f"lag{r['lag']} rho_res={r['rho_residual']:.3f} top25={r['top25_residual']:.2f}" for r in pers_rows if r["model"] == m))
        rv = d["revival"]; o = d["ols"]
        print(f"   revival {rv['revived']}/{rv['ever_dormant']} ({rv['events']} events); OLS R2 {o.get('r2', float('nan')):.3f} "
              f"(age only {o.get('r2_age_only', float('nan')):.3f}); partial R2 " +
              ", ".join(f"{k}={v:.3f}" for k, v in o.get("partial_r2", {}).items()))
        if "replay_floor_kl_mean" in d:
            print(f"   base-vs-live floor KL(live || rebuilt/replayed base): mean {d['replay_floor_kl_mean']:.2e}, "
                  f"median {d['replay_floor_kl_median']:.2e}, p90 {d['replay_floor_kl_p90']:.2e}")
        if m.startswith("LightNav-0"):
            print(f"   sanity: p_mass {d['p_mass_mean']:.3f}, free-run prefix top-1 p_apos {d['p_apos_mean']:.3f} / "
                  f"p_opos {d['p_opos_mean']:.3f}, unprobed (old) tubelets/step {d['n_tubelets_unprobed_mean']:.1f}, "
                  f"probes {d['probe_seconds_total']:.0f} s")
        if m == "LightNav-0" and agree_rows:
            fa = d["full_agreement"]
            print(f"   lone tubelet vs FULL decision (6-bin argmax agreement, balanced; majority class "
                  f"{fa['majority_class']} {fa['majority_class_rate']:.2f}, no-visual prior {fa['reference_agree_bin']:.2f}): "
                  + "  ".join(f"age{r['age']}={r['agree_bin_balanced']:.2f}" for r in agree_rows if r["age"] in (0, 2, 4, 8, 16, 32)))
            print("   KL(full || only tubelet), 256 codes: "
                  + "  ".join(f"age{r['age']}={r['kl_full_only_balanced']:.2f}" for r in agree_rows if r["age"] in (0, 2, 4, 8, 16, 32)))
        if m == "Uni-NaVid":
            print(f"   sanity: sampled token in set {d['sampled_first_token_in_set_rate']:.3f}, argmax==sampled "
                  f"{d['argmax_vs_sampled_agreement']:.3f}, p_mass {d['p_mass_mean']:.3f}, merged_control KL "
                  f"{d['merged_control_kl_mean']:.2e} (n={d['merged_control_n']}), probes {d['probe_seconds_total']:.0f} s")
    if leak_rows:
        print("\n== LongNav-R1 eviction leakage at the decision level (evict-style minus leak-free horizon_K, paired):")
        for r in leak_rows:
            print(f"   K={r['K']:>2}: evict {r['mean_a']:.4f}  leak-free {r['mean_b']:.4f}  delta {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  n={r['n_steps']} steps / {r['n_episodes']} eps")
    print(f"\nwrote {args.out}/{{influence.csv.gz, steps.csv, age_curve.csv, cumulative_share.csv, horizon.csv, "
          f"persistence.csv, covariates.csv, flips.csv, summary.csv, table.tex" + (", lightnav_agreement.csv, lightnav_agreement_episodes.csv" if agree_rows else "") + "}")


if __name__ == "__main__":
    main()
