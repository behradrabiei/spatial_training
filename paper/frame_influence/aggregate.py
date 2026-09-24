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

Runs in longnav_vlm (numpy only):  python paper/frame_influence/aggregate.py [--measure keep_one]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import influence as inf
from longnav.analysis.results_io import assert_split, load_manifest, load_r1_run, load_uninavid_run

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
        "Uni-NaVid": ("uninavid_influence_keepone", inf.load_uninavid_influence, load_uninavid_run, MANIFEST_10)}),
}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", choices=list(MEASURES), default="leave_one",
                    help="leave_one (default) or keep_one (leave-one-in; picks the _keepone results root and arms)")
    ap.add_argument("--runs", type=Path, default=None, help="default <results root>/runs")
    ap.add_argument("--out", type=Path, default=None, help="default <results root>/results")
    ap.add_argument("--allow-partial", action="store_true", help="aggregate incomplete arms (interim looks only)")
    ap.add_argument("--skip-partial", action="store_true",
                    help="leave out arms that are still running instead of failing (interim looks only)")
    args = ap.parse_args()
    results_root, ARMS = MEASURES[args.measure]
    args.runs = args.runs or results_root / "runs"
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
        if model.startswith("LongNav-R1"):
            floor = np.array([s["kl_live_vs_replay"] for s in steps if math.isfinite(s["kl_live_vs_replay"])])
            summ.update({"replay_floor_kl_mean": float(floor.mean()) if floor.size else float("nan"),
                         "replay_floor_kl_median": float(np.median(floor)) if floor.size else float("nan"),
                         "replay_floor_kl_p90": float(np.percentile(floor, 90)) if floor.size else float("nan")})
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
    single = ("single-frame (leave-one-out) influence" if args.measure == "leave_one"
              else "single-frame leave-one-in influence, KL(only frame || no visual),")
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
        tag = "LOO" if args.measure == "leave_one" else "keep-one"
        print(f"   mean {tag} {d['infl_mean']:.5f} (median {d['infl_median']:.5f}); age0 {d['age0_mean']:.4f}; "
              f"history_all {d['history_all_mean']:.4f} vs sum {tag} {d['sum_loo_mean']:.4f}; visual_all {d['visual_all_mean']:.4f}; "
              f"flip rate {d['flip_rate']:.3f}")
        print("   age bins: " + "  ".join(f"{r['age_bin']}:{r['mean']:.4f}({r['mass_share']:.2f})" for r in bin_rows if r["model"] == m))
        print("   share<=k: " + "  ".join(f"k{r['k']}={r['pooled_share']:.2f}/{r['per_step_share']:.2f}" for r in share_rows if r["model"] == m))
        print("   horizon : " + "  ".join(f"K{r['K']}={r['mean']:.4f}(sumLOO {r['loo_sum_mean']:.4f})" for r in hor_rows if r["model"] == m))
        print("   persist : " + "  ".join(f"lag{r['lag']} rho_res={r['rho_residual']:.3f} top25={r['top25_residual']:.2f}" for r in pers_rows if r["model"] == m))
        rv = d["revival"]; o = d["ols"]
        print(f"   revival {rv['revived']}/{rv['ever_dormant']} ({rv['events']} events); OLS R2 {o.get('r2', float('nan')):.3f} "
              f"(age only {o.get('r2_age_only', float('nan')):.3f}); partial R2 " +
              ", ".join(f"{k}={v:.3f}" for k, v in o.get("partial_r2", {}).items()))
        if m.startswith("LongNav-R1"):
            print(f"   base-vs-live floor KL(live || rebuilt/replayed base): mean {d['replay_floor_kl_mean']:.2e}, "
                  f"median {d['replay_floor_kl_median']:.2e}, p90 {d['replay_floor_kl_p90']:.2e}")
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
          f"persistence.csv, covariates.csv, flips.csv, summary.csv, table.tex}}")


if __name__ == "__main__":
    main()
