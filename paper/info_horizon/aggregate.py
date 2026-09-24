#!/usr/bin/env python3
"""Aggregate the information-horizon experiment into paper_results/info_horizon/results/.

Reads the KL probe arms (runs/r1_probe_leaky, runs/r1_probe: rollout/*/*/sequence.json;
runs/uninavid_probe: horizon/*.jsonl), the visual-blind SR/SPL arms (runs/r1_blind_l<L>,
runs/uninavid_blind_l<L>) and the references (paper_results/ctx_ablation/runs/r1_full for
LongNav-R1, runs/uninavid_ref for Uni-NaVid), and writes:
  horizon.csv      long form, one row per (curve, layer): keep-one-in / leave-one-out KL of the
                   current frame (pooled mean over scored steps, episode-cluster bootstrap CI,
                   balanced mean) and SR / SPL of the blind arm at that layer (mean, CI, paired
                   delta vs the reference); the reference sits at layer = n_layers ("none").
  probe_steps.csv  every (curve, episode, step, layer) probe value -- the tracked copy of the
                   per-step data whose raw files are gitignored.
  table.tex        booktabs table of the same numbers.
  summary.json     arm completeness, sanity checks.
Runs in longnav_vlm (numpy only):  python paper/info_horizon/aggregate.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import horizon as hz
from longnav.analysis.results_io import load_manifest, load_r1_run, load_uninavid_run, split_status
from longnav.analysis.stats import bootstrap_mean_ci, paired_delta, stars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = {  # results root, SR/SPL manifest, KL-probe manifest, LongNav-R1 reference run
    "hm3d": (ROOT / "paper_results/info_horizon", "hm3d_v2_100.json", "hm3d_v2_10_seed17.json",
             ROOT / "paper_results/ctx_ablation/runs/r1_full"),
    "mp3d": (ROOT / "paper_results/info_horizon_mp3d", "mp3d_100_seed17.json", "mp3d_10_seed17.json",
             ROOT / "paper_results/info_horizon_mp3d/runs/r1_ref"),
}
RESULTS_ROOT = DATASETS["hm3d"][0]
MANIFEST_100 = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_100.json"
MANIFEST_9 = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json"
N_LAYERS = {"LongNav-R1": 28, "Uni-NaVid": 32}
# KL curves: label -> (arm dir, loader, model the SR/SPL arms belong to)
PROBES = {"LongNav-R1": ("r1_probe_leaky", hz.load_r1_horizon, "LongNav-R1"),
          "LongNav-R1 (leak-free)": ("r1_probe", hz.load_r1_horizon, "LongNav-R1"),
          "Uni-NaVid": ("uninavid_probe", hz.load_uninavid_horizon, "Uni-NaVid")}
# SR/SPL arms: model -> (blind arm template, run loader, reference dir)
BLIND = {"LongNav-R1": ("r1_blind_l{L}", load_r1_run, ROOT / "paper_results/ctx_ablation/runs/r1_full"),
         "Uni-NaVid": ("uninavid_blind_l{L}", load_uninavid_run, RESULTS_ROOT / "runs/uninavid_ref")}
LAYERS = {"LongNav-R1": [8, 12, 14, 16, 18, 20, 24], "Uni-NaVid": [8, 12, 16, 18, 20, 24, 28]}
OPTIONAL = {"r1_probe"}  # the leak-free LongNav-R1 probe: drawn if present, not required


def fmt(v, nd=3):
    return "" if v is None or v == "" or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_csv(path: Path, rows: list[dict], fields=None):
    if not rows:
        return
    fields = list(fields or rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def trajectory_check(rows: dict, ref: dict, labels: list[str], name: str) -> list[str]:
    msgs = []
    for label in labels:
        a, b = rows.get(label), ref.get(label)
        if a is None or b is None:
            continue
        if (a["success"], a["n_steps"]) != (b["success"], b["n_steps"]):
            msgs.append(f"{name} {label}: success={a['success']} steps={a['n_steps']} vs reference "
                        f"success={b['success']} steps={b['n_steps']}")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hm3d")
    ap.add_argument("--runs", type=Path, default=None, help="default: <results root>/runs of --dataset")
    ap.add_argument("--out", type=Path, default=None, help="default: <results root>/results of --dataset")
    ap.add_argument("--allow-partial", action="store_true", help="aggregate incomplete arms (interim looks only)")
    args = ap.parse_args()
    root, m100, m9, r1_ref = DATASETS[args.dataset]
    args.runs = args.runs or root / "runs"
    args.out = args.out or root / "results"
    BLIND["LongNav-R1"] = (BLIND["LongNav-R1"][0], BLIND["LongNav-R1"][1], r1_ref)
    BLIND["Uni-NaVid"] = (BLIND["Uni-NaVid"][0], BLIND["Uni-NaVid"][1], args.runs / "uninavid_ref")
    args.out.mkdir(parents=True, exist_ok=True)
    labels_100 = load_manifest(ROOT / "src/longnav/conf/episode_jsons" / m100)
    labels_9 = load_manifest(ROOT / "src/longnav/conf/episode_jsons" / m9)
    summary: dict = {"arms": {}, "checks": []}
    incomplete = []

    # -- KL curves ----------------------------------------------------------------------
    curves: dict[str, dict] = {}
    step_rows = []
    for curve, (arm, loader, model) in PROBES.items():
        run_dir = args.runs / arm
        if not run_dir.is_dir():
            if arm in OPTIONAL:
                summary["arms"][arm] = "not run (optional)"
            else:
                incomplete.append(f"{arm}: missing")
            continue
        rows, layers = loader(run_dir, model=curve)
        done = sorted({r["episode"] for r in rows})
        missing = [l for l in labels_9 if l not in done]
        summary["arms"][arm] = {"episodes": len(done), "missing": missing, "steps": len({(r['episode'], r['step']) for r in rows})}
        if missing:
            incomplete.append(f"{arm}: {len(missing)} of {len(labels_9)} episodes missing")
        if not rows:
            continue
        step_rows += rows
        curves[curve] = {"model": model, "layers": layers,
                         "keep": {r["layer"]: r for r in hz.layer_curve(rows, "keep_one")},
                         "leave": {r["layer"]: r for r in hz.layer_curve(rows, "leave_one")}}
        if arm.startswith("r1_"):
            run_rows = load_r1_run(run_dir)
            ref = load_r1_run(BLIND["LongNav-R1"][2]) if BLIND["LongNav-R1"][2].is_dir() else {}
            summary["checks"] += trajectory_check(run_rows, ref, labels_9, arm) or [f"{arm}: trajectories match the reference"]

    # -- SR / SPL -----------------------------------------------------------------------
    perf: dict[str, dict[int, dict]] = {}
    for model, (template, run_loader, ref_dir) in BLIND.items():
        perf[model] = {}
        ref = run_loader(ref_dir) if ref_dir.is_dir() else {}
        if not ref:
            incomplete.append(f"{model} reference {ref_dir.name}: missing")
        else:
            missing, extra = split_status(ref, labels_100)
            if missing or extra:
                incomplete.append(f"{ref_dir.name}: {len(missing)} of 100 episodes missing, {len(extra)} extra")
            summary["arms"][ref_dir.name] = {"episodes": len(ref), "missing": missing, "extra": extra}
        for L in LAYERS[model] + [N_LAYERS[model]]:
            if L == N_LAYERS[model]:
                rows = ref
                name = ref_dir.name
            else:
                run_dir = args.runs / template.format(L=L)
                name = run_dir.name
                if not run_dir.is_dir():
                    incomplete.append(f"{name}: missing"); continue
                rows = run_loader(run_dir)
                missing, extra = split_status(rows, labels_100)
                summary["arms"][name] = {"episodes": len(rows), "missing": missing, "extra": extra}
                if missing or extra:
                    incomplete.append(f"{name}: {len(missing)} of 100 episodes missing, {len(extra)} extra")
            if not rows:
                continue
            rec = {"arm": name, "n": len(rows)}
            for key, col in (("success", "sr"), ("spl", "spl")):
                m, lo, hi = bootstrap_mean_ci([r[key] for r in rows.values()])
                rec |= {col: m, f"{col}_lo": lo, f"{col}_hi": hi}
                if ref and rows is not ref:
                    d = paired_delta(rows, ref, key)
                    rec |= {f"d{col}": d["delta"], f"p_{col}": d["p"]}
            m, lo, hi = bootstrap_mean_ci([r["n_steps"] for r in rows.values()])
            rec |= {"steps": m}
            perf[model][L] = rec

    if incomplete and not args.allow_partial:
        sys.exit("incomplete arms (pass --allow-partial for an interim look):\n  " + "\n  ".join(incomplete))
    summary["incomplete"] = incomplete

    # -- long-form table ----------------------------------------------------------------
    out_rows = []
    for curve, (arm, _, model) in PROBES.items():
        n = N_LAYERS[model]
        c = curves.get(curve)
        if c is None and arm in OPTIONAL:
            continue
        # KL rows at every probed layer (the probes are cheap and ran the full grid); SR/SPL only
        # where a blind arm exists.
        probed = sorted(set(c["layers"]) | set(LAYERS[model])) if c else list(LAYERS[model])
        for L in [x for x in probed if x < n] + [n]:
            row = {"curve": curve, "model": model, "arm": arm, "layer": L, "n_layers": n, "layer_frac": L / n,
                   "source": "reference (nothing hidden)" if L == n else "probe"}
            if c is not None:
                if L == n:
                    row |= {"keep_one_mean": 0.0, "keep_one_lo": 0.0, "keep_one_hi": 0.0,
                            "leave_one_mean": 0.0, "leave_one_lo": 0.0, "leave_one_hi": 0.0, "exact": "by construction"}
                else:
                    k, l = c["keep"].get(L), c["leave"].get(L)
                    if k:
                        row |= {x: k[x] for x in ("n_steps", "n_episodes", "keep_one_mean", "keep_one_lo", "keep_one_hi",
                                                  "keep_one_median", "keep_one_balanced_mean")}
                    if l:
                        row |= {x: l[x] for x in ("leave_one_mean", "leave_one_lo", "leave_one_hi",
                                                  "leave_one_median", "leave_one_balanced_mean")}
            if curve != "LongNav-R1 (leak-free)":  # the blind rollout is leak-free by construction: one SR/SPL curve per model
                p = perf.get(model, {}).get(L)
                if p:
                    row |= {x: p.get(x) for x in ("n", "sr", "sr_lo", "sr_hi", "dsr", "p_sr", "spl", "spl_lo", "spl_hi",
                                                  "dspl", "p_spl", "steps")}
                    row["blind_arm"] = p["arm"]
            out_rows.append(row)
    fields = ["curve", "model", "arm", "layer", "n_layers", "layer_frac", "source", "n_steps", "n_episodes",
              "keep_one_mean", "keep_one_lo", "keep_one_hi", "keep_one_median", "keep_one_balanced_mean",
              "leave_one_mean", "leave_one_lo", "leave_one_hi", "leave_one_median", "leave_one_balanced_mean",
              "blind_arm", "n", "sr", "sr_lo", "sr_hi", "dsr", "p_sr", "spl", "spl_lo", "spl_hi", "dspl", "p_spl",
              "steps", "exact"]
    write_csv(args.out / "horizon.csv", out_rows, fields)
    write_csv(args.out / "probe_steps.csv", sorted(step_rows, key=lambda r: (r["model"], r["episode"], r["step"], r["layer"])),
              hz.ROW_FIELDS)

    # -- tex ----------------------------------------------------------------------------
    lines = ["% Generated by paper/info_horizon/aggregate.py -- do not edit by hand.",
             r"\begin{tabular}{l r cc cc}", r"\toprule",
             r"model & $L$ & keep-one-in & leave-one-out & SR & SPL \\", r"\midrule"]
    for curve, (arm, _, model) in PROBES.items():
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{curve}}}}} \\")
        for r in (x for x in out_rows if x["curve"] == curve):
            L = "none" if r["layer"] == r["n_layers"] else str(r["layer"])
            sr = fmt(r.get("sr"), 2) + (stars(r["p_sr"]) if r.get("p_sr") is not None and np.isfinite(r.get("p_sr", np.nan)) else "")
            spl = fmt(r.get("spl"), 3) + (stars(r["p_spl"]) if r.get("p_spl") is not None and np.isfinite(r.get("p_spl", np.nan)) else "")
            lines.append(f"{model} & {L} & {fmt(r.get('keep_one_mean'), 4)} & {fmt(r.get('leave_one_mean'), 4)} & {sr} & {spl} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              f"% KL in nats, pooled mean over every step of the {len(labels_9)} probe episodes; SR/SPL over the",
              "% 100-episode split with all visual tokens hidden from layer L on; stars: paired sign-flip",
              "% test vs the unblinded reference, *p<0.05 **p<0.01 ***p<0.001."]
    (args.out / "table.tex").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"{'curve':24}{'L':>5}{'keep-one':>11}{'leave-one':>11}{'SR':>7}{'SPL':>8}{'n_steps':>9}")
    for r in out_rows:
        print(f"{r['curve']:24}{r['layer']:>5}{fmt(r.get('keep_one_mean'), 4):>11}{fmt(r.get('leave_one_mean'), 4):>11}"
              f"{fmt(r.get('sr'), 2):>7}{fmt(r.get('spl'), 3):>8}{str(r.get('n_steps', '')):>9}")
    for m in summary["checks"]:
        print("check:", m)
    if incomplete:
        print("INCOMPLETE:", *incomplete, sep="\n  ")
    print(f"wrote {args.out / 'horizon.csv'}, {args.out / 'probe_steps.csv'}, {args.out / 'table.tex'}, {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
