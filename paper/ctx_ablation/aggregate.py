#!/usr/bin/env python3
"""Aggregate the context-window ablation into paper_results/ctx_ablation/results/.

Reads the LongNav-R1 arms from paper_results/ctx_ablation/runs/<arm>/rollout/results_* and
the Uni-NaVid arms from paper_results/ctx_ablation/runs/<arm>/episodes/*.json, checks every
arm against the 100-episode manifest, and writes:

  results/per_episode.csv  one row per (arm, episode): success, spl, n_steps, kv/mem/latency
  results/summary.csv      one row per arm: means with 95% bootstrap CIs, paired sign-flip
                           p-values vs the model's full-context baseline, and for R1 the
                           eviction-leakage gap (evict - recompute at matched K)
  results/table.tex        booktabs table, stars = significance vs the model's full baseline

Arms that have not been run yet are reported and left out (--strict makes that an error).
An arm that exists but is incomplete is an error, so the final numbers can never rest on a
smaller denominator; --skip-partial downgrades that to a warning for interim looks while the
sweep is still running. Runs in longnav_vlm (numpy only):

    python paper/ctx_ablation/aggregate.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from longnav.analysis.results_io import assert_split, load_manifest, load_r1_run, load_uninavid_run, split_status
from longnav.analysis.stats import bootstrap_mean_ci, paired_delta, stars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/ctx_ablation"
K_LIST = (32, 16, 8, 4, 1)
MODELS = {"r1": "LongNav-R1", "uninavid": "Uni-NaVid"}


def arm_specs():
    """(arm, model, mode, K) for every arm of the experiment. K=None is full context."""
    yield "r1_full", "r1", "full", None
    for mode in ("evict", "recompute"):
        for k in K_LIST:
            yield f"r1_{mode}_k{k}", "r1", mode, k
    yield "uninavid_full", "uninavid", "full", None
    for k in K_LIST:
        yield f"uninavid_k{k}", "uninavid", "native", k


def nanmean(rows, key):
    v = np.array([float(r.get(key, np.nan)) for r in rows.values()])
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def nanmax(rows, key):
    v = np.array([float(r.get(key, np.nan)) for r in rows.values()])
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1-root", type=Path, default=RESULTS_ROOT / "runs")
    ap.add_argument("--uninavid-root", type=Path, default=RESULTS_ROOT / "runs")
    ap.add_argument("--manifest", type=Path, default=ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_100.json")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "results")
    ap.add_argument("--strict", action="store_true", help="fail if any arm is missing")
    ap.add_argument("--skip-partial", action="store_true",
                    help="leave out arms that are still running instead of failing (interim plots only)")
    args = ap.parse_args()

    labels = load_manifest(args.manifest)
    runs, meta, missing = {}, {}, []
    for arm, model, mode, k in arm_specs():
        root = args.r1_root if model == "r1" else args.uninavid_root
        run_dir = root / arm
        if not run_dir.is_dir():
            missing.append(arm)
            continue
        rows = load_r1_run(run_dir) if model == "r1" else load_uninavid_run(run_dir)
        if args.skip_partial and split_status(rows, labels)[0]:
            print(f"[partial] {arm}: {len(rows)}/{len(labels)} episodes, skipped")
            missing.append(arm)
            continue
        assert_split(rows, labels, arm)  # a partial arm is an error, never a smaller denominator
        runs[arm] = rows
        meta[arm] = (model, mode, k)
    if missing:
        print(f"[missing] {len(missing)} arm(s) not run yet: {' '.join(missing)}")
        if args.strict:
            sys.exit(1)
    if not runs:
        sys.exit("no arms found")

    args.out.mkdir(parents=True, exist_ok=True)

    # --- per-episode ---------------------------------------------------------------------
    ep_cols = ["arm", "model", "mode", "K", "episode_label", "success", "spl", "soft_spl",
               "distance_to_goal", "n_steps", "kv_len_mean", "kv_len_max", "peak_mem_GB", "vlm_latency_ms"]
    with (args.out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_cols)
        w.writeheader()
        for arm, rows in runs.items():
            model, mode, k = meta[arm]
            for label in labels:
                r = rows[label]
                lat = r.get("sup/mean_vlm_latency")
                w.writerow({
                    "arm": arm, "model": MODELS[model], "mode": mode, "K": "full" if k is None else k,
                    "episode_label": label, "success": r["success"], "spl": r["spl"],
                    "soft_spl": r.get("soft_spl", ""), "distance_to_goal": r.get("distance_to_goal", ""),
                    "n_steps": r["n_steps"],
                    "kv_len_mean": r.get("sup/mean_kv_len", ""), "kv_len_max": r.get("sup/max_kv_len", ""),
                    "peak_mem_GB": r.get("sup/max_vlm_mem_GB", r.get("peak_mem_GB", "")),
                    "vlm_latency_ms": 1000 * lat if lat is not None else "",
                })

    # --- per-arm summary -----------------------------------------------------------------
    summary = []
    for arm, rows in runs.items():
        model, mode, k = meta[arm]
        rec = {"arm": arm, "model": MODELS[model], "mode": mode, "K": "full" if k is None else k, "n": len(rows)}
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            m, lo, hi = bootstrap_mean_ci([r[key] for r in rows.values()])
            rec.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
        rec["kv_len_mean"] = nanmean(rows, "sup/mean_kv_len")
        rec["kv_len_max"] = nanmax(rows, "sup/max_kv_len")
        rec["peak_mem_GB"] = nanmax(rows, "sup/max_vlm_mem_GB") if model == "r1" else nanmax(rows, "peak_mem_GB")
        rec["vlm_latency_ms"] = 1000 * nanmean(rows, "sup/mean_vlm_latency")

        full = runs.get(f"{model}_full")
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            if full is not None and arm != f"{model}_full":
                d = paired_delta(rows, full, key, labels)
                rec.update({f"d{name}_vs_full": d["delta"], f"p_{name}_vs_full": d["p"]})
            else:
                rec.update({f"d{name}_vs_full": "", f"p_{name}_vs_full": ""})

        # Eviction leakage: evict minus leak-free recompute at the same K (R1 only).
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            for col in (f"leak_{name}", f"leak_{name}_lo", f"leak_{name}_hi", f"leak_{name}_p"):
                rec[col] = ""
        if mode == "evict":
            partner = runs.get(f"r1_recompute_k{k}")
            if partner is not None:
                for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
                    d = paired_delta(rows, partner, key, labels)
                    rec.update({f"leak_{name}": d["delta"], f"leak_{name}_lo": d["lo"],
                                f"leak_{name}_hi": d["hi"], f"leak_{name}_p": d["p"]})
        summary.append(rec)

    with (args.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # --- LaTeX table ---------------------------------------------------------------------
    by_arm = {r["arm"]: r for r in summary}

    def cell(arm, name, fmt):
        r = by_arm.get(arm)
        if r is None:
            return "--"
        s = stars(float(r[f"p_{name}_vs_full"])) if r[f"p_{name}_vs_full"] != "" else ""
        return f"{r[name]:{fmt}}" + (f"$^{{{s}}}$" if s else "")

    lines = [
        r"% Generated by paper/ctx_ablation/aggregate.py -- do not edit by hand.",
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{LongNav-R1 (KV eviction)} & \multicolumn{2}{c}{LongNav-R1 (leak-free)} & \multicolumn{2}{c}{Uni-NaVid} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"$K$ & SR & SPL & SR & SPL & SR & SPL \\",
        r"\midrule",
    ]
    for k in ("full",) + K_LIST:
        if k == "full":
            arms = ("r1_full", "r1_full", "uninavid_full")
        else:
            arms = (f"r1_evict_k{k}", f"r1_recompute_k{k}", f"uninavid_k{k}")
        cells = []
        for a in arms:
            cells += [cell(a, "sr", ".2f"), cell(a, "spl", ".3f")]
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $K$ = frames visible when the action is read (current frame included); full = whole episode.",
              r"% Stars: paired sign-flip test vs the same model's full-context run, *p<0.05 **p<0.01 ***p<0.001.",
              f"% n = {len(labels)} HM3D-v2 val episodes; SR/SPL are episode means."]
    (args.out / "table.tex").write_text("\n".join(lines) + "\n")

    # --- console -------------------------------------------------------------------------
    hdr = f"{'arm':<20}{'n':>4}{'SR':>7}{'SPL':>8}{'steps':>7}{'kv_len':>8}{'memGB':>7}{'dSR':>8}{'p':>7}{'dSPL':>8}{'p':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        f = lambda v, fmt: (f"{v:{fmt}}" if v != "" and np.isfinite(float(v)) else "-")
        print(f"{r['arm']:<20}{r['n']:>4}{r['sr']:>7.2f}{r['spl']:>8.3f}{r['steps']:>7.0f}"
              f"{f(r['kv_len_mean'], '.0f'):>8}{f(r['peak_mem_GB'], '.1f'):>7}"
              f"{f(r['dsr_vs_full'], '+.3f'):>8}{f(r['p_sr_vs_full'], '.3f'):>7}"
              f"{f(r['dspl_vs_full'], '+.3f'):>8}{f(r['p_spl_vs_full'], '.3f'):>7}")
    ctrl = by_arm.get("r1_evict_k1")
    if ctrl is not None and ctrl["leak_sr"] != "":
        print(f"\nK=1 control (evict - recompute, should be ~0): dSR={ctrl['leak_sr']:+.3f} "
              f"(p={ctrl['leak_sr_p']:.3f}), dSPL={ctrl['leak_spl']:+.3f} (p={ctrl['leak_spl_p']:.3f})")
    print(f"\nwrote {args.out / 'per_episode.csv'}, {args.out / 'summary.csv'}, {args.out / 'table.tex'}")


if __name__ == "__main__":
    main()
