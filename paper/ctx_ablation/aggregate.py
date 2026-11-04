#!/usr/bin/env python3
"""Aggregate the context-window ablation into paper_results/ctx_ablation/results/.

Reads the LongNav-R1 arms from paper_results/ctx_ablation/runs/<arm>/rollout/results_* and
the Uni-NaVid and LightNav-0 arms from paper_results/ctx_ablation/runs/<arm>/episodes/*.json
(the same per-episode layout), checks every arm against the 100-episode manifest, and writes:
The LightNav-0 "old blocks kept" control arms lightnav_keep_k{32,16,8,4,2,1} (2026-10-27) are the
flat token-prune study's lightnav_recency_b{30,14,6,2} runs (paper_results/token_prune_flat/runs,
eval_prune.py --prune-method recency): the same image tokens as the window at K (the newest K
frames; history pairs at the model's 28-token pooling) but every older segment's timestamp and
vision markers left in the prompt. They are read from there, paired against lightnav_full like
every other arm, and against the window arm at the same K (gap_* columns). Two points below
the window grid: K = 2 is lightnav_random_b0 (the full-resolution current pair, no history
tokens; there is no K = 2 window arm, so its gap_* columns are empty) and K = 1 is
lightnav_recency_b0_cur1 (no history tokens and the current frame doubled as the current
tubelet, --prune-current-frames 1, the K = 1 window's own layout).

  results/per_episode.csv  one row per (arm, episode): success, spl, n_steps, kv/mem/latency
  results/summary.csv      one row per arm: means with 95% bootstrap CIs, paired sign-flip
                           p-values vs the model's full-context baseline, and for R1 the
                           eviction-leakage gap (evict - recompute at matched K)
  results/table.tex        booktabs table, stars = significance vs the model's full baseline

Arms that have not been run yet are reported and left out (--strict makes that an error).
An arm that exists but is incomplete is an error, so the final numbers can never rest on a
smaller denominator; --skip-partial downgrades that to a warning for interim looks while the
sweep is still running. Runs in longnav_vlm (numpy only):

    python paper/ctx_ablation/aggregate.py [--dataset navverse]

--dataset navverse aggregates the NavVerse indoor replicate: the LongNav-R1 arms (nv_full,
nv_evict_k<K>) and the LightNav-0 old-blocks-kept arms (nvl_full, nvl_keep_k<K>: the model behind
the same NavVerse client via baselines/lightnav_hm3d/serve_navverse.py, every arm's rows read from
paper_results/ctx_ablation_navverse/runs/<arm>/<arm>/result.json via results_io.load_navverse_run:
the benchmark's own measures, forced stops at the 500-step cap re-scored as failures, the
server's per-step log for kv/latency/memory) into paper_results/ctx_ablation_navverse/results/
with the same columns plus per-arm forced-stop and time-out counts, each episode's outcome class
(results_io.navverse_failure_mode: success / fell / cap_never_near_goal / cap_near_goal_no_stop /
wrong_stop) in per_episode.csv and summary.csv, and results/failure_modes.csv with one count column
per class and arm (the stacked-bar figure of plot.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from longnav.analysis.results_io import (NAVVERSE_FAILURE_MODES, assert_split, load_lightnav_run, load_manifest,
                                         load_navverse_run, load_r1_run, load_uninavid_run, split_status)
from longnav.analysis.stats import bootstrap_mean_ci, paired_delta, stars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/ctx_ablation"
K_LIST = (32, 16, 8, 4, 1)
KEEP_ROOT = ROOT / "paper_results/token_prune_flat/runs"  # LightNav-0 old-blocks-kept arms live there
KEEP_SOURCE = {f"lightnav_keep_k{k}": KEEP_ROOT / f"lightnav_recency_b{k - 2}" for k in (32, 16, 8, 4)}
KEEP_SOURCE["lightnav_keep_k2"] = KEEP_ROOT / "lightnav_random_b0"  # current pair only, no history (no K = 2 window)
KEEP_SOURCE["lightnav_keep_k1"] = KEEP_ROOT / "lightnav_recency_b0_cur1"  # no history, current frame doubled
KEEP_K = (32, 16, 8, 4, 2, 1)  # K of the keep arms (K_LIST plus the K = 2 point)
NVL_KEEP_K = (32, 16, 8, 4, 1)  # NavVerse LightNav-0 keep arms (run_navverse_lightnav.sh; K = 1: no history, current
# frame doubled, as HM3D lightnav_keep_k1; no K = 2 point)
MODELS = {"r1": "LongNav-R1", "uninavid": "Uni-NaVid", "lightnav": "LightNav-0", "nv": "LongNav-R1",
          "nvl": "LightNav-0"}  # nv* = the NavVerse arms (model key carries the dataset)
LOADERS = {"r1": load_r1_run, "uninavid": load_uninavid_run, "lightnav": load_lightnav_run,
           "nv": load_navverse_run, "nvl": load_navverse_run}  # LightNav-0 / NavVerse forced stops at the cap are re-scored as failures
DATASETS = {  # dataset -> (results root, manifest, models with arms, episode-set note for the table)
    "hm3d": (RESULTS_ROOT, ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_100.json",
             ("r1", "uninavid", "lightnav"), "HM3D-v2 val episodes"),
    "navverse": (ROOT / "paper_results/ctx_ablation_navverse",
                 ROOT / "src/longnav/conf/episode_jsons/navverse_indoor_100_seed17.json",
                 ("nv", "nvl"), "NavVerse indoor (GRScenes commercial) ObjectNav episodes, benchmark500_indoor seed-17 subset"),
}


def arm_specs(models=("r1", "uninavid", "lightnav")):
    """(arm, model, mode, K) for every arm of the experiment. K=None is full context."""
    if "r1" in models:
        yield "r1_full", "r1", "full", None
        for mode in ("evict", "recompute"):
            for k in K_LIST:
                yield f"r1_{mode}_k{k}", "r1", mode, k
    if "uninavid" in models:
        yield "uninavid_full", "uninavid", "full", None
        for k in K_LIST:
            yield f"uninavid_k{k}", "uninavid", "native", k
    if "lightnav" in models:
        yield "lightnav_full", "lightnav", "full", None
        for k in K_LIST:
            yield f"lightnav_k{k}", "lightnav", "native", k
        for k in KEEP_K:  # window content with the old segments' skeleton kept (prune wrapper)
            yield f"lightnav_keep_k{k}", "lightnav", "keep", k
    if "nv" in models:  # NavVerse indoor replicate of the LongNav-R1 arms
        yield "nv_full", "nv", "full", None
        for mode in ("evict", "recompute"):
            for k in K_LIST:
                yield f"nv_{mode}_k{k}", "nv", mode, k
    if "nvl" in models:  # NavVerse: LightNav-0, old blocks kept (serve_navverse.py --prune-method recency)
        yield "nvl_full", "nvl", "full", None
        for k in NVL_KEEP_K:
            yield f"nvl_keep_k{k}", "nvl", "keep", k


def count(rows, key, value=True):
    return int(sum(1 for r in rows.values() if r.get(key) == value))


def nanmean(rows, key):
    v = np.array([float(r.get(key, np.nan)) for r in rows.values()])
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def nanmax(rows, key):
    v = np.array([float(r.get(key, np.nan)) for r in rows.values()])
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="hm3d")
    ap.add_argument("--r1-root", type=Path, default=None, help="default: <results root>/runs")
    ap.add_argument("--uninavid-root", type=Path, default=None,
                    help="root of the Uni-NaVid and LightNav-0 arms (episodes/ layout); default: <results root>/runs")
    ap.add_argument("--manifest", type=Path, default=None, help="default: the dataset's 100-episode manifest")
    ap.add_argument("--out", type=Path, default=None, help="default: <results root>/results")
    ap.add_argument("--strict", action="store_true", help="fail if any arm is missing")
    ap.add_argument("--skip-partial", action="store_true",
                    help="leave out arms that are still running instead of failing (interim plots only)")
    args = ap.parse_args()
    results_root, manifest, models, episode_note = DATASETS[args.dataset]
    r1_root = args.r1_root or results_root / "runs"
    uninavid_root = args.uninavid_root or results_root / "runs"
    args.manifest = args.manifest or manifest
    args.out = args.out or results_root / "results"

    labels = load_manifest(args.manifest)
    runs, meta, missing = {}, {}, []
    for arm, model, mode, k in arm_specs(models):
        root = r1_root if model in ("r1", "nv", "nvl") else uninavid_root
        run_dir = KEEP_SOURCE[arm] if arm in KEEP_SOURCE else root / arm  # HM3D keep arms live in the prune study
        if not run_dir.is_dir():
            missing.append(arm)
            continue
        # LightNav-0 / NavVerse forced stops at the 500-step cap are re-scored as failures (results_io).
        rows = LOADERS[model](run_dir)
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
    # R1 rows carry Habitat's sup/* keys; NavVerse rows carry the server-log keys directly.
    kv_keys = {"kv_len_mean": "sup/mean_kv_len", "kv_len_max": "sup/max_kv_len", "peak_mem_GB": "sup/max_vlm_mem_GB"}

    def ep_extras(r, model):
        if model in ("nv", "nvl"):
            return {c: r.get(c, "") for c in ("kv_len_mean", "kv_len_max", "peak_mem_GB", "vlm_latency_ms")}
        lat = r.get("sup/mean_vlm_latency")
        return {"kv_len_mean": r.get("sup/mean_kv_len", ""), "kv_len_max": r.get("sup/max_kv_len", ""),
                "peak_mem_GB": r.get("sup/max_vlm_mem_GB", r.get("peak_mem_GB", "")),
                "vlm_latency_ms": 1000 * lat if lat is not None else ""}

    navverse = args.dataset == "navverse"  # extra step-cap columns; the HM3D outputs keep their layout
    ep_cols = ["arm", "model", "mode", "K", "episode_label", "success", "spl", "soft_spl",
               "distance_to_goal", "n_steps", "kv_len_mean", "kv_len_max", "peak_mem_GB", "vlm_latency_ms"]
    if navverse:
        ep_cols += ["forced_stop", "termination_reason", "oracle_success", "failure_mode"]
    with (args.out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_cols, extrasaction="ignore")
        w.writeheader()
        for arm, rows in runs.items():
            model, mode, k = meta[arm]
            for label in labels:
                r = rows[label]
                w.writerow({
                    "arm": arm, "model": MODELS[model], "mode": mode, "K": "full" if k is None else k,
                    "episode_label": label, "success": r["success"], "spl": r["spl"],
                    "soft_spl": r.get("soft_spl", ""), "distance_to_goal": r.get("distance_to_goal", ""),
                    "n_steps": r["n_steps"], **ep_extras(r, model),
                    "forced_stop": int(r["forced_stop"]) if "forced_stop" in r else "",
                    "termination_reason": r.get("termination_reason", ""),
                    "oracle_success": r.get("oracle_success", ""), "failure_mode": r.get("failure_mode", ""),
                })

    # --- per-arm summary -----------------------------------------------------------------
    summary = []
    for arm, rows in runs.items():
        model, mode, k = meta[arm]
        rec = {"arm": arm, "model": MODELS[model], "mode": mode, "K": "full" if k is None else k, "n": len(rows)}
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            m, lo, hi = bootstrap_mean_ci([r[key] for r in rows.values()])
            rec.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
        if model in ("nv", "nvl"):  # nvl: kv_len = the step's prompt tokens (LightNav-0 re-prefills every step)
            rec["kv_len_mean"] = nanmean(rows, "kv_len_mean")
            rec["kv_len_max"] = nanmax(rows, "kv_len_max")
            rec["peak_mem_GB"] = nanmax(rows, "peak_mem_GB")
            rec["vlm_latency_ms"] = nanmean(rows, "vlm_latency_ms")
        else:
            rec["kv_len_mean"] = nanmean(rows, kv_keys["kv_len_mean"])
            rec["kv_len_max"] = nanmax(rows, kv_keys["kv_len_max"])
            rec["peak_mem_GB"] = nanmax(rows, kv_keys["peak_mem_GB"]) if model == "r1" else nanmax(rows, "peak_mem_GB")
            rec["vlm_latency_ms"] = 1000 * nanmean(rows, "sup/mean_vlm_latency")
        if navverse:
            # Step-cap bookkeeping: forced stops at the cap (re-scored as failures), NavVerse's
            # simulated-time limit, and every other simulator termination (bad_orientation, ...).
            rec["forced_stops"] = count(rows, "forced_stop")
            rec["rescored"] = count(rows, "rescored")
            rec["time_outs"] = count(rows, "termination_reason", "time_out")
            reasons = sorted({r.get("termination_reason") for r in rows.values()} - {None})
            rec["terminations"] = " ".join(f"{reason}={count(rows, 'termination_reason', reason)}" for reason in reasons)
            # Outcome classes (results_io.navverse_failure_mode); "other" only if some row fits no class.
            modes = list(NAVVERSE_FAILURE_MODES) + (["other"] if count(rows, "failure_mode", "other") else [])
            rec["failure_modes"] = " ".join(f"{m}={count(rows, 'failure_mode', m)}" for m in modes)

        full = runs.get(f"{model}_full")
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            if full is not None and arm != f"{model}_full":
                d = paired_delta(rows, full, key, labels)
                rec.update({f"d{name}_vs_full": d["delta"], f"p_{name}_vs_full": d["p"]})
            else:
                rec.update({f"d{name}_vs_full": "", f"p_{name}_vs_full": ""})

        # Eviction leakage: evict minus leak-free recompute at the same K (R1 only); skeleton gap:
        # LightNav-0 old-blocks-kept minus the window at the same K.
        for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            for pre in ("leak", "gap"):
                for col in (f"{pre}_{name}", f"{pre}_{name}_lo", f"{pre}_{name}_hi", f"{pre}_{name}_p"):
                    rec[col] = ""
        pairings = {"evict": ("leak", f"{model}_recompute_k{k}"), "keep": ("gap", f"{model}_k{k}")}
        if mode in pairings:
            pre, partner_arm = pairings[mode]
            partner = runs.get(partner_arm)
            if partner is not None:
                for key, name in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
                    d = paired_delta(rows, partner, key, labels)
                    rec.update({f"{pre}_{name}": d["delta"], f"{pre}_{name}_lo": d["lo"],
                                f"{pre}_{name}_hi": d["hi"], f"{pre}_{name}_p": d["p"]})
        summary.append(rec)

    with (args.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    if navverse:
        # Failure-mode counts per arm (every row is in exactly one class, so each arm sums to n).
        modes = list(NAVVERSE_FAILURE_MODES)
        if any(count(rows, "failure_mode", "other") for rows in runs.values()):
            modes.append("other")
        with (args.out / "failure_modes.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "K", "n", *modes])
            w.writeheader()
            for arm, rows in runs.items():
                counts = {m: count(rows, "failure_mode", m) for m in modes}
                assert sum(counts.values()) == len(rows), f"{arm}: failure modes do not cover every episode"
                w.writerow({"arm": arm, "K": "full" if meta[arm][2] is None else meta[arm][2], "n": len(rows), **counts})

    # --- LaTeX table ---------------------------------------------------------------------
    by_arm = {r["arm"]: r for r in summary}

    def cell(arm, name, fmt):
        r = by_arm.get(arm)
        if r is None:
            return "--"
        s = stars(float(r[f"p_{name}_vs_full"])) if r[f"p_{name}_vs_full"] != "" else ""
        return f"{r[name]:{fmt}}" + (f"$^{{{s}}}$" if s else "")

    # Column groups: (header, arm template) -- the HM3D table has all three models, the NavVerse
    # replicate only the LongNav-R1 arms.
    if args.dataset == "navverse":
        groups = [("LongNav-R1 (KV eviction)", lambda k: "nv_full" if k == "full" else f"nv_evict_k{k}")]
        if any(arm.startswith("nv_recompute_") for arm in runs):  # leak-free arms are optional on NavVerse
            groups.append(("LongNav-R1 (leak-free)", lambda k: "nv_full" if k == "full" else f"nv_recompute_k{k}"))
        if any(arm.startswith("nvl_") for arm in runs):
            groups.append(("LightNav-0 (old blocks kept)", lambda k: "nvl_full" if k == "full" else f"nvl_keep_k{k}"))
    else:
        groups = [("LongNav-R1 (KV eviction)", lambda k: "r1_full" if k == "full" else f"r1_evict_k{k}"),
                  ("LongNav-R1 (leak-free)", lambda k: "r1_full" if k == "full" else f"r1_recompute_k{k}"),
                  ("Uni-NaVid", lambda k: "uninavid_full" if k == "full" else f"uninavid_k{k}"),
                  ("LightNav-0", lambda k: "lightnav_full" if k == "full" else f"lightnav_k{k}"),
                  ("LightNav-0 (old blocks kept)", lambda k: "lightnav_full" if k == "full" else f"lightnav_keep_k{k}")]
    lines = [
        r"% Generated by paper/ctx_ablation/aggregate.py -- do not edit by hand.",
        r"\begin{tabular}{l " + " ".join("cc" for _ in groups) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h, _ in groups) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(groups))),
        r"$K$ & " + " & ".join("SR & SPL" for _ in groups) + r" \\",
        r"\midrule",
    ]
    table_ks = K_LIST if navverse else tuple(sorted(set(K_LIST) | set(KEEP_K), reverse=True))  # K = 2 row: keep arm only
    for k in ("full",) + table_ks:
        cells = []
        for _, arm_of in groups:
            cells += [cell(arm_of(k), "sr", ".2f"), cell(arm_of(k), "spl", ".3f")]
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")
    if navverse:
        keep_note = [
            r"% LightNav-0 (old blocks kept): the image tokens of the current pair and the K-2 newest history frames",
            r"% kept, every older block's image tokens hidden with its timestamp and vision markers left in the prompt",
            r"% (serve_navverse.py --prune-method recency --prune-k K-2 --prune-tokens-per-frame 15: frames at their",
            r"% native 4:3 aspect, 384x288, 30 tokens per pooled history pair; eager masked attention, nvl_full on the",
            r"% compiled kernel). LightNav's 1 s velocity step is discretized to the client's STOP / FORWARD / TURN.",
            r"% K = 1: no history and the current frame doubled (--prune-k 0 --prune-current-frames 1, as the HM3D",
            r"% lightnav_recency_b0_cur1 point); no K = 2 arm for LightNav-0 on NavVerse.",
        ]
    else:
        keep_note = [
            r"% LightNav-0 (old blocks kept): the window's image tokens at K, with every older segment's timestamp",
            r"% and vision markers left in the prompt (flat token-prune study, lightnav_recency_b{K-2}; K = 2: the",
            r"% current pair and no history, lightnav_random_b0, no window arm at K = 2; K = 1: no history and the",
            r"% current frame doubled, lightnav_recency_b0_cur1).",
        ]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $K$ = frames visible when the action is read (current frame included); full = whole episode.",
              *keep_note,
              r"% Stars: paired sign-flip test vs the same model's full-context run, *p<0.05 **p<0.01 ***p<0.001.",
              f"% n = {len(labels)} {episode_note}; SR/SPL are episode means."]
    if args.dataset == "navverse":
        falls = " ".join(f"{r['arm']}={r['terminations']}" for r in summary)
        lines.append(r"% NavVerse measures: success = STOP within 1.6 m; forced STOPs at the 500-step cap are "
                     r"re-scored as failures; bad_orientation = the robot tipped over (a failure with no Habitat "
                     r"analogue); the simulated-time limit was raised to 1800 s so no episode timed out.")
        lines.append(f"% Terminations per arm: {falls}")
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
    if args.dataset == "navverse":
        print("\n" + f"{'arm':<20}{'forced':>8}{'rescored':>10}{'time_out':>10}{'latency_ms':>12}  terminations")
        for r in summary:
            print(f"{r['arm']:<20}{r['forced_stops']:>8}{r['rescored']:>10}{r['time_outs']:>10}"
                  f"{f(r['vlm_latency_ms'], '.0f'):>12}  {r['terminations']}")
    ctrl = by_arm.get("nv_evict_k1" if args.dataset == "navverse" else "r1_evict_k1")
    if ctrl is not None and ctrl["leak_sr"] != "":
        print(f"\nK=1 control (evict - recompute, should be ~0): dSR={ctrl['leak_sr']:+.3f} "
              f"(p={ctrl['leak_sr_p']:.3f}), dSPL={ctrl['leak_spl']:+.3f} (p={ctrl['leak_spl_p']:.3f})")
    print(f"\nwrote {args.out / 'per_episode.csv'}, {args.out / 'summary.csv'}, {args.out / 'table.tex'}")


if __name__ == "__main__":
    main()
