#!/usr/bin/env python3
"""Aggregate the hybrid-pruning comparison into paper_results/hybrid_prune/results/.

Reads the LongNav-R1 arms from runs/<arm>/rollout/results_* and the Uni-NaVid arms from
runs/<arm>/episodes/*.json (+ the per-inference stage counts in prune/<label>.jsonl), and writes:

  results/per_episode.csv  one row per (arm, episode): success, spl, n_steps and the memory columns
  results/summary.csv      one row per arm: means with 95% bootstrap CIs and the memory columns
  results/paired.csv       hybrid - pyramiddrop at each (model, k): paired delta, bootstrap CI and
                           sign-flip p on SR, SPL and steps
  results/table.tex        booktabs table (stars = hybrid vs pyramiddrop at the same k)

Budget. k bounds the history visual tokens AVERAGED OVER ALL DECODER LAYERS, in frames' worth
(--budget model, the experiment; arms <model>_<protocol>_mp<k>: whole-model budget, schedule anchored
at what the cache holds). held_frames is what each arm
actually held by that measure -- it must sit at (or, early in an episode, under) k. The first
run bounded the last scheduled stage only (--budget last_stage, arms ..._b<k>, written to
results_last_stage/); its held_frames shows how far from k those arms really were.

Other memory columns: LongNav-R1 -- kv_len_mean (ALL resident KV slots, text included, layer
mean), kv_len_master (the longest layer), stage_visual_<s> (history visual slots on each stage's
layers), kv_text_slots; Uni-NaVid -- stage_kept_<s> and attended_tokens (history tokens a query
can attend to, layer mean; its layers 8+ hide key columns instead of deleting rows, so this is
attention width, not resident memory). vlm_latency_ms is reported but the sweep runs two arms
at a time on one GPU, so it is not comparable across arms.

Arms not run yet are reported and left out (--strict makes that an error); an incomplete arm is
an error unless --skip-partial (drop it) or --allow-partial (keep it, flagged partial=1) is
given. --status prints per-arm progress. --dataset mp3d aggregates the MP3D replicate
(paper_results/hybrid_prune_mp3d/, mp3d_100_seed17.json). Runs in longnav_vlm:

    python paper/hybrid_prune/aggregate.py [--dataset mp3d] [--allow-partial]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis.results_io import assert_split, load_manifest, load_r1_run, load_uninavid_run, split_status
from longnav.analysis.stats import bootstrap_mean_ci, paired_delta, stars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/hybrid_prune"
MANIFEST = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_100.json"
DATASETS = {  # dataset -> results root, manifest
    "hm3d": (RESULTS_ROOT, MANIFEST),
    "mp3d": (ROOT / "paper_results/hybrid_prune_mp3d", ROOT / "src/longnav/conf/episode_jsons/mp3d_100_seed17.json"),
}
MODELS = {"r1": "LongNav-R1", "uninavid": "Uni-NaVid"}
PROTOCOLS = ("pyramiddrop", "hybrid")
# budget scope -> arm tag, budgets, output dir, R1 layers per band (unpruned layers first)
BUDGETS = {
    "model": dict(tag="mp", k_list=(16, 8, 4, 2), out="results",
                  r1_bands={"pyramiddrop": (0, 7, 7, 7, 7), "hybrid": (0, 7, 7, 6, 8)}, uninavid_unpruned=0),
    "last_stage": dict(tag="b", k_list=(16, 8, 4, 1), out="results_last_stage",
                       r1_bands={"pyramiddrop": (7, 7, 7, 7), "hybrid": (7, 7, 6, 8)}, uninavid_unpruned=8),
}
BUDGET = BUDGETS["model"]
K_LIST = BUDGET["k_list"]
N_STAGES = 4
UNINAVID_LAYERS = 32
UNINAVID_TOKENS_PER_FRAME = 4

R1_MEMORY = {"kv_len_mean": "sup/mean_kv_len", "kv_len_max": "sup/max_kv_len",
             "kv_len_master": "sup/mean_kv_len_master", "kv_text_slots": "sup/mean_kv_text_slots",
             "n_per_frame": "sup/mean_vtp_n_new", "vlm_mem_GB_max": "sup/max_vlm_mem_GB",
             **{f"stage_visual_{s}": f"sup/mean_kv_stage_visual_{s}" for s in range(N_STAGES)}}
MEMORY_COLS = ["held_frames"] + list(R1_MEMORY) + [f"stage_kept_{s}" for s in range(N_STAGES)] \
    + ["attended_tokens", "vlm_latency_ms"]
EP_COLS = ["arm", "model", "protocol", "k", "episode_label", "success", "spl", "soft_spl", "distance_to_goal",
           "n_steps"] + MEMORY_COLS + ["partial"]
METRICS = ("success", "spl", "n_steps")


def arm_specs():
    """(arm, model, protocol, k) for every arm of the experiment."""
    for model in MODELS:
        for k in K_LIST:
            for protocol in PROTOCOLS:
                yield f"{model}_{protocol}_{BUDGET['tag']}{k}", model, protocol, k


def fnum(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def nanmean(values):
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def uninavid_stage_stats(run_dir: Path):
    """{label: {stage_kept_<s>, attended_tokens}}: per-episode means over inferences of the history
    tokens each stage's layers attend to, and their layer-mean (layers below the first stage see
    the whole history)."""
    out = {}
    for path in sorted((run_dir / "prune").glob("*.jsonl")):
        per_stage, attended, label = [], [], path.stem
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                label = rec.get("episode_label", label)
                kept = rec.get("stage_kept")
                if kept is None:
                    continue
                # Layers below the first stage (none under a whole-model budget) see every history row.
                unpruned = BUDGET["uninavid_unpruned"]
                width = (UNINAVID_LAYERS - unpruned) // len(kept)
                per_stage.append(kept)
                attended.append((unpruned * sum(rec["n_tokens"]) + width * sum(kept)) / UNINAVID_LAYERS)
        if per_stage:
            mean = np.mean(np.asarray(per_stage, dtype=float), axis=0)
            out[label] = {**{f"stage_kept_{s}": float(v) for s, v in enumerate(mean)},
                          "attended_tokens": float(np.mean(attended))}
    return out


def load_arms(runs_root: Path, labels, mode: str):
    """{arm: (rows, model, protocol, k, partial)}; mode in {strict-complete, skip, allow}."""
    runs, missing = {}, []
    for arm, model, protocol, k in arm_specs():
        run_dir = runs_root / arm
        if not run_dir.is_dir():
            missing.append(arm)
            continue
        rows = load_r1_run(run_dir) if model == "r1" else load_uninavid_run(run_dir)
        if model == "uninavid":
            for label, extra in uninavid_stage_stats(run_dir).items():
                if label in rows:
                    rows[label].update(extra)
        partial = bool(split_status(rows, labels)[0])
        if partial:
            if mode == "skip":
                print(f"[partial] {arm}: {len(rows)}/{len(labels)} episodes, skipped")
                missing.append(arm)
                continue
            if mode == "allow":
                print(f"[partial] {arm}: {len(rows)}/{len(labels)} episodes, kept (flagged)")
                rows = {l: rows[l] for l in labels if l in rows}
            else:
                assert_split(rows, labels, arm)  # a partial arm is an error, never a smaller denominator
        runs[arm] = (rows, model, protocol, k, partial)
    return runs, missing


def memory_fields(model, protocol, r):
    out = {c: "" for c in MEMORY_COLS}
    if model == "r1":
        for col, key in R1_MEMORY.items():
            out[col] = r.get(key, "")
        lat = r.get("sup/mean_vlm_latency")
        out["vlm_latency_ms"] = 1000 * lat if lat is not None else ""
        out["held_frames"] = r.get("sup/mean_kv_visual_model_frames", "")
        if out["held_frames"] == "":  # runs that predate the log: rebuild it from the per-stage means
            bands = BUDGET["r1_bands"][protocol]
            counts = [fnum(r.get("sup/mean_kv_len_master")) - fnum(r.get("sup/mean_kv_text_slots"))] \
                + [fnum(r.get(f"sup/mean_kv_stage_visual_{s}")) for s in range(len(bands) - 1)]
            held = sum(w * n for w, n in zip(bands, counts) if w) / sum(bands)
            out["held_frames"] = held / fnum(r.get("sup/mean_vtp_n_new"))
    else:
        for col in [f"stage_kept_{s}" for s in range(N_STAGES)] + ["attended_tokens"]:
            out[col] = r.get(col, "")
        if out["attended_tokens"] != "":
            out["held_frames"] = out["attended_tokens"] / UNINAVID_TOKENS_PER_FRAME
    return out


def write_csv(path: Path, rows, cols=None):
    rows = list(rows)
    cols = cols or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def status(runs_root: Path, labels):
    print(f"{'arm':<28}{'episodes':>10}")
    for arm, model, *_ in arm_specs():
        run_dir = runs_root / arm
        if not run_dir.is_dir():
            print(f"{arm:<28}{'-':>10}")
            continue
        rows = load_r1_run(run_dir) if model == "r1" else load_uninavid_run(run_dir)
        done = len(set(rows) & set(labels))
        print(f"{arm:<28}{done:>6}/{len(labels)}")


def fmt(value, digits=3):
    return "--" if not np.isfinite(fnum(value)) else f"{fnum(value):.{digits}f}"


def write_table(path: Path, summary, paired):
    """Rows = model x protocol, columns = k; stars on the hybrid row (vs pyramiddrop, same k)."""
    by_arm = {r["arm"]: r for r in summary}
    p_of = {(r["model"], r["k"], r["metric"]): r["p"] for r in paired}
    lines = [r"\begin{tabular}{ll" + "cc" * len(K_LIST) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in K_LIST) + r" \\",
             "Model & Protocol & " + " & ".join("SR & SPL" for _ in K_LIST) + r" \\", r"\midrule"]
    for model, name in MODELS.items():
        for protocol in PROTOCOLS:
            cells = []
            for k in K_LIST:
                rec = by_arm.get(f"{model}_{protocol}_{BUDGET['tag']}{k}")
                for metric, col in (("success", "sr"), ("spl", "spl")):
                    if rec is None:
                        cells.append("--")
                        continue
                    mark = stars(fnum(p_of.get((name, k, metric)))) if protocol == "hybrid" else ""
                    cells.append(f"{rec[col]:.2f}" + (f"$^{{{mark}}}$" if mark else ""))
            lines.append(f"{name if protocol == PROTOCOLS[0] else ''} & {protocol} & " + " & ".join(cells) + r" \\")
            mem = [fmt(by_arm.get(f"{model}_{protocol}_{BUDGET['tag']}{k}", {}).get("held_frames"), 1) for k in K_LIST]
            lines.append(" & \\quad held & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in mem) + r" \\")
        lines.append(r"\midrule" if model != list(MODELS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}",
              "% held: history visual tokens averaged over all decoder layers, in frames' worth (the budget is k). "
              "Stars: paired sign-flip test, hybrid vs pyramiddrop at the same k "
              "(* p<0.05, ** p<0.01, *** p<0.001)."]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", choices=list(BUDGETS), default="model",
                    help="which run to aggregate: the whole-model budget (default) or the first, last-stage one")
    ap.add_argument("--dataset", choices=list(DATASETS), default="hm3d",
                    help="hm3d (default) or the MP3D replicate in paper_results/hybrid_prune_mp3d/")
    ap.add_argument("--runs-root", type=Path, default=None, help="default: <results root>/runs")
    ap.add_argument("--out", type=Path, default=None, help="default: results/ or results_last_stage/")
    ap.add_argument("--manifest", type=Path, default=None, help="default: the dataset's 100-episode manifest")
    ap.add_argument("--status", action="store_true", help="print per-arm progress and exit")
    ap.add_argument("--strict", action="store_true", help="every arm must exist and be complete")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-partial", action="store_true", help="drop incomplete arms")
    group.add_argument("--allow-partial", action="store_true", help="keep incomplete arms, flagged partial=1")
    args = ap.parse_args()
    global BUDGET, K_LIST
    BUDGET = BUDGETS[args.budget]
    K_LIST = BUDGET["k_list"]
    results_root, manifest = DATASETS[args.dataset]
    args.runs_root = args.runs_root or results_root / "runs"
    args.manifest = args.manifest or manifest
    if args.out is None:
        args.out = results_root / BUDGET["out"]

    labels = load_manifest(args.manifest)
    if args.status:
        status(args.runs_root, labels)
        return
    mode = "skip" if args.skip_partial else "allow" if args.allow_partial else "strict-complete"
    runs, missing = load_arms(args.runs_root, labels, mode)
    if missing:
        print(f"[missing] {len(missing)} arms left out: {' '.join(missing)}")
        if args.strict:
            sys.exit(1)
    if not runs:
        sys.exit("no arm to aggregate")
    args.out.mkdir(parents=True, exist_ok=True)

    episodes, summary = [], []
    for arm, (rows, model, protocol, k, partial) in runs.items():
        for label in labels:
            if label not in rows:
                continue
            r = rows[label]
            episodes.append({"arm": arm, "model": MODELS[model], "protocol": protocol, "k": k,
                             "episode_label": label, "success": r["success"], "spl": r["spl"],
                             "soft_spl": r.get("soft_spl", ""), "distance_to_goal": r.get("distance_to_goal", ""),
                             "n_steps": r["n_steps"], **memory_fields(model, protocol, r), "partial": int(partial)})
        rec = {"arm": arm, "model": MODELS[model], "protocol": protocol, "k": k, "n": len(rows), "partial": int(partial)}
        for metric, col in (("success", "sr"), ("spl", "spl"), ("n_steps", "steps")):
            m, lo, hi = bootstrap_mean_ci([fnum(r.get(metric)) for r in rows.values()])
            rec |= {col: m, f"{col}_lo": lo, f"{col}_hi": hi}
        mem = [memory_fields(model, protocol, r) for r in rows.values()]
        rec |= {c: nanmean(m[c] for m in mem) for c in MEMORY_COLS}
        summary.append(rec)

    paired = []
    for model, name in MODELS.items():
        for k in K_LIST:
            a, b = (runs.get(f"{model}_{p}_{BUDGET['tag']}{k}") for p in ("hybrid", "pyramiddrop"))
            if a is None or b is None:
                continue
            for metric in METRICS:
                d = paired_delta(a[0], b[0], metric)  # hybrid - pyramiddrop on the shared episodes
                paired.append({"model": name, "k": k, "metric": metric, **d, "partial": int(a[4] or b[4])})

    write_csv(args.out / "per_episode.csv", episodes, EP_COLS)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "paired.csv", paired, ["model", "k", "metric", "delta", "lo", "hi", "p", "n", "partial"])
    write_table(args.out / "table.tex", summary, paired)

    print(f"\n{'arm':<26}{'n':>4}{'SR':>7}{'SPL':>8}{'steps':>8}{'held (frames)':>15}{'kv slots':>10}")
    for rec in summary:
        print(f"{rec['arm']:<26}{rec['n']:>4}{rec['sr']:>7.2f}{rec['spl']:>8.3f}{rec['steps']:>8.1f}"
              f"{fmt(rec['held_frames'], 2):>15}{fmt(rec['kv_len_mean'], 0):>10}")
    print("\nhybrid - pyramiddrop (paired):")
    for r in paired:
        if r["metric"] != "n_steps":
            print(f"  {r['model']:<11} k={r['k']:<3}{r['metric']:<8} {r['delta']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.4f}  n={r['n']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
