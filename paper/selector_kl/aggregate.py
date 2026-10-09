#!/usr/bin/env python3
"""Aggregate the selector-criterion vs KL probe into paper_results/selector_kl/results/.

Reads both arms -- runs/r1_selector_probe (LongNav-R1: rollout/*/*/sequence.json, the sup/selector_*
rows) and runs/uninavid_selector_probe (Uni-NaVid: selector/<label>.jsonl) -- and writes (a `model`
column everywhere; the budget statistics exist for LongNav-R1 only):
  per_token.parquet   one row per (episode, step, token): raster index, keep_all (no-history keep-one-in KL), attn,
                      positions and keep ranks of every method -- the tracked copy of the
                      per-step data whose raw files are gitignored
  per_decision.csv    one row per (episode, step, measure): Spearman rho of each criterion
                      with the measure, and at every binding budget the KL-mass share kept /
                      token share kept / lift per method
  summary.csv         pooled statistics: per (method, measure) the rho distribution; per
                      (method, measure, k) mass share, token share, lift, each with an
                      episode-cluster bootstrap CI, the balanced mean and the delta vs random
  table.tex           booktabs table (keep-one-in only: mean rho per model, mean lift at every k; no CIs)
  showcase.json       the decision the figure draws: every budget binding, largest keep-one-in mass
  summary.json        completeness, trajectory checks (R1 vs ctx_ablation/r1_full -- differs, filter off;
                      Uni-NaVid vs info_horizon/runs/uninavid_probe -- must match, same seed), consistency
Runs in longnav_vlm:  python paper/selector_kl/aggregate.py [--allow-partial] [--showcase LABEL:STEP]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis import selector as sel
from longnav.analysis.results_io import load_manifest, load_run, split_status

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/selector_kl"
MANIFEST = ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_10_short.json"
# model -> (arm dir, loader, trajectory reference run, must the trajectory match the reference?)
ARMS = {"LongNav-R1": ("r1_selector_probe", sel.load_r1_selector, ROOT / "paper_results/ctx_ablation/runs/r1_full", False),
        "Uni-NaVid": ("uninavid_selector_probe", sel.load_uninavid_selector,
                      ROOT / "paper_results/info_horizon/runs/uninavid_probe", True)}
TOKEN_COLS = ["model", "episode", "step", "n_frames", "token", "raster", "keep_all", "keep_cur", "leave", "attn", "divprune_nn",
              "dart_dis", "attn_pos", "divprune_pos", "random_pos", "attn_rank", "divprune_rank", "dart_rank", "random_rank"]


def write_csv(path: Path, rows: list[dict], fields=None):
    if not rows:
        return
    if fields is None:  # ordered union: rows of different models carry different columns
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_parquet(path: Path, rows, cols):
    try:
        import pandas as pd
    except ImportError:
        print(f"[warn] pandas unavailable, {path.name} not written")
        return
    pd.DataFrame(rows, columns=cols).to_parquet(path, index=False)


def fmt(v, nd=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def token_rows(recs):
    for rec in recs:
        for i in range(rec["n_cur"]):
            yield {"model": rec["model"], "episode": rec["episode"], "step": rec["step"], "n_frames": rec["n_frames"], "token": i,
                   **{k: rec[k][i].item() for k in ("raster", "attn", "divprune_nn", "dart_dis", "attn_pos", "divprune_pos",
                                                    "random_pos", "attn_rank", "divprune_rank", "dart_rank", "random_rank")},
                   **{m: (rec[m][i].item() if m in rec else float("nan")) for m in ("keep_all", "keep_cur", "leave")}}


def summary_rows(dec_rows, budgets, model):
    out = []
    for measure in sel.MEASURES:
        rows = [r for r in dec_rows if r["model"] == model and r["measure"] == measure and r["valid"]]
        for m in sel.METHODS:
            kinds = [("rho", f"rho_{m}"), ("rho_rank", f"rho_rank_{m}")] + ([("rho_order", "rho_divprune_order")] if m == "divprune" else [])
            for kind, key in kinds:
                s = sel.summarise(rows, key)
                d = sel.paired_vs_random(rows, key, key.replace(m, "random")) if m != "random" else {}
                out.append({"model": model, "measure": measure, "method": m, "label": sel.LABELS[m], "stat": kind, "k": "", **s, **d})
            for k in budgets:
                for kind in ("mass", "tokshare", "lift"):
                    key = f"{kind}_{m}_k{k}"
                    s = sel.summarise(rows, key)
                    d = sel.paired_vs_random(rows, key, f"{kind}_random_k{k}") if m != "random" else {}
                    out.append({"model": model, "measure": measure, "method": m, "label": sel.LABELS[m], "stat": kind, "k": k, **s, **d})
    return out


def table_tex(summary, budgets, models):
    def get(model, measure, m, stat, k=""):
        for r in summary:
            if (r["model"], r["measure"], r["method"], r["stat"], r["k"]) == (model, measure, m, stat, k):
                return r
        return {}
    cols = "l" + "c" * (len(models) + len(budgets))
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "method & " + " & ".join(f"$\\rho$ vs keep-one-in KL, {model}" for model in models) +
             ("".join(f" & KL share / token share, $k{{=}}{k}$ (LongNav-R1)" for k in budgets)) + r" \\", r"\midrule"]
    for m in sel.METHODS:
        cells = [sel.LABELS[m]]
        for model in models:
            r = get(model, sel.MEASURE, m, "rho")
            cells.append(fmt(r.get('mean')))
        for k in budgets:
            r = get("LongNav-R1", sel.MEASURE, m, "lift", k)
            cells.append(fmt(r.get('mean'), 2))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def pick_showcase(recs, override):
    if override:
        label, step = override.rsplit(":", 1)
        for rec in recs:
            if rec["episode"] == label and rec["step"] == int(step):
                return rec, "override"
        raise SystemExit(f"--showcase {override}: no such probed decision")
    k_max = max(recs[0]["budget_frames"])
    pool = [r for r in recs if r["n_frames"] > k_max and r["n_cur"] >= sel.MIN_TOKENS and sel.MEASURE in r]
    if not pool:
        return None, "no decision with every budget binding"
    best = max(pool, key=lambda r: float(r[sel.MEASURE].sum()))
    return best, f"n_frames > {k_max} and max total {sel.MEASURE} (the frame's keep-one-in mass)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RESULTS_ROOT / "runs")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "results")
    ap.add_argument("--allow-partial", action="store_true",
                    help="aggregate incomplete or missing arms (interim looks)")
    ap.add_argument("--showcase", help="LABEL:STEP (0-based step) to override the showcase rule (LongNav-R1)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = load_manifest(MANIFEST)
    summary: dict = {"arms": {}, "checks": {}}
    all_recs: dict[str, list[dict]] = {}
    for model, (arm, loader, reference, must_match) in ARMS.items():
        run_dir = args.runs / arm
        run_rows = load_run(run_dir) if run_dir.is_dir() else {}
        missing, extra = split_status(run_rows, labels)
        summary["arms"][model] = {"arm": arm, "episodes_done": len(run_rows), "missing": missing, "extra": extra}
        if (missing or not run_dir.is_dir()) and not args.allow_partial:
            sys.exit(f"{arm}: {len(missing)} of {len(labels)} episodes missing ({missing}); "
                     "pass --allow-partial for an interim look")
        checks: dict = {}
        if reference.is_dir() and run_rows:
            ref = load_run(reference)
            mism = [f"{l}: success={run_rows[l]['success']} steps={run_rows[l]['n_steps']} vs reference "
                    f"success={ref[l]['success']} steps={ref[l]['n_steps']}"
                    for l in labels if l in run_rows and l in ref
                    and (run_rows[l]["success"], run_rows[l]["n_steps"]) != (ref[l]["success"], ref[l]["n_steps"])]
            checks[f"trajectory_vs_{reference.name}"] = mism or "match"
            if mism and must_match:
                print(f"[warn] {arm}: {len(mism)} trajectories differ from {reference.name}, which shares its seed")
        recs = loader(run_dir) if run_dir.is_dir() else []
        if not recs:
            summary["checks"][model] = checks
            continue
        bad = [m for rec in recs for m in sel.consistency(rec)]
        checks["consistency"] = bad[:20] or "ok"
        checks["n_bad"] = len(bad)
        budgets = recs[0]["budget_frames"]
        checks["decisions"] = len(recs)
        checks["decisions_valid"] = sum(1 for r in recs if r["n_cur"] >= sel.MIN_TOKENS and float(r[sel.MEASURE].sum()) > 0)
        if budgets:
            checks["decisions_per_budget"] = {k: sum(1 for r in recs if k in sel.binding_budgets(r)) for k in budgets}
        summary["checks"][model] = checks
        all_recs[model] = recs
    if not all_recs:
        sys.exit("no selector rows in any arm")

    dec_rows, token_rows_all, srows = [], [], []
    for model, recs in all_recs.items():
        measures = [m for m in ("keep_all", "keep_cur", "leave") if m in recs[0]]
        rows = [sel.decision_stats(rec, m) for rec in recs for m in measures if m in rec]
        dec_rows += rows
        token_rows_all += list(token_rows(recs))
        srows += summary_rows(rows, recs[0]["budget_frames"], model)
    write_csv(args.out / "per_decision.csv", dec_rows)
    write_parquet(args.out / "per_token.parquet", token_rows_all, TOKEN_COLS)
    write_csv(args.out / "summary.csv", srows)
    r1_budgets = all_recs["LongNav-R1"][0]["budget_frames"] if "LongNav-R1" in all_recs else []
    (args.out / "table.tex").write_text(table_tex(srows, r1_budgets, tuple(all_recs)))

    rule = "no LongNav-R1 records"
    if "LongNav-R1" in all_recs:
        best, rule = pick_showcase(all_recs["LongNav-R1"], args.showcase)
        if best is not None:
            (args.out / "showcase.json").write_text(json.dumps({
                "episode_label": best["episode"], "step": best["step"], "n_frames": best["n_frames"],
                "n_cur": best["n_cur"], "total_keep_one": float(best[sel.MEASURE].sum()),
                "text_all_kl": best["text_all_kl"], "action": best["action"], "rule": rule}, indent=1))
    summary["showcase"] = rule
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    for model, recs in all_recs.items():
        c = summary["checks"][model]
        print(f"{ARMS[model][0]}: {summary['arms'][model]['episodes_done']} episodes, {c['decisions']} decisions "
              f"({c['decisions_valid']} valid); consistency issues: {c['n_bad']}")
        for r in srows:
            if r["model"] == model and r["stat"] == "rho" and r["measure"] == sel.MEASURE:
                print(f"  rho({sel.MEASURE}) {r['label']:9s} mean {fmt(r['mean'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] "
                      f"balanced {fmt(r['balanced_mean'])} n={r['n_decisions']}")
    for r in srows:
        if r["model"] == "LongNav-R1" and r["stat"] == "lift" and r["measure"] == sel.MEASURE:
            print(f"  lift({sel.MEASURE}) {r['label']:9s} k={r['k']:<3} {fmt(r['mean'], 2)} [{fmt(r['lo'], 2)}, {fmt(r['hi'], 2)}] n={r['n_decisions']}")
    print(f"showcase: {rule}")


if __name__ == "__main__":
    main()
