#!/usr/bin/env python3
"""Aggregate the visual-token pruning experiment into paper_results/token_prune/results/.

Reads the LongNav-R1 arms from runs/<arm>/rollout/results_* (+ the per-step keep masks in
rollout/<scene>/<label>.*/sequence.json, keys sup/vtp_*) and the Uni-NaVid arms from
runs/<arm>/episodes/*.json (+ prune/<label>.jsonl), merges the recency and 32-frame reference
rows of the context-window ablation (paper_results/ctx_ablation/results/per_episode.csv), and
writes:

  results/per_episode.{csv,parquet}  one row per (arm, episode): success, spl, n_steps, kv/latency,
                                     tokens per history frame and mean tokens kept
  results/summary.csv                one row per arm: means with 95% bootstrap CIs, paired sign-flip
                                     deltas/p vs the same-k recency arm and vs the k32 reference
  results/table.tex                  booktabs table (stars = significance vs the same-k recency arm)
  results/n_tokens.csv               tokens per history frame and mean tokens kept, per arm
  results/masks.parquet              one row per (arm, episode, step, history frame): tokens, kept,
                                     packed keep bits -- joins Exp 3 on (model, episode, step, frame)
  results/retention.csv              fraction of a frame's tokens retained vs its age, per arm

Budgets are "k frames' worth" of the model's own history tokens; the ctx-ablation recency arm
K holds K-1 history frames (K counts the current frame), so recency rows carry
frames_worth = K-1 and pruning rows frames_worth = k. Arms not run yet are reported and left out
(--strict makes that an error); an arm that exists but is incomplete is an error unless
--skip-partial (drop it) or --allow-partial (keep it, flagged partial=1 with its n) is given.
--status prints per-arm progress; --check-identity compares the identity arms with the
ctx-ablation references and exits 1 on a mismatch. Runs in longnav_vlm:

    python paper/token_prune/aggregate.py [--allow-partial]

--variant flat aggregates the flat variant (README "Flat variant": full context for LongNav-R1,
64 frames for Uni-NaVid, every method at layer 0) from paper_results/token_prune_flat/: methods
without pyramiddrop, k in {32,16,8,4}, no recency / reference rows; the paired deltas are
against the same-k random arm (`*_vs_random`) and the family's identity arm (`*_vs_ref`), and
--check-identity compares r1_identity with the ctx ablation's r1_full.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

from longnav.analysis.results_io import assert_split, load_manifest, load_r1_run, load_uninavid_run, split_status
from longnav.analysis.stats import bootstrap_mean_ci, paired_delta, stars

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CTX_PER_EPISODE = ROOT / "paper_results/ctx_ablation/results/per_episode.csv"
MODELS = {"r1": "LongNav-R1", "uninavid": "Uni-NaVid"}
# window: history frames the arm can hold (None = the whole episode); baseline: the tag of the
# same-k comparison arm the table's stars refer to; identity: (arm, ctx-ablation arm, exact).
VARIANTS = {
    "window": dict(root=ROOT / "paper_results/token_prune",
                   methods=("random", "fastv", "pyramiddrop", "divprune", "dart"), k_list=(16, 8, 4, 1),
                   window={"r1": 31, "uninavid": 31}, merge_ctx=True, baseline="recency",
                   identity=(("r1_identity", "r1_evict_k32", True), ("uninavid_identity", "uninavid_k32", False))),
    "flat": dict(root=ROOT / "paper_results/token_prune_flat",
                 methods=("random", "fastv", "divprune", "dart"), k_list=(32, 16, 8, 4),
                 window={"r1": None, "uninavid": 63}, merge_ctx=False, baseline="random",
                 identity=(("r1_identity", "r1_full", True),)),
}
V = VARIANTS["window"]  # set from --variant in main()
METRICS = (("success", "sr"), ("spl", "spl"), ("n_steps", "steps"))
EP_COLS = ["arm", "model", "method", "k", "frames_worth", "episode_label", "success", "spl", "soft_spl",
           "distance_to_goal", "n_steps", "kv_len_mean", "kv_len_max", "vlm_latency_ms", "n_per_frame",
           "tokens_kept_mean", "partial"]
MASK_COLS = ["arm", "model", "method", "k", "episode", "step", "frame", "age", "n_tokens", "n_kept", "keep_bits"]


def arm_specs():
    """(arm, model key, method, k) for every arm this experiment runs. Identity has k=None."""
    for m in MODELS:
        yield f"{m}_identity", m, "identity", None
        for k in V["k_list"]:
            for method in V["methods"]:
                yield f"{m}_{method}_b{k}", m, method, k


def ctx_specs():
    """(ctx arm, model key, method, k, frames_worth) of the ctx-ablation rows merged in."""
    if not V["merge_ctx"]:
        return
    for m, tpl in (("r1", "r1_evict_k{}"), ("uninavid", "uninavid_k{}")):
        yield tpl.format(32), m, "reference", None, V["window"][m]
        for k in V["k_list"]:
            yield tpl.format(k), m, "recency", k, k - 1


def frames_worth(model, k):
    """x position of an arm; an unpruned arm sits at its window (inf = the whole episode)."""
    if k is not None:
        return k
    return float("inf") if V["window"][model] is None else V["window"][model]


def partner_arm(model, k, tag):
    """The arm a paired delta of `tag` is taken against (None = not applicable)."""
    ctx = "r1_evict_k{}" if model == "r1" else "uninavid_k{}"
    if tag == "recency":
        return ctx.format(k) if k is not None else None
    if tag == "random":
        return f"{model}_random_b{k}" if k is not None else None
    return ctx.format(32) if V["merge_ctx"] else f"{model}_identity"


def fnum(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def nanmean(values):
    v = np.array([fnum(x) for x in values])
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def nanmax(values):
    v = np.array([fnum(x) for x in values])
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


# --- keep-mask logs ----------------------------------------------------------------------
def unpack_bits(b64: str, counts) -> list[bytes]:
    """Split one base64 packbits blob (frames concatenated in order) into per-frame packed bytes."""
    total = int(sum(counts))
    if total == 0:
        return [b"" for _ in counts]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), dtype=np.uint8))[:total]
    out, pos = [], 0
    for n in counts:
        out.append(np.packbits(bits[pos:pos + int(n)]).tobytes())
        pos += int(n)
    return out


def mask_rows_r1(run_dir: Path, arm: str, method: str, k):
    """Rows from sequence.json: the mask logged after env step t is in effect for the decision at
    t+1, so step = t+1; the entry after the final decision is dropped."""
    for seq in sorted(run_dir.glob("rollout/*/*/sequence.json")):
        label = seq.parent.name.split(".")[0]
        d = json.loads(seq.read_text())
        frames, ntok, kept, bits = (d.get(f"sup/vtp_{key}") for key in ("frames", "ntok", "kept", "bits"))
        if not frames:
            continue
        n_dec = len(d.get("action_history", frames))
        for t in range(min(len(frames), n_dec - 1)):
            step = t + 1
            per_frame = unpack_bits(bits[t], ntok[t])
            for f, n, kp, kb in zip(frames[t], ntok[t], kept[t], per_frame):
                yield {"arm": arm, "model": MODELS["r1"], "method": method, "k": k, "episode": label,
                       "step": step, "frame": int(f), "age": step - int(f), "n_tokens": int(n),
                       "n_kept": int(kp), "keep_bits": kb}


def mask_rows_uninavid(run_dir: Path, arm: str, method: str, k):
    for path in sorted(run_dir.glob("prune/*.jsonl")):
        label = path.stem
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("frames"):
                    continue
                step = int(rec["step"])
                per_frame = unpack_bits(rec["bits"], rec["n_tokens"])
                for fr, n, kp, kb in zip(rec["frames"], rec["n_tokens"], rec["n_kept"], per_frame):
                    yield {"arm": arm, "model": MODELS["uninavid"], "method": method, "k": k, "episode": label,
                           "step": step, "frame": int(fr), "age": step - int(fr), "n_tokens": int(n),
                           "n_kept": int(kp), "keep_bits": kb}


def per_episode_token_stats(mask_rows):
    """{episode: (n_per_frame, tokens_kept_mean)}: mean original tokens per history frame and the
    mean over steps of the total tokens kept in the window."""
    by_ep = {}
    for r in mask_rows:
        by_ep.setdefault(r["episode"], {}).setdefault(r["step"], []).append((r["n_tokens"], r["n_kept"]))
    out = {}
    for ep, steps in by_ep.items():
        n_tok = [n for s in steps.values() for n, _ in s]
        kept = [sum(kp for _, kp in s) for s in steps.values()]
        out[ep] = (float(np.mean(n_tok)) if n_tok else float("nan"), float(np.mean(kept)) if kept else float("nan"))
    return out


def retention_rows(mask_rows, arm, model, method, k, labels):
    """Per age: mean over episodes of the episode's mean kept fraction at that age, bootstrap CI."""
    frac = {}  # age -> {episode: [fractions]}
    n_steps = {}
    for r in mask_rows:
        if r["n_tokens"] <= 0:
            continue
        frac.setdefault(r["age"], {}).setdefault(r["episode"], []).append(r["n_kept"] / r["n_tokens"])
        n_steps.setdefault(r["age"], set()).add((r["episode"], r["step"]))
    for age in sorted(frac):
        per_ep = [float(np.mean(v)) for ep, v in frac[age].items()]
        m, lo, hi = bootstrap_mean_ci(per_ep)
        yield {"arm": arm, "model": model, "method": method, "k": "" if k is None else k, "age": age,
               "frac_mean": m, "lo": lo, "hi": hi, "n_episodes": len(per_ep), "n_steps": len(n_steps[age])}


# --- loading ------------------------------------------------------------------------------
def load_ctx_rows(path: Path):
    """{ctx arm: {label: row}} from the ctx-ablation per-episode table (float fields)."""
    if not path.is_file():
        return {}
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for key in ("success", "spl", "soft_spl", "distance_to_goal", "n_steps", "kv_len_mean", "kv_len_max", "vlm_latency_ms"):
                row[key] = fnum(r.get(key))
            out.setdefault(r["arm"], {})[r["episode_label"]] = row
    return out


def load_arms(runs_root: Path, labels, mode: str):
    """{arm: (rows, model, method, k, partial)}; mode in {strict-complete, skip, allow}."""
    runs, missing = {}, []
    for arm, model, method, k in arm_specs():
        run_dir = runs_root / arm
        if not run_dir.is_dir():
            missing.append(arm)
            continue
        rows = load_r1_run(run_dir) if model == "r1" else load_uninavid_run(run_dir)
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
        runs[arm] = (rows, model, method, k, partial)
    return runs, missing


def episode_row(arm, model, method, k, label, r, tok, partial):
    lat = r.get("sup/mean_vlm_latency")
    n_pf, kept = tok.get(label, (np.nan, np.nan))
    return {"arm": arm, "model": MODELS[model], "method": method, "k": "" if k is None else k,
            "frames_worth": frames_worth(model, k), "episode_label": label,
            "success": r["success"], "spl": r["spl"], "soft_spl": r.get("soft_spl", ""),
            "distance_to_goal": r.get("distance_to_goal", ""), "n_steps": r["n_steps"],
            "kv_len_mean": r.get("sup/mean_kv_len", ""), "kv_len_max": r.get("sup/max_kv_len", ""),
            "vlm_latency_ms": 1000 * lat if lat is not None else "",
            "n_per_frame": r.get("sup/mean_vtp_n_new", n_pf if np.isfinite(n_pf) else ""),
            "tokens_kept_mean": kept if np.isfinite(kept) else "", "partial": int(partial)}


def ctx_episode_row(arm, model, method, k, fw, label, r):
    return {"arm": arm, "model": MODELS[model], "method": method, "k": "" if k is None else k, "frames_worth": fw,
            "episode_label": label, "success": r["success"], "spl": r["spl"], "soft_spl": r.get("soft_spl", ""),
            "distance_to_goal": r.get("distance_to_goal", ""), "n_steps": r["n_steps"],
            "kv_len_mean": r.get("kv_len_mean", ""), "kv_len_max": r.get("kv_len_max", ""),
            "vlm_latency_ms": r.get("vlm_latency_ms", ""), "n_per_frame": "", "tokens_kept_mean": "", "partial": 0}


def write_csv(path: Path, rows, cols=None):
    rows = list(rows)
    cols = cols or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_parquet(path: Path, rows, cols):
    try:
        import pandas as pd
    except ImportError:
        print(f"[warn] pandas/pyarrow unavailable, {path.name} not written")
        return
    df = pd.DataFrame(list(rows), columns=cols)
    for c in df.columns:  # csv-style blanks become nulls; numeric-looking object columns become numbers
        if df[c].dtype == object and c != "keep_bits":
            col = df[c].replace("", None)
            try:
                df[c] = pd.to_numeric(col)
            except (ValueError, TypeError):
                df[c] = col
    df.to_parquet(path, index=False)


# --- modes --------------------------------------------------------------------------------
def status(runs_root: Path, labels):
    print(f"{'arm':<26}{'episodes':>10}")
    for arm, *_ in arm_specs():
        run_dir = runs_root / arm
        if not run_dir.is_dir():
            print(f"{arm:<26}{'-':>10}")
            continue
        rows = load_r1_run(run_dir) if arm.startswith("r1_") else load_uninavid_run(run_dir)
        n = len([l for l in labels if l in rows])
        print(f"{arm:<26}{n:>6}/{len(labels)}" + ("" if n == len(labels) else "  (partial)"))


def check_identity(runs, ctx, labels, tolerance: bool) -> bool:
    """Identity arms must reproduce the ctx-ablation K=32 arms episode by episode."""
    ok = True
    for arm, ref, exact in V["identity"]:
        if arm not in runs:
            print(f"[identity] {arm}: not run yet")
            continue
        if ref not in ctx:
            print(f"[identity] {arm}: reference {ref} missing from {CTX_PER_EPISODE}")
            ok = False
            continue
        rows = runs[arm][0]
        common = [l for l in labels if l in rows and l in ctx[ref]]
        mism = {key: [l for l in common if not np.isclose(fnum(rows[l][key]), ctx[ref][l][key], atol=1e-6)]
                for key, _ in METRICS}
        n_bad = len(set().union(*mism.values()))
        d_sr = paired_delta(rows, ctx[ref], "success", common)
        d_spl = paired_delta(rows, ctx[ref], "spl", common)
        print(f"[identity] {arm} vs {ref}: {len(common)} episodes, {n_bad} mismatching "
              f"(success {len(mism['success'])}, spl {len(mism['spl'])}, n_steps {len(mism['n_steps'])}); "
              f"dSR={d_sr['delta']:+.3f} (p={d_sr['p']:.3f}) dSPL={d_spl['delta']:+.3f} (p={d_spl['p']:.3f})")
        if n_bad == 0:
            continue
        if not exact and tolerance and abs(d_sr["delta"]) <= 0.04 and d_sr["p"] > 0.05 and d_spl["p"] > 0.05:
            print(f"[identity] {arm}: within Uni-NaVid's documented sampling noise, accepted (--identity-tolerance)")
            continue
        for key, bad in mism.items():
            for l in bad[:5]:
                print(f"    {l}: {key} {rows[l][key]} vs {ctx[ref][l][key]}")
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="window",
                    help="window = Exp 4 as published; flat = full context / layer 0 (token_prune_flat)")
    ap.add_argument("--root", type=Path, default=None, help="experiment root holding runs/ results/ "
                    "(default: the variant's paper_results directory)")
    ap.add_argument("--ctx-per-episode", type=Path, default=CTX_PER_EPISODE,
                    help="ctx-ablation per-episode table supplying the recency and reference rows")
    ap.add_argument("--manifest", type=Path, default=ROOT / "src/longnav/conf/episode_jsons/hm3d_v2_100.json")
    ap.add_argument("--out", type=Path, default=None, help="results dir (default <root>/results)")
    ap.add_argument("--strict", action="store_true", help="fail if any arm is missing")
    ap.add_argument("--skip-partial", action="store_true", help="leave out arms that are still running")
    ap.add_argument("--allow-partial", action="store_true", help="keep arms that are still running, flagged partial=1")
    ap.add_argument("--status", action="store_true", help="print per-arm progress and exit")
    ap.add_argument("--check-identity", action="store_true",
                    help="compare the identity arms with the ctx-ablation K=32 arms and exit 1 on a mismatch")
    ap.add_argument("--identity-tolerance", action="store_true",
                    help="with --check-identity: accept Uni-NaVid differences within its +-0.04 SR sampling noise")
    args = ap.parse_args()
    global V
    V = VARIANTS[args.variant]
    args.root = args.root or V["root"]
    runs_root = args.root / "runs"
    out = args.out or args.root / "results"
    labels = load_manifest(args.manifest)

    if args.status:
        status(runs_root, labels)
        return

    mode = "allow" if (args.allow_partial or args.check_identity) else "skip" if args.skip_partial else "strict"
    runs, missing = load_arms(runs_root, labels, mode)
    ctx = load_ctx_rows(args.ctx_per_episode)
    if not ctx and V["merge_ctx"]:
        print(f"[warn] no ctx-ablation rows at {args.ctx_per_episode}; recency/reference rows left out")

    if args.check_identity:
        sys.exit(0 if check_identity(runs, ctx, labels, args.identity_tolerance) else 1)

    if missing:
        print(f"[missing] {len(missing)} arm(s) not run yet: {' '.join(missing)}")
        if args.strict:
            sys.exit(1)
    if not runs:
        sys.exit("no arms found")
    out.mkdir(parents=True, exist_ok=True)

    # --- keep masks, retention, per-episode token stats ---------------------------------
    masks, retention, tok_stats = [], [], {}
    for arm, (rows, model, method, k, partial) in runs.items():
        gen = mask_rows_r1 if model == "r1" else mask_rows_uninavid
        arm_masks = [r for r in gen(runs_root / arm, arm, method, k) if r["episode"] in rows]
        tok_stats[arm] = per_episode_token_stats(arm_masks)
        retention.extend(retention_rows(arm_masks, arm, MODELS[model], method, k, labels))
        masks.extend(arm_masks)
        if not arm_masks:
            print(f"[warn] {arm}: no keep-mask logs found")
    write_parquet(out / "masks.parquet", masks, MASK_COLS)
    write_csv(out / "retention.csv", retention,
              ["arm", "model", "method", "k", "age", "frac_mean", "lo", "hi", "n_episodes", "n_steps"])

    # --- per-episode ---------------------------------------------------------------------
    episodes = []
    meta = {}  # arm -> (model, method, k, frames_worth, partial)
    for arm, (rows, model, method, k, partial) in runs.items():
        meta[arm] = (model, method, k, frames_worth(model, k), partial)
        episodes += [episode_row(arm, model, method, k, l, rows[l], tok_stats[arm], partial) for l in labels if l in rows]
    ctx_rows = {}
    for arm, model, method, k, fw in ctx_specs():
        if arm in ctx:
            ctx_rows[arm] = ctx[arm]
            meta[arm] = (model, method, k, fw, False)
            episodes += [ctx_episode_row(arm, model, method, k, fw, l, ctx[arm][l]) for l in labels if l in ctx[arm]]
    write_csv(out / "per_episode.csv", episodes, EP_COLS)
    write_parquet(out / "per_episode.parquet", episodes, EP_COLS)

    # --- per-arm summary -----------------------------------------------------------------
    all_rows = {arm: v[0] for arm, v in runs.items()} | ctx_rows
    summary = []
    for arm, rows in all_rows.items():
        model, method, k, fw, partial = meta[arm]
        rec = {"arm": arm, "model": MODELS[model], "method": method, "k": "" if k is None else k,
               "frames_worth": fw, "n": len(rows), "partial": int(partial)}
        for key, name in METRICS:
            m, lo, hi = bootstrap_mean_ci([r[key] for r in rows.values()])
            rec.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
        if arm in runs:
            rec["kv_len_mean"] = nanmean(r.get("sup/mean_kv_len") for r in rows.values())
            rec["kv_len_max"] = nanmax(r.get("sup/max_kv_len") for r in rows.values())
            rec["vlm_latency_ms"] = 1000 * nanmean(r.get("sup/mean_vlm_latency") for r in rows.values())
            rec["n_per_frame"] = nanmean([r.get("sup/mean_vtp_n_new", tok_stats[arm].get(l, (np.nan,))[0])
                                          for l, r in rows.items()])
            rec["tokens_kept_mean"] = nanmean(v[1] for v in tok_stats[arm].values())
        else:
            rec["kv_len_mean"] = nanmean(r.get("kv_len_mean") for r in rows.values())
            rec["kv_len_max"] = nanmax(r.get("kv_len_max") for r in rows.values())
            rec["vlm_latency_ms"] = nanmean(r.get("vlm_latency_ms") for r in rows.values())
            rec["n_per_frame"] = rec["tokens_kept_mean"] = float("nan")
        window = V["window"][model]
        rec["tokens_kept_frac"] = (rec["tokens_kept_mean"] / (window * rec["n_per_frame"])
                                   if window and np.isfinite(rec["tokens_kept_mean"]) and rec["n_per_frame"] > 0
                                   else float("nan"))
        # Paired deltas vs the same-k baseline arm (recency window / random) and vs the unpruned
        # reference (labels both hold).
        for tag in (V["baseline"], "ref"):
            partner = partner_arm(model, k, tag)
            other = all_rows.get(partner) if partner and partner != arm else None
            common = [l for l in labels if other is not None and l in rows and l in other]
            for key, name in METRICS:
                if other is None or not common:
                    rec.update({f"d{name}_vs_{tag}": "", f"p_{name}_vs_{tag}": ""})
                    continue
                d = paired_delta(rows, other, key, common)
                rec.update({f"d{name}_vs_{tag}": d["delta"], f"p_{name}_vs_{tag}": d["p"]})
        summary.append(rec)
    summary.sort(key=lambda r: (r["model"], -r["frames_worth"], r["method"]))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "n_tokens.csv", [{k: r[k] for k in ("arm", "model", "method", "k", "frames_worth", "n",
                                                          "n_per_frame", "tokens_kept_mean", "tokens_kept_frac",
                                                          "kv_len_mean", "vlm_latency_ms")} for r in summary])

    # --- LaTeX table ---------------------------------------------------------------------
    by_arm = {r["arm"]: r for r in summary}

    def cell(arm, name, fmt):
        r = by_arm.get(arm)
        if r is None:
            return "--"
        p = r.get(f"p_{name}_vs_{V['baseline']}", "")
        s = stars(float(p)) if p != "" else ""
        return f"{r[name]:{fmt}}" + (f"$^{{{s}}}$" if s else "") + (r"$^\dagger$" if r["partial"] else "")

    K_LIST, METHODS, flat = V["k_list"], V["methods"], not V["merge_ctx"]
    lines = [r"% Generated by paper/token_prune/aggregate.py -- do not edit by hand.",
             r"\begin{tabular}{l l " + "cc " * len(K_LIST) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in K_LIST) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(K_LIST))),
             "model & method & " + " & ".join("SR & SPL" for _ in K_LIST) + r" \\", r"\midrule"]
    for m, tpl in (("r1", "r1_evict_k{}"), ("uninavid", "uninavid_k{}")):
        ref = by_arm.get(partner_arm(m, None, "ref"))
        ref_txt = f"{ref['sr']:.2f} / {ref['spl']:.3f}" if ref else "--"
        ref_name = "unpruned reference" if flat else "32-frame reference"
        lines.append(rf"\multicolumn{{{2 + 2 * len(K_LIST)}}}{{l}}{{\emph{{{MODELS[m]}}}, {ref_name} SR / SPL = {ref_txt}}} \\")
        for method in (() if flat else ("recency",)) + METHODS:
            cells = []
            for k in K_LIST:
                arm = tpl.format(k) if method == "recency" else f"{m}_{method}_b{k}"
                cells += [cell(arm, "sr", ".2f"), cell(arm, "spl", ".3f")]
            lines.append(f"{MODELS[m]} & {method} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]
    lines += [r"% $k$ = frames' worth of history tokens kept; every method scores and deletes at decoder layer 0",
              r"% (LongNav-R1: full episode context, Uni-NaVid: 64-frame window). Stars: paired sign-flip test vs",
              r"% the same-$k$ random arm, *p<0.05 **p<0.01 ***p<0.001; $\dagger$ = arm still partial.",
              ] if flat else [
              r"% $k$ = frames' worth of history tokens kept (recency row: the ctx-ablation arm with $k$ frames",
              r"% visible including the current one, i.e. $k-1$ history frames). Stars: paired sign-flip test vs",
              r"% the same-$k$ recency arm, *p<0.05 **p<0.01 ***p<0.001; $\dagger$ = arm still partial.",
              ]
    lines += [f"% n = {len(labels)} HM3D-v2 val episodes; SR/SPL are episode means."]
    (out / "table.tex").write_text("\n".join(lines) + "\n")

    # --- console -------------------------------------------------------------------------
    base = V["baseline"]
    hdr = (f"{'arm':<26}{'n':>4}{'fw':>4}{'SR':>7}{'SPL':>8}{'steps':>7}{'n/fr':>7}{'kept':>7}"
           f"{'dSR_' + base[:3]:>9}{'p':>7}{'dSPL_' + base[:3]:>10}{'p':>7}")
    print(hdr)
    print("-" * len(hdr))
    f = lambda v, fmt: (f"{v:{fmt}}" if v != "" and np.isfinite(float(v)) else "-")
    for r in summary:
        print(f"{r['arm'] + ('*' if r['partial'] else ''):<26}{r['n']:>4}{r['frames_worth']:>4}{r['sr']:>7.2f}"
              f"{r['spl']:>8.3f}{r['steps']:>7.0f}{f(r['n_per_frame'], '.0f'):>7}{f(r['tokens_kept_mean'], '.0f'):>7}"
              f"{f(r[f'dsr_vs_{base}'], '+.3f'):>9}{f(r[f'p_sr_vs_{base}'], '.3f'):>7}"
              f"{f(r[f'dspl_vs_{base}'], '+.3f'):>10}{f(r[f'p_spl_vs_{base}'], '.3f'):>7}")
    if any(r["partial"] for r in summary):
        print("* = partial arm (interim numbers on fewer episodes)")
    print(f"\nwrote {out / 'per_episode.csv'}, {out / 'summary.csv'}, {out / 'table.tex'}, {out / 'n_tokens.csv'}, "
          f"{out / 'masks.parquet'}, {out / 'retention.csv'}")


if __name__ == "__main__":
    main()
