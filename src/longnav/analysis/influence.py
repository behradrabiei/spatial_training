"""Leave-one-frame-out influence: loaders and statistics shared by paper/frame_influence.

Both models are brought to one long-row schema so nothing downstream branches on the model:

    model, episode, step, frame, age, infl, tier, weight, n_slots, logp_full, logp_removed,
    chosen_action, argmax_action, argmax_removed, flip, success, n_history, T,
    action_at_frame, entropy_t, dtg_t

``infl`` is KL(π_full || π_without_frame) in nats over the four actions at temperature 1.
Rows whose frame has no tokens in the context (LongNav-R1 frames fully removed by the sparse
filter, ``n_slots == 0``) are kept in the table but carry ``infl = nan`` so statistics skip them.

Per-step records (``load_*_steps``) hold the joint probes of each decision: ``history_all``
(every history frame removed), ``visual_all`` (no visual tokens), ``age0`` (current frame
removed), ``horizon[K]`` (frames of age >= K removed) and ``sum_loo`` (sum of the single-frame
influences), plus the decision's entropy, chosen action and distance to goal.

Pure numpy/stdlib so it imports in both conda envs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from longnav.analysis.results_io import load_r1_run, load_uninavid_run
from longnav.analysis.stats import N_BOOT_CI, SEED

ACTIONS = ("stop", "forward", "left", "right")
ROW_FIELDS = ("model", "episode", "step", "frame", "age", "infl", "tier", "weight", "n_slots",
              "logp_full", "logp_removed", "chosen_action", "argmax_action", "argmax_removed", "flip",
              "success", "n_history", "T", "action_at_frame", "entropy_t", "dtg_t")
AGE_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 9))
KS = (1, 4, 8, 16, 32)
MIN_FRAMES = 8  # per-step rank correlations need this many scored frames


def age_bin_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")


def entropy(logp: Iterable[float]) -> float:
    lp = np.asarray(list(logp), dtype=float)
    return float(-(np.exp(lp) * lp).sum())


def kl_from_probs(p, q, eps: float = 1e-12) -> float:
    """KL(p || q) in nats between two probability vectors."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def _argmax_action(logp) -> str:
    return ACTIONS[int(np.argmax(np.asarray(logp, dtype=float)))]


def _action_name(value) -> str:
    """LongNav logs action ids or names depending on the version; normalise to names."""
    if isinstance(value, str):
        return value
    return ACTIONS[int(value)] if 0 <= int(value) < len(ACTIONS) else str(value)


# ------------------------------------------------------------------------------ LongNav-R1
def _r1_sequences(run_dir: Path):
    for seq in sorted(run_dir.glob("rollout/*/*/sequence.json")):
        label = seq.parent.name.split(".")[0]
        yield label, json.loads(seq.read_text())


MEASURES = ("leave_one", "keep_one")


def _check_measure(measure: str) -> None:
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")


def load_r1_influence(run_dir: Path | str, model: str = "LongNav-R1",
                      measure: str = "leave_one") -> tuple[list[dict], list[dict]]:
    """(rows, steps) from a LongNav run logged with vlm.kv_prune_log_influence.

    ``measure`` picks the per-frame rows: ``leave_one`` (sup/kl_influence: KL(full || frame
    hidden), ``logp_full`` = the replayed full decision, ``logp_removed`` = with the frame hidden,
    ``flip`` = argmax moved from the full decision) or ``keep_one`` (sup/kl_influence_keep_one:
    KL(only that frame's visual slots || no visual); then ``logp_full`` holds the REFERENCE, the
    no-visual decision, ``logp_removed`` the only-frame decision and ``flip`` says whether the
    frame alone moves the argmax off the no-visual one). ``age0`` is the row's last entry (the
    current frame) either way; the joint probes and ``sum_loo`` (sum of the chosen measure over
    the history frames) are unchanged in meaning.
    """
    _check_measure(measure)
    run_dir = Path(run_dir)
    results = load_r1_run(run_dir)
    rows, steps = [], []
    row_key, logp_key = (("sup/kl_influence", "sup/kl_influence_logp") if measure == "leave_one"
                         else ("sup/kl_influence_keep_one", "sup/kl_influence_keep_one_logp"))
    for label, d in _r1_sequences(run_dir):
        kls = d.get(row_key)
        if not kls:
            continue
        logps = d.get(logp_key) or [[] for _ in kls]
        bases = d.get("sup/kl_influence_base_logp") or [[] for _ in kls]
        hist = d.get("sup/kl_influence_history_all") or [[] for _ in kls]
        vis = d.get("sup/kl_influence_visual_all") or [[] for _ in kls]
        horiz = d.get("sup/kl_influence_horizon") or [[] for _ in kls]
        vis_new = d.get("kv_visual_new") or []
        live = d.get("sup/action_probs") or []
        actions = [_action_name(a) for a in (d.get("action_history") or [])]
        dtg = d.get("distance_to_goal") or []
        T = len(kls)
        success = results.get(label, {}).get("success")
        for t, row in enumerate(kls):
            if len(row) != t + 1:
                continue  # stride-skipped or malformed step
            base = bases[t] if t < len(bases) else []
            ent = entropy(base) if base else float("nan")
            chosen = actions[t] if t < len(actions) else ""
            argmax = _argmax_action(base) if base else ""
            step_rec = {
                "model": model, "episode": label, "step": t, "n_history": t, "T": T, "success": success,
                "chosen_action": chosen, "argmax_action": argmax, "entropy_t": ent,
                "dtg_t": dtg[t] if t < len(dtg) else float("nan"),
                "p_chosen": math.exp(base[ACTIONS.index(chosen)]) if base and chosen in ACTIONS else float("nan"),
                "sum_loo": float("nan"), "n_scored": 0,
                "history_all": hist[t][0] if t < len(hist) and hist[t] else float("nan"),
                "visual_all": vis[t][0] if t < len(vis) and vis[t] else float("nan"),
                "age0": row[t] if (t < len(vis_new) and vis_new[t]) or t >= len(vis_new) else float("nan"),
                "horizon": {int(h[0]): float(h[1]) for h in (horiz[t] if t < len(horiz) else [])},
                "p_mass": float("nan"), "sampled_in_set": None, "probe_seconds": float("nan"),
                # bf16 single-token replay vs the live multi-token forward of the same decision:
                # the numerical floor below which a logged KL cannot be read as influence.
                "kl_live_vs_replay": (kl_from_probs(live[t], np.exp(base)) if base and t < len(live) else float("nan")),
            }
            if math.isfinite(step_rec["history_all"]):
                step_rec["horizon"].setdefault(1, step_rec["history_all"])
            # keep_one rows are scored against the no-visual decision (the visual_all logp)
            ref = base if measure == "leave_one" else (vis[t][1] if t < len(vis) and vis[t] else [])
            ref_argmax = _argmax_action(ref) if ref else ""
            loo = []
            for f in range(t):  # history frames only; the current frame is the age0 probe
                n_slots = vis_new[f] if f < len(vis_new) else None
                scored = n_slots is None or n_slots > 0
                lp = logps[t][f] if t < len(logps) and f < len(logps[t]) else []
                infl = float(row[f]) if scored else float("nan")
                if scored:
                    loo.append(infl)
                rows.append({
                    "model": model, "episode": label, "step": t, "frame": f, "age": t - f, "infl": infl,
                    "tier": "cache", "weight": 1, "n_slots": n_slots if n_slots is not None else "",
                    "logp_full": ref, "logp_removed": lp, "chosen_action": chosen, "argmax_action": argmax,
                    "argmax_removed": _argmax_action(lp) if lp else "",
                    "flip": (int(_argmax_action(lp) != ref_argmax) if lp and ref_argmax else ""),
                    "success": success, "n_history": t, "T": T,
                    "action_at_frame": actions[f] if f < len(actions) else "",
                    "entropy_t": ent, "dtg_t": step_rec["dtg_t"],
                })
            step_rec["sum_loo"] = float(np.sum(loo)) if loo else (0.0 if t == 0 else float("nan"))
            step_rec["n_scored"] = len(loo)
            steps.append(step_rec)
    return rows, steps


# -------------------------------------------------------------------------------- Uni-NaVid
def _uninavid_traces(run_dir: Path, label: str) -> dict[int, dict]:
    path = run_dir / "traces" / f"{label}.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["step"])] = rec
    return out


def load_uninavid_influence(run_dir: Path | str, model: str = "Uni-NaVid",
                            measure: str = "leave_one") -> tuple[list[dict], list[dict]]:
    """(rows, steps) from a Uni-NaVid run made with --frame-influence.

    ``measure`` as in load_r1_influence: ``leave_one`` reads ``frames`` / ``age0``, ``keep_one``
    reads ``keep_one`` / ``keep_one_age0`` (reference = the ``visual_all`` prompt).
    """
    _check_measure(measure)
    run_dir = Path(run_dir)
    results = load_uninavid_run(run_dir)
    rows, steps = [], []
    for path in sorted((run_dir / "influence").glob("*.jsonl")):
        label = path.stem
        trace = _uninavid_traces(run_dir, label)
        success = results.get(label, {}).get("success")
        T = results.get(label, {}).get("n_steps")
        actions_by_step = {s: r["action"] for s, r in trace.items()}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = int(rec["step"])
            base = rec["logp_full"]
            chosen = rec["action"]
            argmax = rec["argmax_action"]
            if measure == "leave_one":
                frames, age0_rec, ref = rec["frames"], rec.get("age0"), base
            else:
                frames, age0_rec = rec.get("keep_one", []), rec.get("keep_one_age0")
                # masked form: the reference is the column-masked no-visual prompt
                vis = rec.get("visual_all_mask") or rec.get("visual_all")
                ref = vis["logp"] if vis else []
            ref_argmax = _argmax_action(ref) if ref else ""
            dtg = rec.get("distance_to_goal")
            if dtg is None:  # older records: fall back to the (gitignored) trace
                dtg = trace.get(t, {}).get("metrics_before_action", {}).get("distance_to_goal", float("nan"))
            ent = entropy(base)
            step_rec = {
                "model": model, "episode": label, "step": t, "n_history": t, "T": T, "success": success,
                "chosen_action": chosen, "argmax_action": argmax, "entropy_t": ent, "dtg_t": dtg,
                "p_chosen": math.exp(base[ACTIONS.index(chosen)]) if chosen in ACTIONS else float("nan"),
                "sum_loo": float(sum(fr["kl"] for fr in frames)), "n_scored": len(frames),
                "history_all": rec["history_all"]["kl"] if rec.get("history_all") else float("nan"),
                "visual_all": rec["visual_all"]["kl"] if rec.get("visual_all") else float("nan"),
                "age0": age0_rec["kl"] if age0_rec else float("nan"),
                "horizon": {int(h["K"]): float(h["kl"]) for h in rec.get("horizon", [])},
                "p_mass": rec.get("p_mass", float("nan")), "sampled_in_set": rec.get("sampled_first_token_in_set"),
                "probe_seconds": rec.get("probe_seconds", float("nan")),
                "kl_live_vs_replay": float("nan"),  # Uni-NaVid scores base and probes on one code path
            }
            if math.isfinite(step_rec["history_all"]):
                step_rec["horizon"].setdefault(1, step_rec["history_all"])
            steps.append(step_rec)
            for fr in frames:
                f = int(fr["frame"])
                rows.append({
                    "model": model, "episode": label, "step": t, "frame": f, "age": t - f, "infl": float(fr["kl"]),
                    "tier": fr["tier"], "weight": fr.get("weight") or 1, "n_slots": 4 if fr["tier"] == "short" else 1,
                    "logp_full": ref, "logp_removed": fr["logp"], "chosen_action": chosen, "argmax_action": argmax,
                    "argmax_removed": fr["argmax_action"],
                    "flip": int(fr["argmax_action"] != ref_argmax) if ref_argmax else "",
                    "success": success, "n_history": t, "T": T,
                    "action_at_frame": actions_by_step.get(f, ""), "entropy_t": ent, "dtg_t": dtg,
                })
    return rows, steps


# ------------------------------------------------------------------------------- statistics
def _finite(rows: list[dict], key: str = "infl") -> list[dict]:
    return [r for r in rows if isinstance(r[key], float) and math.isfinite(r[key])]


def _episode_bootstrap(per_episode: dict[str, np.ndarray], n_boot: int = N_BOOT_CI, seed: int = SEED):
    """CI of the pooled mean under resampling whole episodes (clusters), 95% percentile."""
    eps = list(per_episode)
    if not eps:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sums = np.array([per_episode[e].sum() for e in eps])
    counts = np.array([per_episode[e].size for e in eps], dtype=float)
    idx = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    boots = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def age_curve(rows: list[dict], max_age: int = 256) -> list[dict]:
    """Per age: pooled mean/std/n with an episode-cluster bootstrap CI, and the
    episode-balanced mean (mean over episodes of that episode's mean at that age)."""
    rows = _finite(rows)
    by_age: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        a = min(int(r["age"]), max_age)
        by_age.setdefault(a, {}).setdefault(r["episode"], []).append(r["infl"])
    out = []
    for a in sorted(by_age):
        per_ep = {e: np.asarray(v) for e, v in by_age[a].items()}
        pooled = np.concatenate(list(per_ep.values()))
        lo, hi = _episode_bootstrap(per_ep)
        ep_means = np.array([v.mean() for v in per_ep.values()])
        out.append({"age": a, "n": int(pooled.size), "n_episodes": len(per_ep),
                    "mean": float(pooled.mean()), "std": float(pooled.std(ddof=0)), "median": float(np.median(pooled)),
                    "lo": lo, "hi": hi, "balanced_mean": float(ep_means.mean()),
                    "balanced_sem": float(ep_means.std(ddof=1) / math.sqrt(len(ep_means))) if len(ep_means) > 1 else float("nan")})
    return out


def age_bin_table(rows: list[dict]) -> list[dict]:
    rows = _finite(rows)
    ages = np.array([r["age"] for r in rows]); vals = np.array([r["infl"] for r in rows])
    total = vals.sum()
    out = []
    for lo, hi in AGE_BINS:
        m = (ages >= lo) & (ages <= hi)
        if not m.any():
            continue
        out.append({"age_bin": age_bin_label(lo, hi), "n": int(m.sum()), "mean": float(vals[m].mean()),
                    "median": float(np.median(vals[m])), "mass_share": float(vals[m].sum() / total) if total > 0 else float("nan")})
    return out


def cumulative_share(rows: list[dict], ks: tuple[int, ...] = KS) -> list[dict]:
    """Share of leave-one-out influence held by the k most recent history frames (ages 1..k).

    ``pooled``: mass share over all (t, i) pairs of steps with at least max(ks) history frames
    (a fixed step set, so the k values are comparable). ``per_step``: mean over steps with
    >= k history frames of that step's own share (steps with zero total mass are skipped).
    """
    rows = _finite(rows)
    kmax = max(ks)
    by_step: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for r in rows:
        by_step.setdefault((r["episode"], r["step"]), []).append((r["age"], r["infl"]))
    out = []
    for k in ks:
        pooled_in = pooled_tot = 0.0
        shares = []
        n_fixed = 0
        for (ep, t), pairs in by_step.items():
            ages = np.array([a for a, _ in pairs]); vals = np.array([v for _, v in pairs])
            tot = vals.sum()
            if t >= kmax:
                pooled_in += vals[ages <= k].sum(); pooled_tot += tot; n_fixed += 1
            if t >= k and tot > 0:
                shares.append(vals[ages <= k].sum() / tot)
        out.append({"k": k, "pooled_share": pooled_in / pooled_tot if pooled_tot > 0 else float("nan"),
                    "n_steps_pooled": n_fixed, "per_step_share": float(np.mean(shares)) if shares else float("nan"),
                    "per_step_share_median": float(np.median(shares)) if shares else float("nan"),
                    "n_steps_per_step": len(shares)})
    return out


def horizon_table(steps: list[dict], ks: tuple[int, ...] = KS) -> list[dict]:
    """Joint influence of the frames older than K (age >= K), i.e. what the last K frames do
    not explain, next to the sum of the single-frame influences over the same frames."""
    out = []
    for k in ks:
        joint, loo_sum, ratio = [], [], []
        for s in steps:
            if k not in s["horizon"] or s["n_history"] < k:
                continue
            joint.append(s["horizon"][k])
        per_ep: dict[str, list[float]] = {}
        for s in steps:
            if k in s["horizon"] and s["n_history"] >= k:
                per_ep.setdefault(s["episode"], []).append(s["horizon"][k])
        lo, hi = _episode_bootstrap({e: np.asarray(v) for e, v in per_ep.items()})
        out.append({"K": k, "n_steps": len(joint), "mean": float(np.mean(joint)) if joint else float("nan"),
                    "median": float(np.median(joint)) if joint else float("nan"), "lo": lo, "hi": hi})
    return out


def loo_sum_by_horizon(rows: list[dict], steps: list[dict], ks: tuple[int, ...] = KS) -> dict[int, float]:
    """Mean over steps (with >= K history frames) of the sum of single-frame influences of the
    frames with age >= K -- the additive prediction the joint horizon probe is compared with."""
    rows = _finite(rows)
    by_step: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for r in rows:
        by_step.setdefault((r["episode"], r["step"]), []).append((r["age"], r["infl"]))
    out = {}
    for k in ks:
        vals = [sum(v for a, v in pairs if a >= k) for (ep, t), pairs in by_step.items() if t >= k]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def flip_rate_by_age(rows: list[dict]) -> list[dict]:
    out = []
    rows = [r for r in _finite(rows) if r["flip"] != ""]
    ages = np.array([r["age"] for r in rows]); flips = np.array([int(r["flip"]) for r in rows])
    for lo, hi in AGE_BINS:
        m = (ages >= lo) & (ages <= hi)
        if m.any():
            out.append({"age_bin": age_bin_label(lo, hi), "n": int(m.sum()), "flip_rate": float(flips[m].mean())})
    return out


# ------------------------------------------------------------------------------ persistence
def influence_matrix(rows: list[dict], steps: list[dict], episode: str) -> np.ndarray:
    """Lower-triangular M[t, i] (NaN where unscored), history frames only; the diagonal holds
    the age-0 probe so the matrix reads like the legacy tools/kl_influence.py output."""
    T = max([s["T"] or 0 for s in steps if s["episode"] == episode] +
            [r["step"] + 1 for r in rows if r["episode"] == episode] + [1])
    M = np.full((T, T), np.nan)
    for r in rows:
        if r["episode"] == episode and isinstance(r["infl"], float) and math.isfinite(r["infl"]):
            M[r["step"], r["frame"]] = r["infl"]
    for s in steps:
        if s["episode"] == episode and math.isfinite(s.get("age0", float("nan"))):
            M[s["step"], s["step"]] = s["age0"]
    return M


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=ranks)[inv] / counts[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def age_pools(mats: list[np.ndarray], max_age: int = 256) -> dict[int, np.ndarray]:
    """Sorted pooled influence values per age (ages beyond max_age share one pool), history only."""
    pools: dict[int, list] = {}
    for M in mats:
        T = M.shape[0]
        t_idx, f_idx = np.tril_indices(T, k=-1)
        v = M[t_idx, f_idx]; ok = np.isfinite(v)
        for a, x in zip(np.minimum(t_idx - f_idx, max_age)[ok], v[ok]):
            pools.setdefault(int(a), []).append(float(x))
    return {a: np.sort(np.asarray(v)) for a, v in pools.items()}


def age_percentiles(M: np.ndarray, pools: dict[int, np.ndarray], max_age: int = 256) -> np.ndarray:
    """Each history entry mapped to its percentile among the pooled values of the same age, so
    every age is uniform on [0, 1] by construction and what remains is 'unusual for its age'."""
    T = M.shape[0]
    out = np.full_like(M, np.nan)
    t_idx, f_idx = np.tril_indices(T, k=-1)
    for t, f in zip(t_idx, f_idx):
        v = M[t, f]
        if not np.isfinite(v):
            continue
        pool = pools.get(min(t - f, max_age))
        if pool is None or pool.size < 2:
            continue
        lo = np.searchsorted(pool, v, side="left"); hi = np.searchsorted(pool, v, side="right")
        out[t, f] = ((lo + hi) / 2) / pool.size  # mid-rank handles ties (e.g. many exact zeros)
    return out


def _lag_stats(mats: list[np.ndarray], k: int) -> dict:
    step_rhos, retained, per_ep = [], [], []
    for M in mats:
        T = M.shape[0]
        ep_rhos = []
        for t in range(T - k):
            x, y = M[t, :t], M[t + k, :t]  # history frames of step t, seen again k steps later
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_FRAMES:
                continue
            rho = spearman(x, y)
            step_rhos.append(rho); ep_rhos.append(rho)
            xo, yo = x[ok], y[ok]
            q = max(1, int(round(0.25 * xo.size)))
            top_x = set(np.argsort(-xo)[:q].tolist()); top_y = set(np.argsort(-yo)[:q].tolist())
            retained.append(len(top_x & top_y) / q)
        per_ep.append(float(np.nanmean(ep_rhos)) if ep_rhos else float("nan"))
    return {"rho_step_mean": float(np.nanmean(step_rhos)) if step_rhos else float("nan"),
            "top_quartile_retention": float(np.mean(retained)) if retained else float("nan"),
            "rho_per_episode": per_ep, "n_steps": len(step_rhos)}


def persistence(mats: list[np.ndarray], lags: tuple[int, ...] = (1, 2, 4, 8, 16)) -> list[dict]:
    """Per lag: raw and age-adjusted Spearman rho of a frame's influence between step t and
    t+k (mean over steps), top-quartile retention (chance 0.25), and the per-episode spread.
    Age adjustment = percentile within the pooled values of the same age (``age_percentiles``)."""
    pools = age_pools(mats)
    resid = [age_percentiles(M, pools) for M in mats]
    out = []
    for k in lags:
        raw, res = _lag_stats(mats, k), _lag_stats(resid, k)
        per = np.array([v for v in res["rho_per_episode"] if math.isfinite(v)])
        out.append({"lag": k, "rho_raw": raw["rho_step_mean"], "rho_residual": res["rho_step_mean"],
                    "rho_residual_ep_lo": float(np.percentile(per, 2.5)) if per.size else float("nan"),
                    "rho_residual_ep_hi": float(np.percentile(per, 97.5)) if per.size else float("nan"),
                    "top25_raw": raw["top_quartile_retention"], "top25_residual": res["top_quartile_retention"],
                    "n_steps": res["n_steps"], "n_episodes": int(per.size)})
    return out


def revival(mats: list[np.ndarray], eps: float = 1e-3, dormant: int = 5, thr: float = 1e-2) -> dict:
    """Frames whose influence stays < eps for >= dormant steps and later exceeds thr."""
    ever, revived, events = 0, 0, 0
    for M in mats:
        T = M.shape[0]
        for f in range(T):
            s = M[f + 1:, f]  # history rows only
            s = s[np.isfinite(s)]
            if s.size < dormant + 1:
                continue
            run, was_dormant, did = 0, False, False
            for v in s:
                if v < eps:
                    run += 1
                    was_dormant = was_dormant or run >= dormant
                else:
                    if was_dormant and v > thr:
                        events += 1; did = True
                    run = 0
            ever += was_dormant; revived += did
    return {"ever_dormant": ever, "revived": revived, "events": events,
            "rate": revived / ever if ever else float("nan")}


# ------------------------------------------------------------------------------ covariates
def covariate_bins(rows: list[dict]) -> list[dict]:
    """Mean influence by (age bin x covariate level) for the discrete/binned covariates."""
    rows = _finite(rows)
    out = []

    def add(name, level_of):
        groups: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            lvl = level_of(r)
            if lvl is None:
                continue
            for lo, hi in AGE_BINS:
                if lo <= r["age"] <= hi:
                    groups.setdefault((age_bin_label(lo, hi), lvl), []).append(r["infl"])
                    break
        for (ab, lvl), v in sorted(groups.items()):
            out.append({"covariate": name, "age_bin": ab, "level": lvl, "n": len(v),
                        "mean": float(np.mean(v)), "median": float(np.median(v))})

    def q_level(key, edges, labels):
        def f(r):
            x = r.get(key)
            if x is None or x == "" or not math.isfinite(float(x)):
                return None
            x = float(x)
            for e, lab in zip(edges, labels):
                if x < e:
                    return lab
            return labels[-1]
        return f

    add("action_at_frame", lambda r: r["action_at_frame"] or None)
    add("chosen_action", lambda r: r["chosen_action"] or None)
    add("entropy_t", q_level("entropy_t", (0.1, 0.5, 1.0), ("<0.1", "0.1-0.5", "0.5-1.0", ">=1.0")))
    add("dtg_t", q_level("dtg_t", (1.0, 3.0, 6.0), ("<1m", "1-3m", "3-6m", ">=6m")))
    add("n_slots", q_level("n_slots", (16, 64, 128), ("<16", "16-63", "64-127", ">=128")))
    return out


def ols_log_influence(rows: list[dict]) -> dict:
    """OLS of log10(infl + 1e-6) on age-bin dummies plus the covariates available for the model;
    reports coefficients, R^2, and the partial R^2 of each covariate block (drop-one)."""
    rows = [r for r in _finite(rows) if r["age"] >= 1]
    if len(rows) < 50:
        return {"n": len(rows)}
    y = np.log10(np.array([r["infl"] for r in rows]) + 1e-6)
    blocks: dict[str, np.ndarray] = {}
    bins = [b for b in AGE_BINS if b[0] >= 1]
    age = np.array([r["age"] for r in rows])
    blocks["age_bin"] = np.stack([((age >= lo) & (age <= hi)).astype(float) for lo, hi in bins[1:]], axis=1)
    blocks["log_age"] = np.log2(age)[:, None].astype(float)

    def onehot(key):
        levels = sorted({r[key] for r in rows if r[key] not in ("", None)})
        if len(levels) < 2:
            return None
        return np.stack([np.array([1.0 if r[key] == l else 0.0 for r in rows]) for l in levels[1:]], axis=1)

    for key in ("action_at_frame", "chosen_action"):
        X = onehot(key)
        if X is not None:
            blocks[key] = X
    for key in ("entropy_t", "dtg_t", "n_slots"):
        vals = np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])
        if np.isfinite(vals).sum() == len(vals) and vals.std() > 0:
            blocks[key] = (np.log1p(vals) if key == "n_slots" else vals)[:, None]

    def fit(names):
        X = np.concatenate([np.ones((len(rows), 1))] + [blocks[n] for n in names], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return beta, 1 - resid.var() / y.var()

    names = list(blocks)
    beta, r2 = fit(names)
    partial = {}
    for n in names:
        _, r2_wo = fit([m for m in names if m != n])
        partial[n] = r2 - r2_wo
    _, r2_age_only = fit(["age_bin", "log_age"])
    return {"n": len(rows), "r2": float(r2), "r2_age_only": float(r2_age_only),
            "partial_r2": {k: float(v) for k, v in partial.items()},
            "log_age_slope": float(beta[1 + blocks["age_bin"].shape[1]])}


def pick_representative_episode(*steps_lists: list[dict]) -> str | None:
    """Label present for every model, successful under all of them, with the most decisions in
    the model that has the fewest."""
    def per(steps):
        out: dict[str, dict] = {}
        for s in steps:
            d = out.setdefault(s["episode"], {"n": 0, "success": s["success"]})
            d["n"] += 1
        return out
    tables = [per(st) for st in steps_lists if st]
    if not tables:
        return None
    common = set(tables[0])
    for t in tables[1:]:
        common &= set(t)
    good = [e for e in common if all((t[e]["success"] or 0) >= 1 for t in tables)]
    pool = good or sorted(common)
    return max(pool, key=lambda e: min(t[e]["n"] for t in tables)) if pool else None


def paired_horizon_gap(steps_a: list[dict], steps_b: list[dict], ks: tuple[int, ...] = KS) -> list[dict]:
    """Per K: mean of horizon_K(a) - horizon_K(b) over decisions both arms scored (paired on
    (episode, step)), with an episode-cluster bootstrap CI. Used for evict-style minus leak-free
    on LongNav-R1: the information the removed frames had already written into later K/V."""
    by_b = {(s["episode"], s["step"]): s for s in steps_b}
    out = []
    for k in ks:
        per_ep: dict[str, list[float]] = {}
        a_vals, b_vals = [], []
        for s in steps_a:
            o = by_b.get((s["episode"], s["step"]))
            if o is None or k not in s["horizon"] or k not in o["horizon"] or s["n_history"] < k:
                continue
            a_vals.append(s["horizon"][k]); b_vals.append(o["horizon"][k])
            per_ep.setdefault(s["episode"], []).append(s["horizon"][k] - o["horizon"][k])
        if not a_vals:
            continue
        lo, hi = _episode_bootstrap({e: np.asarray(v) for e, v in per_ep.items()})
        out.append({"K": k, "n_steps": len(a_vals), "n_episodes": len(per_ep),
                    "mean_a": float(np.mean(a_vals)), "mean_b": float(np.mean(b_vals)),
                    "delta": float(np.mean(a_vals) - np.mean(b_vals)), "lo": lo, "hi": hi})
    return out
