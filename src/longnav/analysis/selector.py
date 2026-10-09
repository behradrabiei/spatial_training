"""Loaders and statistics for the selector-criterion vs keep-one-in KL probe (paper/selector_kl).

One record per decision, for either model: LongNav-R1 from a ``vlm.selector_probe`` run
(``rollout/*/*/sequence.json``, ``sup/selector_*`` rows written by ``longnav.utils.selector_probe``),
Uni-NaVid from a ``--selector-probe`` run (``selector/<label>.jsonl``, written by
``baselines/uninavid_hm3d/selector_probe.py``). For every visual token of the current frame the
record holds its layer-0 no-history keep-one-in KL (``keep_all``) and the flat selectors' continuous
criteria -- FastV's layer-0 attention (``attn``), DivPrune's nearest-neighbour distance
(``divprune_nn``), DART's pivot dissimilarity (``dart_dis``), the random control's permutation
position (``random_pos``) -- and, on LongNav-R1 only, each method's greedy position and the
smallest frames'-worth budget at which it keeps the token (``*_rank``; -1 = never; Uni-NaVid records
carry no budgets). Pure numpy/stdlib so both conda envs can import it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from longnav.analysis.influence import _episode_bootstrap, _r1_sequences, spearman

NEVER = -1
METHODS = ("random", "attn", "divprune", "dart")
LABELS = {"random": "Random", "attn": "FastV", "divprune": "DivPrune", "dart": "DART"}
MEASURE = "keep_all"  # keep-one-in with no visual history: KL(P_only k || P_no visual), as the horizon probe
MEASURES = (MEASURE,)
LIST_KEYS = ("raster", "attn", "attn_pos", "divprune_pos", "random_pos", "divprune_nn", "dart_dis",
             "attn_rank", "divprune_rank", "dart_rank", "random_rank")
FLOAT_LISTS = ("attn", "divprune_nn", "dart_dis", "keep_cur", "leave", "keep_all", "leave_nohist")
MIN_TOKENS = 8  # a rank correlation over fewer tokens is noise


def load_r1_selector(run_dir: Path | str) -> list[dict]:
    """One record per (episode, step) with numpy arrays over the current frame's tokens."""
    recs = []
    for label, d in _r1_sequences(Path(run_dir)):
        n_cur = d.get("sup/selector_n_cur")
        if not n_cur:
            continue
        measures = [m for m in ("keep_cur", "leave", "keep_all", "leave_nohist") if f"sup/selector_{m}" in d]
        actions = d.get("action_history") or []
        for t in range(len(n_cur)):
            rec = {"model": "LongNav-R1", "episode": label, "step": t, "n_frames": int(d["sup/selector_n_frames"][t]),
                   "n_cand": int(d["sup/selector_n_cand"][t]), "n_cur": int(n_cur[t]),
                   "dart_nested": int(d["sup/selector_dart_nested"][t]),
                   "text_cur_kl": float(d["sup/selector_text_cur_kl"][t]),
                   "text_all_kl": float(d["sup/selector_text_all_kl"][t]),
                   "budget_frames": [int(k) for k in d["sup/selector_budget_frames"][t]],
                   "budgets": [int(c) for c in d["sup/selector_budgets"][t]],
                   "grid_thw": [int(x) for x in d["sup/selector_grid_thw"][t]],
                   "base_logp": np.asarray(d["sup/selector_base_logp"][t], dtype=np.float64),
                   "action": actions[t] if t < len(actions) else None,
                   "measures": measures}
            for key in LIST_KEYS + tuple(measures):
                dtype = np.float64 if key in FLOAT_LISTS else np.int64
                rec[key] = np.asarray(d[f"sup/selector_{key}"][t], dtype=dtype)
            recs.append(rec)
    return recs


def load_uninavid_selector(run_dir: Path | str, model: str = "Uni-NaVid") -> list[dict]:
    """One record per inference from ``<run_dir>/selector/<label>.jsonl`` with the same keys as
    :func:`load_r1_selector` (no budgets: the positions / keep ranks are all ``NEVER``)."""
    import json
    recs = []
    for path in sorted((Path(run_dir) / "selector").glob("*.jsonl")):
        label = path.stem
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n = int(r["n_current"])
            never = np.full(n, NEVER, dtype=np.int64)
            rec = {"model": model, "episode": label, "step": int(r["step"]), "n_frames": int(r["n_frames"]),
                   "n_cand": int(r["n_visual"]), "n_cur": n, "dart_nested": 1,
                   "text_cur_kl": float("nan"), "text_all_kl": float(r["kl_no_visual"]),
                   "budget_frames": [], "budgets": [], "grid_thw": [1, 16, 16],
                   "base_logp": np.asarray(r["logp_full"], dtype=np.float64),
                   "action": r.get("action"), "measures": ["keep_all"],
                   "is_nav": np.asarray(r["is_nav"], dtype=bool),
                   "raster": np.asarray(r["current_cols"], dtype=np.int64),
                   "keep_all": np.asarray(r["keep_all"], dtype=np.float64),
                   "attn": np.asarray(r["attn"], dtype=np.float64),
                   "divprune_nn": np.asarray(r["divprune_nn"], dtype=np.float64),
                   "dart_dis": np.asarray(r["dart_dis"], dtype=np.float64),
                   "random_pos": np.asarray(r["random_pos"], dtype=np.int64),
                   "attn_pos": never.copy(), "divprune_pos": never.copy(), "attn_rank": never.copy(),
                   "divprune_rank": never.copy(), "dart_rank": never.copy(), "random_rank": never.copy()}
            recs.append(rec)
    return recs


def binding_budgets(rec: dict) -> list[int]:
    return [k for k, c in zip(rec["budget_frames"], rec["budgets"]) if c != NEVER]


def kept_at(rec: dict, method: str, k: int) -> np.ndarray:
    """Bool mask over the frame's tokens: kept by ``method`` at budget ``k`` (rank = smallest
    binding budget that keeps the token, so kept at k iff 0 <= rank <= k)."""
    r = rec[f"{method}_rank"]
    return (r != NEVER) & (r <= k)


def rank_score(rec: dict, method: str) -> np.ndarray:
    """The shared 5-level scale, higher = kept at a smaller budget; never-kept lowest."""
    r = rec[f"{method}_rank"].astype(np.float64)
    worst = float(max(rec["budget_frames"])) * 2
    return np.where(r == NEVER, -worst, -r)


def criteria(rec: dict) -> dict[str, np.ndarray]:
    """Each method's continuous per-token criterion, higher = keep: attn = the layer-0 attention
    score; divprune = the token's nearest-neighbour cosine distance in the pool (DivPrune's
    max-min objective); dart = its dissimilarity to DART's pivots (``1 - min_p cos``, pivots
    2.0); random = minus the permutation position (a control, no criterion)."""
    return {"attn": rec["attn"].astype(np.float64),
            "divprune": rec["divprune_nn"].astype(np.float64),
            "dart": rec["dart_dis"].astype(np.float64),
            "random": -rec["random_pos"].astype(np.float64)}


def greedy_position_score(rec: dict) -> np.ndarray:
    """DivPrune's realised greedy order as a score (minus the position; beyond the order = lowest)."""
    pos = rec["divprune_pos"].astype(np.float64)
    return np.where(pos == NEVER, -float(rec["n_cand"]), -pos)


def decision_stats(rec: dict, measure: str, min_tokens: int = MIN_TOKENS) -> dict:
    """Per-decision statistics of one measure: Spearman rho of each criterion (and of each
    method's rank scale) with the measure, and at every binding budget the share of the frame's
    measure mass the kept tokens carry vs the share of tokens kept (lift = their ratio)."""
    y = rec[measure]
    n = int(y.size)
    total = float(y.sum())
    row = {"model": rec.get("model", "LongNav-R1"), "episode": rec["episode"], "step": rec["step"], "n_frames": rec["n_frames"], "n_cur": n,
           "n_cand": rec["n_cand"], "measure": measure, "total": total,
           "text_cur_kl": rec["text_cur_kl"], "valid": int(n >= min_tokens and total > 0)}
    crit = criteria(rec)
    ok = row["valid"]
    for m in METHODS:
        row[f"rho_{m}"] = spearman(crit[m], y) if ok else float("nan")
        # the rank scale is defined only where a budget binds
        row[f"rho_rank_{m}"] = spearman(rank_score(rec, m), y) if ok and binding_budgets(rec) else float("nan")
    row["rho_divprune_order"] = spearman(greedy_position_score(rec), y) if ok and binding_budgets(rec) else float("nan")
    for k in rec["budget_frames"]:
        bind = k in binding_budgets(rec)
        for m in METHODS:
            if not (ok and bind):
                row[f"mass_{m}_k{k}"] = row[f"tokshare_{m}_k{k}"] = row[f"lift_{m}_k{k}"] = float("nan")
                continue
            kept = kept_at(rec, m, k)
            mass = float(y[kept].sum()) / total
            share = float(kept.mean())
            row[f"mass_{m}_k{k}"] = mass
            row[f"tokshare_{m}_k{k}"] = share
            row[f"lift_{m}_k{k}"] = mass / share if share > 0 else float("nan")
    return row


def summarise(rows: list[dict], key: str) -> dict:
    """Pooled mean with episode-cluster bootstrap CI, episode-balanced mean, median, counts."""
    per_ep: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(key, float("nan"))
        if isinstance(v, float) and not np.isfinite(v):
            continue
        per_ep.setdefault(r["episode"], []).append(float(v))
    if not per_ep:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "balanced_mean": float("nan"),
                "median": float("nan"), "n_decisions": 0, "n_episodes": 0}
    arrs = {e: np.asarray(v) for e, v in per_ep.items()}
    pooled = np.concatenate(list(arrs.values()))
    lo, hi = _episode_bootstrap(arrs)
    return {"mean": float(pooled.mean()), "lo": lo, "hi": hi,
            "balanced_mean": float(np.mean([a.mean() for a in arrs.values()])),
            "median": float(np.median(pooled)), "n_decisions": int(pooled.size), "n_episodes": len(arrs)}


def paired_vs_random(rows: list[dict], key_method: str, key_random: str) -> dict:
    """Mean per-decision difference method - random with an episode-cluster bootstrap CI."""
    per_ep: dict[str, list[float]] = {}
    for r in rows:
        a, b = r.get(key_method, float("nan")), r.get(key_random, float("nan"))
        if np.isfinite(a) and np.isfinite(b):
            per_ep.setdefault(r["episode"], []).append(float(a - b))
    if not per_ep:
        return {"delta": float("nan"), "delta_lo": float("nan"), "delta_hi": float("nan")}
    arrs = {e: np.asarray(v) for e, v in per_ep.items()}
    lo, hi = _episode_bootstrap(arrs)
    return {"delta": float(np.concatenate(list(arrs.values())).mean()), "delta_lo": lo, "delta_hi": hi}


def consistency(rec: dict) -> list[str]:
    """Internal checks of one record: rank <-> position agreement, DART nesting flag, budgets."""
    msgs = []
    bind = binding_budgets(rec)
    caps = dict(zip(rec["budget_frames"], rec["budgets"]))
    for m in ("attn", "divprune", "random"):
        pos, rank = rec[f"{m}_pos"], rec[f"{m}_rank"]
        exp = np.full(pos.shape, NEVER, dtype=np.int64)
        for k in sorted(bind, reverse=True):
            exp[(pos >= 0) & (pos < caps[k])] = k
        if not np.array_equal(exp, rank):
            msgs.append(f"{rec['episode']} t={rec['step']}: {m}_rank disagrees with {m}_pos")
    if not rec["dart_nested"]:
        msgs.append(f"{rec['episode']} t={rec['step']}: DART sets not nested")
    caps_b = [caps[k] for k in bind]
    if caps_b != sorted(caps_b):
        msgs.append(f"{rec['episode']} t={rec['step']}: budgets not monotone {caps_b}")
    for k, c in caps.items():
        if (rec["n_frames"] <= k) != (c == NEVER):
            msgs.append(f"{rec['episode']} t={rec['step']}: budget k={k} binding state wrong")
    return msgs
