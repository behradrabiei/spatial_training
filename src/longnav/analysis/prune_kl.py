"""Loaders and statistics for the pruning-KL experiment (paper/prune_kl).

    Prune_t(M, k) = KL(P(action | full context) || P(action | the visual tokens method M holds at
                                                                 k frames' worth))

on the full-context trajectory, the expectation under the full-context decision (T = 1, nats): the
pruning analogue of the recent-window (paper/frame_influence) and blind-from-layer
(paper/info_horizon) measures. One shadow pruner per (method, k) is advanced along the trajectory
with the flat token-pruning arm's own rule (permanent deletion).

``load_r1_prune_kl`` reads a LongNav-R1 run made with ``vlm.prune_kl_probe`` (leaky decision
replay, 4 actions; ``sup/prune_kl_*`` rows of ``rollout/*/*/sequence.json``),
``load_lightnav_prune_kl`` a LightNav-0 run made with ``eval_influence.py --measure prune_kl``
(full re-prefill, 256 ``<act_l0>`` codes and their six executed-action bins). Both return one row
per (episode, step, method, k) with the same columns (``PRUNE_KL_FIELDS``). Pure numpy/stdlib so
both conda envs can import it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from longnav.analysis.horizon import _values
from longnav.analysis.influence import (_action_name, _argmax_action, _episode_bootstrap, _nan, _r1_sequences,
                                        entropy, kl_from_probs)
from longnav.analysis.results_io import load_lightnav_run, load_r1_run

METHODS = ("random", "fastv", "divprune", "dart")
PRUNE_KL_FIELDS = [
    "model", "episode", "step", "T", "success", "method", "k",
    "kl",          # KL(full || pruned), the model's own action space (4 actions / 256 codes)
    "kl_exec",     # the same over the executed-action classes (LongNav-R1: identical; LightNav-0: 6 bins)
    "flip",        # argmax moved off the full decision's (action / code)
    "flip_exec",   # executed-action class moved (LongNav-R1: identical; LightNav-0: argmax of the 6-bin fold)
    "binding",     # 1 when some method's shadow hides a token at this (step, k): the budget binds
    "n_hidden",    # visual tokens hidden in this probe
    "n_kept",      # history visual tokens visible in this probe (the frame decided on is extra)
    "novis",       # KL(full || no visual token) of the same decision, the measure's ceiling
    "novis_exec",
    "argmax_full", "argmax_pruned", "chosen_action", "entropy_full", "entropy_pruned", "kl_floor",
    "p_mass", "probe_seconds",
]
KEYS = ("kl", "kl_exec", "flip", "flip_exec")


def _mark_binding(rows: list[dict]) -> list[dict]:
    """``binding`` = some method hides a token at this (model, episode, step, k): one decision set
    per budget, shared by every method."""
    binds: dict[tuple, bool] = {}
    for r in rows:
        key = (r["model"], r["episode"], r["step"], r["k"])
        binds[key] = binds.get(key, False) or r["n_hidden"] > 0
    for r in rows:
        r["binding"] = int(binds[(r["model"], r["episode"], r["step"], r["k"])])
    return rows


def load_r1_prune_kl(run_dir: Path | str, model: str = "LongNav-R1") -> list[dict]:
    """Rows from a LongNav-R1 run with ``vlm.prune_kl_probe``. ``kl`` is the logged value; the flips
    and entropies come from the logged log-probs. ``kl_floor`` = KL(live decision || replayed full
    decision), the replay's noise floor (the reference is the replayed decision)."""
    run_dir = Path(run_dir)
    results = load_r1_run(run_dir)
    rows = []
    for label, d in _r1_sequences(run_dir):
        kls = d.get("sup/prune_kl_kl")
        if not kls:
            continue
        methods, budgets = d["sup/prune_kl_methods"], d["sup/prune_kl_budgets"]
        logps, base, vis = d["sup/prune_kl_logp"], d["sup/prune_kl_base_logp"], d["sup/prune_kl_visual_all"]
        hidden, kept = d["sup/prune_kl_hidden"], d["sup/prune_kl_kept"]
        live = d.get("sup/action_probs") or []
        actions = [_action_name(a) for a in (d.get("action_history") or [])]
        T, success = len(kls), results.get(label, {}).get("success")
        for t, row in enumerate(kls):
            if not row:  # malformed / skipped step
                continue
            full = base[t]
            arg_full, ent_full = _argmax_action(full), entropy(full)
            floor = kl_from_probs(live[t], np.exp(full)) if t < len(live) else float("nan")
            novis = float(vis[t][0]) if vis[t] else float("nan")
            for j, (m, k) in enumerate(zip(methods[t], budgets[t])):
                arg = _argmax_action(logps[t][j])
                rows.append({
                    "model": model, "episode": label, "step": t, "T": T, "success": success,
                    "method": str(m), "k": int(k), "kl": float(row[j]), "kl_exec": float(row[j]),
                    "flip": int(arg != arg_full), "flip_exec": int(arg != arg_full),
                    "n_hidden": int(hidden[t][j]), "n_kept": int(kept[t][j]), "novis": novis, "novis_exec": novis,
                    "argmax_full": arg_full, "argmax_pruned": arg,
                    "chosen_action": actions[t] if t < len(actions) else "",
                    "entropy_full": ent_full, "entropy_pruned": entropy(logps[t][j]), "kl_floor": floor,
                    "p_mass": float("nan"), "probe_seconds": float("nan"),
                })
    return _mark_binding(rows)


def lightnav_records(run_dir: Path | str):
    """``(label, record)`` over a converted LightNav run's ``influence/<label>.jsonl`` files."""
    for path in sorted((Path(run_dir) / "influence").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                yield path.stem, json.loads(line)


def load_lightnav_prune_kl(run_dir: Path | str, model: str = "LightNav-0",
                           methods: tuple[str, ...] | None = METHODS) -> list[dict]:
    """Rows from a LightNav-0 run made with ``eval_influence.py --measure prune_kl``. ``kl`` /
    ``flip`` are over the 256 ``<act_l0>`` codes, ``kl_exec`` / ``flip_exec`` over the six
    executed-action bins (argmax of the folded distribution); ``chosen_action`` = the bin of the
    command actually executed. A probe whose generate ended before the action position has NaN
    values. ``kl_floor`` = the replay with nothing hidden. ``methods`` filters the arms (None =
    every arm of the run, the recency check arms included)."""
    run_dir = Path(run_dir)
    results = load_lightnav_run(run_dir)
    rows = []
    for label, rec in lightnav_records(run_dir):
        if "prune_kl" not in rec:
            raise ValueError(f"{run_dir}: {label} is not a prune_kl record (run eval_influence.py --measure prune_kl)")
        success, T = results.get(label, {}).get("success"), results.get(label, {}).get("n_steps")
        full, novis = rec["full"], rec.get("visual_all_mask") or {}
        has_full = full.get("code") is not None
        for pr in rec["prune_kl"]:
            if methods is not None and pr["method"] not in methods:
                continue
            ok = has_full and pr.get("code") is not None
            rows.append({
                "model": model, "episode": label, "step": int(rec["step"]), "T": T, "success": success,
                "method": pr["method"], "k": int(pr["k"]),
                "kl": _nan(pr.get("kl")), "kl_exec": _nan(pr.get("kl_bin")),
                "flip": int(pr["code"] != full["code"]) if ok else "",
                "flip_exec": int(pr["bin_argmax"] != full["bin_argmax"]) if ok else "",
                "n_hidden": int(pr["n_hidden"]), "n_kept": int(pr["n_kept"]),
                "novis": _nan(novis.get("kl")), "novis_exec": _nan(novis.get("kl_bin")),
                "argmax_full": full.get("code", ""), "argmax_pruned": pr.get("code", ""),
                "chosen_action": full.get("exec_bin") or "",
                "entropy_full": _nan(full.get("entropy")), "entropy_pruned": _nan(pr.get("entropy")),
                "kl_floor": _nan((rec.get("replay") or {}).get("kl")),
                "p_mass": _nan(pr.get("p_mass")), "probe_seconds": _nan(rec.get("probe_seconds")),
            })
    return _mark_binding(rows)


def _mean(v: np.ndarray) -> float:
    return float(v.mean()) if v.size else float("nan")


def prune_kl_episode_table(rows: list[dict]) -> list[dict]:
    """One row per (model, episode, method, k): the episode's mean of every measure over the
    decisions where the budget binds (``n_steps`` of them; an episode no longer than the budget
    has none), ``kl_all`` over every decision, and ``kl_norm`` / ``kl_exec_norm`` = the binding
    mean over the mean no-visual KL of the same decisions."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["episode"], r["method"], r["k"]), []).append(r)
    out = []
    for (model, ep, method, k), rs in sorted(groups.items()):
        bind = [r for r in rs if r["binding"]]
        row = {"model": model, "episode": ep, "method": method, "k": k, "T": rs[0]["T"], "success": rs[0]["success"],
               "n_steps": int(_values(bind, "kl").size), "n_steps_all": int(_values(rs, "kl").size)}
        for key in KEYS:
            row[key] = _mean(_values(bind, key))
        row["kl_median"] = float(np.median(_values(bind, "kl"))) if row["n_steps"] else float("nan")
        row["kl_all"], row["kl_exec_all"] = _mean(_values(rs, "kl")), _mean(_values(rs, "kl_exec"))
        row["n_kept"] = _mean(_values(bind, "n_kept"))
        for key, ref in (("kl", "novis"), ("kl_exec", "novis_exec")):
            denom = _mean(_values(bind, ref))
            row[ref] = denom
            row[f"{key}_norm"] = row[key] / denom if np.isfinite(denom) and denom > 0 else float("nan")
        out.append(row)
    return out


def prune_kl_curve(rows: list[dict]) -> list[dict]:
    """One row per (model, method, k), over the decisions where the budget binds. Per measure: the
    pooled mean (``*_mean``, with an episode-cluster bootstrap CI for the two KLs), the
    episode-balanced mean and its SEM across episodes; ``kl_median`` over steps; ``kl_norm*`` = the
    balanced mean (SEM, median) of each episode's mean divided by its mean no-visual KL;
    ``novis_*`` = the no-visual KL of the same decisions (the ceiling); ``kl_all_*`` = the same
    statistics over every decision, the non-binding ones included."""
    episodes = prune_kl_episode_table(rows)
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["model"], r["method"], r["k"]), []).append(r)
    out = []
    for (model, method, k), rs in sorted(by.items()):
        bind = [r for r in rs if r["binding"]]
        eps = [e for e in episodes if (e["model"], e["method"], e["k"]) == (model, method, k)]
        scored = [e for e in eps if e["n_steps"] > 0]
        row = {"model": model, "method": method, "k": k, "n_steps": int(_values(bind, "kl").size),
               "n_episodes": len(scored), "n_steps_all": int(_values(rs, "kl").size), "n_episodes_all": len(eps),
               "n_kept_mean": _mean(_values(bind, "n_kept"))}

        def balanced(name, key, pool):
            em = np.array([e[key] for e in pool], dtype=float)
            em = em[np.isfinite(em)]
            row[f"{name}_balanced"] = _mean(em)
            row[f"{name}_sem"] = float(em.std(ddof=1) / np.sqrt(em.size)) if em.size > 1 else float("nan")
            return em

        for key in KEYS:
            row[f"{key}_mean"] = _mean(_values(bind, key))
            balanced(key, key, scored)
            if key in ("kl", "kl_exec"):
                per_ep: dict[str, list[float]] = {}
                for r in bind:
                    x = float(r[key]) if r[key] != "" else float("nan")
                    if np.isfinite(x):
                        per_ep.setdefault(r["episode"], []).append(x)
                row[f"{key}_lo"], row[f"{key}_hi"] = _episode_bootstrap({e: np.asarray(x) for e, x in per_ep.items()})
                v = _values(bind, key)
                row[f"{key}_median"] = float(np.median(v)) if v.size else float("nan")
                nm = balanced(f"{key}_norm", f"{key}_norm", scored)
                row[f"{key}_norm_median"] = float(np.median(nm)) if nm.size else float("nan")
                row[f"{key}_all_mean"] = _mean(_values(rs, key))
                balanced(f"{key}_all", f"{key}_all", eps)
        row["novis_mean"] = _mean(_values(bind, "novis"))
        balanced("novis", "novis", scored)
        row["novis_exec_mean"] = _mean(_values(bind, "novis_exec"))
        out.append(row)
    return out
