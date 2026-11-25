"""Loaders and statistics for the information-horizon experiment (paper/info_horizon).

Per step and probe layer L, both models log the current frame's frame-level
``keep_one = KL(P_only_current || P_no_visual)`` and ``leave_one = KL(P_full || P_current_hidden)``
with the frame's visual tokens hidden from decoder layer L on. LongNav-R1 rows come from
``rollout/*/*/sequence.json`` (``sup/horizon_*``, written by ``vlm.horizon_probe_layers``);
Uni-NaVid rows from ``horizon/<label>.jsonl`` (``--layer-probe``). Pure numpy/stdlib so both
conda envs can import it.

Blind-from-layer KL (2026-11-23), the depth analogue of the recent-window measure:
``Blind_t(L) = KL(P_full || P_every visual token hidden from decoder layer L on)``, expectation
under the full-context decision. ``load_r1_blind`` reads a LongNav-R1 run made with
``vlm.horizon_probe_blind`` (leaky decision replay, 4 actions), ``load_lightnav_blind`` a
LightNav-0 run made with ``eval_influence.py --measure blind_layer`` (full re-prefill, 256
``<act_l0>`` codes and their six executed-action bins); both return one row per
(episode, step, layer) with the same columns (``BLIND_FIELDS``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from longnav.analysis.influence import (ACTIONS, _action_name, _argmax_action, _episode_bootstrap, _nan, _r1_sequences,
                                        entropy, kl_from_logp, kl_from_probs)
from longnav.analysis.results_io import load_lightnav_run, load_r1_run

ROW_FIELDS = ["model", "episode", "step", "n_frames", "layer", "keep_one", "leave_one", "n_cur_tokens"]


def load_r1_horizon(run_dir: Path | str, model: str = "LongNav-R1") -> tuple[list[dict], list[int]]:
    """(rows, layers): one row per (episode, step, layer) from a LongNav horizon run."""
    rows, layers = [], None
    for label, d in _r1_sequences(Path(run_dir)):
        keep, leave = d.get("sup/horizon_keep_one"), d.get("sup/horizon_leave_one")
        if not keep:
            continue
        ls = d.get("sup/horizon_layers") or []
        vis_new = d.get("kv_visual_new") or []
        for t, (k_row, l_row) in enumerate(zip(keep, leave)):
            if not k_row:  # malformed / skipped step
                continue
            step_layers = [int(x) for x in ls[t]] if ls and t < len(ls) and ls[t] else None
            if layers is None and step_layers:
                layers = step_layers
            n_cur = int(vis_new[t]) if t < len(vis_new) else -1
            for j, L in enumerate(step_layers or layers or []):
                rows.append({"model": model, "episode": label, "step": t, "n_frames": t + 1, "layer": int(L),
                             "keep_one": float(k_row[j]), "leave_one": float(l_row[j]), "n_cur_tokens": n_cur})
    return rows, list(layers or [])


def load_uninavid_horizon(run_dir: Path | str, model: str = "Uni-NaVid") -> tuple[list[dict], list[int]]:
    rows, layers = [], None
    for path in sorted((Path(run_dir) / "horizon").glob("*.jsonl")):
        label = path.stem
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if layers is None:
                layers = [int(x) for x in r["layers"]]
            for j, L in enumerate(r["layers"]):
                rows.append({"model": model, "episode": label, "step": int(r["step"]), "n_frames": int(r["n_frames"]),
                             "layer": int(L), "keep_one": float(r["keep_one"][j]),
                             "leave_one": float(r["leave_one"][j]), "n_cur_tokens": int(r["n_current_tokens"])})
    return rows, list(layers or [])


def layer_curve(rows: list[dict], key: str) -> list[dict]:
    """Per layer: pooled mean over scored steps, episode-cluster bootstrap CI, balanced mean, n."""
    by_layer: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["n_cur_tokens"] == 0:  # nothing to hide at this step: vacuous, skip
            continue
        v = float(r[key])
        if not np.isfinite(v):
            continue
        by_layer.setdefault(r["layer"], {}).setdefault(r["episode"], []).append(v)
    out = []
    for L in sorted(by_layer):
        per_ep = {e: np.asarray(v) for e, v in by_layer[L].items()}
        pooled = np.concatenate(list(per_ep.values()))
        lo, hi = _episode_bootstrap(per_ep)
        ep_means = np.array([v.mean() for v in per_ep.values()])
        out.append({"layer": L, "n_steps": int(pooled.size), "n_episodes": len(per_ep),
                    f"{key}_mean": float(pooled.mean()), f"{key}_lo": lo, f"{key}_hi": hi,
                    f"{key}_median": float(np.median(pooled)), f"{key}_balanced_mean": float(ep_means.mean())})
    return out



# ------------------------------------------------------------------ blind-from-layer KL
BLIND_FIELDS = [
    "model", "episode", "step", "T", "success", "layer", "n_layers",
    "kl",          # KL(full || no visual from L on), the model's own action space (4 actions / 256 codes)
    "kl_exec",     # the same over the executed-action classes (LongNav-R1: identical; LightNav-0: 6 bins)
    "flip",        # argmax moved off the full decision's (action / code)
    "flip_exec",   # executed-action class moved (LongNav-R1: identical; LightNav-0: argmax of the 6-bin fold)
    "argmax_full", "argmax_blind", "exec_full", "exec_blind", "chosen_action",
    "entropy_full", "entropy_blind", "kl_floor",
    # LongNav-R1 only (the probe's other two hidden sets, same layer): keep-one-in, leave-one-out and
    # KL(full || only the current frame visible from L on), i.e. the history hidden from L on
    "keep_one", "leave_one", "kl_only_cur",
    "n_cur_tokens", "n_visual_tokens", "p_mass", "probe_seconds",
]


def load_r1_blind(run_dir: Path | str, model: str = "LongNav-R1", n_layers: int = 28) -> tuple[list[dict], list[int]]:
    """(rows, layers) from a LongNav-R1 run with ``vlm.horizon_probe_blind``. ``kl`` is the logged
    value; the flips, entropies and ``kl_only_cur`` come from the logged per-layer log-probs.
    ``kl_floor`` = KL(live decision || replayed full decision), the replay's noise floor."""
    run_dir = Path(run_dir)
    results = load_r1_run(run_dir)
    rows, layers = [], None
    for label, d in _r1_sequences(run_dir):
        blind = d.get("sup/horizon_blind")
        if not blind:
            continue
        ls, base = d.get("sup/horizon_layers") or [], d.get("sup/horizon_base_logp") or []
        lp_b, lp_o = d.get("sup/horizon_logp_blind") or [], d.get("sup/horizon_logp_only_cur") or []
        keep, leave = d.get("sup/horizon_keep_one") or [], d.get("sup/horizon_leave_one") or []
        vis_new, vis_tot = d.get("kv_visual_new") or [], d.get("kv_visual_total") or []
        live = d.get("sup/action_probs") or []
        actions = [_action_name(a) for a in (d.get("action_history") or [])]
        T, success = len(blind), results.get(label, {}).get("success")
        for t, row in enumerate(blind):
            if not row:  # malformed / skipped step
                continue
            step_layers = [int(x) for x in ls[t]]
            if layers is None:
                layers = step_layers
            full = base[t]
            arg_full, ent_full = _argmax_action(full), entropy(full)
            floor = kl_from_probs(live[t], np.exp(full)) if t < len(live) else float("nan")
            for j, L in enumerate(step_layers):
                arg_b = _argmax_action(lp_b[t][j])
                rows.append({
                    "model": model, "episode": label, "step": t, "T": T, "success": success, "layer": L,
                    "n_layers": n_layers, "kl": float(row[j]), "kl_exec": float(row[j]),
                    "flip": int(arg_b != arg_full), "flip_exec": int(arg_b != arg_full),
                    "argmax_full": arg_full, "argmax_blind": arg_b, "exec_full": arg_full, "exec_blind": arg_b,
                    "chosen_action": actions[t] if t < len(actions) else "",
                    "entropy_full": ent_full, "entropy_blind": entropy(lp_b[t][j]), "kl_floor": floor,
                    "keep_one": float(keep[t][j]), "leave_one": float(leave[t][j]),
                    "kl_only_cur": max(kl_from_logp(full, lp_o[t][j]), 0.0),
                    "n_cur_tokens": int(vis_new[t]) if t < len(vis_new) else -1,
                    "n_visual_tokens": int(vis_tot[t]) if t < len(vis_tot) else -1,
                    "p_mass": float("nan"), "probe_seconds": float("nan"),
                })
    return rows, list(layers or [])


def load_lightnav_blind(run_dir: Path | str, model: str = "LightNav-0") -> tuple[list[dict], list[int]]:
    """(rows, layers) from a LightNav-0 run made with ``eval_influence.py --measure blind_layer``.
    ``kl`` / ``flip`` are over the 256 ``<act_l0>`` codes, ``kl_exec`` / ``flip_exec`` over the six
    executed-action bins (argmax of the folded distribution, as the recent-window loader);
    ``chosen_action`` = the bin of the command actually executed. A probe whose generate ended
    before the action position has NaN values. ``kl_floor`` = the replay with nothing hidden."""
    run_dir = Path(run_dir)
    results = load_lightnav_run(run_dir)
    rows, layers = [], None
    for path in sorted((run_dir / "influence").glob("*.jsonl")):
        label = path.stem
        success, T = results.get(label, {}).get("success"), results.get(label, {}).get("n_steps")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "blind_layer" not in rec:
                raise ValueError(f"{path}: not a blind_layer record (run eval_influence.py --measure blind_layer)")
            if layers is None:
                layers = [int(x) for x in rec["layers"]]
            full = rec["full"]
            has_full = full.get("code") is not None
            for pr in rec["blind_layer"]:
                ok = has_full and pr.get("code") is not None
                rows.append({
                    "model": model, "episode": label, "step": int(rec["step"]), "T": T, "success": success,
                    "layer": int(pr["layer"]), "n_layers": int(rec["n_layers"]),
                    "kl": _nan(pr.get("kl")), "kl_exec": _nan(pr.get("kl_bin")),
                    "flip": int(pr["code"] != full["code"]) if ok else "",
                    "flip_exec": int(pr["bin_argmax"] != full["bin_argmax"]) if ok else "",
                    "argmax_full": full.get("code", ""), "argmax_blind": pr.get("code", ""),
                    "exec_full": full.get("bin_argmax", ""), "exec_blind": pr.get("bin_argmax", ""),
                    "chosen_action": full.get("exec_bin") or "",
                    "entropy_full": _nan(full.get("entropy")), "entropy_blind": _nan(pr.get("entropy")),
                    "kl_floor": _nan((rec.get("replay") or {}).get("kl")),
                    "keep_one": float("nan"), "leave_one": float("nan"), "kl_only_cur": float("nan"),
                    "n_cur_tokens": int(rec.get("n_current_tokens", -1)), "n_visual_tokens": int(rec.get("n_visual", -1)),
                    "p_mass": _nan(pr.get("p_mass")), "probe_seconds": _nan(rec.get("probe_seconds")),
                })
    return rows, list(layers or [])


BLIND_KEYS = ("kl", "kl_exec", "flip", "flip_exec", "keep_one", "leave_one", "kl_only_cur")


def _values(rows: list[dict], key: str) -> np.ndarray:
    vals = np.array([float(r[key]) if r[key] != "" else np.nan for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def blind_episode_table(rows: list[dict]) -> list[dict]:
    """One row per (model, episode, layer): the episode's mean of every measure over its scored
    steps, ``kl_median``, and ``kl_norm`` / ``kl_exec_norm`` = that mean over the episode's own
    mean at the shallowest probed layer (KL(full || no visual) when that layer is 0)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["episode"], r["layer"]), []).append(r)
    first = {}
    for (model, _, layer) in groups:
        first[model] = min(first.get(model, layer), layer)
    out = []
    for (model, ep, layer), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        row = {"model": model, "episode": ep, "layer": layer, "n_layers": rs[0]["n_layers"], "T": rs[0]["T"],
               "success": rs[0]["success"], "n_steps": int(_values(rs, "kl").size)}
        for key in BLIND_KEYS:
            v = _values(rs, key)
            row[key] = float(v.mean()) if v.size else float("nan")
        v = _values(rs, "kl")
        row["kl_median"] = float(np.median(v)) if v.size else float("nan")
        ref = groups[(model, ep, first[model])]
        for key in ("kl", "kl_exec"):
            denom = _values(ref, key)
            row[f"{key}_norm"] = row[key] / float(denom.mean()) if denom.size and denom.mean() > 0 else float("nan")
        out.append(row)
    return out


def blind_curve(rows: list[dict]) -> list[dict]:
    """One row per (model, layer). Per measure: pooled mean over scored steps (``*_mean``, with an
    episode-cluster bootstrap CI for the two KLs), the episode-balanced mean and its SEM across
    episodes; ``kl_median`` over steps; ``kl_norm*`` = the balanced mean (and SEM, median, quartiles
    across episodes) of each episode's curve divided by its own value at the shallowest layer."""
    episodes = blind_episode_table(rows)
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["model"], r["layer"]), []).append(r)
    out = []
    for (model, layer), rs in sorted(by.items()):
        eps = [e for e in episodes if e["model"] == model and e["layer"] == layer and e["n_steps"] > 0]
        row = {"model": model, "layer": layer, "n_layers": rs[0]["n_layers"],
               "layer_frac": layer / rs[0]["n_layers"], "n_steps": int(_values(rs, "kl").size), "n_episodes": len(eps)}
        for key in BLIND_KEYS:
            v = _values(rs, key)
            em = np.array([e[key] for e in eps], dtype=float)
            em = em[np.isfinite(em)]
            row[f"{key}_mean"] = float(v.mean()) if v.size else float("nan")
            row[f"{key}_balanced"] = float(em.mean()) if em.size else float("nan")
            row[f"{key}_sem"] = float(em.std(ddof=1) / np.sqrt(em.size)) if em.size > 1 else float("nan")
            if key in ("kl", "kl_exec"):
                per_ep: dict[str, list[float]] = {}
                for r in rs:
                    x = float(r[key]) if r[key] != "" else float("nan")
                    if np.isfinite(x):
                        per_ep.setdefault(r["episode"], []).append(x)
                row[f"{key}_lo"], row[f"{key}_hi"] = _episode_bootstrap({e: np.asarray(x) for e, x in per_ep.items()})
                row[f"{key}_median"] = float(np.median(v)) if v.size else float("nan")
                nm = np.array([e[f"{key}_norm"] for e in eps], dtype=float)
                nm = nm[np.isfinite(nm)]
                row[f"{key}_norm_balanced"] = float(nm.mean()) if nm.size else float("nan")
                row[f"{key}_norm_sem"] = float(nm.std(ddof=1) / np.sqrt(nm.size)) if nm.size > 1 else float("nan")
                for name, q in (("q25", 25), ("median", 50), ("q75", 75)):
                    row[f"{key}_norm_{name}"] = float(np.percentile(nm, q)) if nm.size else float("nan")
        out.append(row)
    return out
