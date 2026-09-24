"""Loaders and statistics for the information-horizon experiment (paper/info_horizon).

Per step and probe layer L, both models log the current frame's frame-level
``keep_one = KL(P_only_current || P_no_visual)`` and ``leave_one = KL(P_full || P_current_hidden)``
with the frame's visual tokens hidden from decoder layer L on. LongNav-R1 rows come from
``rollout/*/*/sequence.json`` (``sup/horizon_*``, written by ``vlm.horizon_probe_layers``);
Uni-NaVid rows from ``horizon/<label>.jsonl`` (``--layer-probe``). Pure numpy/stdlib so both
conda envs can import it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from longnav.analysis.influence import _episode_bootstrap, _r1_sequences

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
