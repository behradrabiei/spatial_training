"""Convert LightNav's ``results.jsonl`` into the Uni-NaVid run layout.

Writes ``<run_dir>/episodes/<label>.json`` (one record per episode with ``metrics``) and
``<run_dir>/summary.json`` so ``longnav.analysis.results_io.load_uninavid_run`` and the
paper aggregators read the run like any Uni-NaVid run. Labels come from
``"<scene>_<raw_episode_id>"``: the subset generator rewrote each episode id to its index
in the source shard, and LightNav's loader keeps that id as ``raw_episode_id`` before
renumbering (``habitat_episode_id`` is the renumbered one and must not be used).

When the client ran the pruning wrapper (``eval_prune.py``) or the influence wrapper
(``eval_influence.py``), its per-episode records ``raw/<name>/episode_NNN.jsonl`` (plus ``meta.json``)
are moved to ``<run_dir>/<name>/<label>.jsonl``, the Uni-NaVid layout the paper aggregators read.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .make_subset import read_labels, scene_label


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def record_from_result(res: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    raw_id = str(res.get("raw_episode_id") or "")
    if not raw_id:
        raise ValueError(f"result without raw_episode_id: {res.get('episode_id')}")
    scene_id = str(res.get("scene_id", ""))
    label = f"{scene_label(scene_id)}_{raw_id}"
    steps = int(res.get("steps", 0))
    forced = bool(res.get("forced_stop", False))
    metrics: Dict[str, float] = {
        "success": 1.0 if res.get("success") else 0.0,
        "spl": float(res.get("spl", 0.0) or 0.0),
        "soft_spl": float(res.get("soft_spl", 0.0) or 0.0),
    }
    dtg = _finite(res.get("final_distance"))
    if dtg is not None:
        metrics["distance_to_goal"] = dtg
    rec = {
        "episode_label": label,
        "episode_id": raw_id,
        "scene_id": scene_id,
        "goal": str(res.get("object_category", "")),
        "instruction": str(res.get("instruction", "")),
        "steps": steps,
        "inference_calls": max(steps - 1, 0) if forced else steps,
        "termination_reason": str(res.get("termination_reason", "unknown")),
        "forced_stop": forced,
        "oracle_success": bool(res.get("oracle_success", False)),
        "min_distance": _finite(res.get("min_distance")),
        "source": source,
        "metrics": metrics,
    }
    return rec


def move_logs(run_dir: Path, records: Dict[str, Dict[str, Any]], name: str, allow_missing: bool = False) -> int:
    """``raw/<name>/episode_NNN.jsonl`` -> ``<name>/<label>.jsonl`` by the episode id LightNav wrote in
    ``results.jsonl``; ``meta.json`` moves along. Returns the number of files moved (0 when the
    client wrote no such records). A record without a results row (client died mid-episode) is an
    error unless ``allow_missing``, in which case it stays under raw/<name>/."""
    src = run_dir / "raw" / name
    if not src.is_dir():
        return 0
    by_id = {str(rec["source"].get("lightnav_episode_id")): label for label, rec in records.items()}
    dst = run_dir / name
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in sorted(src.glob("episode_*.jsonl")):
        label = by_id.get(path.stem)
        if label is None:
            if allow_missing:
                continue
            raise ValueError(f"{path}: no results row for {path.stem}")
        path.replace(dst / f"{label}.jsonl")
        moved += 1
    meta = src / "meta.json"
    if meta.is_file():
        meta.replace(dst / "meta.json")
    if not any(src.iterdir()):
        src.rmdir()
    return moved


def move_prune_logs(run_dir: Path, records: Dict[str, Dict[str, Any]], allow_missing: bool = False) -> int:
    return move_logs(run_dir, records, "prune", allow_missing)


def convert(run_dir: Path, labels_path: Path, results_files: List[Path], expected_count: int = 100,
            allow_missing: bool = False) -> Dict[str, Any]:
    labels = read_labels(labels_path, expected_count)
    wanted = set(labels)
    records: Dict[str, Dict[str, Any]] = {}
    for path in results_files:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                res = json.loads(line)
                src = {"results_jsonl": str(path), "line": lineno,
                       "lightnav_episode_id": res.get("episode_id"),
                       "habitat_episode_id": res.get("habitat_episode_id")}
                rec = record_from_result(res, src)
                label = rec["episode_label"]
                if label not in wanted:
                    raise ValueError(f"{path}:{lineno}: label {label} is not in {labels_path}")
                if label in records:
                    raise ValueError(f"{path}:{lineno}: duplicate label {label}")
                records[label] = rec

    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for label, rec in records.items():
        tmp = episodes_dir / f"{label}.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        tmp.replace(episodes_dir / f"{label}.json")

    missing = [l for l in labels if l not in records]
    done = [records[l] for l in labels if l in records]
    for name in ("prune", "influence"):
        move_logs(run_dir, records, name, allow_missing)

    def mean(key: str, where: str = "metrics") -> float | None:
        vals = [r[where][key] if where else r[key] for r in done
                if (r[where] if where else r).get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    lightnav_summary = None
    for path in results_files:
        s = path.with_name("summary.json")
        if s.is_file():
            lightnav_summary = json.loads(s.read_text()).get("metrics")
    summary = {
        "expected_episodes": len(labels),
        "completed_episodes": len(done),
        "missing_episodes": missing,
        "extra_episodes": [],
        "success_rate": mean("success"),
        "spl": mean("spl"),
        "soft_spl": mean("soft_spl"),
        "average_final_distance": mean("distance_to_goal"),
        "average_steps": mean("steps", ""),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in done)),
        "forced_stops": sum(1 for r in done if r["forced_stop"]),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "results_files": [str(p) for p in results_files],
        "lightnav_summary": lightnav_summary,
    }
    if lightnav_summary and summary["success_rate"] is not None and not missing:
        sr_pct = float(lightnav_summary.get("SR_success_rate_pct", -1))
        if abs(sr_pct - 100.0 * summary["success_rate"]) > 0.01:
            raise ValueError(f"SR mismatch: LightNav summary {sr_pct} vs converted {100 * summary['success_rate']}")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} episodes missing (first: {missing[:5]}); summary written anyway")
    return summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--labels", type=Path, default=Path("dump/hm3d_v2_100_labels.json"))
    p.add_argument("--results", type=Path, nargs="*", default=None,
                   help="results.jsonl files (default: <run_dir>/raw/results.jsonl)")
    p.add_argument("--expected-count", type=int, default=100)
    p.add_argument("--allow-missing", action="store_true")
    a = p.parse_args(argv)
    files = a.results or [a.run_dir / "raw" / "results.jsonl"]
    s = convert(a.run_dir.resolve(), a.labels.resolve(), [f.resolve() for f in files],
                a.expected_count, a.allow_missing)
    print(json.dumps({k: s[k] for k in ("completed_episodes", "success_rate", "spl", "average_steps",
                                        "termination_reasons", "forced_stops")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
