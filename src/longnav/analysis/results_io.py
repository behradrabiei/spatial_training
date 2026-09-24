"""Load per-episode results from LongNav eval runs and Uni-NaVid baseline runs.

Both loaders return ``{episode_label: row}`` with a common core schema so downstream code
never branches on the model:

    episode_label, success (0/1), spl, soft_spl, distance_to_goal, n_steps, duration

LongNav rows keep every original key (``sup/*`` metrics, artifact paths, ...); Uni-NaVid rows
add ``inference_calls`` and ``peak_mem_GB``.

CLI, used by the run scripts to skip finished arms:

    python -m longnav.analysis.results_io --complete <run_dir> --manifest <labels.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

Row = dict


def load_manifest(path: Path | str) -> list[str]:
    """Episode labels of a split, in file order."""
    labels = json.loads(Path(path).read_text())
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"{path}: expected a JSON list of episode_label strings")
    return labels


def load_r1_run(run_dir: Path | str) -> dict[str, Row]:
    """One row per episode from ``<run_dir>/rollout/results_*`` (JSONL, one file per sim).

    Resumed runs can log an episode twice; the first record wins, as in every plot script
    this replaces, and a warning names the duplicate so it is never silent.
    """
    run_dir = Path(run_dir)
    rows: dict[str, Row] = {}
    for path in sorted((run_dir / "rollout").glob("results_*")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = row["episode_label"]
                if label in rows:
                    warnings.warn(f"{run_dir.name}: duplicate episode {label} in {path.name}; keeping first")
                    continue
                row.setdefault("duration", float("nan"))
                rows[label] = row
    return rows


def load_uninavid_run(run_dir: Path | str) -> dict[str, Row]:
    """One row per episode from ``<run_dir>/episodes/<label>.json`` (the harness's per-episode
    record; ``metrics`` holds the final habitat measures)."""
    run_dir = Path(run_dir)
    rows: dict[str, Row] = {}
    for path in sorted((run_dir / "episodes").glob("*.json")):
        rec = json.loads(path.read_text())
        m = rec["metrics"]
        label = rec["episode_label"]
        rows[label] = {
            "episode_label": label,
            "episode_id": rec.get("episode_id"),
            "scene_id": rec.get("scene_id"),
            "goal": rec.get("goal"),
            "success": float(m["success"]),
            "spl": float(m["spl"]),
            "soft_spl": float(m.get("soft_spl", float("nan"))),
            "distance_to_goal": float(m.get("distance_to_goal", float("nan"))),
            "n_steps": int(rec["steps"]),
            "duration": float(rec.get("elapsed_seconds", float("nan"))),
            "inference_calls": rec.get("inference_calls"),
            "peak_mem_GB": rec["peak_cuda_bytes"] / 1024 ** 3 if rec.get("peak_cuda_bytes") else float("nan"),
        }
    return rows


def load_uninavid_manifest(run_dir: Path | str) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def load_run(run_dir: Path | str) -> dict[str, Row]:
    """Dispatch on layout: LongNav runs have ``rollout/``, Uni-NaVid runs ``episodes/``."""
    run_dir = Path(run_dir)
    if (run_dir / "rollout").is_dir():
        return load_r1_run(run_dir)
    if (run_dir / "episodes").is_dir():
        return load_uninavid_run(run_dir)
    return {}


def split_status(rows: dict[str, Row], labels: list[str]) -> tuple[list[str], list[str]]:
    """(missing, extra) labels of ``rows`` relative to the manifest."""
    want, have = set(labels), set(rows)
    return sorted(want - have), sorted(have - want)


def assert_split(rows: dict[str, Row], labels: list[str], name: str) -> None:
    """A paper arm is complete or it is not used: partial arms would silently change the
    denominator the other arms are compared against."""
    missing, extra = split_status(rows, labels)
    if missing or extra:
        raise ValueError(f"{name}: {len(rows)} episodes vs {len(labels)} in the manifest "
                         f"(missing {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}; "
                         f"extra {len(extra)}: {extra[:5]}{'...' if len(extra) > 5 else ''})")


def is_complete(run_dir: Path | str, labels: list[str]) -> bool:
    missing, _ = split_status(load_run(run_dir), labels)
    return not missing


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete", metavar="RUN_DIR", type=Path,
                    help="exit 0 if RUN_DIR holds every episode of --manifest, 1 otherwise")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    labels = load_manifest(args.manifest)
    if args.complete is not None:
        rows = load_run(args.complete)
        missing, _ = split_status(rows, labels)
        print(f"{args.complete}: {len(rows)}/{len(labels)} episodes" + (f", {len(missing)} missing" if missing else ""))
        sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    _main()
