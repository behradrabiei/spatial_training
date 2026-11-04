"""Load per-episode results from LongNav eval runs and Uni-NaVid baseline runs.

Both loaders return ``{episode_label: row}`` with a common core schema so downstream code
never branches on the model:

    episode_label, success (0/1), spl, soft_spl, distance_to_goal, n_steps, duration

LongNav rows keep every original key (``sup/*`` metrics, artifact paths, ...); Uni-NaVid rows
add ``inference_calls`` and ``peak_mem_GB``.

LightNav-0 runs share the Uni-NaVid layout but are loaded with ``load_lightnav_run``: upstream's
evaluator forces an explicit STOP on the last step of the budget and lets Habitat score the final
pose, so an agent parked inside the success radius when the 500-step cap hits counts as a success
there. LongNav-R1 and Uni-NaVid cannot earn success at the cap (their episodes are truncated
without a stop call), so ``load_lightnav_run`` re-scores every ``forced_stop`` episode as a
failure (``success`` = ``spl`` = 0; the harness values stay in ``success_raw`` / ``spl_raw``).

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
        if "forced_stop" in rec:  # LightNav-0 records only (see load_lightnav_run)
            rows[label]["forced_stop"] = bool(rec["forced_stop"])
    return rows


def load_lightnav_run(run_dir: Path | str) -> dict[str, Row]:
    """LightNav-0 rows (``episodes/`` layout written by ``baselines/lightnav_hm3d/convert_results.py``)
    with the step-cap rule of the other two models: an episode the harness ended by a forced
    STOP at the budget is a failure, whatever Habitat measured at that final pose.

    Every LightNav record carries ``forced_stop``; a run without the flag is not a LightNav run.
    ``success_raw`` / ``spl_raw`` keep the harness's own scores, ``rescored`` marks changed rows.
    """
    rows = load_uninavid_run(run_dir)
    for label, row in rows.items():
        if "forced_stop" not in row:
            raise ValueError(f"{Path(run_dir).name}: episode {label} has no forced_stop flag; "
                             "load_lightnav_run is for LightNav-0 runs only")
        row["success_raw"], row["spl_raw"] = row["success"], row["spl"]
        row["rescored"] = bool(row["forced_stop"] and (row["success"] > 0 or row["spl"] > 0))
        if row["forced_stop"]:
            row["success"], row["spl"] = 0.0, 0.0
    return rows


def load_uninavid_manifest(run_dir: Path | str) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


NAVVERSE_STEP_CAP = 500  # --longnav_max_steps of the NavVerse arms (paper/ctx_ablation/run_navverse.sh)

# Outcome taxonomy of a NavVerse episode (paper/ctx_ablation failure-mode figure), mutually
# exclusive in this priority order; "near goal" is NavVerse's oracle_success (the agent was
# within the 1.6 m success radius at some step). "other" catches anything else (e.g. the
# simulator's time_out, which never occurred in the 2026-10-20/22 sweep).
NAVVERSE_FAILURE_MODES = ("success", "fell", "cap_never_near_goal", "cap_near_goal_no_stop", "wrong_stop")


def navverse_failure_mode(row: Row) -> str:
    """One of ``NAVVERSE_FAILURE_MODES`` (or ``"other"``) for a re-scored ``load_navverse_run`` row."""
    if row["success"] > 0:
        return "success"
    if row.get("termination_reason") == "bad_orientation":  # the robot fell over
        return "fell"
    near_goal = float(row.get("oracle_success", 0) or 0) > 0
    if row.get("forced_stop"):  # the client's own STOP at the step cap
        return "cap_near_goal_no_stop" if near_goal else "cap_never_near_goal"
    if row.get("termination_reason") == "stop_called":  # the model stopped early, outside the radius
        return "wrong_stop"
    return "other"


def _navverse_result_files(run_dir: Path) -> list[Path]:
    """NavVerse writes ``dump/<exp_name>/result.json``; an arm directory is mounted as ``dump``."""
    return sorted(run_dir.glob("*/result.json")) + ([run_dir / "result.json"] if (run_dir / "result.json").is_file() else [])


def load_navverse_run(run_dir: Path | str, step_cap: int = NAVVERSE_STEP_CAP) -> dict[str, Row]:
    """LongNav-R1 rows from a NavVerse benchmark arm (``<run_dir>/<exp_name>/result.json``, the
    evaluator's per-episode records with the benchmark's own measures: success radius 1.6 m,
    SPL over the 3D path length).

    NavVerse's LongNav client returns STOP itself when the step cap hits and the evaluator then
    scores the final pose, exactly LightNav-0's forced stop; those episodes are re-scored as
    failures (``success`` = ``spl`` = 0, harness values kept in ``success_raw`` / ``spl_raw``,
    ``rescored`` marks changed rows) so the cap means the same thing as in the Habitat arms.
    ``termination_reason`` is kept (``stop_called``, ``time_out`` at 300 s of simulated time, ...),
    ``oracle_success`` (ever within the success radius) is carried over, and ``failure_mode`` is
    the episode's outcome class (``navverse_failure_mode``).
    The server's per-step log (``<run_dir>/server_steps.jsonl``, written by
    ``longnav.scripts.serve --step-log``) adds ``vlm_latency_ms``, ``kv_len_mean/max`` and
    ``peak_mem_GB`` per episode when present.
    """
    run_dir = Path(run_dir)
    rows: dict[str, Row] = {}
    for path in _navverse_result_files(run_dir):
        for rec in json.loads(path.read_text()):
            label = rec["episode_label"]
            if label in rows:
                warnings.warn(f"{run_dir.name}: duplicate episode {label} in {path}; keeping first")
                continue
            steps = int(rec["step"])
            reason = rec.get("termination_reason")
            # The client returns STOP itself once l_step reaches the cap and counts that action,
            # so a forced stop records step_cap + 1; a model STOP as the cap-th action records step_cap.
            forced = bool(steps > step_cap and reason == "stop_called")
            row = {
                "episode_label": label,
                "episode_id": rec.get("episode_id"),
                "scene_id": rec.get("scene_id"),
                "goal": rec.get("instruction"),
                "success": float(rec["success"]),
                "spl": float(rec["spl"]),
                "soft_spl": float(rec.get("soft_spl", float("nan"))),
                "distance_to_goal": float(rec.get("distance_to_goal", float("nan"))),
                "oracle_success": float(rec.get("oracle_success", float("nan"))),
                # The client's own STOP at the cap is counted by NavVerse (step_cap + 1); the
                # Habitat arms record the cap itself, so clamp for a comparable steps column.
                "n_steps": min(steps, step_cap) if forced else steps,
                "n_steps_raw": steps,
                "duration": float(rec.get("episode_duration", rec.get("sim_duration", float("nan")))),
                "sim_duration": float(rec.get("sim_duration", float("nan"))),
                "termination_reason": reason,
                "forced_stop": forced,
                "success_raw": float(rec["success"]),
                "spl_raw": float(rec["spl"]),
            }
            row["rescored"] = bool(forced and (row["success"] > 0 or row["spl"] > 0))
            if forced:
                row["success"], row["spl"] = 0.0, 0.0
            row["failure_mode"] = navverse_failure_mode(row)
            rows[label] = row
    step_logs = sorted(run_dir.glob("server_steps*.jsonl"))  # one per server (worker)
    if step_logs and rows:
        _merge_navverse_step_logs(rows, step_logs, run_dir)
    return rows


NAVVERSE_ACTION_IDS = {"STOP": 0, "FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3,  # --longnav_action_map (R1 arms)
                       "TURN_LEFT_SMALL": 2, "TURN_RIGHT_SMALL": 3}  # the LightNav-0 arms' 15 deg turns, same ids


def _navverse_action_ids(run_dir: Path, label: str) -> list[int] | None:
    """The evaluator's recorded action sequence for an episode (``<exp>/data/<label>.json``)."""
    paths = list(run_dir.glob(f"*/data/{label}.json"))
    if not paths:
        return None
    history = json.loads(paths[0].read_text()).get("action_history") or {}
    try:
        return [NAVVERSE_ACTION_IDS[history[k]["action"]] for k in sorted(history, key=int)]
    except (KeyError, TypeError, ValueError):
        return None


def _merge_navverse_step_logs(rows: dict[str, Row], step_logs: list[Path], run_dir: Path) -> None:
    """Attach per-episode VLM latency / cache / memory from the servers' step logs.

    A server numbers episodes by goal announcements in arrival order and never learns the
    episode label, so server episodes are matched to results by their action sequence: the
    evaluator's ``data/<label>.json`` action history must start with the server's action ids
    and be at most one action longer (the client's own forced STOP at the step cap is never
    sent to the server). With a single server and no action histories, matching falls back to
    completion order when the counts agree. Unmatched episodes keep NaN diagnostics.
    """
    server_eps: list[list[dict]] = []
    for path in step_logs:
        # The log is appended across resumes and each server process numbers its episodes from
        # 1, so a new server episode starts wherever the step counter restarts at 1.
        current: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec.get("step", 1)) == 1 and current:
                    server_eps.append(current)
                    current = []
                current.append(rec)
        if current:
            server_eps.append(current)
    by_actions: dict[tuple, list[int]] = {}
    for i, recs in enumerate(server_eps):
        by_actions.setdefault(tuple(int(r["action_id"]) for r in recs), []).append(i)

    def attach(row, recs):
        lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
        kv = [r["kv_len"] for r in recs if r.get("kv_len") is not None]
        mem = [r["peak_mem_GB"] for r in recs if r.get("peak_mem_GB") is not None]
        row["vlm_latency_ms"] = float(sum(lat) / len(lat)) if lat else float("nan")
        row["kv_len_mean"] = float(sum(kv) / len(kv)) if kv else float("nan")
        row["kv_len_max"] = float(max(kv)) if kv else float("nan")
        row["peak_mem_GB"] = float(max(mem)) if mem else float("nan")
        row["inference_calls"] = len(recs)

    for row in rows.values():
        row.setdefault("vlm_latency_ms", float("nan"))
        row.setdefault("kv_len_mean", float("nan"))
        row.setdefault("kv_len_max", float("nan"))
        row.setdefault("peak_mem_GB", float("nan"))
    used: set[int] = set()
    unmatched = []
    for label, row in rows.items():
        ids = _navverse_action_ids(run_dir, label)
        if ids is None:
            unmatched.append(label)
            continue
        candidates = [i for n in (len(ids), len(ids) - 1) if n > 0
                      for i in by_actions.get(tuple(ids[:n]), []) if i not in used]
        if not candidates:
            unmatched.append(label)
            continue
        # Identical sequences (e.g. two episodes that turn in place to the cap) are tied; results
        # are appended in completion order, so the earliest unused server episode is the match.
        used.add(candidates[0])
        attach(row, server_eps[candidates[0]])
    if unmatched and len(step_logs) == 1 and len(server_eps) == len(rows) and not used:
        for row, recs in zip(rows.values(), server_eps):  # legacy: single server, completion order
            if row["goal"] is not None and recs[0].get("goal") not in (None, row["goal"]):
                warnings.warn(f"{run_dir.name}: goal mismatch while merging {step_logs[0].name}; step log not merged")
                return
        for row, recs in zip(rows.values(), server_eps):
            attach(row, recs)
    elif unmatched:
        warnings.warn(f"{run_dir.name}: {len(unmatched)} episode(s) without a matching server step log "
                      f"(e.g. {unmatched[:3]}); their kv/latency/memory are NaN")


def load_run(run_dir: Path | str) -> dict[str, Row]:
    """Dispatch on layout: LongNav runs have ``rollout/``, Uni-NaVid runs ``episodes/``,
    NavVerse arms ``<exp_name>/result.json``."""
    run_dir = Path(run_dir)
    if (run_dir / "rollout").is_dir():
        return load_r1_run(run_dir)
    if (run_dir / "episodes").is_dir():
        return load_uninavid_run(run_dir)
    if _navverse_result_files(run_dir):
        return load_navverse_run(run_dir)
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
