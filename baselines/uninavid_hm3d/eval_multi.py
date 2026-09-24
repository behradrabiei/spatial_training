"""Evaluate Uni-NaVid on the sequential OneMap multi-object benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from uninavid_hm3d.eval import (
    ACTION_TO_ID,
    UniNaVidPolicy,
    annotate_frame,
    atomic_json,
    episode_label,
    json_value,
    make_config,
    nonnegative_int,
    package_versions,
    positive_int,
    read_labels,
    select_episodes,
    sha256_file,
)


INITIAL_TASK_TEMPLATE = "Find the {goal}."
ALL_GOALS_TASK_TEMPLATE = "Find {goals}, in that order."
NEW_TASK_TEMPLATE = (
    "You have found the previous target. Your new task is to find the {goal}. "
    "Continue navigating from your current position in the same environment. "
    "Use your observation history: if you have already seen the {goal} earlier, "
    "head back to it directly."
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


class SequentialGoalTracker:
    """Pure bookkeeping for one sequential multi-object episode."""

    def __init__(
        self,
        goals: list[str],
        initial_position: np.ndarray,
        initial_distance: float,
        *,
        leg_max_steps: int = 500,
        success_distance: float = 1.0,
        reveal_all_goals: bool = False,
    ) -> None:
        if not goals:
            raise ValueError("A sequential episode must contain at least one goal")
        self.goals = list(goals)
        self.goal_idx = 0
        self.leg_max_steps = leg_max_steps
        self.success_distance = success_distance
        self.last_position = np.asarray(initial_position, dtype=np.float64)
        self.leg_steps = 0
        self.leg_path_length = 0.0
        self.leg_start_distance = float(initial_distance)
        self.legs: list[dict[str, Any]] = []
        self.done_reason: str | None = None
        if reveal_all_goals:
            ordered_goals = ", then ".join(self.goals)
            initial_task = ALL_GOALS_TASK_TEMPLATE.format(goals=ordered_goals)
        else:
            initial_task = INITIAL_TASK_TEMPLATE.format(goal=self.current_goal)
        self.task_history = [initial_task]

    @property
    def current_goal(self) -> str:
        return self.goals[self.goal_idx]

    @property
    def instruction_context(self) -> str:
        return " ".join(self.task_history)

    @property
    def done(self) -> bool:
        return self.done_reason is not None

    def _record_leg(self, result: str) -> None:
        success = result == "success"
        shortest = self.leg_start_distance
        if success and math.isfinite(shortest):
            denominator = max(self.leg_path_length, shortest)
            spl = min(1.0, shortest / denominator) if denominator > 0 else 1.0
        else:
            spl = 0.0
        self.legs.append(
            {
                "category": self.current_goal,
                "result": result,
                "steps": self.leg_steps,
                "path_length": self.leg_path_length,
                "start_distance": shortest if math.isfinite(shortest) else None,
                "spl": spl,
            }
        )

    def apply_move(self, position_after: np.ndarray) -> str:
        if self.done:
            raise RuntimeError("Cannot advance a completed sequential episode")
        position = np.asarray(position_after, dtype=np.float64)
        self.leg_path_length += float(np.linalg.norm(position - self.last_position))
        self.last_position = position
        self.leg_steps += 1
        if self.leg_steps >= self.leg_max_steps:
            self._record_leg("oot")
            self.done_reason = "oot"
            return "oot"
        return "move"

    def apply_stop(self, distance: float, next_goal_distance: float | None = None) -> str:
        if self.done:
            raise RuntimeError("Cannot stop a completed sequential episode")
        self.leg_steps += 1
        success = math.isfinite(distance) and distance < self.success_distance
        if not success:
            self._record_leg("wrong_stop")
            self.done_reason = "wrong_stop"
            return "wrong_stop"

        self._record_leg("success")
        if self.goal_idx == len(self.goals) - 1:
            self.done_reason = "all_success"
            return "all_success"

        if next_goal_distance is None:
            raise ValueError("next_goal_distance is required after an intermediate success")
        self.goal_idx += 1
        self.leg_steps = 0
        self.leg_path_length = 0.0
        self.leg_start_distance = float(next_goal_distance)
        self.task_history.append(NEW_TASK_TEMPLATE.format(goal=self.current_goal))
        return "goal_advanced"

    def finalize_unexpected(self, reason: str = "habitat_done") -> None:
        if not self.done:
            self._record_leg(reason)
            self.done_reason = reason

    def metrics(self) -> dict[str, Any]:
        successes = sum(leg["result"] == "success" for leg in self.legs)
        n_goals = len(self.goals)
        progress = successes / n_goals
        ppl = sum(float(leg["spl"]) for leg in self.legs) / n_goals
        all_success = float(successes == n_goals)
        return {
            "progress": progress,
            "ppl": ppl,
            "all_success": all_success,
            "spl": all_success * ppl,
            "legs_succeeded": successes,
            "goal_sequence": list(self.goals),
            "done_reason": self.done_reason,
            "legs": json_value(self.legs),
        }


def goal_view_positions(env: Any, category: str) -> list[Any]:
    episode = env.current_episode
    key = f"{Path(episode.scene_id).name}_{category}"
    goals = env._dataset.goals_by_category.get(key, [])
    positions = [view.agent_state.position for goal in goals for view in goal.view_points]
    if not positions:
        raise RuntimeError(f"No goal viewpoints for {key}")
    return positions


def geodesic_to_goal(env: Any, category: str) -> float:
    position = env.sim.get_agent_state().position
    return float(env.sim.geodesic_distance(position, goal_view_positions(env, category)))


def validate_sequential_goals(dataset: Any) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for episode in dataset.episodes:
        goals = (episode.info or {}).get("object_goals")
        label = episode_label(episode)
        if not isinstance(goals, list) or not goals:
            invalid.append(label)
            continue
        for category in goals:
            key = f"{Path(episode.scene_id).name}_{category}"
            records = dataset.goals_by_category.get(key, [])
            if not any(goal.view_points for goal in records):
                missing.append(key)
    if invalid:
        raise ValueError(f"Episodes missing object_goals: {invalid}")
    if missing:
        raise ValueError(f"Goals missing viewpoints: {sorted(set(missing))}")


def completed_results(output_dir: Path, labels: list[str]) -> list[dict[str, Any]]:
    results = []
    for label in labels:
        path = output_dir / "episodes" / f"{label}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def build_multi_summary(
    results: list[dict[str, Any]], expected_labels: list[str]
) -> dict[str, Any]:
    completed = {result["episode_label"] for result in results}
    missing = [label for label in expected_labels if label not in completed]
    denominator = len(expected_labels)

    def padded_mean(metric: str) -> float | None:
        if not denominator:
            return None
        return sum(float(result["metrics"][metric]) for result in results) / denominator

    per_leg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = Counter()
    for result in results:
        metrics = result["metrics"]
        failures[metrics["done_reason"]] += 1
        recorded_legs = metrics["legs"]
        for index, category in enumerate(metrics["goal_sequence"]):
            leg = (
                recorded_legs[index]
                if index < len(recorded_legs)
                else {
                    "category": category,
                    "result": "not_attempted",
                    "steps": 0,
                    "path_length": 0.0,
                    "start_distance": None,
                    "spl": 0.0,
                }
            )
            per_leg[index].append(leg)
            per_category[category].append(leg)
    if missing:
        failures["missing"] += len(missing)

    def breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
        attempted = sum(record["result"] != "not_attempted" for record in records)
        return {
            "total": len(records),
            "attempted": attempted,
            "success_rate": (
                sum(record["result"] == "success" for record in records) / len(records)
                if records
                else None
            ),
            "spl": (
                sum(float(record["spl"]) for record in records) / len(records)
                if records
                else None
            ),
        }

    return {
        "expected_episodes": denominator,
        "completed_episodes": len(results),
        "missing_episodes": missing,
        "pr": padded_mean("progress"),
        "ppl": padded_mean("ppl"),
        "sr": padded_mean("all_success"),
        "spl": padded_mean("spl"),
        "per_leg_index": {
            str(index + 1): breakdown(records) for index, records in sorted(per_leg.items())
        },
        "per_category": {
            category: breakdown(records) for category, records in sorted(per_category.items())
        },
        "failure_modes": dict(sorted(failures.items())),
        "total_steps": sum(int(result["steps"]) for result in results),
        "total_inference_calls": sum(int(result["inference_calls"]) for result in results),
        "invalid_output_events": sum(int(result["invalid_output_events"]) for result in results),
        "fallback_stops": sum(int(result["fallback_stops"]) for result in results),
        "elapsed_episode_seconds": sum(float(result["elapsed_seconds"]) for result in results),
    }


def write_multi_manifest(args: argparse.Namespace, labels: list[str]) -> None:
    manifest = {
        "created_at_unix": time.time(),
        "command": sys.argv,
        "labels_path": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "requested_labels": labels,
        "dataset_path": str(args.dataset),
        "scenes_path": str(args.scenes),
        "model_path": str(args.model_path),
        "benchmark": {
            "name": "OneMap sequential multi-object HM3D",
            "goals_per_episode": 3,
            "goal_disclosure": (
                "all_ordered_at_start" if args.reveal_all_goals else "sequential"
            ),
            "task_update": "append_to_instruction_context",
            "action_history_in_prompt": False,
            "preserve_visual_history_on_goal_switch": True,
            "oracle_stopping": False,
            "false_positive_stop_guard": False,
            "false_negative_stop_guard": False,
            "post_goal_stop_veto": 0,
            "leg_max_steps": args.leg_max_steps,
            "success_distance_m": args.success_distance,
        },
        "inference": {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "actions_per_inference": args.actions_per_inference,
            "context_window_previous_frames": args.context_window,
            "visual_tokens_enabled": not args.no_visual_tokens,
            "current_frame_only_64": args.current_frame_only_64,
            "max_episode_steps": args.max_steps,
            "policy_observations": ["rgb"],
        },
        "packages": package_versions(
            [
                "torch",
                "torchvision",
                "transformers",
                "habitat-lab",
                "habitat-sim",
                "numpy",
                "opencv-python",
                "uninavid",
            ]
        ),
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(args.output / "input_labels.json", labels)


def run_episode(
    env: Any,
    policy: UniNaVidPolicy,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    observation = env.reset()
    current_label = episode_label(env.current_episode)
    if current_label != label:
        raise RuntimeError(f"Expected episode {label}, Habitat reset to {current_label}")
    episode = env.current_episode
    goals = list((episode.info or {}).get("object_goals", []))
    if not goals:
        raise RuntimeError(f"Episode {label} has no sequential goals")

    position = np.asarray(env.sim.get_agent_state().position)
    tracker = SequentialGoalTracker(
        goals,
        position,
        geodesic_to_goal(env, goals[0]),
        leg_max_steps=args.leg_max_steps,
        success_distance=args.success_distance,
        reveal_all_goals=args.reveal_all_goals,
    )
    policy.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trace_path = args.output / "traces" / f"{label}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_tmp = trace_path.with_name(f".{trace_path.name}.{os.getpid()}.tmp")
    video_path = args.output / "videos" / f"{label}.mp4"
    video_writer = None
    if args.video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = observation["rgb"].shape[:2]
        video_writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")

    steps = 0
    inference_calls = 0
    invalid_output_events = 0
    fallback_stops = 0
    started = time.perf_counter()
    try:
        with trace_tmp.open("w", encoding="utf-8") as trace:
            while not tracker.done and steps < args.max_steps:
                goal_before = tracker.current_goal
                goal_idx_before = tracker.goal_idx
                distance_before = geodesic_to_goal(env, goal_before)
                prompt_context = tracker.instruction_context
                decision = policy.act(observation["rgb"], prompt_context)
                inference_calls += int(decision["inference"])
                invalid_output_events += int(bool(decision["invalid_tokens"]))
                fallback_stops += int(decision["fallback_stop"])

                if video_writer is not None:
                    video_writer.write(
                        annotate_frame(
                            observation["rgb"],
                            label,
                            goal_before,
                            steps,
                            decision["action"],
                            distance_before,
                        )
                    )

                if decision["action_id"] == ACTION_TO_ID["stop"]:
                    inside_goal = (
                        math.isfinite(distance_before)
                        and distance_before < args.success_distance
                    )
                    next_distance = None
                    if inside_goal and goal_idx_before < len(goals) - 1:
                        next_distance = geodesic_to_goal(env, goals[goal_idx_before + 1])
                    outcome = tracker.apply_stop(distance_before, next_distance)
                    if outcome == "goal_advanced":
                        policy.discard_pending_actions()
                    else:
                        observation = env.step(ACTION_TO_ID["stop"])
                else:
                    observation = env.step(decision["action_id"])
                    position_after = np.asarray(env.sim.get_agent_state().position)
                    outcome = tracker.apply_move(position_after)
                    if env.episode_over and not tracker.done:
                        tracker.finalize_unexpected()
                        outcome = tracker.done_reason

                steps += 1
                record = {
                    "step": steps - 1,
                    "episode_label": label,
                    "goal": goal_before,
                    "goal_idx": goal_idx_before,
                    "instruction_context": prompt_context,
                    "distance_to_goal_before_action": distance_before,
                    "action": decision["action"],
                    "action_id": decision["action_id"],
                    "inference": decision["inference"],
                    "raw_output": decision["raw_output"],
                    "parsed_actions": decision["parsed_actions"],
                    "invalid_tokens": decision["invalid_tokens"],
                    "fallback_stop": decision["fallback_stop"],
                    "inference_seconds": decision["inference_seconds"],
                    "outcome": outcome,
                    "next_goal_idx": tracker.goal_idx,
                }
                trace.write(json.dumps(json_value(record), sort_keys=True, allow_nan=False) + "\n")
                trace.flush()

            if not tracker.done:
                tracker.finalize_unexpected("episode_cap")
        os.replace(trace_tmp, trace_path)
    finally:
        if video_writer is not None:
            video_writer.release()

    elapsed = time.perf_counter() - started
    return {
        "episode_label": label,
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "goal_sequence": goals,
        "task_history": tracker.task_history,
        "final_instruction_context": tracker.instruction_context,
        "steps": steps,
        "inference_calls": inference_calls,
        "invalid_output_events": invalid_output_events,
        "fallback_stops": fallback_stops,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "metrics": tracker.metrics(),
        "trace": str(trace_path),
        "video": str(video_path) if args.video else None,
    }


def parse_args() -> argparse.Namespace:
    baseline_root = Path(__file__).resolve().parents[1]
    project_root = baseline_root.parent
    uninavid_root = baseline_root / "Uni-NaVid"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninavid-root", type=Path, default=uninavid_root)
    parser.add_argument(
        "--config", type=Path, default=project_root / "habitat_configs/objectnav_hm3d_multi.yaml"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=project_root / "src/longnav/conf/episode_jsons/onemap_multi.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/"
            "onemap_multi/val/val.json.gz"
        ),
    )
    parser.add_argument(
        "--scenes", type=Path, default=Path("/home/brabiei/vault/habitat_data/scenes/HM3D/v2")
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=uninavid_root / "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
    )
    parser.add_argument(
        "--output", type=Path, default=baseline_root / "results/onemap_multi_uninavid_sequential"
    )
    parser.add_argument("--expected-count", type=int, default=236)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model-load-only", action="store_true")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--actions-per-inference", type=positive_int, default=2)
    parser.add_argument("--context-window", type=nonnegative_int)
    parser.add_argument("--no-visual-tokens", action="store_true")
    parser.add_argument("--current-frame-only-64", action="store_true")
    parser.add_argument("--max-steps", type=positive_int, default=1600)
    parser.add_argument("--leg-max-steps", type=positive_int, default=500)
    parser.add_argument("--success-distance", type=positive_float, default=1.0)
    parser.add_argument(
        "--reveal-all-goals",
        action="store_true",
        help="Tell Uni-NaVid the complete ordered goal sequence in its initial task",
    )
    args = parser.parse_args()
    if args.current_frame_only_64:
        if args.context_window not in (None, 0):
            parser.error("--current-frame-only-64 is incompatible with a nonzero context window")
        args.context_window = 0
    if args.max_steps < args.leg_max_steps:
        parser.error("--max-steps must be at least --leg-max-steps")
    return args


def main() -> None:
    args = parse_args()
    args.uninavid_root = args.uninavid_root.resolve()
    args.config = args.config.resolve()
    args.labels = args.labels.resolve()
    args.dataset = args.dataset.resolve()
    args.scenes = args.scenes.resolve()
    args.model_path = args.model_path.resolve()
    args.output = args.output.resolve()
    os.chdir(args.uninavid_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    labels = read_labels(args.labels, args.expected_count)
    config = make_config(args)
    dataset = select_episodes(config, labels)
    validate_sequential_goals(dataset)
    scenes = {Path(episode.scene_id).name.split(".")[0] for episode in dataset.episodes}
    print(
        f"Validated {len(dataset.episodes)} sequential episodes across {len(scenes)} scenes; "
        "all sub-goals have viewpoints",
        flush=True,
    )
    if args.validate_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Uni-NaVid inference")
    write_multi_manifest(args, labels)
    policy = UniNaVidPolicy(
        args.model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        actions_per_inference=args.actions_per_inference,
        context_window=args.context_window,
        no_visual_tokens=args.no_visual_tokens,
        current_frame_only_64=args.current_frame_only_64,
    )
    print(f"Loaded Uni-NaVid on {torch.cuda.get_device_name(0)}", flush=True)
    if args.model_load_only:
        return

    selected_labels = labels[: args.limit] if args.limit is not None else labels
    dataset.episodes = dataset.episodes[: len(selected_labels)]
    if args.resume:
        pending_episodes = [
            episode
            for label, episode in zip(selected_labels, dataset.episodes)
            if not (args.output / "episodes" / f"{label}.json").is_file()
        ]
        pending_labels = [episode_label(episode) for episode in pending_episodes]
    else:
        pending_episodes = list(dataset.episodes)
        pending_labels = list(selected_labels)

    if pending_labels:
        import habitat

        dataset.episodes = pending_episodes
        env = habitat.Env(config=config.habitat, dataset=dataset)
        try:
            for index, label in enumerate(pending_labels, start=1):
                print(f"[{index}/{len(pending_labels)}] {label}", flush=True)
                try:
                    result = run_episode(env, policy, args, label)
                    atomic_json(args.output / "episodes" / f"{label}.json", result)
                    metrics = result["metrics"]
                    print(
                        f"  progress={metrics['progress']:.3f} ppl={metrics['ppl']:.3f} "
                        f"all_success={metrics['all_success']:.0f} "
                        f"reason={metrics['done_reason']} steps={result['steps']} "
                        f"seconds={result['elapsed_seconds']:.1f}",
                        flush=True,
                    )
                except Exception as error:
                    atomic_json(
                        args.output / "errors" / f"{label}.json",
                        {
                            "episode_label": label,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                            "time": time.time(),
                        },
                    )
                    raise
        finally:
            env.close()

    results = completed_results(args.output, selected_labels)
    summary = build_multi_summary(results, selected_labels)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_episodes"]:
        raise RuntimeError(f"Evaluation is incomplete: {summary['missing_episodes']}")


if __name__ == "__main__":
    main()
