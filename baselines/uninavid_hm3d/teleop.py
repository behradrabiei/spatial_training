"""Interactively drive one HM3D episode, then hand control to Uni-NaVid."""

from __future__ import annotations

import argparse
import os
import random
import sys
import termios
import time
import traceback
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from uninavid_hm3d.eval import (
    ACTION_TO_ID,
    UniNaVidPolicy,
    atomic_json,
    episode_label,
    json_value,
    make_config,
    nonnegative_int,
    positive_int,
    read_labels,
    select_episodes,
)


ACTION_NAMES = list(ACTION_TO_ID)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def select_episode_label(labels: list[str], episode_index: int) -> str:
    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    if episode_index >= len(labels):
        raise IndexError(
            f"episode index {episode_index} is out of range for {len(labels)} labels"
        )
    return labels[episode_index]


def safe_component(value: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
    return sanitized or "episode"


def configure_top_down_map(config: Any, max_steps: int) -> Any:
    """Enable the same visualization-only Habitat map used by LongNav teleop."""
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )

    with read_write(config):
        config.habitat.task.measurements.top_down_map = TopDownMapMeasurementConfig(
            max_episode_steps=max_steps,
            map_padding=3,
            map_resolution=512,
            draw_goal_positions=True,
            draw_goal_aabbs=False,
            draw_shortest_path=True,
            draw_view_points=True,
            draw_border=True,
            fog_of_war=FogOfWarConfig(draw=True, visibility_dist=20, fov=79),
        )
    return config


def configure_habitat_seed(config: Any, seed: int) -> Any:
    """Use the teleop seed for episode iteration, the simulator, and the task."""
    from habitat.config import read_write

    with read_write(config):
        config.habitat.seed = seed
    return config


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep serializable episode metrics without embedding the full map arrays."""
    return json_value(
        {key: value for key, value in metrics.items() if key != "top_down_map"}
    )


def interpret_key(key: str, action_space: list[str] = ACTION_NAMES) -> tuple[str, int | None]:
    """Return an action, handoff, abort, or invalid keyboard command."""
    if key == " ":
        return "handoff", None
    normalized = key.lower()
    if normalized == "q":
        return "abort", None
    key_actions = {
        "w": "forward",
        "a": "left",
        "d": "right",
        "x": "stop",
        "r": "up",
        "f": "down",
    }
    action = key_actions.get(normalized)
    if action is None or action not in action_space:
        return "invalid", None
    return "action", action_space.index(action)


def read_teleop_command(action_space: list[str] = ACTION_NAMES) -> tuple[str, int | None]:
    if not sys.stdin.isatty():
        raise RuntimeError("Teleoperation requires an interactive terminal (stdin is not a TTY)")
    controls = ["W=forward", "A=left", "D=right", "X=stop"]
    if "up" in action_space:
        controls.append("R=up")
    if "down" in action_space:
        controls.append("F=down")
    prompt = "  ".join([*controls, "Space=model", "Q=abort"])
    descriptor = sys.stdin.fileno()
    while True:
        print(f"\n{prompt}\nteleop> ", end="", flush=True)
        previous = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        if key == "":
            raise EOFError("Interactive input closed during teleoperation")
        kind, action_id = interpret_key(key, action_space)
        if kind != "invalid":
            print("Space" if key == " " else key.upper())
            return kind, action_id
        print(f"Unsupported key {key!r}; choose one of the displayed controls.")


@dataclass(frozen=True)
class ControlChoice:
    next_mode: str
    executed_action_id: int | None
    action_source: str | None
    termination_reason: str | None = None
    handoff: bool = False


def resolve_control_choice(
    control_mode: str,
    model_action_id: int,
    command: str | None = None,
    manual_action_id: int | None = None,
) -> ControlChoice:
    """Resolve one prediction into the action Habitat should execute."""
    if control_mode == "model":
        return ControlChoice("model", model_action_id, "model")
    if control_mode != "teleop":
        raise ValueError(f"Unknown control mode {control_mode!r}")
    if command == "abort":
        return ControlChoice("teleop", None, "teleop", termination_reason="aborted")
    if command == "handoff":
        return ControlChoice("model", model_action_id, "model", handoff=True)
    if command == "action" and manual_action_id is not None:
        return ControlChoice("teleop", int(manual_action_id), "teleop")
    raise ValueError(f"Invalid teleop command {command!r} with action {manual_action_id!r}")


def predict_for_mode(
    policy: UniNaVidPolicy,
    rgb: np.ndarray,
    instruction: str,
    control_mode: str,
) -> dict[str, Any]:
    """Refresh manual previews while preserving native buffering after handoff."""
    if control_mode == "teleop":
        policy.discard_pending_actions()
    decision = policy.act(rgb, instruction)
    if control_mode == "teleop" and not decision["inference"]:
        raise RuntimeError("Teleop preview unexpectedly used a buffered action")
    return decision


def annotate_teleop_frame(
    rgb: np.ndarray,
    label: str,
    goal: str,
    step: int,
    control_mode: str,
    decision: dict[str, Any],
    metrics: dict[str, Any],
) -> np.ndarray:
    frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    action = str(decision["action"])
    inference = "fresh inference" if decision["inference"] else "buffered action"
    parsed_actions = decision.get("parsed_actions")
    plan_text = (
        "model plan: " + " ".join(str(item) for item in parsed_actions)
        if parsed_actions
        else f"model plan: {action} (buffered)"
    )
    distance = metrics.get("distance_to_goal")
    lines = [
        f"episode: {label}",
        f"goal: {goal}",
        f"step: {step}  mode: {control_mode}",
        f"model action: {action}  ({inference})",
        plan_text,
        f"distance: {float(distance):.3f}" if distance is not None else "distance: n/a",
        f"distance reward: {metrics.get('distance_to_goal_reward', 'n/a')}",
        f"success: {metrics.get('success', 'n/a')}  "
        f"spl: {metrics.get('spl', 'n/a')}  soft_spl: {metrics.get('soft_spl', 'n/a')}",
    ]
    overlay = frame.copy()
    overlay_height = min(frame.shape[0], 200)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 22 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def render_top_down_map(metrics: dict[str, Any], output_height: int) -> np.ndarray:
    if "top_down_map" not in metrics:
        raise ValueError("Habitat top_down_map is required for live rendering")
    from habitat.utils.visualizations import maps

    top_down_rgb = maps.colorize_draw_agent_and_fit_to_height(
        metrics["top_down_map"], output_height
    )
    top_down_bgr = cv2.cvtColor(top_down_rgb, cv2.COLOR_RGB2BGR)
    overlay = top_down_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (top_down_bgr.shape[1], 34), (0, 0, 0), -1)
    top_down_bgr = cv2.addWeighted(overlay, 0.65, top_down_bgr, 0.35, 0)
    cv2.putText(
        top_down_bgr, "Top-down map | goals + trajectory", (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return top_down_bgr


def atomic_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), frame):
        raise RuntimeError(f"Could not write live image to {temporary}")
    os.replace(temporary, path)


class LiveRecorder:
    """Write the live preview atomically and append the same frames to a video."""

    def __init__(self, output_dir: Path, fps: float) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.current_path = output_dir / "current.png"
        self.video_path = output_dir / "episode.mp4"
        self.partial_video_path = output_dir / ".episode.partial.mp4"
        self.writer: Any | None = None
        self.frames = 0
        self.closed = False
        output_dir.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        rgb: np.ndarray,
        label: str,
        goal: str,
        step: int,
        control_mode: str,
        decision: dict[str, Any],
        metrics: dict[str, Any],
    ) -> str:
        if self.closed:
            raise RuntimeError("Cannot append to a closed recorder")
        frame = annotate_teleop_frame(
            rgb, label, goal, step, control_mode, decision, metrics
        )
        top_down = render_top_down_map(metrics, frame.shape[0])
        frame = np.concatenate((frame, top_down), axis=1)
        atomic_png(self.current_path, frame)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.partial_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Could not open video writer for {self.video_path}")
        self.writer.write(frame)
        self.frames += 1
        return str(self.current_path)

    def close(self) -> str | None:
        if self.closed:
            return str(self.video_path) if self.frames else None
        self.closed = True
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not self.frames:
            return None
        os.replace(self.partial_video_path, self.video_path)
        return str(self.video_path)


def decision_trace(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_action": decision["action"],
        "model_action_id": int(decision["action_id"]),
        "inference": bool(decision["inference"]),
        "raw_output": decision["raw_output"],
        "parsed_actions": decision["parsed_actions"],
        "invalid_tokens": decision["invalid_tokens"],
        "fallback_stop": bool(decision["fallback_stop"]),
        "inference_seconds": float(decision["inference_seconds"]),
        "inference_seed": decision.get("inference_seed"),
    }


def run_teleop_episode(
    env: Any,
    policy: UniNaVidPolicy,
    *,
    output_dir: Path,
    label: str,
    episode_index: int,
    fps: float,
    max_steps: int,
    settings: dict[str, Any] | None = None,
    command_reader: Callable[[list[str]], tuple[str, int | None]] = read_teleop_command,
    recorder_factory: Callable[[Path, float], Any] = LiveRecorder,
) -> dict[str, Any]:
    """Run one interactive episode and always finalize its trace and recording."""
    recorder = recorder_factory(output_dir, fps)
    trace_steps: list[dict[str, Any]] = []
    control_mode = "teleop"
    handoff_step: int | None = None
    termination_reason = "error"
    goal: str | None = None
    started = time.perf_counter()
    raised: BaseException | None = None
    error_traceback: str | None = None
    video_path: str | None = None
    final_metrics: dict[str, Any] = {}
    steps = 0

    try:
        observation = env.reset()
        current_label = episode_label(env.current_episode)
        if current_label != label:
            raise RuntimeError(f"Expected episode {label}, Habitat reset to {current_label}")
        goal = str(env.current_episode.object_category)
        instruction = f"Find the {goal}."
        policy.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"\nEpisode {episode_index}: {label}")
        print(f"Goal: {goal}")
        print(f"Live image: {recorder.current_path}")

        while not env.episode_over and steps < max_steps:
            metrics_before_raw = env.get_metrics()
            metrics_before = metric_snapshot(metrics_before_raw)
            mode_before_action = control_mode
            decision = predict_for_mode(
                policy, observation["rgb"], instruction, mode_before_action
            )
            current_path = recorder.append(
                observation["rgb"],
                label,
                goal,
                steps,
                mode_before_action,
                decision,
                metrics_before_raw,
            )
            print(
                f"step {steps}: current image updated at {current_path}; "
                f"model would take {decision['action']}"
            )
            if decision["inference"] and decision["raw_output"]:
                print(f"model output: {decision['raw_output']}")

            if mode_before_action == "teleop":
                command, manual_action_id = command_reader(ACTION_NAMES)
            else:
                command, manual_action_id = None, None
            choice = resolve_control_choice(
                mode_before_action,
                int(decision["action_id"]),
                command,
                manual_action_id,
            )
            record = {
                "step": steps,
                "episode_label": label,
                "goal": goal,
                "mode_before_action": mode_before_action,
                "control_mode": choice.action_source,
                **decision_trace(decision),
                "executed_action": (
                    ACTION_NAMES[choice.executed_action_id]
                    if choice.executed_action_id is not None
                    else None
                ),
                "executed_action_id": choice.executed_action_id,
                "metrics_before_action": metrics_before,
            }
            if choice.termination_reason is not None:
                record["metrics_after_action"] = metrics_before
                trace_steps.append(record)
                termination_reason = choice.termination_reason
                break

            if choice.handoff:
                handoff_step = steps
                print("Control handed to Uni-NaVid.")
            control_mode = choice.next_mode
            observation = env.step(int(choice.executed_action_id))
            steps += 1
            record["metrics_after_action"] = metric_snapshot(env.get_metrics())
            trace_steps.append(record)

        if termination_reason == "error":
            termination_reason = "habitat_done" if env.episode_over else "max_steps"
    except KeyboardInterrupt as error:
        termination_reason = "interrupted"
        raised = error
        error_traceback = traceback.format_exc()
    except BaseException as error:
        termination_reason = "error"
        raised = error
        error_traceback = traceback.format_exc()
    finally:
        try:
            final_metrics = metric_snapshot(env.get_metrics())
        except Exception:
            final_metrics = {}
        try:
            video_path = recorder.close()
        except BaseException as error:
            if raised is None:
                raised = error
                error_traceback = traceback.format_exc()
                termination_reason = "error"
        payload = {
            "episode_index": episode_index,
            "episode_label": label,
            "goal": goal,
            "handoff_step": handoff_step,
            "termination_reason": termination_reason,
            "elapsed_seconds": time.perf_counter() - started,
            "attention_visualization": False,
            "current_image": str(recorder.current_path),
            "video": video_path,
            "final_metrics": final_metrics,
            "settings": settings or {},
            "error": repr(raised) if raised is not None else None,
            "traceback": error_traceback,
            "steps": trace_steps,
        }
        atomic_json(output_dir / "trace.json", json_value(payload))

    print(f"\nRecording: {video_path or 'not created'}")
    print(f"Trace: {output_dir / 'trace.json'}")
    if raised is not None:
        raise raised
    return payload


def parse_args() -> argparse.Namespace:
    baseline_root = Path(__file__).resolve().parents[1]
    project_root = baseline_root.parent
    uninavid_root = baseline_root / "Uni-NaVid"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninavid-root", type=Path, default=uninavid_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config/objectnav_hm3d_v2.yaml",
    )
    parser.add_argument(
        "--labels", type=Path, default=project_root / "dump/hm3d_v2_100_labels.json"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/"
            "objectnav_hm3d_v2/val/val.json.gz"
        ),
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        default=Path("/home/brabiei/vault/habitat_data/scenes/HM3D/v2"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=uninavid_root / "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
    )
    parser.add_argument(
        "--output", type=Path, default=baseline_root / "results/uninavid_teleop"
    )
    parser.add_argument("--expected-count", type=positive_int, default=100)
    parser.add_argument("--episode-index", type=nonnegative_int, default=0)
    parser.add_argument("--fps", type=positive_float, default=4.0)
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=30,
        help="Seed Habitat and every Uni-NaVid sampling call (default: 30)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--actions-per-inference", type=positive_int, default=2)
    parser.add_argument(
        "--context-window",
        type=nonnegative_int,
        help="Number of previous observations to retain; the current frame is additional",
    )
    parser.add_argument(
        "--no-visual-tokens",
        action="store_true",
        help="Remove all visual embeddings from the LLM input",
    )
    parser.add_argument(
        "--current-frame-only-64",
        action="store_true",
        help="Use the current image's 64 detailed tokens without history tokens",
    )
    parser.add_argument("--max-steps", type=positive_int, default=500)
    args = parser.parse_args()
    if args.current_frame_only_64:
        if args.context_window not in (None, 0):
            parser.error("--current-frame-only-64 is incompatible with a nonzero context window")
        args.context_window = 0
    return args


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("teleop must be launched from an interactive terminal")

    for field in ("uninavid_root", "config", "labels", "dataset", "scenes", "model_path", "output"):
        setattr(args, field, getattr(args, field).resolve())
    os.chdir(args.uninavid_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    labels = read_labels(args.labels, args.expected_count)
    label = select_episode_label(labels, args.episode_index)
    config = configure_habitat_seed(
        configure_top_down_map(make_config(args), args.max_steps), args.seed
    )
    dataset = select_episodes(config, labels)
    selected_episode = dataset.episodes[args.episode_index]
    if episode_label(selected_episode) != label:
        raise RuntimeError("Selected Habitat episode does not match the requested label")
    dataset.episodes = [selected_episode]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Uni-NaVid inference")

    policy = UniNaVidPolicy(
        args.model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        actions_per_inference=args.actions_per_inference,
        context_window=args.context_window,
        no_visual_tokens=args.no_visual_tokens,
        current_frame_only_64=args.current_frame_only_64,
        sampling_seed=args.seed,
    )
    print(f"Loaded Uni-NaVid on {torch.cuda.get_device_name(0)}", flush=True)

    import habitat

    output_dir = args.output / f"{args.episode_index}_{safe_component(label)}"
    settings = {
        "command": sys.argv,
        "model_path": str(args.model_path),
        "dataset": str(args.dataset),
        "scenes": str(args.scenes),
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "actions_per_inference": args.actions_per_inference,
        "context_window": args.context_window,
        "no_visual_tokens": args.no_visual_tokens,
        "current_frame_only_64": args.current_frame_only_64,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "top_down_map": True,
        "reproducibility": {
            "habitat_seeded": True,
            "per_inference_sampling_seed": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }
    env = habitat.Env(config=config.habitat, dataset=dataset)
    env.seed(args.seed)
    try:
        run_teleop_episode(
            env,
            policy,
            output_dir=output_dir,
            label=label,
            episode_index=args.episode_index,
            fps=args.fps,
            max_steps=args.max_steps,
            settings=settings,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
