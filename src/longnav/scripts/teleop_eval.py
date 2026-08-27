"""Interactively drive one eval episode, then hand control to the model."""

from __future__ import annotations

import json
import os
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig
from longnav.utils.rollout_core import substitute_convo_template


@dataclass
class TeleopOptions:
    episode_index: int = 0
    fps: int = 4


@dataclass
class TeleopEvalConfig(RLConfig):
    teleop: TeleopOptions = field(default_factory=TeleopOptions)


register_configs()
ConfigStore.instance().store(name="teleop_eval_config", node=TeleopEvalConfig)


def resolve_episode_labels(cfg: TeleopEvalConfig, dataset_labels=None) -> list[str]:
    """Resolve episodes with the same source precedence as normal evaluation."""
    if cfg.task.subset_label:
        from longnav.constants import episode_labels_table

        try:
            return list(episode_labels_table[cfg.task.subset_label])
        except KeyError as exc:
            raise ValueError(f"Unknown episode subset {cfg.task.subset_label!r}") from exc
    if cfg.task.episode_json:
        episode_path = Path(to_absolute_path(cfg.task.episode_json))
        with episode_path.open() as handle:
            labels = json.load(handle)
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"{episode_path} must contain a JSON list of episode labels")
        return labels
    if dataset_labels is None:
        raise ValueError("Dataset labels are required when no subset or episode JSON is configured")
    return list(dataset_labels)


def select_episode_label(labels: list[str], episode_index: int) -> str:
    if episode_index < 0:
        raise ValueError("teleop.episode_index must be non-negative")
    if episode_index >= len(labels):
        raise IndexError(
            f"teleop.episode_index={episode_index} is out of range for {len(labels)} episodes"
        )
    return labels[episode_index]


def interpret_key(key: str, action_space: list[str]):
    """Return (kind, action_id), where kind is action/handoff/abort/invalid."""
    normalized = key.lower()
    if key == " ":
        return "handoff", None
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
    action_name = key_actions.get(normalized)
    if action_name is None or action_name not in action_space:
        return "invalid", None
    return "action", action_space.index(action_name)


def read_teleop_command(action_space: list[str]):
    if not sys.stdin.isatty():
        raise RuntimeError("Teleoperation requires an interactive terminal (stdin is not a TTY)")
    available = ["W=forward", "A=left", "D=right", "X=stop"]
    if "up" in action_space:
        available.append("R=up")
    if "down" in action_space:
        available.append("F=down")
    prompt = "  ".join(available + ["Space=model", "Q=abort"])
    descriptor = sys.stdin.fileno()
    while True:
        print(f"\n{prompt}\nteleop> ", end="", flush=True)
        previous = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        kind, action_id = interpret_key(key, action_space)
        if kind != "invalid":
            print("Space" if key == " " else key.upper())
            return kind, action_id
        print(f"Unsupported key {key!r}; choose one of the displayed controls.")


def unpack_actor_state(actor_result):
    if len(actor_result) == 2:
        rgb, state = actor_result
        return rgb, state, {"mode": "standard"}
    if len(actor_result) == 3:
        rgb, patch_coords, state = actor_result
        return rgb, state, {"mode": "bev", "patch_coords": patch_coords}
    raise ValueError(f"Unexpected simulator result with {len(actor_result)} elements")


def metric_snapshot(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_label", "episode_id", "scene_id", "distance_to_goal",
        "distance_to_goal_reward", "success", "spl", "soft_spl", "oracle_action",
    )
    return {key: info.get(key) for key in keys if key in info}


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return str(value)


def write_trace(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    os.replace(temporary, path)


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


@hydra.main(version_base=None, config_name="teleop_eval_config", config_path="../config")
def main(cfg: TeleopEvalConfig):
    import ray

    from longnav.utils.factories import ExpBootstrapper

    if not sys.stdin.isatty():
        raise RuntimeError("teleop_eval must be launched from an interactive terminal")
    if cfg.teleop.fps <= 0:
        raise ValueError("teleop.fps must be positive")

    # This command intentionally owns one model and one simulator and never starts
    # WandB or the normal eval log-flush pipeline.
    cfg.resources.num_vlms = 1
    cfg.resources.num_sims = 1
    cfg.task.wandb_project = None
    cfg.rollout.visualize_attention_3d = True
    cfg.rollout.attn3d_layers = [-1]
    cfg.rollout.visualize_attention = False
    cfg.rollout.visualize_attention_heads = False
    cfg.sim.visualize_attn3d = False
    cfg.sim.visualize_3d = False
    cfg.sim.add_top_down_map = True
    # TopDownMap is needed only for visualization; do not let enabling the
    # measurement change the episode reward relative to the previous teleop path.
    cfg.sim.explr_bonus = None
    cfg.sim.auto_flush = False

    explicit_labels: Optional[list[str]] = None
    if cfg.task.subset_label or cfg.task.episode_json:
        explicit_labels = resolve_episode_labels(cfg)
        select_episode_label(explicit_labels, cfg.teleop.episode_index)

    bootstrapper = ExpBootstrapper(cfg)
    model = None
    simulator = None
    recording_started = False
    output_dir: Optional[Path] = None
    trace_steps: list[dict[str, Any]] = []
    termination_reason = "error"
    handoff_step = None
    final_info: dict[str, Any] = {}
    actual_label: Optional[str] = None
    goal: Optional[str] = None
    raised: Optional[BaseException] = None

    try:
        bootstrapper.setup_cluster()
        models = bootstrapper.bootstrap_vlms_rl(training=False)
        simulators = bootstrapper.bootstrap_sims(logger=None)
        model, simulator = models[0], simulators[0]

        if explicit_labels is not None:
            episode_label = select_episode_label(explicit_labels, cfg.teleop.episode_index)
            ray.get(simulator.assign_shard.remote([episode_label]))
        else:
            # No label list: teleop.episode_index addresses the whole dataset by
            # position (scene files alphabetically, episodes in file order). Labels
            # cannot do that on datasets whose episode_id restarts per goal category.
            catalog = ray.get(simulator.export_dataset_catalog.remote())
            if not 0 <= cfg.teleop.episode_index < len(catalog):
                raise IndexError(
                    f"teleop.episode_index={cfg.teleop.episode_index} is out of range for {len(catalog)} episodes"
                )
            entry = catalog[cfg.teleop.episode_index]
            episode_label = entry["label"]
            print(f"Episode index {cfg.teleop.episode_index} of {len(catalog)}: {episode_label} "
                  f"(scene {entry['scene']}, goal {entry['object_category']})")
            ray.get(simulator.assign_shard.remote(None, [cfg.teleop.episode_index]))
        rgb, state, pos_id_kwargs = unpack_actor_state(ray.get(simulator.reset.remote()))
        actual_label = state["info"]["episode_label"]
        final_info = state["info"]
        if actual_label != episode_label:
            raise RuntimeError(
                f"Requested episode {episode_label!r}, but Habitat loaded {actual_label!r}"
            )

        output_root = Path(to_absolute_path(cfg.task.output_dir))
        output_dir = (
            output_root / cfg.task.run_name / "teleop" /
            f"{cfg.teleop.episode_index}_{_safe_component(actual_label)}"
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ray.get(simulator.start_live_recording.remote(str(output_dir), cfg.teleop.fps))
        recording_started = True
        ray.get(model.reset.remote())

        rollout_config = bootstrapper.resolved_dict["rollout"]
        action_space = list(rollout_config["action_space"])
        if len(action_space) != len(cfg.vlm.vocab):
            raise ValueError("rollout.action_space and vlm.vocab must have the same length")
        goal = state["obs"]["instr_or_goal"]
        messages = substitute_convo_template(
            rollout_config["convo_start_template"],
            state["obs"] | rollout_config,
        )
        teleop = True
        done = False
        step = 0
        print(f"\nEpisode {cfg.teleop.episode_index}: {actual_label}")
        print(f"Goal: {goal}")
        print(f"Live image: {output_dir / 'current.png'}")

        while not done and step < cfg.rollout.max_steps:
            prediction = ray.get(
                model.predict_rollout_step.remote(rgb, messages, pos_id_kwargs)
            )
            prediction["goal"] = goal
            mode_at_render = "teleop" if teleop else "model"
            current_path = ray.get(
                simulator.render_live_step.remote(
                    prediction, mode_at_render, step,
                    cfg.teleop.episode_index, action_space,
                )
            )
            model_action = int(prediction["model_action_id"])
            print(
                f"step {step}: current image updated at {current_path}; "
                f"model would take {action_space[model_action]}"
            )

            if teleop:
                command, manual_action = read_teleop_command(action_space)
                if command == "abort":
                    trace_steps.append({
                        "step": step,
                        "control_mode": "teleop",
                        "model_action_id": model_action,
                        "model_action": action_space[model_action],
                        "executed_action_id": None,
                        "executed_action": None,
                        "action_probs": prediction["action_probs"],
                        "action_logprobs": prediction["action_logprobs"],
                        "spguard_triggered": prediction["spguard_triggered"],
                        "metrics_before": metric_snapshot(state["info"]),
                        "metrics_after": metric_snapshot(state["info"]),
                    })
                    termination_reason = "aborted"
                    break
                if command == "handoff":
                    teleop = False
                    handoff_step = step
                    executed_action = model_action
                    action_source = "model"
                else:
                    executed_action = int(manual_action)
                    action_source = "teleop"
            else:
                executed_action = model_action
                action_source = "model"

            supplementary_logs = dict(prediction["supplementary_logs"])
            supplementary_logs.update({
                "control_mode": action_source,
                "model_action_id": model_action,
                "executed_action_id": executed_action,
            })
            pre_metrics = metric_snapshot(state["info"])
            rgb, next_state, pos_id_kwargs = unpack_actor_state(
                ray.get(
                    simulator.step.remote(
                        executed_action, supplementary_logs=supplementary_logs
                    )
                )
            )
            trace_steps.append({
                "step": step,
                "control_mode": action_source,
                "model_action_id": model_action,
                "model_action": action_space[model_action],
                "executed_action_id": executed_action,
                "executed_action": action_space[executed_action],
                "action_probs": prediction["action_probs"],
                "action_logprobs": prediction["action_logprobs"],
                "spguard_triggered": prediction["spguard_triggered"],
                "metrics_before": pre_metrics,
                "metrics_after": metric_snapshot(next_state["info"]),
            })
            messages = substitute_convo_template(
                rollout_config["convo_turn_template"],
                {"action": action_space[executed_action]},
            )
            state = next_state
            final_info = state["info"]
            done = bool(state["done"])
            step += 1

        if done:
            termination_reason = "habitat_done"
        elif step >= cfg.rollout.max_steps:
            termination_reason = "max_steps"

    except KeyboardInterrupt as exc:
        termination_reason = "interrupted"
        raised = exc
    except BaseException as exc:
        termination_reason = "error"
        raised = exc
    finally:
        video_path = None
        if simulator is not None and recording_started:
            try:
                video_path = ray.get(simulator.finish_live_recording.remote())
            except Exception as exc:
                print(f"Failed to finalize live recording: {exc}", file=sys.stderr)
        if output_dir is not None:
            write_trace(output_dir / "trace.json", {
                "episode_index": int(cfg.teleop.episode_index),
                "episode_label": final_info.get("episode_label", actual_label),
                "goal": goal,
                "handoff_step": handoff_step,
                "termination_reason": termination_reason,
                "video": video_path,
                "final_metrics": metric_snapshot(final_info),
                "steps": trace_steps,
            })
        for actor in (model, simulator):
            if actor is not None:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        if ray.is_initialized():
            ray.shutdown()

    if output_dir is not None:
        print(f"\nRecording: {output_dir / 'episode.mp4'}")
        print(f"Trace: {output_dir / 'trace.json'}")
    if raised is not None:
        raise raised


if __name__ == "__main__":
    main()
