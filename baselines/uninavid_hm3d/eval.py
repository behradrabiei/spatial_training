"""Evaluate Uni-NaVid on a label-selected HM3D-v2 ObjectNav subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


ACTION_TO_ID = {"stop": 0, "forward": 1, "left": 2, "right": 3}
PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given "
    "a video of historical observations and an image of the current observation "
    "<image>. Your assigned task is: '{}'. Analyze this series of images to determine "
    "your next four actions. The predicted action should be one of the following: "
    "forward, left, right, or stop."
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def episode_label(episode: Any) -> str:
    scene_name = Path(episode.scene_id).name.split(".")[0]
    return f"{scene_name}_{episode.episode_id}"


def read_labels(path: Path, expected_count: int) -> list[str]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise TypeError("Episode label file must contain a JSON list of strings")
    duplicates = sorted({item for item in labels if labels.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate requested episode labels: {duplicates}")
    if len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} labels, found {len(labels)}")
    return labels


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


@contextmanager
def isolated_torch_rng(seed: int, device: torch.device):
    """Seed one generation call, then restore the caller's Torch RNG state."""
    devices: list[int] = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


class UniNaVidPolicy:
    """Inference-only policy preserving the official two-action evaluator behavior."""

    def __init__(
        self,
        model_path: Path,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        actions_per_inference: int = 2,
        context_window: int | None = None,
        no_visual_tokens: bool = False,
        current_frame_only_64: bool = False,
        sampling_seed: int | None = None,
    ) -> None:
        from uninavid.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from uninavid.conversation import SeparatorStyle, conv_templates
        from uninavid.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_token,
        )
        from uninavid.model.builder import load_pretrained_model

        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.separator_style = SeparatorStyle
        self.conv_templates = conv_templates
        self.keywords_stopping_criteria = KeywordsStoppingCriteria
        self.tokenizer_image_token = tokenizer_image_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.actions_per_inference = actions_per_inference
        self.context_window = context_window
        self.no_visual_tokens = no_visual_tokens
        self.current_frame_only_64 = current_frame_only_64
        self.sampling_seed = sampling_seed

        model_name = get_model_name_from_path(str(model_path))
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(str(model_path), None, model_name)
        )
        if self.image_processor is None:
            raise RuntimeError("Uni-NaVid did not initialize its image processor")
        self.model.eval()
        if self.current_frame_only_64:
            self.model.online_process_tensor = self._current_only_history
        self.device = next(self.model.parameters()).device
        self._special_tokens = {
            name: self.tokenizer(token, return_tensors="pt").input_ids[0][1:].to(self.device)
            for name, token in {
                "image_start": "<image_special>",
                "image_end": "</image_special>",
                "video_start": "<video_special>",
                "video_end": "</video_special>",
                "navigation": "[Navigation]",
                "image_separator": "<image_sep>",
            }.items()
        }
        self.pending_actions: list[str] = []
        self.new_rgb_frames: list[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        self.pending_actions = []
        self.new_rgb_frames = []
        self.inference_index = 0
        self.last_inference_seed: int | None = None
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def discard_pending_actions(self) -> None:
        """Force the next observation through inference without resetting history."""
        self.pending_actions = []

    @staticmethod
    def inference_seed_for_call(base_seed: int, inference_index: int) -> int:
        """Derive the stable RNG seed for one model-generation call."""
        if inference_index < 0:
            raise ValueError("inference_index must be non-negative")
        return (int(base_seed) + inference_index) % (2**63 - 1)

    @staticmethod
    def parse_actions(output: str, limit: int) -> tuple[list[str], list[str], bool]:
        words = re.findall(r"[A-Za-z]+", output.lower())
        actions = [word for word in words if word in ACTION_TO_ID][:limit]
        invalid = [word for word in words if word not in ACTION_TO_ID]
        fallback = not actions
        if fallback:
            actions = ["stop"]
        return actions, invalid, fallback

    @staticmethod
    def remove_image_placeholders(
        input_ids: torch.Tensor, image_token_index: int
    ) -> torch.Tensor:
        """Return the same prompt token stream with all visual insertion sites removed."""
        return input_ids[input_ids != image_token_index]

    @staticmethod
    def empty_visual_history(model_state: Any) -> tuple[torch.Tensor, list[int]]:
        """Return a zero-token history with the model's feature width and dtype."""
        feat_cache = model_state.feat_cache
        if feat_cache is None:
            raise RuntimeError("Current-frame features have not been cached")
        return feat_cache.new_empty((0, feat_cache.shape[-1])), []

    def _current_only_history(
        self,
        nav_size: int,
        length_threshold: int = 64,
        similarity_threshold: float = 0.985,
    ) -> tuple[torch.Tensor, list[int]]:
        del nav_size, length_threshold, similarity_threshold
        return self.empty_visual_history(self.model.get_model())

    @staticmethod
    def trim_visual_history(
        model_state: Any,
        new_frame_count: int,
        context_window: int | None,
    ) -> None:
        """Bound Uni-NaVid to N previous frames plus the current frame.

        ``feat_cache`` stores one row per observation before each row is flattened
        into the four-token historical-video representation. New observations are
        appended inside Uni-NaVid, so retain only enough old rows to leave room for
        the incoming batch. A finite window must also remove the compressed
        long-term cache, otherwise evicted observations would still reach the LLM.
        """
        if context_window is None:
            return
        if context_window < 0:
            raise ValueError("context_window must be at least 0")

        max_frames = context_window + 1
        if new_frame_count < 1 or new_frame_count > max_frames:
            raise ValueError(
                f"new_frame_count must be in [1, {max_frames}], got {new_frame_count}"
            )

        keep_previous = max_frames - new_frame_count
        feat_cache = model_state.feat_cache
        if feat_cache is not None and feat_cache.shape[0] > keep_previous:
            model_state.feat_cache = (
                feat_cache[-keep_previous:] if keep_previous else None
            )

        model_state.long_feat_cache = None
        model_state.weight = 1

    def _process_images(self) -> list[torch.Tensor]:
        processing_window = 0 if self.no_visual_tokens else self.context_window
        if processing_window is not None:
            max_frames = processing_window + 1
            self.new_rgb_frames = self.new_rgb_frames[-max_frames:]

        model_state = self.model.get_model()
        self.trim_visual_history(
            model_state,
            new_frame_count=len(self.new_rgb_frames),
            context_window=processing_window,
        )
        batch = np.asarray(self.new_rgb_frames)
        model_state.new_frames = len(self.new_rgb_frames)
        video = self.image_processor.preprocess(batch, return_tensors="pt")[
            "pixel_values"
        ].to(device=self.device, dtype=torch.float16)
        self.new_rgb_frames = []
        return [video]

    def _predict(self, prompt: str) -> str:
        question = prompt.replace(self.default_image_token, "").replace("\n", "")
        if self.model.config.mm_use_im_start_end:
            query = (
                self.default_im_start_token
                + self.default_image_token
                + self.default_im_end_token
                + "\n"
                + prompt.replace("<image>", "")
            )
        else:
            query = self.default_image_token + "\n" + prompt.replace("<image>", "")

        conversation = self.conv_templates["vicuna_v1"].copy()
        conversation.append_message(conversation.roles[0], query)
        conversation.append_message(conversation.roles[1], None)
        token_prompt = self.tokenizer_image_token(
            conversation.get_prompt(),
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).to(self.device)

        pieces: list[torch.Tensor] = []
        remaining = token_prompt
        image_locations = torch.where(remaining == self.image_token_index)[0]
        while image_locations.numel() > 0:
            index = image_locations[0]
            pieces.extend(
                [
                    remaining[:index],
                    self._special_tokens["video_start"],
                    self._special_tokens["image_separator"],
                    remaining[index : index + 1],
                    self._special_tokens["video_end"],
                    self._special_tokens["image_start"],
                    self._special_tokens["image_end"],
                    self._special_tokens["navigation"],
                ]
            )
            remaining = remaining[index + 1 :]
            image_locations = torch.where(remaining == self.image_token_index)[0]
        if remaining.numel() > 0:
            pieces.append(remaining)
        input_ids = torch.cat(pieces)
        if self.no_visual_tokens:
            input_ids = self.remove_image_placeholders(
                input_ids, self.image_token_index
            )
        input_ids = input_ids.unsqueeze(0)

        stop_string = (
            conversation.sep
            if conversation.sep_style != self.separator_style.TWO
            else conversation.sep2
        )
        stopping = self.keywords_stopping_criteria(
            [stop_string], self.tokenizer, input_ids
        )
        images = self._process_images()
        self.model.update_prompt([[question]])
        inference_seed = None
        rng_context: Any = nullcontext()
        if self.sampling_seed is not None:
            inference_seed = self.inference_seed_for_call(
                self.sampling_seed, self.inference_index
            )
            self.inference_index += 1
            rng_context = isolated_torch_rng(inference_seed, self.device)
        self.last_inference_seed = inference_seed
        with torch.inference_mode(), rng_context:
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )

        output = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()
        if output.endswith(stop_string):
            output = output[: -len(stop_string)].strip()
        return output

    def act(self, rgb: np.ndarray, instruction: str) -> dict[str, Any]:
        self.new_rgb_frames.append(np.asarray(rgb))
        if self.pending_actions:
            action = self.pending_actions.pop(0)
            return {
                "action": action,
                "action_id": ACTION_TO_ID[action],
                "inference": False,
                "raw_output": None,
                "parsed_actions": None,
                "invalid_tokens": [],
                "fallback_stop": False,
                "inference_seconds": 0.0,
                "inference_seed": None,
            }

        prompt = PROMPT_TEMPLATE.format(instruction)
        started = time.perf_counter()
        raw_output = self._predict(prompt)
        elapsed = time.perf_counter() - started
        actions, invalid, fallback = self.parse_actions(
            raw_output, self.actions_per_inference
        )
        action = actions.pop(0)
        self.pending_actions = actions
        return {
            "action": action,
            "action_id": ACTION_TO_ID[action],
            "inference": True,
            "raw_output": raw_output,
            "parsed_actions": [action, *actions],
            "invalid_tokens": invalid,
            "fallback_stop": fallback,
            "inference_seconds": elapsed,
            "inference_seed": self.last_inference_seed,
        }


def make_config(args: argparse.Namespace) -> Any:
    from habitat.config import read_write
    from habitat.config.default import get_config

    config = get_config(str(args.config))
    with read_write(config):
        config.habitat.dataset.data_path = str(args.dataset)
        config.habitat.dataset.scenes_dir = str(args.scenes)
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
    return config


def select_episodes(config: Any, labels: list[str]) -> Any:
    import habitat

    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    indexed: dict[str, Any] = {}
    duplicates: list[str] = []
    for episode in dataset.episodes:
        label = episode_label(episode)
        if label in indexed:
            duplicates.append(label)
        indexed[label] = episode
    if duplicates:
        raise ValueError(f"Dataset contains duplicate labels: {sorted(duplicates)}")
    missing = [label for label in labels if label not in indexed]
    if missing:
        raise ValueError(f"Requested labels absent from HM3D-v2: {missing}")
    dataset.episodes = [indexed[label] for label in labels]
    return dataset


def annotate_frame(
    rgb: np.ndarray,
    label: str,
    goal: str,
    step: int,
    action: str,
    distance: float | None,
) -> np.ndarray:
    frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    lines = [
        f"episode: {label}",
        f"goal: {goal}",
        f"step: {step}  action: {action}",
        f"distance: {distance:.3f}" if distance is not None else "distance: n/a",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 104), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 23 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def completed_results(output_dir: Path, labels: list[str]) -> list[dict[str, Any]]:
    results = []
    for label in labels:
        path = output_dir / "episodes" / f"{label}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def build_summary(results: list[dict[str, Any]], expected_labels: list[str]) -> dict[str, Any]:
    def average(key: str) -> float | None:
        values = [result["metrics"].get(key) for result in results]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return sum(finite) / len(finite) if finite else None

    completed = {result["episode_label"] for result in results}
    return {
        "expected_episodes": len(expected_labels),
        "completed_episodes": len(results),
        "missing_episodes": [label for label in expected_labels if label not in completed],
        "success_rate": average("success"),
        "spl": average("spl"),
        "soft_spl": average("soft_spl"),
        "average_final_distance": average("distance_to_goal"),
        "average_steps": (
            sum(result["steps"] for result in results) / len(results) if results else None
        ),
        "total_inference_calls": sum(result["inference_calls"] for result in results),
        "invalid_output_events": sum(result["invalid_output_events"] for result in results),
        "fallback_stops": sum(result["fallback_stops"] for result in results),
        "elapsed_episode_seconds": sum(result["elapsed_seconds"] for result in results),
    }


def write_manifest(args: argparse.Namespace, labels: list[str]) -> None:
    baseline_root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_at_unix": time.time(),
        "command": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "environment_prefix": sys.prefix,
        "labels_path": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "requested_labels": labels,
        "dataset_path": str(args.dataset),
        "scenes_path": str(args.scenes),
        "model_path": str(args.model_path),
        "source_revisions": {
            "uninavid": git_revision(args.uninavid_root),
            "habitat_lab": git_revision(baseline_root / "habitat-lab"),
        },
        "inference": {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "actions_per_inference": args.actions_per_inference,
            "context_window_previous_frames": args.context_window,
            "max_visual_history_frames": (
                0
                if args.no_visual_tokens or args.current_frame_only_64
                else args.context_window + 1
                if args.context_window is not None
                else None
            ),
            "visual_tokens_enabled": not args.no_visual_tokens,
            "current_frame_history_tokens": (
                0 if args.no_visual_tokens or args.current_frame_only_64 else 4
            ),
            "current_frame_detail_tokens": 0 if args.no_visual_tokens else 64,
            "encoded_visual_cache_max_frames": (
                1 if args.no_visual_tokens or args.current_frame_only_64 else None
            ),
            "max_episode_steps": args.max_steps,
            "forward_step_m": 0.25,
            "turn_degrees": 30,
            "success_distance_m": 1.0,
            "allow_sliding": True,
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
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
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
    goal = str(env.current_episode.object_category)
    instruction = f"Find the {goal}."
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
            while not env.episode_over:
                before = json_value(env.get_metrics())
                decision = policy.act(observation["rgb"], instruction)
                inference_calls += int(decision["inference"])
                invalid_output_events += int(bool(decision["invalid_tokens"]))
                fallback_stops += int(decision["fallback_stop"])
                record = {
                    "step": steps,
                    "episode_label": label,
                    "goal": goal,
                    "action": decision["action"],
                    "action_id": decision["action_id"],
                    "inference": decision["inference"],
                    "raw_output": decision["raw_output"],
                    "parsed_actions": decision["parsed_actions"],
                    "invalid_tokens": decision["invalid_tokens"],
                    "fallback_stop": decision["fallback_stop"],
                    "inference_seconds": decision["inference_seconds"],
                    "metrics_before_action": before,
                }
                trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                trace.flush()

                if video_writer is not None:
                    video_writer.write(
                        annotate_frame(
                            observation["rgb"],
                            label,
                            goal,
                            steps,
                            decision["action"],
                            before.get("distance_to_goal"),
                        )
                    )

                observation = env.step(decision["action_id"])
                steps += 1
                if steps >= args.max_steps and not env.episode_over:
                    observation = env.step(ACTION_TO_ID["stop"])
                    steps += 1
        os.replace(trace_tmp, trace_path)
    finally:
        if video_writer is not None:
            video_writer.release()

    elapsed = time.perf_counter() - started
    metrics = json_value(env.get_metrics())
    return {
        "episode_label": label,
        "episode_id": str(env.current_episode.episode_id),
        "scene_id": str(env.current_episode.scene_id),
        "goal": goal,
        "instruction": instruction,
        "steps": steps,
        "inference_calls": inference_calls,
        "invalid_output_events": invalid_output_events,
        "fallback_stops": fallback_stops,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "metrics": metrics,
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
        "--config", type=Path, default=Path(__file__).resolve().parent / "config/objectnav_hm3d_v2.yaml"
    )
    parser.add_argument("--labels", type=Path, default=project_root / "dump/hm3d_v2_100_labels.json")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/"
            "objectnav_hm3d_v2/val/val.json.gz"
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
    parser.add_argument("--output", type=Path, default=baseline_root / "results/hm3d_v2_100")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model-load-only", action="store_true")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--actions-per-inference", type=int, default=2)
    parser.add_argument(
        "--context-window",
        type=nonnegative_int,
        help="Number of previous observations to retain; the current frame is additional",
    )
    parser.add_argument(
        "--no-visual-tokens",
        action="store_true",
        help="Remove all visual embeddings from the LLM input (text and action decoding only)",
    )
    parser.add_argument(
        "--current-frame-only-64",
        action="store_true",
        help="Use only the current image's 64 detailed tokens and no 4-token history stream",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    if args.current_frame_only_64:
        if args.context_window not in (None, 0):
            parser.error("--current-frame-only-64 is incompatible with a nonzero context window")
        args.context_window = 0
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
    if len(dataset.episodes) != args.expected_count:
        raise RuntimeError(
            f"Selected {len(dataset.episodes)} episodes; expected {args.expected_count}"
        )
    print(
        f"Validated {len(dataset.episodes)} unique episodes across "
        f"{len({Path(ep.scene_id).name.split('.')[0] for ep in dataset.episodes})} scenes"
    )
    if args.validate_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Uni-NaVid inference")
    write_manifest(args, labels)
    policy = UniNaVidPolicy(
        args.model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        actions_per_inference=args.actions_per_inference,
        context_window=args.context_window,
        no_visual_tokens=args.no_visual_tokens,
        current_frame_only_64=args.current_frame_only_64,
    )
    print(
        f"Loaded Uni-NaVid on {torch.cuda.get_device_name(0)}; "
        f"compute capability {torch.cuda.get_device_capability(0)}"
    )
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
                    print(
                        f"  success={result['metrics'].get('success')} "
                        f"spl={result['metrics'].get('spl')} steps={result['steps']} "
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
    summary = build_summary(results, selected_labels)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_episodes"]:
        raise RuntimeError(f"Evaluation is incomplete: {summary['missing_episodes']}")


if __name__ == "__main__":
    main()
