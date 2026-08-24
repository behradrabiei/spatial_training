import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

from uninavid_hm3d.eval import UniNaVidPolicy, isolated_torch_rng
from uninavid_hm3d.teleop import (
    ACTION_NAMES,
    LiveRecorder,
    interpret_key,
    predict_for_mode,
    resolve_control_choice,
    run_teleop_episode,
    safe_component,
    select_episode_label,
)


def model_decision(action: str, *, inference: bool) -> dict:
    return {
        "action": action,
        "action_id": ACTION_NAMES.index(action),
        "inference": inference,
        "raw_output": "forward left" if inference else None,
        "parsed_actions": ["forward", "left"] if inference else None,
        "invalid_tokens": [],
        "fallback_stop": False,
        "inference_seconds": 0.1 if inference else 0.0,
        "inference_seed": 30 if inference else None,
    }


def top_down_map(size: int = 24) -> dict:
    return {
        "map": np.ones((size, size), dtype=np.uint8),
        "fog_of_war_mask": np.ones((size, size), dtype=np.uint8),
        "agent_map_coord": [(size // 2, size // 2)],
        "agent_angle": [0.0],
    }


class FakePolicy:
    def __init__(self) -> None:
        self.pending_actions: list[str] = []
        self.discards = 0
        self.resets = 0

    def reset(self) -> None:
        self.pending_actions = []
        self.resets += 1

    def discard_pending_actions(self) -> None:
        self.pending_actions = []
        self.discards += 1

    def act(self, _rgb, _instruction) -> dict:
        if self.pending_actions:
            return model_decision(self.pending_actions.pop(0), inference=False)
        self.pending_actions = ["left"]
        return model_decision("forward", inference=True)


class FakeEnv:
    def __init__(self, episode_steps: int = 3) -> None:
        self.current_episode = SimpleNamespace(
            scene_id="/scenes/scene.glb",
            episode_id="0",
            object_category="chair",
        )
        self.episode_steps = episode_steps
        self.steps: list[int] = []
        self.episode_over = False

    def observation(self) -> dict:
        rgb = np.full((24, 32, 3), len(self.steps), dtype=np.uint8)
        return {"rgb": rgb}

    def reset(self) -> dict:
        self.steps = []
        self.episode_over = False
        return self.observation()

    def step(self, action_id: int) -> dict:
        self.steps.append(action_id)
        self.episode_over = len(self.steps) >= self.episode_steps
        return self.observation()

    def get_metrics(self) -> dict:
        return {
            "distance_to_goal": float(self.episode_steps - len(self.steps)),
            "success": float(self.episode_over),
            "top_down_map": top_down_map(),
        }


class FakeRecorder:
    instances: list["FakeRecorder"] = []

    def __init__(self, output_dir: Path, _fps: float) -> None:
        self.output_dir = output_dir
        self.current_path = output_dir / "current.png"
        self.frames: list[tuple[int, str, str]] = []
        self.closed = False
        output_dir.mkdir(parents=True, exist_ok=True)
        self.instances.append(self)

    def append(self, _rgb, _label, _goal, step, mode, decision, _metrics) -> str:
        self.frames.append((step, mode, decision["action"]))
        return str(self.current_path)

    def close(self) -> str:
        self.closed = True
        return str(self.output_dir / "episode.mp4")


class TeleopHelpersTests(unittest.TestCase):
    def test_key_interpretation_and_vertical_rejection(self):
        self.assertEqual(interpret_key("w"), ("action", 1))
        self.assertEqual(interpret_key("A"), ("action", 2))
        self.assertEqual(interpret_key(" "), ("handoff", None))
        self.assertEqual(interpret_key("q"), ("abort", None))
        self.assertEqual(interpret_key("r"), ("invalid", None))

    def test_episode_selection_bounds(self):
        self.assertEqual(select_episode_label(["a", "b"], 1), "b")
        with self.assertRaises(ValueError):
            select_episode_label(["a"], -1)
        with self.assertRaises(IndexError):
            select_episode_label(["a"], 1)

    def test_safe_output_component(self):
        self.assertEqual(safe_component("scene/name 1"), "scene_name_1")
        self.assertEqual(safe_component(""), "episode")

    def test_per_inference_seeds_are_stable_and_distinct(self):
        self.assertEqual(UniNaVidPolicy.inference_seed_for_call(30, 0), 30)
        self.assertEqual(UniNaVidPolicy.inference_seed_for_call(30, 1), 31)
        self.assertEqual(UniNaVidPolicy.inference_seed_for_call(30, 100), 130)
        with self.assertRaises(ValueError):
            UniNaVidPolicy.inference_seed_for_call(30, -1)

    def test_isolated_rng_repeats_sampling_and_restores_caller_state(self):
        probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])

        torch.manual_seed(91)
        expected_before = torch.rand(4)
        expected_after = torch.rand(4)

        torch.manual_seed(91)
        actual_before = torch.rand(4)
        with isolated_torch_rng(30, torch.device("cpu")):
            first = torch.multinomial(probabilities, num_samples=4, replacement=True)
        actual_after = torch.rand(4)
        with isolated_torch_rng(30, torch.device("cpu")):
            second = torch.multinomial(probabilities, num_samples=4, replacement=True)

        torch.testing.assert_close(actual_before, expected_before)
        torch.testing.assert_close(actual_after, expected_after)
        torch.testing.assert_close(first, second)

    def test_manual_preview_refresh_and_handoff_buffer(self):
        policy = FakePolicy()
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)

        first = predict_for_mode(policy, rgb, "Find chair.", "teleop")
        manual = resolve_control_choice("teleop", first["action_id"], "action", 2)
        self.assertEqual(manual.executed_action_id, 2)
        self.assertEqual(manual.next_mode, "teleop")

        second = predict_for_mode(policy, rgb, "Find chair.", "teleop")
        handoff = resolve_control_choice("teleop", second["action_id"], "handoff")
        self.assertEqual(handoff.executed_action_id, ACTION_NAMES.index("forward"))
        self.assertTrue(handoff.handoff)

        buffered = predict_for_mode(policy, rgb, "Find chair.", "model")
        self.assertEqual(buffered["action"], "left")
        self.assertFalse(buffered["inference"])
        self.assertEqual(policy.discards, 2)

    def test_abort_has_no_executed_action(self):
        choice = resolve_control_choice("teleop", 1, "abort")
        self.assertEqual(choice.termination_reason, "aborted")
        self.assertEqual(choice.action_source, "teleop")
        self.assertIsNone(choice.executed_action_id)


class TeleopEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRecorder.instances.clear()

    def test_manual_action_then_handoff_and_native_buffering(self):
        env = FakeEnv(episode_steps=3)
        policy = FakePolicy()
        commands = iter([("action", ACTION_NAMES.index("right")), ("handoff", None)])

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = run_teleop_episode(
                env,
                policy,
                output_dir=output_dir,
                label="scene_0",
                episode_index=0,
                fps=4,
                max_steps=10,
                command_reader=lambda _actions: next(commands),
                recorder_factory=FakeRecorder,
            )
            trace = json.loads((output_dir / "trace.json").read_text())

        self.assertEqual(
            env.steps,
            [ACTION_NAMES.index("right"), ACTION_NAMES.index("forward"), ACTION_NAMES.index("left")],
        )
        self.assertEqual(result["handoff_step"], 1)
        self.assertEqual(result["termination_reason"], "habitat_done")
        self.assertEqual(trace["steps"][0]["inference_seed"], 30)
        self.assertIsNone(trace["steps"][2]["inference_seed"])
        self.assertEqual(
            [step["control_mode"] for step in trace["steps"]],
            ["teleop", "model", "model"],
        )
        self.assertNotIn("top_down_map", trace["final_metrics"])
        for step in trace["steps"]:
            self.assertNotIn("top_down_map", step["metrics_before_action"])
            self.assertNotIn("top_down_map", step["metrics_after_action"])
        self.assertTrue(FakeRecorder.instances[0].closed)

    def test_abort_does_not_step_environment(self):
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as directory:
            result = run_teleop_episode(
                env,
                FakePolicy(),
                output_dir=Path(directory),
                label="scene_0",
                episode_index=0,
                fps=4,
                max_steps=10,
                command_reader=lambda _actions: ("abort", None),
                recorder_factory=FakeRecorder,
            )
        self.assertEqual(env.steps, [])
        self.assertEqual(result["termination_reason"], "aborted")
        self.assertIsNone(result["steps"][0]["executed_action_id"])

    def test_error_finalizes_recorder_and_trace(self):
        env = FakeEnv()

        def fail(_actions):
            raise RuntimeError("input failed")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "input failed"):
                run_teleop_episode(
                    env,
                    FakePolicy(),
                    output_dir=output_dir,
                    label="scene_0",
                    episode_index=0,
                    fps=4,
                    max_steps=10,
                    command_reader=fail,
                    recorder_factory=FakeRecorder,
                )
            trace = json.loads((output_dir / "trace.json").read_text())
        self.assertEqual(trace["termination_reason"], "error")
        self.assertIn("input failed", trace["error"])
        self.assertTrue(FakeRecorder.instances[0].closed)


class LiveRecorderTests(unittest.TestCase):
    def test_writes_current_image_and_finalizes_video_atomically(self):
        writers = []

        class Writer:
            def __init__(self, path, *_args):
                self.path = Path(path)
                self.path.touch()
                self.released = False
                writers.append(self)

            def isOpened(self):
                return True

            def write(self, _frame):
                pass

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "uninavid_hm3d.teleop.cv2.VideoWriter", Writer
        ):
            recorder = LiveRecorder(Path(directory), 4)
            recorder.append(
                np.zeros((24, 32, 3), dtype=np.uint8),
                "scene_0",
                "chair",
                0,
                "teleop",
                model_decision("forward", inference=True),
                {"distance_to_goal": np.float32(1.5), "top_down_map": top_down_map()},
            )
            video = recorder.close()
            self.assertTrue((Path(directory) / "current.png").is_file())
            current = cv2.imread(str(Path(directory) / "current.png"))
            self.assertEqual(current.shape, (24, 56, 3))
            self.assertEqual(video, str(Path(directory) / "episode.mp4"))
            self.assertTrue((Path(directory) / "episode.mp4").is_file())
            self.assertFalse((Path(directory) / ".episode.partial.mp4").exists())
            self.assertTrue(writers[0].released)
            self.assertEqual(recorder.close(), video)


if __name__ == "__main__":
    unittest.main()
