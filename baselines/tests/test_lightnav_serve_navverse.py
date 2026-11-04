"""Unit tests for the LightNav-0 NavVerse server (baselines/lightnav_hm3d/serve_navverse.py).

No weights, no GPU, no vLLM: the velocity -> discrete action rule (against hand-computed cases
and a round trip through LightNav's own waypoint -> velocity conversion when lightnav imports),
the /act contract behind a fake policy, the step log, and the merge of that step log into
NavVerse results by ``longnav.analysis.results_io.load_navverse_run``. Needs fastapi + httpx, so
run it with the client runtime:

    PYTHONPATH=baselines/LightNav-0/src:src baselines/.envs/lightnav/bin/python \\
        -m unittest baselines.tests.test_lightnav_serve_navverse
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from baselines.lightnav_hm3d.serve_navverse import (
    ACTIONS,
    FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    create_app,
    discretization_constants,
    discretize,
    instruction_for,
    one_hot,
    parse_args,
    prune_log_dir_for,
    raw_to_velocity,
    server_info,
    velocity_from_cmd,
    waypoint_fields,
)

HAVE_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None
HAVE_LIGHTNAV = importlib.util.find_spec("lightnav") is not None
HAVE_RESULTS_IO = importlib.util.find_spec("longnav") is not None


def cmd_for(fwd_m: float, yaw_deg: float) -> dict:
    """A Habitat velocity_control dict as LightNav's ``first_waypoint_to_velocity_cmd`` would build it
    (dt = 1 s, lin [0, 0.25] m/s, ang [-30, 30] deg/s; raw = normalized to [-1, 1] and clipped)."""
    lin_raw = float(np.clip(2.0 * (fwd_m - 0.0) / 0.25 - 1.0, -1.0, 1.0))
    ang_raw = float(np.clip(2.0 * (yaw_deg + 30.0) / 60.0 - 1.0, -1.0, 1.0))
    return {"action": "velocity_control", "action_args": {"linear_velocity": lin_raw, "angular_velocity": ang_raw}}


CASES = [  # (fwd_m, yaw_deg) of the executed waypoint -> action id
    ((0.0, 0.0), STOP), ((0.02, 0.5), STOP), ((0.02, 5.0), TURN_LEFT), ((0.02, -5.0), TURN_RIGHT),
    ((0.25, 0.0), FORWARD), ((0.25, 30.0), FORWARD), ((0.10, 20.0), TURN_LEFT), ((0.50, 10.0), FORWARD),
    ((-0.3, 0.0), STOP), ((-0.3, 10.0), TURN_LEFT), ((0.2, -25.0), TURN_RIGHT), ((0.0, 0.9), STOP),
    ((0.0, 1.5), TURN_LEFT), ((0.03, 0.0), FORWARD),
]


class DiscretizeTest(unittest.TestCase):
    def test_table(self):
        for (fwd, yaw), expected in CASES:
            lin, ang = velocity_from_cmd(cmd_for(fwd, yaw))
            got, reason = discretize(lin, ang)
            self.assertEqual(got, expected, f"fwd={fwd} yaw={yaw}: {ACTIONS[got]} ({reason})")

    def test_velocity_from_cmd_matches_habitat_scaling(self):
        lin, ang = velocity_from_cmd(cmd_for(0.10, -12.0))
        self.assertAlmostEqual(lin, 0.10, places=6)
        self.assertAlmostEqual(ang, -12.0, places=6)
        lin, ang = velocity_from_cmd(cmd_for(0.50, 45.0))  # clipped by the raw clamp
        self.assertAlmostEqual(lin, 0.25, places=6)
        self.assertAlmostEqual(ang, 30.0, places=6)
        self.assertEqual(raw_to_velocity(-1.0, 0.0, 0.25), 0.0)
        with self.assertRaises(ValueError):
            velocity_from_cmd({"action": "STOP"})

    def test_reasons(self):
        self.assertEqual(discretize(0.0, 0.0), (STOP, "stop_guard"))
        self.assertEqual(discretize(0.25, 29.0), (FORWARD, "forward"))
        self.assertEqual(discretize(0.10, 20.0), (TURN_LEFT, "turn_left"))
        self.assertEqual(discretize(0.10, -20.0), (TURN_RIGHT, "turn_right"))
        # 15 deg turns: the same request is 1.33 turns vs 0.4 forwards -> still a turn; a 0.2 m / 10 deg
        # step is 0.8 forwards vs 0.67 turns -> FORWARD at 15 deg, and 0.8 vs 0.33 at 30 deg.
        self.assertEqual(discretize(0.10, -20.0, turn_deg=15.0)[0], TURN_RIGHT)
        self.assertEqual(discretize(0.20, 10.0, turn_deg=15.0)[0], FORWARD)
        self.assertEqual(discretize(0.05, 10.0, turn_deg=15.0)[0], TURN_LEFT)  # 0.2 vs 0.67
        self.assertEqual(discretize(0.05, 10.0, turn_deg=30.0)[0], TURN_LEFT)  # 0.2 vs 0.33
        self.assertEqual(discretize(0.10, 10.0, turn_deg=30.0)[0], FORWARD)  # 0.4 vs 0.33
        self.assertEqual(discretize(0.10, 10.0, turn_deg=15.0)[0], TURN_LEFT)  # 0.4 vs 0.67

    @unittest.skipUnless(HAVE_LIGHTNAV, "lightnav not importable")
    def test_round_trip_through_lightnav_velocity(self):
        from lightnav.velocity import first_waypoint_to_velocity_cmd
        for (fwd, yaw), expected in CASES:
            cmd = first_waypoint_to_velocity_cmd(np.array([fwd, 0.0, math.radians(yaw)], dtype=np.float32),
                                                 1.0, (0.0, 0.25), (-30.0, 30.0))
            lin, ang = velocity_from_cmd(cmd)
            self.assertEqual(discretize(lin, ang)[0], expected, f"fwd={fwd} yaw={yaw}")
            lin2, ang2 = velocity_from_cmd(cmd_for(fwd, yaw))
            self.assertAlmostEqual(lin, lin2, places=4)
            self.assertAlmostEqual(ang, ang2, places=3)

    def test_helpers(self):
        self.assertEqual(one_hot(TURN_RIGHT), [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(instruction_for("tv_monitor"), "Find the tv monitor.")
        self.assertEqual(instruction_for(" chair "), "Find the chair.")
        self.assertEqual(discretization_constants()["action_ids"], dict(zip(ACTIONS, range(4))))
        json.dumps(discretization_constants())
        fields = waypoint_fields({"raw_text": "x", "action_waypoint": np.array([0.1, 0.0, math.radians(15)]),
                                  "action_waypoint_index": 2, "predicted_traj": np.zeros((10, 3))})
        self.assertTrue(fields["decode_ok"])
        self.assertAlmostEqual(fields["yaw_deg"], 15.0, places=3)
        self.assertEqual(fields["fwd_m"], 0.1)
        self.assertFalse(waypoint_fields({"raw_text": "junk"})["decode_ok"])


class FakePolicy:
    """Scripted (fwd_m, yaw_deg) per step; records resets and the frames it saw."""

    def __init__(self, script, video_size=(288, 384)):
        self.script = list(script)
        self.resets = []
        self.frames = []
        self.agent = SimpleNamespace(video_size=video_size)
        self.lin_vel_range = (0.0, 0.25)
        self.ang_vel_range = (-30.0, 30.0)
        self._last = None

    def reset(self, obs=None):
        self.resets.append(obs["instruction"]["text"])

    def act(self, obs, info=None):
        rgb = obs["rgb"]
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, rgb.shape
        self.frames.append(rgb.shape)
        fwd, yaw = self.script.pop(0)
        self._last = (fwd, yaw)
        return cmd_for(fwd, yaw)

    def get_info(self):
        fwd, yaw = self._last
        return {"raw_text": "<apos_1><act_l0_1>", "predicted_traj": np.zeros((10, 3)), "rvq_codes": [1, 2, 3],
                "action_waypoint_index": 0, "action_waypoint": np.array([fwd, 0.0, math.radians(yaw)])}


def jpeg_bytes(width=640, height=480):
    buf = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), 120, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def served_for(policy):
    return SimpleNamespace(policy=policy, prune=None, prompt_stats={}, last_prune={}, facts={})


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class ServerTest(unittest.TestCase):
    def test_contract_and_step_log(self):
        from fastapi.testclient import TestClient
        policy = FakePolicy([(0.0, 18.0), (0.25, 2.0), (0.0, 0.0), (0.1, -25.0)])
        builds = []

        def factory():
            builds.append(True)
            return served_for(policy)

        files = {"file": ("obs.jpg", jpeg_bytes(), "image/jpeg")}
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "steps.jsonl"
            app = create_app(factory, info={"arm": "t"}, step_log=log, expect_video_size=(288, 384))
            self.assertEqual(builds, [])
            with TestClient(app) as client:
                self.assertTrue(client.get("/health").json()["ok"])
                self.assertEqual(client.post("/act", files=files).status_code, 400)
                r = client.post("/act", files=files, data={"instr_or_goal": "bookshelf"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json(), {"action_id": TURN_LEFT, "action_probs": one_hot(TURN_LEFT)})
                self.assertEqual(client.post("/act", files=files).json()["action_id"], FORWARD)
                self.assertEqual(client.post("/act", files=files).json()["action_id"], STOP)
                self.assertEqual(policy.resets, ["Find the bookshelf."])
                self.assertEqual(client.post("/act", files=files, data={"instr_or_goal": "tv_monitor"}).json()["action_id"],
                                 TURN_RIGHT)
                self.assertEqual(policy.resets, ["Find the bookshelf.", "Find the tv monitor."])
                self.assertEqual(client.post("/reset").json(), {"ok": True})
                self.assertEqual(client.post("/act", files=files).status_code, 400)
            records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(builds), 1)
        self.assertEqual([(r["episode"], r["step"], r["goal"]) for r in records],
                         [(1, 1, "bookshelf"), (1, 2, "bookshelf"), (1, 3, "bookshelf"), (2, 1, "tv_monitor")])
        for key in ("episode", "goal", "step", "latency_ms", "action_id", "action_probs", "kv_len", "kv_len_layer_max",
                    "peak_mem_GB", "time", "instruction", "reason", "fwd_m", "yaw_deg", "raw_text", "video_size"):
            self.assertIn(key, records[0])
        self.assertEqual([r["action_id"] for r in records], [TURN_LEFT, FORWARD, STOP, TURN_RIGHT])
        self.assertEqual([r["reason"] for r in records], ["turn_left", "forward", "stop_guard", "turn_right"])
        self.assertIsNone(records[0]["kv_len"])  # no engine behind the fake
        self.assertEqual(records[0]["instruction"], "Find the bookshelf.")
        self.assertEqual(records[0]["frame_hw"], [480, 640])
        self.assertEqual(policy.frames[0], (480, 640, 3))
        self.assertAlmostEqual(records[0]["yaw_deg"], 18.0, places=3)

    def test_video_size_mismatch_fails_loudly(self):
        from fastapi.testclient import TestClient
        policy = FakePolicy([(0.25, 0.0)], video_size=(256, 448))
        app = create_app(lambda: served_for(policy), expect_video_size=(288, 384))
        files = {"file": ("obs.jpg", jpeg_bytes(), "image/jpeg")}
        with TestClient(app, raise_server_exceptions=True) as client:
            with self.assertRaises(RuntimeError):
                client.post("/act", files=files, data={"instr_or_goal": "chair"})

    def test_prune_fields(self):
        from fastapi.testclient import TestClient
        policy = FakePolicy([(0.25, 0.0), (0.25, 0.0)])
        served = served_for(policy)
        served.prune = SimpleNamespace(episode=0)
        served.last_prune["rec"] = {"step": 5, "n_kept": [30, 0], "budget": 30, "n_pool": 60, "ledger": 30}
        served.prompt_stats.update(prompt_tokens=700, n_visual=250)
        files = {"file": ("obs.jpg", jpeg_bytes(), "image/jpeg")}
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "steps.jsonl"
            with TestClient(create_app(lambda: served, step_log=log)) as client:
                client.post("/act", files=files, data={"instr_or_goal": "chair"})
            rec = json.loads(log.read_text().splitlines()[0])
        self.assertEqual((rec["kv_len"], rec["n_visual"], rec["n_kept"], rec["budget"], rec["prune_episode"]),
                         (700, 250, 30, 30, 0))


@unittest.skipUnless(HAVE_FASTAPI and HAVE_RESULTS_IO, "fastapi/httpx or longnav.analysis not importable")
class MergeTest(unittest.TestCase):
    """The step log merges into NavVerse results like the LongNav-R1 server's does."""

    def test_load_navverse_run_merges_step_log(self):
        from fastapi.testclient import TestClient
        from longnav.analysis.results_io import load_navverse_run

        # Episode A: 2 model steps, a model STOP at the cap -> exactly cap steps, not forced. Episode B:
        # 3 model steps, then the client's own STOP once l_step reaches the cap -> recorded as cap + 1
        # steps with stop_called, a forced stop (re-scored as a failure).
        policy = FakePolicy([(0.25, 0.0), (0.0, 0.0), (0.25, 0.0), (0.0, -20.0), (0.25, 0.0)])
        files = {"file": ("obs.jpg", jpeg_bytes(), "image/jpeg")}
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp) / "nvl_keep_k4"
            (arm / "nvl_keep_k4" / "data").mkdir(parents=True)
            with TestClient(create_app(lambda: served_for(policy), step_log=arm / "server_steps.jsonl")) as client:
                client.post("/act", files=files, data={"instr_or_goal": "chair"})
                client.post("/act", files=files)
                client.post("/act", files=files, data={"instr_or_goal": "bed"})
                client.post("/act", files=files)
                client.post("/act", files=files)
            # The LightNav-0 arms map ids 2/3 to NavVerse's 15 deg TURN_*_SMALL names.
            names = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT_SMALL", 3: "TURN_RIGHT_SMALL"}
            histories = {"s1_0": [1, 0], "s1_1": [1, 3, 1, 0]}
            for label, ids in histories.items():
                (arm / "nvl_keep_k4" / "data" / f"{label}.json").write_text(json.dumps(
                    {"action_history": {str(i): {"action": names[a]} for i, a in enumerate(ids)}}))
            (arm / "nvl_keep_k4" / "result.json").write_text(json.dumps([
                {"episode_label": "s1_0", "step": 2, "termination_reason": "stop_called", "success": 1.0, "spl": 0.8,
                 "instruction": "chair", "oracle_success": 1.0},
                {"episode_label": "s1_1", "step": 4, "termination_reason": "stop_called", "success": 1.0, "spl": 0.5,
                 "instruction": "bed", "oracle_success": 1.0},
            ]))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                rows = load_navverse_run(arm, step_cap=3)
        self.assertEqual(rows["s1_0"]["inference_calls"], 2)
        self.assertEqual(rows["s1_1"]["inference_calls"], 3)
        self.assertTrue(all(math.isfinite(r["vlm_latency_ms"]) for r in rows.values()))
        self.assertFalse(rows["s1_0"]["forced_stop"])
        self.assertTrue(rows["s1_1"]["forced_stop"])
        self.assertEqual((rows["s1_1"]["success"], rows["s1_1"]["success_raw"]), (0.0, 1.0))
        self.assertEqual(rows["s1_1"]["n_steps"], 3)


class CliTest(unittest.TestCase):
    def test_parse_and_info(self):
        args = parse_args(["--prune-method", "recency", "--prune-k", "6", "--prune-tokens-per-frame", "15", "--arm", "x"])
        info = server_info(args)
        self.assertEqual(info["prune"]["budget"], 90)
        self.assertEqual(info["aspect_mode"], "keep")
        json.dumps(info)
        self.assertIsNone(server_info(parse_args([]))["prune"])
        # K = 1 arm: no history (k = 0, budget 0) and the current frame doubled.
        k1 = server_info(parse_args(["--prune-method", "recency", "--prune-k", "0", "--prune-tokens-per-frame", "15",
                                     "--prune-current-frames", "1"]))["prune"]
        self.assertEqual((k1["budget"], k1["current_frames"]), (0, 1))
        with self.assertRaises(SystemExit):
            parse_args(["--prune-method", "recency", "--prune-k", "0", "--prune-tokens-per-frame", "15",
                        "--prune-current-frames", "3"])
        for argv in (["--prune-method", "recency", "--prune-k", "6"], ["--prune-k", "6"],
                     ["--prune-method", "random", "--prune-k", "6", "--prune-tokens-per-frame", "0"]):
            with self.assertRaises(SystemExit):
                parse_args(argv)

    def test_prune_log_dir_is_fresh_per_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = prune_log_dir_for(Path(tmp), now=1_000_000.0)
            first.mkdir()
            second = prune_log_dir_for(Path(tmp), now=1_000_000.0)
            self.assertNotEqual(first, second)
            self.assertEqual(second.parent, Path(tmp))


if __name__ == "__main__":
    unittest.main()
