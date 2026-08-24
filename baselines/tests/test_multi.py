import unittest
from types import SimpleNamespace

import numpy as np

from uninavid_hm3d.eval import UniNaVidPolicy
from uninavid_hm3d.eval_multi import SequentialGoalTracker, build_multi_summary


class TaskContextTests(unittest.TestCase):
    def tracker(self):
        return SequentialGoalTracker(
            ["chair", "plant", "bed"],
            np.zeros(3),
            4.0,
            leg_max_steps=3,
            success_distance=1.0,
        )

    def test_new_tasks_append_without_action_history(self):
        tracker = self.tracker()
        first = tracker.instruction_context
        tracker.apply_stop(0.5, next_goal_distance=3.0)
        second = tracker.instruction_context
        self.assertTrue(second.startswith(first))
        self.assertIn("new task is to find the plant", second)
        self.assertNotIn("stop", second.lower())
        self.assertNotIn("forward", second.lower())

    def test_all_goals_are_revealed_in_order_at_start(self):
        tracker = SequentialGoalTracker(
            ["chair", "plant", "bed"],
            np.zeros(3),
            4.0,
            reveal_all_goals=True,
        )
        self.assertEqual(
            tracker.instruction_context,
            "Find chair, then plant, then bed, in that order.",
        )
        tracker.apply_stop(0.5, next_goal_distance=3.0)
        self.assertIn("new task is to find the plant", tracker.instruction_context)
        self.assertNotIn("stop", tracker.instruction_context.lower())

    def test_goal_switch_discards_actions_but_preserves_visual_state(self):
        policy = object.__new__(UniNaVidPolicy)
        policy.pending_actions = ["left"]
        frame = np.ones((2, 2, 3), dtype=np.uint8)
        policy.new_rgb_frames = [frame]
        visual_cache = object()
        policy.model = SimpleNamespace(visual_cache=visual_cache)
        policy.discard_pending_actions()
        self.assertEqual(policy.pending_actions, [])
        self.assertIs(policy.new_rgb_frames[0], frame)
        self.assertIs(policy.model.visual_cache, visual_cache)


class SequentialTransitionTests(unittest.TestCase):
    def tracker(self, *, leg_max_steps=3):
        return SequentialGoalTracker(
            ["chair", "plant", "bed"],
            np.zeros(3),
            4.0,
            leg_max_steps=leg_max_steps,
            success_distance=1.0,
        )

    def test_move_inside_goal_does_not_oracle_stop(self):
        tracker = self.tracker()
        outcome = tracker.apply_move(np.array([0.25, 0.0, 0.0]))
        self.assertEqual(outcome, "move")
        self.assertFalse(tracker.done)
        self.assertEqual(tracker.legs, [])

    def test_successful_intermediate_stop_advances_in_place(self):
        tracker = self.tracker()
        last_position = tracker.last_position.copy()
        outcome = tracker.apply_stop(0.5, next_goal_distance=2.5)
        self.assertEqual(outcome, "goal_advanced")
        self.assertEqual(tracker.current_goal, "plant")
        np.testing.assert_array_equal(tracker.last_position, last_position)
        self.assertEqual(tracker.legs[0]["result"], "success")
        self.assertEqual(tracker.leg_steps, 0)

    def test_wrong_stop_ends_episode_without_guard(self):
        tracker = self.tracker()
        outcome = tracker.apply_stop(1.5)
        self.assertEqual(outcome, "wrong_stop")
        self.assertTrue(tracker.done)
        self.assertEqual(tracker.done_reason, "wrong_stop")

    def test_final_success(self):
        tracker = self.tracker()
        tracker.apply_stop(0.5, next_goal_distance=2.5)
        tracker.apply_stop(0.5, next_goal_distance=1.5)
        outcome = tracker.apply_stop(0.5)
        self.assertEqual(outcome, "all_success")
        self.assertEqual(tracker.metrics()["all_success"], 1.0)

    def test_leg_timeout(self):
        tracker = self.tracker(leg_max_steps=2)
        self.assertEqual(tracker.apply_move(np.array([0.25, 0.0, 0.0])), "move")
        self.assertEqual(tracker.apply_move(np.array([0.5, 0.0, 0.0])), "oot")
        self.assertEqual(tracker.done_reason, "oot")
        self.assertEqual(tracker.legs[0]["steps"], 2)


class MetricTests(unittest.TestCase):
    def test_episode_and_aggregate_metrics(self):
        tracker = SequentialGoalTracker(
            ["chair"], np.zeros(3), 2.0, leg_max_steps=5, success_distance=1.0
        )
        tracker.apply_move(np.array([3.0, 0.0, 0.0]))
        tracker.apply_stop(0.5)
        metrics = tracker.metrics()
        self.assertAlmostEqual(metrics["ppl"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["spl"], 2.0 / 3.0)

        result = {
            "episode_label": "scene_0",
            "steps": 2,
            "inference_calls": 1,
            "invalid_output_events": 0,
            "fallback_stops": 0,
            "elapsed_seconds": 1.0,
            "metrics": metrics,
        }
        summary = build_multi_summary([result], ["scene_0", "scene_1"])
        self.assertAlmostEqual(summary["pr"], 0.5)
        self.assertAlmostEqual(summary["sr"], 0.5)
        self.assertAlmostEqual(summary["spl"], 1.0 / 3.0)
        self.assertEqual(summary["failure_modes"]["missing"], 1)


if __name__ == "__main__":
    unittest.main()
