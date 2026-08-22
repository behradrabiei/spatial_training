import unittest
from types import SimpleNamespace

import torch

from uninavid_hm3d.eval import (
    UniNaVidPolicy,
    build_summary,
    nonnegative_int,
    positive_int,
)


class ActionParsingTests(unittest.TestCase):
    def test_parses_first_two_actions(self):
        actions, invalid, fallback = UniNaVidPolicy.parse_actions(
            "forward left right stop", 2
        )
        self.assertEqual(actions, ["forward", "left"])
        self.assertEqual(invalid, [])
        self.assertFalse(fallback)

    def test_skips_invalid_tokens(self):
        actions, invalid, fallback = UniNaVidPolicy.parse_actions(
            "I suggest LEFT, then banana and forward.", 2
        )
        self.assertEqual(actions, ["left", "forward"])
        self.assertIn("banana", invalid)
        self.assertFalse(fallback)

    def test_falls_back_to_stop(self):
        actions, _, fallback = UniNaVidPolicy.parse_actions("unclear", 2)
        self.assertEqual(actions, ["stop"])
        self.assertTrue(fallback)


class SummaryTests(unittest.TestCase):
    def test_summary_reports_missing_and_means(self):
        results = [
            {
                "episode_label": "scene_0",
                "steps": 10,
                "inference_calls": 5,
                "invalid_output_events": 1,
                "fallback_stops": 0,
                "elapsed_seconds": 2.0,
                "metrics": {
                    "success": 1.0,
                    "spl": 0.5,
                    "soft_spl": 0.75,
                    "distance_to_goal": 0.5,
                },
            }
        ]
        summary = build_summary(results, ["scene_0", "scene_1"])
        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(summary["missing_episodes"], ["scene_1"])
        self.assertEqual(summary["success_rate"], 1.0)


class ContextWindowTests(unittest.TestCase):
    @staticmethod
    def state(frame_count=5):
        cache = torch.arange(frame_count).view(frame_count, 1, 1)
        return SimpleNamespace(
            feat_cache=cache,
            long_feat_cache=torch.tensor([[99]]),
            weight=7,
        )

    def test_unset_window_preserves_native_caches(self):
        state = self.state()
        original_short = state.feat_cache
        original_long = state.long_feat_cache
        UniNaVidPolicy.trim_visual_history(state, 2, None)
        self.assertIs(state.feat_cache, original_short)
        self.assertIs(state.long_feat_cache, original_long)
        self.assertEqual(state.weight, 7)

    def test_one_new_frame_leaves_n_previous_slots(self):
        state = self.state()
        UniNaVidPolicy.trim_visual_history(state, 1, 2)
        self.assertEqual(state.feat_cache.flatten().tolist(), [3, 4])
        self.assertIsNone(state.long_feat_cache)
        self.assertEqual(state.weight, 1)

    def test_two_new_frames_fill_window_one_without_old_history(self):
        state = self.state()
        UniNaVidPolicy.trim_visual_history(state, 2, 1)
        self.assertIsNone(state.feat_cache)
        self.assertIsNone(state.long_feat_cache)

    def test_positive_window_validation(self):
        self.assertEqual(positive_int("1"), 1)
        with self.assertRaisesRegex(Exception, "at least 1"):
            positive_int("0")

    def test_zero_window_drops_all_previous_frames(self):
        state = self.state()
        UniNaVidPolicy.trim_visual_history(state, 1, 0)
        self.assertIsNone(state.feat_cache)
        self.assertIsNone(state.long_feat_cache)
        self.assertEqual(state.weight, 1)

    def test_nonnegative_window_validation(self):
        self.assertEqual(nonnegative_int("0"), 0)
        with self.assertRaisesRegex(Exception, "at least 0"):
            nonnegative_int("-1")

    def test_no_visual_mode_removes_only_image_placeholders(self):
        tokens = torch.tensor([10, -200, 11, -200, 12])
        result = UniNaVidPolicy.remove_image_placeholders(tokens, -200)
        self.assertEqual(result.tolist(), [10, 11, 12])

    def test_current_64_mode_returns_an_empty_history_branch(self):
        state = SimpleNamespace(feat_cache=torch.ones((1, 4, 7), dtype=torch.float16))
        history, lengths = UniNaVidPolicy.empty_visual_history(state)
        self.assertEqual(history.shape, (0, 7))
        self.assertEqual(history.dtype, torch.float16)
        self.assertEqual(lengths, [])


if __name__ == "__main__":
    unittest.main()
