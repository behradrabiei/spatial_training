"""Equivalence test for the LightNav-0 context-window wrapper (baselines/lightnav_hm3d/eval_ctx.py).

Needs torch and the upstream ``lightnav`` package, so run it with the client runtime:

    PYTHONPATH=baselines/LightNav-0/src baselines/.envs/lightnav/bin/python \
        -m unittest baselines.tests.test_lightnav_ctx

It is skipped elsewhere. For every (episode length T, window K) the windowed sampler applied to
the full T-frame buffer must hand the model exactly what the unmodified sampler hands it on a
fresh K-frame episode made of the newest K frames: the same images in the same segments with
the same pooling, ids shifted by T - K (so the age-relative timestamps are unchanged), and the
original output when T <= K.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

HAVE_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("lightnav") is not None

ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG = ROOT / "baselines/LightNav-0/checkpoints/LightNav-0/eval_config.json"


def _tiers():
    from lightnav.slowfast import DEFAULT_SLOWFAST_TIERS_SPAN, validate_slowfast_tiers

    if EVAL_CONFIG.is_file():
        tiers = json.loads(EVAL_CONFIG.read_text())["tasks"]["vlnce"]["slowfast_tiers"]
    else:
        tiers = DEFAULT_SLOWFAST_TIERS_SPAN
    return validate_slowfast_tiers(tiers)


@unittest.skipUnless(HAVE_DEPS, "needs torch + lightnav (run with baselines/.envs/lightnav)")
class WindowedSegmentsTest(unittest.TestCase):
    def setUp(self):
        import torch
        from lightnav.slowfast import slowfast_segments, slowfast_video_segments

        self.torch = torch
        self.original = slowfast_video_segments
        self.layout = slowfast_segments
        self.tiers = _tiers()

    def _buffer(self, total):
        # Frame i is filled with the value i so equality of images implies equality of ids.
        t = self.torch
        return t.arange(total, dtype=t.float32).view(total, 1, 1, 1).expand(total, 3, 4, 6).clone()

    def _windowed(self, buf, k):
        from baselines.lightnav_hm3d.eval_ctx import windowed_segments

        total = int(buf.shape[0])
        return windowed_segments(buf, total - 1, total, self.tiers, k, self.original, self.layout)

    def test_matches_fresh_episode_of_newest_k_frames(self):
        for total in (1, 2, 3, 5, 33, 34, 35, 40, 120, 300):
            buf = self._buffer(total)
            for k in (1, 2, 4, 8, 16, 32, 64, 128):
                with self.subTest(total=total, k=k):
                    got = self._windowed(buf, k)
                    n = min(total, k)
                    ref = self.original(buf[-n:], n - 1, n, self.tiers)
                    offset = total - n
                    self.assertEqual(len(got), len(ref))
                    for g, r in zip(got, ref):
                        self.assertEqual(g["pool_spatial"], r["pool_spatial"])
                        self.assertEqual(g["pool_mode"], r["pool_mode"])
                        self.assertEqual(g["frame_indices"], [i + offset for i in r["frame_indices"]])
                        self.assertTrue(self.torch.equal(g["video"], r["video"]))  # slice keeps absolute values
                        self.assertEqual(g["total_frames"], total)
                    # Timestamp inputs: age of every frame relative to the newest one.
                    newest = max(i for s in got for i in s["frame_indices"])
                    ages_got = [[newest - i for i in s["frame_indices"]] for s in got]
                    newest_ref = max(i for s in ref for i in s["frame_indices"])
                    ages_ref = [[newest_ref - i for i in s["frame_indices"]] for s in ref]
                    self.assertEqual(ages_got, ages_ref)
                    # The window never reaches past the newest k frames.
                    for s in got:
                        self.assertTrue(all(total - n <= i <= total - 1 for i in s["frame_indices"]))

    def test_no_window_when_episode_fits(self):
        for total, k in ((1, 1), (7, 8), (34, 34), (40, 1000)):
            buf = self._buffer(total)
            got = self._windowed(buf, k)
            ref = self.original(buf, total - 1, total, self.tiers)
            self.assertEqual([s["frame_indices"] for s in got], [s["frame_indices"] for s in ref])

    def test_sweep_windows_are_dense(self):
        # K <= 34 lies inside the current (0-1, pool 1) + fast (2-33, pool 2) tiers: the model
        # sees exactly the newest K frames, nothing sparse, no anchor.
        buf = self._buffer(200)
        for k in (1, 4, 8, 16, 32):
            segs = self._windowed(buf, k)
            ids = sorted({i for s in segs for i in s["frame_indices"]})
            self.assertEqual(ids, list(range(200 - k, 200)))
            pools = [s["pool_spatial"] for s in segs]
            self.assertEqual(pools, [2, 1] if k > 2 else [1])

    def test_rejects_other_call_conventions(self):
        from baselines.lightnav_hm3d.eval_ctx import windowed_segments

        buf = self._buffer(10)
        with self.assertRaises(AssertionError):
            windowed_segments(buf, 8, 10, self.tiers, 4, self.original, self.layout)
        with self.assertRaises(AssertionError):
            windowed_segments(buf, 9, 5, self.tiers, 4, self.original, self.layout)


if __name__ == "__main__":
    unittest.main()
