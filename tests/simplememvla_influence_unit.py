"""CPU checks of the SimpleMemVLA velocity-influence probe (no model, no simulator).

    robomme_benchmark/.venv-simplememvla/bin/python tests/simplememvla_influence_unit.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROBE = (ROOT / "robomme_benchmark/scripts/plus_eval/simplememvla/SimpleMemVLA"
         / "robomme_sim/influence_probe.py")
spec = importlib.util.spec_from_file_location("influence_probe", PROBE)
ip = importlib.util.module_from_spec(spec)
sys.modules["influence_probe"] = ip
spec.loader.exec_module(ip)

H, D, HID = 30, 8, 4


class FakeHead:
    """v = a * x + mean of the valid condition states + state, a velocity field with a known gap."""
    action_horizon, action_dim = H, D

    def forward(self, hidden, x, t, condition_attention_mask, condition_position_ids,
                condition_token_embeds=None, state=None):
        w = condition_attention_mask.to(hidden.dtype).unsqueeze(-1)
        c = (hidden * w).sum(1) / w.sum(1)                      # (B, HID)
        shift = c.mean(-1).view(-1, 1, 1) + t.view(-1, 1, 1)
        if state is not None:
            shift = shift + state.mean(-1).view(-1, 1, 1)
        return 0.5 * x + shift


class FakePolicy:
    """Stands in for RoboMMEPolicy: a frame counter and the window rule's patch keys."""
    n_frames, stride = 8, 10
    history_image_keys = ["front"]

    def __init__(self, n):
        self._buffers = {"front": list(range(n))}
        self._frames_seen = {"front": n}
        self._prompt_cache = {"key": ("p", 8)}
        self._patch_feat_cache = {}
        self.context_frames = None

    def _positions(self):
        cur = len(self._buffers["front"]) - 1
        lo = 0 if self.context_frames is None else max(0, cur - (self.context_frames - 1) * self.stride)
        return [max(lo, cur - (self.n_frames - 1 - i) * self.stride) for i in range(self.n_frames)]

    def _video_patch_keys(self):
        p = self._positions()
        return [("front", p[j], p[j + 1]) for j in range(0, len(p), 2)]

    def _prepare_inputs(self, prompt):
        return {"policy": self, "video_patch_keys": self._video_patch_keys()}


class FakeBatched:
    """generate_batch builds a condition from the oldest visible frame, then honours the tap."""
    device = torch.device("cpu")
    action_dim, eval_temperature, num_denoising_steps = D, 1.0, 10

    def __init__(self):
        self.model = type("M", (), {"action_head": FakeHead()})()
        self.sampled = 0

    def generate_batch(self, processed, state_list):
        n = len(processed)
        oldest = torch.tensor([float(p["video_patch_keys"][0][1]) for p in processed])
        hidden = oldest.view(n, 1, 1).expand(n, 3, HID).clone()
        cond = (hidden, None, torch.ones(n, 3, dtype=torch.long), torch.zeros(3, n, 3, dtype=torch.long))
        tok = [[int(o)] for o in oldest]
        if self.condition_tap(cond=cond, state=torch.stack(state_list), subtasks=[str(t) for t in tok],
                              token_ids=tok):
            return None
        self.sampled += 1
        torch.randn(1)  # the rollout's own noise draw
        return [(None, "s")] * n


def test_trajectory_follows_the_sampler_rule():
    head = FakeHead()
    cond = (torch.ones(2, 3, HID), None, torch.ones(2, 3, dtype=torch.long), None)
    noise = torch.randn(2, H, D)
    pts = ip.full_trajectory(head, cond, None, noise, 10)
    assert len(pts) == 10 and torch.equal(pts[0][0], noise)
    steps = torch.linspace(1.0, 0.0, 11)
    for i, (x, t, v) in enumerate(pts):
        assert torch.equal(t, steps[i].expand(2))
        assert torch.equal(v, head.forward(cond[0], x, t, cond[2], None))
        if i + 1 < len(pts):
            assert torch.equal(pts[i + 1][0], x - (steps[i] - steps[i + 1]) * v)


def test_probe_values_floor_and_random_stream():
    batched = FakeBatched()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "influence.jsonl"
        probe = ip.VelocityInfluenceProbe(batched, [1, 2, 4, 8], n_noise=4, seed=0,
                                          out_path=str(out), execute_horizon=16)
        policies = [FakePolicy(61), FakePolicy(21)]   # cur = 60 and 20 -> 7 and 3 history frames
        states = [torch.zeros(D), torch.ones(D)]
        torch.manual_seed(123)
        before = torch.get_rng_state()
        batched.generate_batch([p._prepare_inputs("p") for p in policies], states)
        after_rollout = torch.get_rng_state()
        assert not torch.equal(before, after_rollout)            # the rollout drew its noise
        probe.begin("T", 0), probe.begin("T", 1)
        probe.measure(policies, ["p", "p"], states,
                      [{"task": "T", "episode": e, "call": 0, "step": 0} for e in (0, 1)])
        assert torch.equal(after_rollout, torch.get_rng_state())  # the probe drew nothing from it
        assert batched.sampled == 1                               # probe passes never sample
        probe.flush("T", 0), probe.flush("T", 1)
        recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["n_history"] for r in recs] == [7, 3]
    for r, cur in zip(recs, (60, 20)):
        assert r["arms"]["replay"]["infl"] == 0.0 and r["arms"]["replay"]["same_input"]
        for k in (1, 2, 4, 8):
            arm = r["arms"][f"k{k}"]
            lo = max(0, cur - (k - 1) * 10)
            # the fake field shifts by (oldest visible frame) -> gap^2 * H * D at every point
            want = float(lo) ** 2 * H * D
            assert abs(arm["infl"] - want) <= 1e-4 * max(1.0, want), (k, arm["infl"], want)
            assert abs(arm["infl_exec"] - float(lo) ** 2 * 16 * D) <= 1e-4 * max(1.0, want)
            assert arm["same_input"] == (lo == 0) and arm["subtask_same"] == (lo == 0)
            assert len(arm["infl_by_t"]) == 10
    assert len({r["block"] for r in recs}) == 2


def test_shadow_shares_frames_and_keeps_its_own_cache():
    main = FakePolicy(61)
    main._patch_feat_cache = {("front", 50, 60): "feat"}
    sh = ip.VelocityInfluenceProbe._shadow(main, "k2", 2)
    assert sh._buffers is main._buffers and sh._frames_seen is main._frames_seen
    assert sh.context_frames == 2 and main.context_frames is None
    assert sh._patch_feat_cache == main._patch_feat_cache
    assert sh._patch_feat_cache is not main._patch_feat_cache
    sh._patch_feat_cache[("front", 50, 50)] = "pad"
    assert ("front", 50, 50) not in main._patch_feat_cache
    assert ip.VelocityInfluenceProbe._shadow(main, "k2", 2) is sh


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
