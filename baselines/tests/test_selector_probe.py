"""Unit tests for uninavid_hm3d.selector_probe (no model weights, no Habitat).

Run from the repo root with the baseline runtime:
    PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python \
        -m unittest baselines.tests.test_selector_probe
"""
import math
import unittest

import torch
import torch.nn.functional as F

from uninavid_hm3d.eval import parse_args
from uninavid_hm3d.influence import kl_divergence
from uninavid_hm3d.layer_blind import ColumnBlinder, visual_columns
from uninavid_hm3d.prune import N_PER_FRAME, NAV_TOKENS, prompt_length
from uninavid_hm3d.selector_probe import (SelectorProbe, dart_dissimilarity, nn_cosine_distance,
                                          random_positions)

from baselines.tests.test_prune import C, IMAGE, FakeLayer, FakeLlama, layer0_draws, synthetic_ids

N_LAYERS = 32
BATCH = 12


class FakeModel:
    """Model stub with an explicit history layout; its state is a FakeLlama so the layer-0
    pre-pass can build a mask through ``_prepare_decoder_attention_mask``."""

    def __init__(self, result_list, n_layers=N_LAYERS):
        self.result_list = list(result_list)
        n_frames = len(self.result_list)
        self.state = FakeLlama(feat_cache=torch.zeros(n_frames, N_PER_FRAME, C), forwards=[],
                               layers=[FakeLayer(i) for i in range(n_layers)])
        self.lm_head = torch.nn.Linear(C, 20, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(20, C, generator=torch.Generator().manual_seed(5)))

    def get_model(self):
        return self.state

    def online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985):
        return torch.zeros(sum(self.result_list), C), list(self.result_list)


def make_probe(result_list=(4, 4, 4)):
    model = FakeModel(result_list)
    probe = SelectorProbe.__new__(SelectorProbe)
    probe.model = model
    probe.blinder = ColumnBlinder(IMAGE)
    probe.batch = BATCH
    probe.seed = 17
    probe.n_pivots = 8
    probe.action_ids = [10, 11, 12, 13]
    probe.label = "scene_0"
    probe._installed = False
    probe._embeds = probe._input_ids = None
    probe._head = None
    probe.blinder.install(model)
    probe._installed = True
    return model, probe


def capture(model, probe, ids):
    """As generate() would: mark the prompt, assemble the layout, store random prompt embeddings."""
    probe.begin_inference(ids)
    model.online_process_tensor(N_PER_FRAME)
    length = prompt_length(ids.numel(), model.result_list)
    g = torch.Generator().manual_seed(11)
    probe._embeds = torch.randn(length, C, generator=g)
    probe._input_ids = ids.clone()
    return length


class CriteriaTests(unittest.TestCase):
    def test_helpers_match_direct_computation(self):
        g = torch.Generator().manual_seed(1)
        rows = torch.randn(30, C, generator=g)
        cur = torch.tensor([3, 7, 29])
        x = F.normalize(rows, dim=-1)
        sim = x @ x.T
        for i, c in enumerate(cur.tolist()):
            others = [j for j in range(30) if j != c]
            self.assertAlmostEqual(float(nn_cosine_distance(rows, cur)[i]), float(1 - sim[c, others].max()), places=5)
        norms = torch.rand(30, generator=g)
        hidden = torch.randn(30, C, generator=g)
        pivots = torch.topk(norms, 8).indices.tolist()
        h = F.normalize(hidden, dim=-1)
        dis = dart_dissimilarity(hidden, norms, cur, 8)
        for i, c in enumerate(cur.tolist()):
            expect = 2.0 if c in pivots else float(1 - min(float(h[c] @ h[p]) for p in pivots))  # 1 - min_p cos
            self.assertAlmostEqual(float(dis[i]), expect, places=5)
        # a current token that is a pivot scores 2.0
        cur2 = torch.tensor([pivots[0]])
        self.assertEqual(float(dart_dissimilarity(hidden, norms, cur2, 8)[0]), 2.0)

    def test_random_positions_are_a_seeded_permutation(self):
        cur = torch.arange(20)
        a = random_positions(20, cur, 17, "ep", 5)
        b = random_positions(20, cur, 17, "ep", 5)
        c = random_positions(20, cur, 17, "ep", 6)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(sorted(a.tolist()), list(range(20)))
        d = random_positions(20, torch.tensor([4, 9]), 17, "ep", 5)
        self.assertEqual(d.tolist(), [a[4].item(), a[9].item()])


class ProbeTests(unittest.TestCase):
    def test_run_record_masks_and_criteria(self):
        model, probe = make_probe([4, 4, 4])
        ids = synthetic_ids()
        length = capture(model, probe, ids)
        embeds = probe._embeds.clone()
        placeholder = int(torch.nonzero(ids == IMAGE)[0])
        all_visual, current, total = visual_columns(placeholder, model.result_list, ids.numel())
        self.assertEqual(total, length)
        forwards = []

        def fake_hidden(batch, hide):
            forwards.append((batch.shape[0], None if hide is None else hide.clone()))
            # the last hidden state depends on what is hidden, so the KLs are non-trivial
            tilt = torch.zeros(batch.shape[0], 1) if hide is None else hide.float().sum(dim=1, keepdim=True) / 10.0
            return torch.ones(batch.shape[0], C) + tilt * torch.arange(C).float()

        probe._last_hidden = fake_hidden
        record = probe.run(ids)
        n_cur = 4 + NAV_TOKENS
        n_vis = 12 + NAV_TOKENS
        self.assertEqual(record["n_current"], n_cur)
        self.assertEqual(record["n_visual"], n_vis)
        self.assertEqual(record["n_frames"], 3)
        self.assertEqual(record["prompt_tokens"], length)
        self.assertEqual(record["is_nav"], [False] * 4 + [True] * NAV_TOKENS)
        for key in ("keep_all", "attn", "divprune_nn", "dart_dis", "random_pos", "is_nav", "current_cols"):
            self.assertEqual(len(record[key]), n_cur, key)
        self.assertTrue(all(v >= 0 for v in record["keep_all"]))
        self.assertTrue(all(math.isfinite(v) for v in record["keep_all"]))
        # forwards: base (unmasked, batch 1) then ceil(69 / BATCH) masked chunks
        self.assertEqual(forwards[0][0], 1)
        self.assertIsNone(forwards[0][1])
        chunks = forwards[1:]
        self.assertEqual(len(chunks), math.ceil((1 + n_cur) / BATCH))
        hides = torch.cat([h for _, h in chunks], dim=0)
        self.assertEqual(tuple(hides.shape), (1 + n_cur, length))
        self.assertTrue(torch.equal(hides[0], all_visual))
        cur_cols = torch.nonzero(current).flatten().tolist()
        self.assertEqual(record["current_cols"], cur_cols)
        for k, c in enumerate(cur_cols):
            expect = all_visual.clone()
            expect[c] = False
            self.assertTrue(torch.equal(hides[1 + k], expect))
        self.assertEqual(record["n_probes"], 1 + 1 + n_cur)
        # criteria against a direct computation from the fake layer 0 (its first call)
        attn, keys = layer0_draws(length)
        row = attn[0, :, -1, :].mean(0)
        pool = torch.nonzero(all_visual).flatten()
        cur_idx = torch.searchsorted(pool, torch.tensor(cur_cols))
        self.assertTrue(torch.allclose(torch.tensor(record["attn"]), row[cur_cols], atol=1e-6))
        nn_ref = nn_cosine_distance(embeds[pool], cur_idx)
        self.assertTrue(torch.allclose(torch.tensor(record["divprune_nn"]), nn_ref, atol=1e-5))
        hidden0 = embeds + 1.0  # FakeLayer returns hidden + 1
        norms = keys.abs().sum(-1)
        dart_ref = dart_dissimilarity(hidden0[pool], norms[pool], cur_idx, 8)
        self.assertTrue(torch.allclose(torch.tensor(record["dart_dis"]), dart_ref, atol=1e-5))
        self.assertEqual(record["random_pos"], random_positions(n_vis, cur_idx, 17, "scene_0", 2).tolist())
        # the KL of the reference against the base is what kl_no_visual holds
        self.assertAlmostEqual(record["kl_no_visual"], kl_divergence(record["logp_full"], record["logp_no_visual"]), places=9)
        # the layer-0 pass ran unmasked and the blinder is clear afterwards
        first_call = model.state.layers[0].calls[0]
        self.assertEqual(first_call[0], length)
        self.assertFalse(probe.blinder._active)
        self.assertIsNone(probe._embeds)

    def test_run_requires_a_capture(self):
        model, probe = make_probe([4, 4])
        with self.assertRaises(RuntimeError):
            probe.run(synthetic_ids())


class CliTests(unittest.TestCase):
    def test_parse_args_guards(self):
        args = parse_args(["--selector-probe", "--probe-batch", "12"])
        self.assertTrue(args.selector_probe)
        self.assertEqual(args.probe_batch, 12)
        self.assertFalse(parse_args([]).selector_probe)
        for argv in (
            ["--selector-probe", "--layer-probe"],
            ["--selector-probe", "--blind-layer", "16"],
            ["--selector-probe", "--context-frames", "32"],
            ["--selector-probe", "--no-visual-tokens"],
            ["--selector-probe", "--frame-influence"],
            ["--selector-probe", "--prune-method", "random", "--prune-k", "4", "--context-frames", "32"],
        ):
            with self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
