"""Unit tests for the LightNav-0 visual-blind wrapper (baselines/lightnav_hm3d/eval_blind.py).

No weights, no GPU, no vLLM: the masked kernel is checked against a naive per-query softmax
attention on a fake paged cache, and the ``FlashAttentionImpl.forward`` replacement is driven
with stub layers / metadata. Needs torch, so run it with the client runtime:

    PYTHONPATH=baselines/LightNav-0/src baselines/.envs/lightnav/bin/python \\
        -m unittest baselines.tests.test_lightnav_blind
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from types import SimpleNamespace

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch

    from baselines.lightnav_hm3d.eval_blind import (
        VIDEO_PAD_ID,
        BlindState,
        attention_mask,
        build_parser,
        check_generate_counts,
        gather_kv,
        layer_index,
        make_blind_forward,
        masked_attention,
        visual_positions,
    )

H, KVH, D = 4, 2, 8
BLOCK = 4
N_LAYERS = 36
VISION_START, VISION_END, TEXT = 151652, 151653, 1000


def naive_attention(query, key, value, scale, allowed):
    """Per-query softmax over the allowed key columns only (float64 reference)."""
    q, k, v = query.double(), key.double(), value.double()
    out = torch.zeros(q.shape, dtype=torch.float64)
    rep = q.shape[1] // k.shape[1]
    for i in range(q.shape[0]):
        cols = torch.nonzero(allowed[i]).flatten()
        for h in range(q.shape[1]):
            kh = k[cols, h // rep]
            w = torch.softmax((kh @ q[i, h]) * scale, dim=0)
            out[i, h] = w @ v[cols, h // rep]
    return out


def paged_cache(seq_total, seed):
    """A fake [2, num_blocks, BLOCK, KVH, D] cache whose logical sequence is scattered over
    non-contiguous blocks; returns (cache, block_table_row, key_seq, value_seq)."""
    g = torch.Generator().manual_seed(seed)
    n_blocks = -(-seq_total // BLOCK)
    perm = torch.randperm(n_blocks + 3, generator=g)[:n_blocks]  # non-identity block table
    cache = torch.randn(2, n_blocks + 3, BLOCK, KVH, D, generator=g)
    key_seq = cache[0, perm].reshape(-1, KVH, D)[:seq_total]
    value_seq = cache[1, perm].reshape(-1, KVH, D)[:seq_total]
    return cache, perm, key_seq, value_seq


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class KernelTest(unittest.TestCase):
    def check(self, seq_len, n_query, hide=None, seed=0):
        cache, row, key, value = paged_cache(seq_len, seed)
        g = torch.Generator().manual_seed(seed + 100)
        query = torch.randn(n_query, H, D, generator=g)
        k, v = gather_kv(cache[0], row, seq_len), gather_kv(cache[1], row, seq_len)
        self.assertTrue(torch.equal(k, key) and torch.equal(v, value))
        mask = attention_mask(n_query, seq_len, hide)
        got = masked_attention(query, k, v, 1 / math.sqrt(D), mask)
        want = naive_attention(query, key, value, 1 / math.sqrt(D), mask)
        self.assertEqual(got.shape, (n_query, H, D))
        torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-5)
        return mask

    def test_causal_full_prefill(self):
        mask = self.check(seq_len=10, n_query=10)
        self.assertTrue(torch.equal(mask, torch.ones(10, 10, dtype=torch.bool).tril()))

    def test_causal_partial_prefill_and_decode(self):
        mask = self.check(seq_len=11, n_query=5)  # prefix-cache hit: queries are positions 6..10
        self.assertEqual(mask[0].sum().item(), 7)
        self.assertEqual(mask[-1].sum().item(), 11)
        mask = self.check(seq_len=13, n_query=1)  # decode: one query sees every key
        self.assertTrue(mask.all())

    def test_hidden_columns_get_zero_weight(self):
        hide = torch.zeros(12, dtype=torch.bool)
        hide[3:7] = True
        mask = self.check(seq_len=12, n_query=12, hide=hide)
        self.assertFalse(mask[:, 3:7].any())  # hidden for every query row, visual rows included
        self.assertTrue(torch.equal(mask[:, :3], torch.ones(12, 3, dtype=torch.bool).tril(0)[:, :3] | (torch.arange(12)[:, None] >= torch.arange(3)[None, :])))
        self.assertTrue(mask.any(dim=1).all())  # position 0 is text, so no row is empty
        # the same hidden set on a decode step and on a partial prefill past the prompt
        self.check(seq_len=12, n_query=1, hide=hide)
        self.check(seq_len=20, n_query=3, hide=hide)  # generated tokens (>= 12) stay visible
        mask = attention_mask(3, 20, hide)
        self.assertTrue(torch.equal(mask[:, 12:], attention_mask(3, 20, None)[:, 12:]))  # causal only there
        self.assertFalse(mask[:, 3:7].any())

    def test_visual_positions(self):
        ids = [TEXT, VISION_START, VIDEO_PAD_ID, VIDEO_PAD_ID, VISION_END, TEXT, VISION_START, VIDEO_PAD_ID, VISION_END, TEXT]
        got = visual_positions(ids)
        self.assertEqual(got.tolist(), [False, False, True, True, False, False, False, True, False, False])

    def test_layer_index(self):
        self.assertEqual(layer_index("language_model.model.layers.17.self_attn.attn"), 17)
        with self.assertRaises(ValueError):
            layer_index("visual.blocks.3.attn")


class FakeImpl:
    """What the replacement reads from a FlashAttentionImpl instance."""

    def __init__(self):
        self.scale = 1 / math.sqrt(D)
        self.kv_cache_dtype = "auto"
        self.sliding_window = (-1, -1)
        self.alibi_slopes = None
        self.logits_soft_cap = 0
        self.attn_type = SimpleNamespace(name="DECODER")


def metadata(seq_len, n_query, row):
    return SimpleNamespace(num_actual_tokens=n_query, query_start_loc=torch.tensor([0, n_query]),
                           seq_lens=torch.tensor([seq_len]), block_table=row[None, :], causal=True)


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class ForwardHookTest(unittest.TestCase):
    def setUp(self):
        self.prompt = [TEXT, TEXT, VISION_START] + [VIDEO_PAD_ID] * 6 + [VISION_END, TEXT, TEXT]  # 12 tokens
        self.original_calls = []

    def original(self, impl, layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
        self.original_calls.append(layer.layer_name)
        return output.fill_(7.0)

    def run_forward(self, state, index, seq_len, n_query, seed=1):
        forward = make_blind_forward(state, self.original)
        cache, row, key, value = paged_cache(seq_len, seed)
        query = torch.randn(n_query, H, D, generator=torch.Generator().manual_seed(seed + 7))
        output = torch.zeros(n_query + 2, H, D)  # padded output buffer as vLLM hands it over
        layer = SimpleNamespace(layer_name=f"language_model.model.layers.{index}.self_attn.attn")
        got = forward(FakeImpl(), layer, query, query[:, :KVH], query[:, :KVH], cache, metadata(seq_len, n_query, row), output=output)
        self.assertIs(got, output)
        return output[:n_query], query, key, value

    def expect(self, out, query, key, value, hide):
        mask = attention_mask(query.shape[0], key.shape[0], hide)
        torch.testing.assert_close(out.double(), naive_attention(query, key, value, 1 / math.sqrt(D), mask), rtol=1e-5, atol=1e-5)

    def test_layers_below_and_above_L(self):
        state = BlindState(16)
        state.n_layers = N_LAYERS
        state.strict = True
        state.set_visual(visual_positions(self.prompt))
        hide = visual_positions(self.prompt)
        for index, hidden in ((0, False), (15, False), (16, True), (35, True)):
            out, query, key, value = self.run_forward(state, index, seq_len=12, n_query=12)
            self.expect(out, query, key, value, hide if hidden else None)
            out, query, key, value = self.run_forward(state, index, seq_len=15, n_query=1)  # decode, 3 generated tokens
            self.expect(out, query, key, value, hide if hidden else None)
        self.assertEqual(self.original_calls, [])
        self.assertEqual((state.calls, state.hidden_calls), (8, 4))

    def test_identity_depth_hides_nothing_but_uses_the_kernel(self):
        state = BlindState(N_LAYERS)
        state.n_layers = N_LAYERS
        state.strict = True
        state.set_visual(visual_positions(self.prompt))
        out, query, key, value = self.run_forward(state, 35, seq_len=12, n_query=12)
        self.expect(out, query, key, value, None)
        self.assertEqual((state.calls, state.hidden_calls), (1, 0))

    def test_text_only_hides_from_layer_zero(self):
        state = BlindState(0)
        state.n_layers = N_LAYERS
        state.set_visual(visual_positions(self.prompt))
        out, query, key, value = self.run_forward(state, 0, seq_len=12, n_query=12)
        self.expect(out, query, key, value, visual_positions(self.prompt))

    def test_warmup_falls_through_and_strict_mode_refuses_unhooked_forwards(self):
        state = BlindState(16)
        forward = make_blind_forward(state, self.original)
        layer = SimpleNamespace(layer_name="language_model.model.layers.3.self_attn.attn")
        out = torch.zeros(2, H, D)
        forward(FakeImpl(), layer, None, None, None, None, None, output=out)  # profiling: no metadata
        self.assertEqual(self.original_calls, [layer.layer_name])
        cache, row, _, _ = paged_cache(8, 0)
        forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)  # before the engine is built
        self.assertEqual(len(self.original_calls), 2)
        state.strict = True  # engine built, no generate in flight: a decoder forward is a bug
        with self.assertRaises(RuntimeError):
            forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)
        encoder = FakeImpl()
        encoder.attn_type = SimpleNamespace(name="ENCODER")  # never masked, never refused
        forward(encoder, layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)
        self.assertEqual(len(self.original_calls), 3)

    def test_rejects_batches(self):
        state = BlindState(16)
        state.set_visual(visual_positions(self.prompt))
        forward = make_blind_forward(state, self.original)
        cache, row, _, _ = paged_cache(12, 0)
        md = metadata(12, 12, row)
        md.query_start_loc = torch.tensor([0, 6, 12])
        layer = SimpleNamespace(layer_name="language_model.model.layers.3.self_attn.attn")
        with self.assertRaises(RuntimeError):
            forward(FakeImpl(), layer, torch.zeros(12, H, D), None, None, cache, md, output=torch.zeros(12, H, D))

    def test_generate_counts(self):
        state = BlindState(20)
        state.n_layers = N_LAYERS
        state.calls, state.hidden_calls = 36 * 5, 16 * 5  # 5 forwards, 16 blind layers each
        self.assertEqual(check_generate_counts(state, 12, 6), 5)
        state.hidden_calls = 16 * 5 - 1
        with self.assertRaises(RuntimeError):
            check_generate_counts(state, 12, 6)
        state.calls, state.hidden_calls = 0, 0  # compiled / graph path: hook never ran
        with self.assertRaises(RuntimeError):
            check_generate_counts(state, 12, 6)
        state.calls, state.hidden_calls = 36, 16
        with self.assertRaises(RuntimeError):
            check_generate_counts(state, 12, 0)  # a prompt without video tokens


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class CliTest(unittest.TestCase):
    def test_blind_layer_required_and_passthrough(self):
        args, rest = build_parser().parse_known_args(["--blind-layer", "16", "--episodes", "-1", "--max_steps", "500"])
        self.assertEqual((args.blind_layer, rest), (16, ["--episodes", "-1", "--max_steps", "500"]))
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--episodes", "-1"])
        with self.assertRaises(ValueError):
            BlindState(-1)


if __name__ == "__main__":
    unittest.main()
