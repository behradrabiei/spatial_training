"""Unit tests for uninavid_hm3d.prune (no model weights, no Habitat).

Run from the repo root with the baseline runtime:
    PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python \
        -m unittest baselines.tests.test_prune
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from uninavid_hm3d.eval import parse_args
from uninavid_hm3d.prune import (
    N_PER_FRAME,
    NAV_TOKENS,
    STAGES,
    UNIFORM_BLIND_LAYER,
    PruneState,
    TokenPruner,
    describe,
    history_columns,
    pack_bits,
    prompt_length,
    select_dart,
    select_divprune,
    select_random,
    select_topk,
    selection_seed,
    model_stage_budgets,
    stage_budgets,
    unpack_bits,
)

IMAGE = -200
SEP, VSTART, VEND, ISTART, IEND, NAV = 32005, 32000, 32001, 32002, 32003, 32004
C = 6


def synthetic_ids(n_text=3, n_tail=2):
    text = list(range(100, 100 + n_text))
    tail = list(range(200, 200 + n_tail))
    return torch.tensor(text + [VSTART, SEP, IMAGE, VEND, ISTART, IEND, NAV] + tail)


def table():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32006, C, generator=g)


def assemble(ids, groups, nav, tab):
    """Prompt embeddings the way uninavid_arch builds them from explicit history groups."""
    s = int(torch.nonzero(ids == IMAGE).flatten()[0])
    parts = [tab[ids[:s]]]
    for j, g in enumerate(groups):
        parts.append(g)
        if j < len(groups) - 1:
            parts.append(tab[ids[s - 1:s]])
    parts += [tab[ids[s + 1:s + 3]], nav, tab[ids[s + 3:]]]
    return torch.cat(parts, dim=0)


def groups_from(tokens, result_list):
    out, pos = [], 0
    for g in result_list:
        out.append(tokens[pos:pos + g])
        pos += g
    return out


class FakeLlama(SimpleNamespace):
    """LlamaModel stub: records what reaches the real forward (the flat pre-pass wraps it)."""

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_length):
        return causal_mask(input_shape[-1])

    def forward(self, **kwargs):
        self.forwards.append(kwargs)
        return kwargs


class FakeModel:
    """Model stub: a feature cache, the upstream early-return compressor and fake layers."""

    def __init__(self, n_layers=32):
        self.state = FakeLlama(feat_cache=None, forwards=[], layers=[FakeLayer(i) for i in range(n_layers)])

    def get_model(self):
        return self.state

    def online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985):
        k, m, c = self.state.feat_cache.shape
        return self.state.feat_cache.reshape(-1, c), [nav_size] * k


class FakeLayer:
    """Decoder-layer stub recording the mask it received; attention rows are random."""

    def __init__(self, index):
        self.index = index
        self.calls = []
        self.gen = torch.Generator().manual_seed(100 + index)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False):
        self.calls.append((hidden_states.shape[1], attention_mask))
        q = hidden_states.shape[1]
        kv = attention_mask.shape[-1]
        attn = torch.softmax(torch.randn(1, 4, q, kv, generator=self.gen), dim=-1)
        present = (torch.randn(1, 4, kv, 3, generator=self.gen), torch.zeros(1, 4, kv, 3))
        out = (hidden_states + 1.0,)
        if output_attentions:
            out += (attn,)
        if use_cache:
            out += (present,)
        return out


def make_pruner(method, k=2, n_frames=6, seed=17, pivots=2, layer=None, scope="last_stage", anchor="original"):
    model = FakeModel()
    g = torch.Generator().manual_seed(3)
    model.state.feat_cache = torch.randn(n_frames, N_PER_FRAME, C, generator=g)
    pruner = TokenPruner(method, k, image_token_index=IMAGE, seed=seed, n_pivots=pivots, layer=layer,
                         scope=scope, anchor=anchor)
    pruner.install(model)
    pruner.begin_episode("scene_0")
    pruner.state.advance(0, list(range(n_frames)))
    return model, pruner


def causal_mask(length):
    return (torch.ones(length, length).triu(1) * torch.finfo(torch.float16).min).half().view(1, 1, length, length)


def run_prefill(model, pruner, ids, step):
    """Drive one inference through the installed hooks without any weights."""
    pruner.begin_inference(ids, step=step)
    tokens, result_list = model.online_process_tensor(N_PER_FRAME)
    length = prompt_length(ids.numel(), result_list)
    hidden = torch.zeros(1, length, C)
    mask = causal_mask(length)
    for layer in model.state.layers:
        out = layer.forward(hidden, attention_mask=mask, position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=True)
        hidden = out[0]
        assert len(out) == 2 and isinstance(out[1], tuple), "layer tuple layout broken"
    decode_mask = torch.zeros(1, 1, 1, length + 3, dtype=torch.float16)
    for layer in model.state.layers:
        layer.forward(torch.zeros(1, 1, C), attention_mask=decode_mask, position_ids=None,
                      past_key_value=("k", "v"), output_attentions=False, use_cache=True)
    return tokens, result_list, mask, decode_mask, pruner.finish()


class SelectorTests(unittest.TestCase):
    def test_random_is_seeded_and_exact(self):
        seed = selection_seed(17, "scene_0", 5)
        a = select_random(50, 8, torch.Generator().manual_seed(seed))
        b = select_random(50, 8, torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(set(a.tolist())), 8)
        self.assertNotEqual(selection_seed(17, "scene_0", 5), selection_seed(17, "scene_0", 6))
        self.assertNotEqual(selection_seed(17, "scene_0", 5), selection_seed(17, "scene_1", 5))

    def test_topk(self):
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        self.assertEqual(select_topk(scores, 2).tolist(), [1, 3])

    def test_divprune_is_greedy_maximin(self):
        g = torch.Generator().manual_seed(4)
        x = torch.randn(30, 8, generator=g)
        keep = select_divprune(x, 10)
        self.assertEqual(len(set(keep.tolist())), 10)
        xn = F.normalize(x, dim=-1)
        dist = 1 - xn @ xn.T
        nn_dist = dist.masked_fill(torch.eye(30, dtype=torch.bool), float("inf")).amin(1)
        first = int(nn_dist.argmax())
        self.assertIn(first, keep.tolist())
        # Replay the greedy rule: every pick maximises the min distance to the picks so far.
        chosen = [first]
        for _ in range(9):
            d = dist[:, chosen].amin(1)
            d[chosen] = -float("inf")
            pick = int(d.argmax())
            chosen.append(pick)
        self.assertEqual(sorted(chosen), keep.tolist())
        self.assertTrue(torch.equal(select_divprune(x, 40), torch.arange(30)))

    def test_dart_pivots_and_quotas(self):
        g = torch.Generator().manual_seed(5)
        keys = torch.randn(20, 12, generator=g)
        hidden = torch.randn(20, 8, generator=g)
        keep = select_dart(keys, hidden, budget=9, n_pivots=2)
        self.assertEqual(len(set(keep.tolist())), 9)
        pivots = torch.topk(keys.abs().sum(-1), 2).indices.tolist()
        for p in pivots:
            self.assertIn(p, keep.tolist())
        # Reference: pivot 0 takes 4 (3 + remainder), pivot 1 takes 3 least-similar of the rest.
        sim = F.normalize(hidden, dim=-1) @ F.normalize(hidden, dim=-1).T
        remaining = set(range(20)) - set(pivots)
        expected = set(pivots)
        for rank, p in enumerate(pivots):
            take = 3 + (1 if rank < 1 else 0)
            picks = sorted(remaining, key=lambda j: float(sim[p, j]))[:take]
            expected |= set(picks)
            remaining -= set(picks)
        self.assertEqual(sorted(expected), keep.tolist())
        self.assertTrue(torch.equal(select_dart(keys, hidden, 25, 8), torch.arange(20)))

    def test_stage_budgets(self):
        b = stage_budgets(120, 32, 3)
        self.assertEqual(b[-1], 32)
        self.assertEqual(b, sorted(b, reverse=True))
        self.assertLess(b[0], 120)
        self.assertEqual(stage_budgets(20, 32, 3), [20, 20, 20])
        self.assertEqual(stage_budgets(120, 32, 1), [32])

    def test_bits_round_trip(self):
        bits = [True, False, False, True, True, False, True, True, True, False, False]
        self.assertEqual(unpack_bits(pack_bits(bits), len(bits)).tolist(), bits)

    def test_layout_helpers(self):
        self.assertEqual(history_columns(5, [4, 4, 4]), [(5, 9), (10, 14), (15, 19)])
        self.assertEqual(history_columns(5, [2, 4]), [(5, 7), (8, 12)])
        ids = synthetic_ids()
        tab = table()
        feat = torch.randn(3, N_PER_FRAME, C)
        embeds = assemble(ids, [feat[i] for i in range(3)], torch.zeros(NAV_TOKENS, C), tab)
        self.assertEqual(prompt_length(ids.numel(), [4, 4, 4]), embeds.shape[0])


class LedgerTests(unittest.TestCase):
    def test_advance_trims_and_purges(self):
        state = PruneState(n_stages=2)
        state.advance(0, [0])
        state.advance(1, [1, 2])
        self.assertEqual(state.frame_ids, [0, 1, 2])
        state.delete(0, [(0, 1)])
        state.delete(1, [(1, 3)])
        self.assertEqual(state.deleted, [{(0, 1)}, {(0, 1), (1, 3)}])  # nested
        # window of 3 frames, two new: keep one old row
        state.advance(1, [3, 4])
        self.assertEqual(state.frame_ids, [2, 3, 4])
        self.assertEqual(state.deleted, [set(), set()])
        self.assertEqual(state.history, [2, 3])
        self.assertEqual(len(state.pool(0)), 8)
        with self.assertRaises(RuntimeError):
            state.advance(5, [6])

    def test_kept_and_bits(self):
        state = PruneState(n_stages=1)
        state.advance(0, [0, 1, 2])
        state.delete(0, [(0, 0), (0, 1), (1, 3)])
        self.assertEqual(state.kept_per_frame(0), [2, 3])
        self.assertEqual(state.keep_bits(0), [False, False, True, True, True, True, True, False])


class InputLevelHookTests(unittest.TestCase):
    def setUp(self):
        self.ids = synthetic_ids()
        self.tab = table()
        self.nav = torch.zeros(NAV_TOKENS, C)

    def test_random_removes_rows_and_empty_groups(self):
        model, pruner = make_pruner("random", k=1, n_frames=6)  # pool 20 tokens, budget 4
        feat = model.state.feat_cache
        tokens, result_list, _, _, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(record["n_pool"], 20)
        self.assertEqual(sum(record["n_kept"]), 4)
        self.assertEqual(record["frames"], [0, 1, 2, 3, 4])
        self.assertEqual(record["n_tokens"], [4] * 5)
        self.assertIsNone(record["stage_kept"])
        # The returned prompt equals the one assembled from the surviving groups only.
        kept_groups = []
        for f in range(6):
            keep = [t for t in range(4) if (f, t) not in pruner.state.deleted[0]]
            if keep:
                kept_groups.append(feat[f][keep])
        self.assertEqual(result_list, [g.shape[0] for g in kept_groups])
        self.assertEqual(result_list[-1], 4)  # current frame untouched
        ref = assemble(self.ids, kept_groups, self.nav, self.tab)
        got = assemble(self.ids, groups_from(tokens, result_list), self.nav, self.tab)
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(got.shape[0], record["prompt_tokens"])
        bits = unpack_bits(record["bits"], 20)
        self.assertEqual(int(bits.sum()), 4)
        self.assertEqual(bits.tolist(), pruner.state.keep_bits(0))

    def test_random_persists_and_is_reproducible(self):
        model, pruner = make_pruner("random", k=1, n_frames=6)
        run_prefill(model, pruner, self.ids, step=5)
        deleted_1 = set(pruner.state.deleted[0])
        # Next inference: two new frames, window of 6 -> drop the two oldest frames.
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        _, result_list, _, _, record = run_prefill(model, pruner, self.ids, step=7)
        self.assertEqual(record["frames"], [2, 3, 4, 5, 6])
        survivors = {tok for tok in deleted_1 if tok[0] >= 2}
        self.assertTrue(survivors <= pruner.state.deleted[0])
        self.assertEqual(sum(record["n_kept"]), 4)
        # Same seed, same episode, same step -> same deletions.
        model2, pruner2 = make_pruner("random", k=1, n_frames=6)
        run_prefill(model2, pruner2, self.ids, step=5)
        self.assertEqual(pruner2.state.deleted[0], deleted_1)

    def test_nothing_pruned_below_budget(self):
        model, pruner = make_pruner("divprune", k=8, n_frames=6)  # budget 32 > pool 20
        tokens, result_list, _, _, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(result_list, [4] * 6)
        self.assertTrue(torch.equal(tokens, model.state.feat_cache.reshape(-1, C)))
        self.assertEqual(sum(record["n_kept"]), 20)

    def test_divprune_uses_feature_rows(self):
        model, pruner = make_pruner("divprune", k=2, n_frames=6)  # budget 8 of 20
        run_prefill(model, pruner, self.ids, step=5)
        pool = [(f, t) for f in range(5) for t in range(4)]
        rows = model.state.feat_cache[:5].reshape(-1, C)
        expected = {pool[i] for i in select_divprune(rows, 8).tolist()}
        self.assertEqual(set(pool) - pruner.state.deleted[0], expected)


class LayerHookTests(unittest.TestCase):
    def setUp(self):
        self.ids = synthetic_ids()

    def hidden_columns(self, mask):
        return set(torch.nonzero((mask[0, 0] == torch.finfo(mask.dtype).min).all(dim=0)).flatten().tolist())

    def test_fastv_hides_from_layer_two_on_prefill_and_decode(self):
        model, pruner = make_pruner("fastv", k=1, n_frames=6)  # keep 4 of 20
        tokens, result_list, mask, decode_mask, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(result_list, [4] * 6)  # rows stay in the prompt
        self.assertEqual(sum(record["n_kept"]), 4)
        spans = history_columns(int(torch.nonzero(self.ids == IMAGE)[0]), result_list)
        deleted_cols = {spans[f][0] + t for f, t in pruner.state.deleted[0]}
        self.assertEqual(len(deleted_cols), 16)
        layers = model.state.layers
        for i in (0, 1):
            self.assertIs(layers[i].calls[0][1], mask)  # untouched below the cut
        for i in range(2, 32):
            got = layers[i].calls[0][1]
            self.assertIsNot(got, mask)
            self.assertEqual(self.hidden_columns(got), deleted_cols)
            unchanged = torch.ones(mask.shape[-1], dtype=torch.bool)
            unchanged[list(deleted_cols)] = False
            self.assertTrue(torch.equal(got[..., unchanged], mask[..., unchanged]))
            dec = layers[i].calls[1][1]
            self.assertEqual(set(torch.nonzero(dec[0, 0, 0] == torch.finfo(dec.dtype).min).flatten().tolist()),
                             deleted_cols)
        self.assertTrue(torch.equal(mask, causal_mask(mask.shape[-1])))  # shared mask never mutated
        self.assertIs(layers[1].calls[1][1], decode_mask)

    def test_fastv_keeps_highest_last_row_attention(self):
        model, pruner = make_pruner("fastv", k=1, n_frames=6)
        run_prefill(model, pruner, self.ids, step=5)
        # Recompute the score the stub produced at layer 1 (same generator state, second draw).
        layer = model.state.layers[1]
        q = layer.calls[0][0]
        gen = torch.Generator().manual_seed(101)
        attn = torch.softmax(torch.randn(1, 4, q, q, generator=gen), dim=-1)
        row = attn[0, :, -1, :].mean(0)
        spans = history_columns(int(torch.nonzero(self.ids == IMAGE)[0]), [4] * 6)
        pool = [(f, t) for f in range(5) for t in range(4)]
        cols = torch.tensor([spans[f][0] + t for f, t in pool])
        expected = {pool[i] for i in torch.topk(row[cols], 4).indices.tolist()}
        self.assertEqual(set(pool) - pruner.state.deleted[0], expected)

    def test_pyramiddrop_stages_are_nested(self):
        model, pruner = make_pruner("pyramiddrop", k=1, n_frames=6)
        _, _, mask, _, record = run_prefill(model, pruner, self.ids, step=5)
        budgets = stage_budgets(20, 4, 3)
        self.assertEqual(record["stage_budgets"], budgets)
        self.assertEqual(record["stage_kept"], budgets)
        d = pruner.state.deleted
        self.assertTrue(d[0] <= d[1] <= d[2])
        self.assertEqual([20 - len(x) for x in d], budgets)
        layers = model.state.layers
        for i in range(8):
            self.assertIs(layers[i].calls[0][1], mask)
        for lo, hi, stage in ((8, 16, 0), (16, 24, 1), (24, 32, 2)):
            for i in range(lo, hi):
                self.assertEqual(len(self.hidden_columns(layers[i].calls[0][1])), len(d[stage]))

    def test_hybrid_budgets_nesting_and_layers(self):
        model, pruner = make_pruner("hybrid", k=2, n_frames=6)  # pool 20, B' = 8, tail 4
        _, result_list, mask, _, record = run_prefill(model, pruner, self.ids, step=5)
        budgets = stage_budgets(20, 8, 2) + [4]
        self.assertEqual(result_list, [4] * 6)  # rows stay in the prompt
        self.assertEqual(record["stage_budgets"], budgets)
        self.assertEqual(record["stage_kept"], budgets)
        self.assertEqual(sum(record["n_kept"]), 4)
        d = pruner.state.deleted
        self.assertTrue(d[0] <= d[1] <= d[2])
        layers = model.state.layers
        for i in range(8):
            self.assertIs(layers[i].calls[0][1], mask)
        for lo, hi, stage in ((8, 16, 0), (16, 24, 1), (24, 32, 2)):
            for i in range(lo, hi):
                self.assertEqual(len(self.hidden_columns(layers[i].calls[0][1])), len(d[stage]))
                dec = layers[i].calls[1][1]
                self.assertEqual(int((dec[0, 0, 0] == torch.finfo(dec.dtype).min).sum()), len(d[stage]))

    def test_hybrid_tail_is_a_seeded_draw_among_survivors(self):
        model, pruner = make_pruner("hybrid", k=2, n_frames=6)
        run_prefill(model, pruner, self.ids, step=5)
        d = pruner.state.deleted
        pool = [(f, t) for f in range(5) for t in range(4) if (f, t) not in d[1]]
        draw = select_random(len(pool), 4, torch.Generator().manual_seed(selection_seed(17, "scene_0", 5)))
        self.assertEqual(set(pool) - d[2], {pool[i] for i in draw.tolist()})
        # No attention is read for the random stage: layer 23 is never asked for its weights.
        self.assertEqual(len(model.state.layers[23].calls), 2)
        other_model, other = make_pruner("hybrid", k=2, n_frames=6, seed=18)
        run_prefill(other_model, other, self.ids, step=5)
        self.assertEqual(other.state.deleted[1], d[1])  # attention stages ignore the seed
        self.assertNotEqual(other.state.deleted[2], d[2])

    def test_hybrid_tail_binds_before_the_schedule(self):
        model, pruner = make_pruner("hybrid", k=16, n_frames=6)  # B' = 64 > pool 20
        _, _, _, _, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(record["stage_budgets"], [20, 20, 4])
        self.assertEqual(record["stage_kept"], [20, 20, 4])
        self.assertEqual(pruner.state.deleted[0], set())

    def test_hybrid_k1_tail_drops_nothing_more_and_persists(self):
        model, pruner = make_pruner("hybrid", k=1, n_frames=6)
        _, _, _, _, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(record["stage_budgets"][1:], [4, 4])
        self.assertEqual(pruner.state.deleted[1], pruner.state.deleted[2])
        first = set(pruner.state.deleted[2])
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        _, _, _, _, record = run_prefill(model, pruner, self.ids, step=7)
        self.assertTrue({t for t in first if t[0] >= 2} <= pruner.state.deleted[2])
        self.assertEqual(record["stage_kept"][-1], 4)

    def test_dart_persists_and_reaches_budget(self):
        model, pruner = make_pruner("dart", k=1, n_frames=6, pivots=2)
        _, _, _, _, record = run_prefill(model, pruner, self.ids, step=5)
        self.assertEqual(sum(record["n_kept"]), 4)
        first = set(pruner.state.deleted[0])
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        _, _, _, _, record = run_prefill(model, pruner, self.ids, step=7)
        self.assertTrue({t for t in first if t[0] >= 2} <= pruner.state.deleted[0])
        self.assertEqual(sum(record["n_kept"]), 4)
        self.assertEqual(record["frames"], [2, 3, 4, 5, 6])

    def test_hooks_are_inert_outside_an_inference(self):
        model, pruner = make_pruner("fastv", k=1, n_frames=6)
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.float16)
        out = model.state.layers[5].forward(torch.zeros(1, 3, C), attention_mask=mask, use_cache=True)
        self.assertIs(model.state.layers[5].calls[0][1], mask)
        self.assertEqual(len(out), 2)
        tokens, result_list = model.online_process_tensor(N_PER_FRAME)
        self.assertEqual(result_list, [4] * 6)


def run_flat(model, pruner, ids, step, nav, tab):
    """One flat (layer-0 pre-pass) inference: assemble the prompt from what the online wrapper
    returns, send it through the wrapped LlamaModel.forward, then one decode call."""
    pruner.begin_inference(ids, step=step)
    tokens, result_list = model.online_process_tensor(N_PER_FRAME)
    embeds = assemble(ids, groups_from(tokens, result_list), nav, tab)[None]
    mask = torch.ones(1, embeds.shape[1], dtype=torch.bool)
    sent = model.state.forward(input_ids=None, attention_mask=mask, past_key_values=None,
                               inputs_embeds=embeds, use_cache=True)
    decode = model.state.forward(input_ids=None, attention_mask=None, past_key_values=("kv",),
                                 inputs_embeds=torch.zeros(1, 1, C), use_cache=True)
    return embeds, result_list, sent, decode, pruner.finish()


def layer0_draws(q):
    """The attention and key tensors FakeLayer(0) produces on its first call."""
    gen = torch.Generator().manual_seed(100)
    attn = torch.softmax(torch.randn(1, 4, q, q, generator=gen), dim=-1)
    keys = torch.randn(1, 4, q, 3, generator=gen)
    return attn, keys.permute(0, 2, 1, 3).reshape(q, -1)


class FlatHookTests(unittest.TestCase):
    """--prune-layer 0: fastv/dart score on a decoder-layer-0 pre-pass and delete by row removal."""

    def setUp(self):
        self.ids = synthetic_ids()
        self.tab = table()
        self.nav = torch.zeros(NAV_TOKENS, C)
        self.placeholder = int(torch.nonzero(self.ids == IMAGE)[0])
        self.pool = [(f, t) for f in range(5) for t in range(4)]

    def surviving_groups(self, feat, frames, deleted):
        groups = []
        for row, f in enumerate(frames):
            keep = [t for t in range(4) if (f, t) not in deleted]
            if keep:
                groups.append(feat[row][keep])
        return groups

    def test_fastv_is_topk_of_the_layer0_row_and_removes_rows(self):
        model, pruner = make_pruner("fastv", k=1, n_frames=6, layer=0)  # keep 4 of 20
        feat = model.state.feat_cache
        embeds, result_list, sent, decode, record = run_flat(model, pruner, self.ids, 5, self.nav, self.tab)
        self.assertEqual(result_list, [4] * 6)  # nothing deleted before this inference
        attn, _ = layer0_draws(embeds.shape[1])
        row = attn[0, :, -1, :].mean(0)
        spans = history_columns(self.placeholder, [4] * 6)
        cols = torch.tensor([spans[f][0] + t for f, t in self.pool])
        expected = {self.pool[i] for i in torch.topk(row[cols], 4).indices.tolist()}
        self.assertEqual(set(self.pool) - pruner.state.deleted[0], expected)
        # What reaches the real prefill is the prompt rebuilt from the surviving groups only:
        # deleted rows gone, emptied frames gone with their separator.
        ref = assemble(self.ids, self.surviving_groups(feat, range(6), pruner.state.deleted[0]), self.nav, self.tab)
        self.assertTrue(torch.equal(sent["inputs_embeds"][0], ref))
        self.assertEqual(sent["attention_mask"].shape[1], ref.shape[0])
        self.assertEqual(record["prompt_tokens"], ref.shape[0])
        self.assertEqual(sum(record["n_kept"]), 4)
        self.assertEqual(record["n_pool"], 20)
        # One pre-pass on layer 0, no decoder-layer hook anywhere, decode passes through.
        self.assertEqual(len(model.state.layers[0].calls), 1)
        self.assertTrue(all("forward" not in vars(layer) for layer in model.state.layers))
        self.assertEqual(decode["inputs_embeds"].shape[1], 1)
        self.assertEqual(len(model.state.forwards), 2)

    def test_dart_reads_layer0_keys_and_output(self):
        model, pruner = make_pruner("dart", k=2, n_frames=6, pivots=2, layer=0)  # keep 8 of 20
        embeds, _, _, _, record = run_flat(model, pruner, self.ids, 5, self.nav, self.tab)
        _, keys = layer0_draws(embeds.shape[1])
        hidden = embeds[0] + 1.0  # FakeLayer output
        spans = history_columns(self.placeholder, [4] * 6)
        cols = torch.tensor([spans[f][0] + t for f, t in self.pool])
        expected = {self.pool[i] for i in select_dart(keys[cols], hidden[cols], 8, 2).tolist()}
        self.assertEqual(set(self.pool) - pruner.state.deleted[0], expected)
        self.assertEqual(sum(record["n_kept"]), 8)

    def test_second_inference_scores_the_row_pruned_prompt(self):
        model, pruner = make_pruner("fastv", k=1, n_frames=6, layer=0)
        run_flat(model, pruner, self.ids, 5, self.nav, self.tab)
        first = set(pruner.state.deleted[0])
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        feat, frames = model.state.feat_cache, [2, 3, 4, 5, 6, 7]
        survivors = {tok for tok in first if tok[0] >= 2}
        embeds, result_list, sent, _, record = run_flat(model, pruner, self.ids, 7, self.nav, self.tab)
        # Earlier deletions are physically absent from the prompt the pre-pass scores.
        before = self.surviving_groups(feat, frames, survivors)
        self.assertEqual(result_list, [g.shape[0] for g in before])
        self.assertTrue(torch.equal(embeds[0], assemble(self.ids, before, self.nav, self.tab)))
        kept_in_window = sum(1 for tok in self.pool if tok not in first and tok[0] >= 2)
        self.assertEqual(record["n_pool"], kept_in_window + 8)  # survivors still in the window + two new frames
        # Independent column map of the variable-width layout, second draw of layer 0's generator.
        gen = torch.Generator().manual_seed(100)
        q1 = prompt_length(self.ids.numel(), [4] * 6)
        torch.randn(1, 4, q1, q1, generator=gen), torch.randn(1, 4, q1, 3, generator=gen)
        q2 = embeds.shape[1]
        row = torch.softmax(torch.randn(1, 4, q2, q2, generator=gen), dim=-1)[0, :, -1, :].mean(0)
        pool, cols, pos = [], [], self.placeholder
        for f in frames:
            alive = [t for t in range(4) if (f, t) not in survivors]
            if not alive:
                continue
            if f != frames[-1]:
                pool += [(f, t) for t in alive]
                cols += list(range(pos, pos + len(alive)))
            pos += len(alive) + 1
        expected = {pool[i] for i in torch.topk(row[torch.tensor(cols)], 4).indices.tolist()}
        self.assertEqual(set(pool) - pruner.state.deleted[0], expected)
        self.assertTrue(survivors <= pruner.state.deleted[0])
        self.assertEqual(sum(record["n_kept"]), 4)
        ref = assemble(self.ids, self.surviving_groups(feat, frames, pruner.state.deleted[0]), self.nav, self.tab)
        self.assertTrue(torch.equal(sent["inputs_embeds"][0], ref))

    def test_below_budget_skips_the_prepass(self):
        model, pruner = make_pruner("fastv", k=8, n_frames=6, layer=0)  # budget 32 > pool 20
        embeds, _, sent, _, record = run_flat(model, pruner, self.ids, 5, self.nav, self.tab)
        self.assertEqual(model.state.layers[0].calls, [])
        self.assertIs(sent["inputs_embeds"], embeds)
        self.assertEqual(sum(record["n_kept"]), 20)

    def test_input_level_methods_ignore_the_flag(self):
        for method in ("random", "divprune"):
            a_model, a = make_pruner(method, k=1, n_frames=6)
            b_model, b = make_pruner(method, k=1, n_frames=6, layer=0)
            run_prefill(a_model, a, self.ids, step=5)
            tokens, result_list = None, None
            b.begin_inference(self.ids, step=5)
            tokens, result_list = b_model.online_process_tensor(N_PER_FRAME)
            b.finish()
            self.assertEqual(a.state.deleted[0], b.state.deleted[0])
            self.assertFalse(b.flat)
            self.assertEqual(sum(result_list), 4 + 4)

    def test_inert_outside_an_inference(self):
        model, pruner = make_pruner("dart", k=1, n_frames=6, layer=0)
        embeds = torch.zeros(1, 9, C)
        sent = model.state.forward(inputs_embeds=embeds, attention_mask=None, past_key_values=None)
        self.assertIs(sent["inputs_embeds"], embeds)
        self.assertEqual(model.state.layers[0].calls, [])

    def test_defaults_are_unchanged_and_bad_layers_refused(self):
        self.assertEqual(STAGES, {"fastv": ((1, 2),), "dart": ((1, 2),),
                                  "pyramiddrop": ((7, 8), (15, 16), (23, 24)),
                                  "hybrid": ((7, 8), (15, 16), (None, 24)),
                                  "uniform": ((0, 0), (None, 20))})
        self.assertEqual(TokenPruner("fastv", 1, IMAGE).stages, ((1, 2),))
        self.assertFalse(TokenPruner("dart", 1, IMAGE).flat)
        old = describe("fastv", 8, 17, 8)
        self.assertEqual((old["placement"], old["first_hidden_layer"], old["stages"], old["prune_layer"],
                          old["removal"]), ("layer", 2, [[1, 2]], None, "columns"))
        flat = describe("fastv", 8, 17, 8, 0)
        self.assertEqual((flat["placement"], flat["first_hidden_layer"], flat["stages"], flat["prune_layer"],
                          flat["removal"]), ("layer0-prepass", 0, [[0, 0]], 0, "rows"))
        self.assertEqual(describe("random", 8, 17, 8, 0)["removal"], "rows")
        for method, layer in (("pyramiddrop", 0), ("hybrid", 0), ("uniform", 0), ("fastv", 2)):
            with self.assertRaises(ValueError):
                TokenPruner(method, 4, IMAGE, layer=layer)
            with self.assertRaises(ValueError):
                describe(method, 4, 17, 8, layer)


def run_whole(model, pruner, ids, step, nav, tab):
    """One whole-model-budget inference: the wrapped LlamaModel.forward runs the layer-0 pre-pass
    and shortens the prompt, then every decoder layer sees the shortened prefill and one decode."""
    pruner.begin_inference(ids, step=step)
    tokens, result_list = model.online_process_tensor(N_PER_FRAME)
    embeds = assemble(ids, groups_from(tokens, result_list), nav, tab)[None]
    sent = model.state.forward(input_ids=None, attention_mask=torch.ones(1, embeds.shape[1], dtype=torch.bool),
                               past_key_values=None, inputs_embeds=embeds, use_cache=True)
    length = sent["inputs_embeds"].shape[1]
    hidden, mask = torch.zeros(1, length, C), causal_mask(length)
    for layer in model.state.layers:
        hidden = layer.forward(hidden, attention_mask=mask, position_ids=None, past_key_value=None,
                               output_attentions=False, use_cache=True)[0]
    decode_mask = torch.zeros(1, 1, 1, length + 3, dtype=torch.float16)
    for layer in model.state.layers:
        layer.forward(torch.zeros(1, 1, C), attention_mask=decode_mask, position_ids=None,
                      past_key_value=("k", "v"), output_attentions=False, use_cache=True)
    return result_list, sent, mask, pruner.finish()


class WholeModelBudgetTests(unittest.TestCase):
    """scope='model': a layer-0 pre-pass stage (row removal) in front of the method's column stages,
    budgets solved so the layer-weighted mean of attended history tokens is B'."""

    def setUp(self):
        self.ids = synthetic_ids()
        self.tab = table()
        self.nav = torch.zeros(NAV_TOKENS, C)
        self.placeholder = int(torch.nonzero(self.ids == IMAGE)[0])

    def hidden_columns(self, mask):
        return set(torch.nonzero((mask[0, 0] == torch.finfo(mask.dtype).min).all(dim=0)).flatten().tolist())

    def columns_after_prepass(self, frames, d0, tokens):
        """Prompt columns of `tokens` once the rows deleted at stage 0 (and emptied frames) are gone."""
        kept = {f: [t for t in range(4) if (f, t) not in d0] for f in frames}
        groups = [f for f in frames if kept[f]]
        spans = history_columns(self.placeholder, [len(kept[f]) for f in groups])
        return {spans[groups.index(f)][0] + kept[f].index(t) for f, t in tokens}

    def test_solver_meets_the_layer_weighted_mean(self):
        b = model_stage_budgets(124, 32, [8, 8, 8, 8])
        self.assertLessEqual(abs(sum(8 * x for x in b) / 32 - 32), 1)
        self.assertEqual(b, sorted(b, reverse=True))
        t = model_stage_budgets(124, 32, [8, 8, 8], tail=(8, 4))
        self.assertLessEqual(abs((sum(8 * x for x in t) + 8 * 4) / 32 - 32), 1)
        self.assertEqual(model_stage_budgets(20, 32, [8, 8, 8, 8]), [20] * 4)  # under budget: nothing pruned
        self.assertEqual(model_stage_budgets(20, 32, [8, 8, 8], tail=(8, 4)), [20] * 3)

    def check_arm(self, method, widths_sched, tail, anchor="original"):
        # pool 20 (= the original count on a first inference), B' = 8 over the model
        model, pruner = make_pruner(method, k=2, n_frames=6, scope="model", anchor=anchor)
        result_list, sent, mask, record = run_whole(model, pruner, self.ids, 5, self.nav, self.tab)
        budgets = model_stage_budgets(20, 8, widths_sched, tail=tail) + ([4] if tail else [])
        self.assertEqual(record["stage_budgets"], budgets)
        self.assertEqual(record["stage_kept"], budgets)
        self.assertAlmostEqual(record["held_tokens"], sum(8 * b for b in budgets) / 32)
        self.assertLessEqual(abs(record["held_tokens"] - 8), 1)
        d = pruner.state.deleted
        self.assertTrue(d[0] <= d[1] <= d[2] <= d[3])
        # Stage 0 removed its rows before the real prefill; emptied frames lost their separator.
        groups = [g for g in ([model.state.feat_cache[f][[t for t in range(4) if (f, t) not in d[0]]]
                               for f in range(6)]) if g.shape[0]]
        self.assertTrue(torch.equal(sent["inputs_embeds"][0], assemble(self.ids, groups, self.nav, self.tab)))
        self.assertEqual(record["prompt_tokens"], sent["inputs_embeds"].shape[1])
        layers = model.state.layers
        self.assertEqual(len(layers[0].calls), 3)  # the pre-pass, the prefill, the decode
        for i in range(8):
            self.assertIs(layers[i].calls[-2][1], mask)  # layers 0-7 attend to every surviving row
        for lo, hi, stage in ((8, 16, 1), (16, 24, 2), (24, 32, 3)):
            want = self.columns_after_prepass(range(5), d[0], d[stage] - d[0])
            for i in range(lo, hi):
                self.assertEqual(self.hidden_columns(layers[i].calls[0][1]), want)
                dec = layers[i].calls[1][1]
                self.assertEqual(set(torch.nonzero(dec[0, 0, 0] == torch.finfo(dec.dtype).min).flatten().tolist()), want)
        # Stage 1 ranks by layer 7's last-row attention READ AT THE SHORTENED PROMPT'S COLUMNS: a stale
        # column map would still hit the budget, so the survivors themselves are checked.
        q = sent["inputs_embeds"].shape[1]
        attn = torch.softmax(torch.randn(1, 4, q, q, generator=torch.Generator().manual_seed(107)), dim=-1)
        row = attn[0, :, -1, :].mean(0)
        pool = [(f, t) for f in range(5) for t in range(4) if (f, t) not in d[0]]
        cols = torch.tensor(sorted(self.columns_after_prepass(range(5), d[0], [tok]).pop() for tok in pool))
        order = {c: tok for tok, c in ((tok, self.columns_after_prepass(range(5), d[0], [tok]).pop()) for tok in pool)}
        expected = {order[int(cols[i])] for i in torch.topk(row[cols], budgets[1]).indices.tolist()}
        self.assertEqual(set(pool) - d[1], expected)
        return model, pruner

    def test_small_budget_spends_the_clamped_tail(self):
        b = model_stage_budgets(124, 8, [8, 8, 8], tail=(8, 4))
        self.assertLess(b[-1], 4)
        self.assertLessEqual(abs((8 * sum(b) + 8 * min(4, b[-1])) / 32 - 8), 1)

    def test_pyramiddrop(self):
        self.check_arm("pyramiddrop", [8, 8, 8, 8], None)

    def test_hybrid_and_persistence(self):
        model, pruner = self.check_arm("hybrid", [8, 8, 8], (8, 4))
        first = [set(x) for x in pruner.state.deleted]
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        result_list, _, _, record = run_whole(model, pruner, self.ids, 7, self.nav, self.tab)
        for s in range(4):
            self.assertTrue({t for t in first[s] if t[0] >= 2} <= pruner.state.deleted[s])
        # Rows deleted at stage 0 by the first inference never reach the prompt again.
        self.assertEqual(sum(result_list), 4 * 6 - len({t for t in first[0] if t[0] >= 2}))
        self.assertEqual(record["stage_kept"][-1], 4)
        self.assertLessEqual(record["held_tokens"], 8 + 1)

    def test_pool_anchor_shrinks_from_what_stage_zero_holds(self):
        model, pruner = self.check_arm("pyramiddrop", [8, 8, 8, 8], None, anchor="pool")
        kept0 = 20 - len(pruner.state.deleted[0])
        records = []
        for step, new in ((7, [6, 7]), (9, [8, 9]), (11, [10, 11])):
            model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
            pruner.state.advance(4, new)
            records.append(run_whole(model, pruner, self.ids, step, self.nav, self.tab)[3])
        # Second inference: stage 0 draws from its survivors in the 3 remaining old frames + 2 new frames.
        self.assertLess(records[0]["n_pool"], 20)
        for rec in records:
            self.assertEqual(rec["stage_budgets"], model_stage_budgets(rec["n_pool"], 8, [8, 8, 8, 8]))
            self.assertLessEqual(rec["held_tokens"], 8 + 1)
        self.assertGreater(kept0, 0)
        original = model_stage_budgets(20, 8, [8, 8, 8, 8])
        self.assertGreater(records[-1]["stage_budgets"][-1], original[-1])  # a gentler pyramid than from h x 4

    def test_scope_guards_and_defaults(self):
        self.assertFalse(TokenPruner("pyramiddrop", 2, IMAGE).whole)
        self.assertEqual(TokenPruner("pyramiddrop", 2, IMAGE).stages, STAGES["pyramiddrop"])
        d = describe("hybrid", 8, 17, 8, None, "model")
        self.assertEqual((d["stages"][0], d["budget_scope"], d["stage_selectors"]),
                         ([0, 0], "model", ["attn", "attn", "attn", "random"]))
        self.assertEqual(describe("hybrid", 8, 17, 8)["budget_scope"], "last_stage")
        for method, layer, scope in (("fastv", None, "model"), ("random", None, "model"),
                                     ("pyramiddrop", None, "layers")):
            with self.assertRaises(ValueError):
                TokenPruner(method, 4, IMAGE, layer=layer, scope=scope)
        for method, anchor in (("fastv", "pool"), ("pyramiddrop", "cache")):
            with self.assertRaises(ValueError):
                TokenPruner(method, 4, IMAGE, anchor=anchor)
        self.assertEqual(describe("hybrid", 8, 17, 8, None, "model", "pool")["schedule_anchor"], "pool")
        self.assertIsNone(describe("fastv", 8, 17, 8)["schedule_anchor"])
        args = parse_args(["--prune-method", "hybrid", "--prune-k", "8", "--context-frames", "32",
                           "--prune-budget-scope", "model"])
        self.assertEqual(args.prune_budget_scope, "model")
        with self.assertRaises(SystemExit):
            parse_args(["--prune-method", "fastv", "--prune-k", "8", "--context-frames", "32",
                        "--prune-budget-scope", "model"])


class UniformProtocolTests(WholeModelBudgetTests):
    """method='uniform' (paper/hybrid_prune): an input-level uniform draw at stage 0 (rows removed, so
    layers 0..19 all hold the same tokens) and a BLIND stage from layer 20 -- every history column
    plus the current frame's group and 64 fine tokens hidden; budget solved over the whole model."""

    def test_solver_with_a_zero_tail(self):
        self.assertEqual(model_stage_budgets(124, 32, [20], tail=(12, 0)), [51])  # 32 * 32 / 20 = 51.2
        self.assertEqual(model_stage_budgets(20, 32, [20], tail=(12, 0)), [20])  # under budget
        self.assertEqual(model_stage_budgets(124, 8, [20], tail=(12, 0)), [13])

    def test_guards_and_describe(self):
        with self.assertRaises(ValueError):
            TokenPruner("uniform", 2, IMAGE)  # last-stage scope has no meaning below a blind cut
        with self.assertRaises(ValueError):
            describe("uniform", 8, 17, 8, None, "last_stage")
        pruner = TokenPruner("uniform", 2, IMAGE, scope="model", anchor="pool")
        self.assertEqual(pruner.stages, ((0, 0), (None, UNIFORM_BLIND_LAYER)))  # no second layer-0 stage
        self.assertTrue(pruner.prepass and pruner.whole and pruner.blind_stage)
        d = describe("uniform", 8, 17, 8, None, "model", "pool")
        self.assertEqual((d["stages"], d["stage_selectors"], d["tail_budget_tokens"], d["blind_layer"], d["seed"],
                          d["removal"], d["schedule_anchor"]),
                         ([[0, 0], [None, 20]], ["random", "blind"], 0, 20, 17, "rows at layer 0, columns after", "pool"))
        args = parse_args(["--prune-method", "uniform", "--prune-k", "8", "--context-frames", "32",
                           "--prune-budget-scope", "model", "--prune-schedule-anchor", "pool"])
        self.assertEqual(args.prune_method, "uniform")
        for argv in (["--prune-method", "uniform", "--prune-k", "8", "--context-frames", "32"],
                     ["--prune-method", "uniform", "--prune-k", "8", "--context-frames", "32",
                      "--prune-budget-scope", "model", "--prune-layer", "0"]):
            with self.assertRaises(SystemExit):
                parse_args(argv)

    def expected_blind_columns(self, d0, history):
        """Columns of every surviving history token, the current frame's group and the fine tokens."""
        frames = list(range(6))
        kept = {f: [t for t in range(4) if (f, t) not in d0] for f in frames}
        groups = [f for f in frames if kept[f]]
        spans = history_columns(self.placeholder, [len(kept[f]) for f in groups])
        cols = self.columns_after_prepass(frames, d0, [tok for tok in history if tok not in d0]
                                          + [(5, t) for t in range(4)])
        nav = spans[-1][1] + 2
        return cols | set(range(nav, nav + NAV_TOKENS))

    def test_uniform_arm(self):
        model, pruner = make_pruner("uniform", k=2, n_frames=6, scope="model", anchor="pool")  # pool 20, B' = 8
        result_list, sent, mask, record = run_whole(model, pruner, self.ids, 5, self.nav, self.tab)
        budgets = model_stage_budgets(20, 8, [20], tail=(12, 0)) + [0]
        self.assertEqual(budgets, [13, 0])
        self.assertEqual(record["stage_budgets"], budgets)
        self.assertEqual(record["stage_kept"], budgets)
        self.assertAlmostEqual(record["held_tokens"], 20 * 13 / 32)
        self.assertEqual(sum(record["n_kept"]), 13)  # reported at the stage below the blind cut
        d = pruner.state.deleted
        history = [(f, t) for f in range(5) for t in range(4)]
        # Stage 0: the seeded uniform draw, rows removed before the prefill (no layer-0 pre-pass forward).
        draw = select_random(20, 13, torch.Generator().manual_seed(selection_seed(17, "scene_0", 5)))
        self.assertEqual(set(history) - d[0], {history[i] for i in draw.tolist()})
        self.assertEqual(d[1], set(history))  # the blind stage holds no history token
        groups = [g for g in ([model.state.feat_cache[f][[t for t in range(4) if (f, t) not in d[0]]]
                               for f in range(6)]) if g.shape[0]]
        self.assertTrue(torch.equal(sent["inputs_embeds"][0], assemble(self.ids, groups, self.nav, self.tab)))
        self.assertEqual(sum(result_list), 4 * 6 - 7)
        layers = model.state.layers
        self.assertEqual(len(layers[0].calls), 2)  # prefill + decode only
        for i in range(20):
            self.assertIs(layers[i].calls[0][1], mask)  # every layer below the cut: the same rows, nothing hidden
            self.assertIs(layers[i].calls[1][1], layers[0].calls[1][1])
        want = self.expected_blind_columns(d[0], history)
        self.assertEqual(len(want), 13 + 4 + NAV_TOKENS)
        for i in range(20, 32):
            self.assertEqual(self.hidden_columns(layers[i].calls[0][1]), want)
            dec = layers[i].calls[1][1]
            self.assertEqual(set(torch.nonzero(dec[0, 0, 0] == torch.finfo(dec.dtype).min).flatten().tolist()), want)
        other_model, other = make_pruner("uniform", k=2, n_frames=6, seed=18, scope="model", anchor="pool")
        run_whole(other_model, other, self.ids, 5, self.nav, self.tab)
        self.assertNotEqual(other.state.deleted[0], d[0])
        # Persistence: rows deleted at stage 0 never come back; the blind stage stays empty.
        first = set(d[0])
        model.state.feat_cache = torch.cat([model.state.feat_cache[2:], torch.randn(2, N_PER_FRAME, C)])
        pruner.state.advance(4, [6, 7])
        result_list, _, _, record = run_whole(model, pruner, self.ids, 7, self.nav, self.tab)
        self.assertTrue({t for t in first if t[0] >= 2} <= pruner.state.deleted[0])
        # Stage 0 selects inside the input hook, so this step's draw is gone from the prompt too.
        self.assertEqual(sum(result_list), 4 * 6 - len(pruner.state.deleted[0]))
        self.assertEqual(record["stage_kept"], [13, 0])
        self.assertLessEqual(record["held_tokens"], 8 + 1)

    def test_first_inference_without_history_hides_the_current_frame(self):
        model, pruner = make_pruner("uniform", k=2, n_frames=1, scope="model", anchor="pool")
        result_list, _, mask, record = run_whole(model, pruner, self.ids, 0, self.nav, self.tab)
        self.assertEqual((result_list, record["stage_kept"], record["held_tokens"]), ([4], [0, 0], 0.0))
        layers = model.state.layers
        self.assertIs(layers[19].calls[0][1], mask)
        spans = history_columns(self.placeholder, [4])
        want = set(range(*spans[-1])) | set(range(spans[-1][1] + 2, spans[-1][1] + 2 + NAV_TOKENS))
        self.assertEqual(self.hidden_columns(layers[20].calls[0][1]), want)


class DescribeAndCliTests(unittest.TestCase):
    def test_describe(self):
        d = describe("pyramiddrop", 8, 17, 8)
        self.assertEqual(d["budget_tokens"], 32)
        self.assertEqual(d["stages"], [list(s) for s in STAGES["pyramiddrop"]])
        self.assertEqual(describe("identity", None, 17, 8)["budget_tokens"], None)
        self.assertEqual(describe("random", 4, 17, 8)["placement"], "input")
        h = describe("hybrid", 8, 17, 8)
        self.assertEqual((h["stages"], h["stage_selectors"], h["tail_budget_tokens"], h["seed"], h["removal"]),
                         ([[7, 8], [15, 16], [None, 24]], ["attn", "attn", "random"], 4, 17, "columns"))
        self.assertIsNone(d["stage_selectors"])
        with self.assertRaises(ValueError):
            describe("bogus", 4, 17, 8)
        with self.assertRaises(ValueError):
            TokenPruner("identity", 4, IMAGE)

    def test_parse_args_guards(self):
        args = parse_args(["--prune-method", "fastv", "--prune-k", "8", "--context-frames", "32"])
        self.assertEqual((args.prune_method, args.prune_k, args.context_window), ("fastv", 8, 31))
        self.assertEqual(parse_args([]).prune_method, "identity")
        self.assertIsNone(args.prune_layer)
        flat = parse_args(["--prune-method", "dart", "--prune-k", "32", "--context-frames", "64", "--prune-layer", "0"])
        self.assertEqual((flat.prune_layer, flat.context_window), (0, 63))
        for argv in (
            ["--prune-method", "fastv", "--context-frames", "32"],  # no k
            ["--prune-method", "fastv", "--prune-k", "8"],  # no window
            ["--prune-method", "random", "--prune-k", "8", "--context-frames", "32", "--no-visual-tokens"],
            ["--prune-method", "random", "--prune-k", "8", "--frame-influence"],
            ["--prune-k", "8", "--context-frames", "32"],  # k without a method
            ["--prune-method", "pyramiddrop", "--prune-k", "8", "--context-frames", "32", "--prune-layer", "0"],
            ["--prune-method", "hybrid", "--prune-k", "8", "--context-frames", "32", "--prune-layer", "0"],
            ["--prune-method", "hybrid", "--prune-k", "8", "--context-frames", "65"],
            ["--prune-layer", "0", "--context-frames", "32"],  # layer without a method
            ["--prune-method", "fastv", "--prune-k", "8", "--context-frames", "32", "--prune-layer", "2"],
            ["--prune-method", "fastv", "--prune-k", "8", "--context-frames", "65"],  # long-term tier
            ["--prune-method", "fastv", "--prune-k", "8", "--context-window", "64"],
        ):
            with self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
