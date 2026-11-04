"""Unit tests for the LightNav-0 pruning wrapper (baselines/lightnav_hm3d/eval_prune.py).

No weights, no GPU, no vLLM: the prompt layout, the deletion ledger, the four selection rules
(against naive references), the layer-0 scoring hook and the layer-1 capture are checked on
synthetic tensors and stub layers. Needs torch (and lightnav for the sampler-parity checks), so
run it with the client runtime:

    PYTHONPATH=baselines/LightNav-0/src baselines/.envs/lightnav/bin/python \\
        -m unittest baselines.tests.test_lightnav_prune
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_LIGHTNAV = HAVE_TORCH and importlib.util.find_spec("lightnav") is not None
if HAVE_TORCH:
    import numpy as np
    import torch

    from baselines.lightnav_hm3d.eval_blind import attention_mask
    from baselines.lightnav_hm3d.eval_prune import (
        TOKENS_PER_FRAME,
        VIDEO_PAD_ID,
        PruneState,
        Tubelet,
        apply_drop,
        build_parser,
        check_generate_counts,
        dart_select,
        decision_row,
        divprune_select,
        fastv_select,
        kept_per_tubelet,
        make_layer1_prehook,
        make_prune_forward,
        output_dir_from,
        pack_bits,
        plan_step,
        random_select,
        recency_select,
        single_current,
        selection_seed,
        split_history,
        token_key,
        tubelet_layout,
    )

H, KVH, D = 4, 2, 8
BLOCK = 4
N_LAYERS = 36
VISION_START, VISION_END, TEXT = 151652, 151653, 1000
MERGE = 2
CKPT_TIERS = Path(__file__).resolve().parents[1] / "LightNav-0/checkpoints/LightNav-0/eval_config.json"


def tiers():
    """The checkpoint's SlowFast tiers if present, otherwise upstream's span defaults."""
    from lightnav.slowfast import DEFAULT_SLOWFAST_TIERS_SPAN, validate_slowfast_tiers
    if CKPT_TIERS.is_file():
        return validate_slowfast_tiers(json.loads(CKPT_TIERS.read_text())["tasks"]["vlnce"]["slowfast_tiers"])
    return validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN)


def grid_for(pool: int, n_tubelets: int):
    """Post-pool video_grid_thw row of a segment of 256x448 frames at `pool` (pre-merge space)."""
    h, w = math.ceil(8 / pool) * MERGE, math.ceil(14 / pool) * MERGE
    return [n_tubelets, h, w]


def prompt_for(tubelets):
    """A prompt whose <|video_pad|> runs follow the tubelet layout: text, then per tubelet
    [TEXT, VISION_START, pads..., VISION_END], then text; returns (ids, pad positions)."""
    ids = [TEXT, TEXT, TEXT]
    for t in tubelets:
        ids += [TEXT, VISION_START] + [VIDEO_PAD_ID] * t.n_tokens + [VISION_END]
    ids += [TEXT, TEXT]
    pads = [i for i, x in enumerate(ids) if x == VIDEO_PAD_ID]
    return ids, pads


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class LayoutTest(unittest.TestCase):
    def test_tubelets_follow_segments_and_grids(self):
        segments = [([0, 1], 4), ([40, 41, 52, 53], 4), ([70, 71, 72, 73, 74, 75], 2), ([98, 99], 1)]
        grids = [grid_for(4, 1), grid_for(4, 2), grid_for(2, 3), grid_for(1, 1)]
        tubs = tubelet_layout(segments, grids)
        self.assertEqual([t.n_tokens for t in tubs], [8, 8, 8, 28, 28, 28, 112])
        self.assertEqual([(t.f0, t.f1) for t in tubs], [(0, 1), (40, 41), (52, 53), (70, 71), (72, 73), (74, 75), (98, 99)])
        self.assertEqual([t.pool for t in tubs], [4, 4, 4, 2, 2, 2, 1])
        self.assertEqual([t.start for t in tubs], [0, 8, 16, 24, 52, 80, 108])
        self.assertEqual([t.segment for t in tubs], [0, 1, 1, 2, 2, 2, 3])
        hist, cur = split_history(tubs, 99)
        self.assertEqual([t.f1 for t in cur], [99])
        self.assertEqual(len(hist), 6)

    def test_layout_errors(self):
        with self.assertRaises(ValueError):
            tubelet_layout([([0, 1], 4)], [])
        with self.assertRaises(ValueError):
            tubelet_layout([([0, 1, 2], 2)], [grid_for(2, 1)])  # odd frame count
        with self.assertRaises(ValueError):
            tubelet_layout([([0, 1], 2)], [[1, 7, 14]])  # not a multiple of the merge size
        with self.assertRaises(RuntimeError):
            split_history(tubelet_layout([([0, 1], 2)], [grid_for(2, 1)]), 5)

    def test_step_zero_has_no_history(self):
        tubs = tubelet_layout([([0, 0], 1)], [grid_for(1, 1)])  # padded single frame
        hist, cur = split_history(tubs, 0)
        self.assertEqual((len(hist), len(cur)), (0, 1))
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 0, set(), pads, len(ids))
        self.assertEqual((plan.cand_ranks, bool(plan.hide.any())), ([], False))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class LedgerTest(unittest.TestCase):
    def layout(self, current, fast_pairs, pool1=True):
        segs = [(sum(([a, a + 1] for a in fast_pairs), []), 2), ([current - 1, current], 1)]
        grids = [grid_for(2, len(fast_pairs)), grid_for(1, 1)]
        return tubelet_layout(segs, grids)

    def test_ledger_hides_only_recorded_tokens_and_recurs(self):
        ledger = set()
        tubs = self.layout(10, [2, 4, 6])  # fast tubelets (2,3) (4,5) (6,7), current (9,10)
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 10, ledger, pads, len(ids))
        self.assertEqual(len(plan.cand_ranks), 3 * 28)
        self.assertEqual(plan.cand_keys[0], (2, 3, 2, 0))
        self.assertEqual(plan.cand_keys[28], (4, 5, 2, 0))
        keep = torch.tensor(sorted(range(84))[:56])  # drop the (6,7) tubelet's 28 tokens
        self.assertEqual(apply_drop(plan, keep, ledger, pads), 28)
        self.assertEqual(len(ledger), 28)
        self.assertTrue(all(k[:2] == (6, 7) for k in ledger))
        n_kept, bits = kept_per_tubelet(plan, pads)
        self.assertEqual(n_kept, [28, 28, 0])
        self.assertEqual(len(bits), 84)
        self.assertEqual(sum(bits), 56)
        # the next step has the other parity: nothing recorded applies, every token is a candidate
        tubs = self.layout(11, [3, 5, 7])
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 11, ledger, pads, len(ids))
        self.assertEqual(len(plan.cand_ranks), 84)
        self.assertFalse(plan.hide.any())
        # two steps on, (6,7) recurs: its tokens are hidden, not candidates
        tubs = self.layout(12, [4, 6, 8])
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 12, ledger, pads, len(ids))
        self.assertEqual(len(plan.cand_ranks), 56)
        self.assertEqual(int(plan.hide.sum()), 28)
        hidden_pos = set(torch.nonzero(plan.hide).flatten().tolist())
        t67 = [t for t in plan.history if (t.f0, t.f1) == (6, 7)][0]
        self.assertEqual(hidden_pos, {pads[t67.start + c] for c in range(28)})
        n_kept, _ = kept_per_tubelet(plan, pads)
        self.assertEqual(n_kept, [28, 0, 28])
        # a re-pooled recurrence of the same frames is a fresh identity
        tubs = tubelet_layout([([6, 7], 4), ([11, 12], 1)], [grid_for(4, 1), grid_for(1, 1)])
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 12, ledger, pads, len(ids))
        self.assertEqual(len(plan.cand_ranks), 8)
        self.assertEqual(token_key(plan.history[0], 3), (6, 7, 4, 3))

    def test_current_tubelet_never_hidden(self):
        ledger = {(9, 10, 1, c) for c in range(112)}  # bogus entries for the current pair
        tubs = self.layout(10, [2])
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 10, ledger, pads, len(ids))
        self.assertFalse(plan.hide.any())
        self.assertEqual(len(plan.cand_ranks), 28)

    def test_budget_zero_hides_every_history_token(self):
        # k = 0 (the no-visual-history control): every history token is dropped, the current
        # tubelet and the text skeleton stay; the bookkeeping handles an empty keep set.
        self.assertEqual(PruneState("random", 0).budget, 0)
        with self.assertRaises(ValueError):
            PruneState("random", -1)
        g = torch.Generator().manual_seed(3)
        self.assertEqual(random_select(56, 0, g).numel(), 0)
        ledger = set()
        tubs = self.layout(10, [2, 4])  # history (2,3) (4,5), current (9,10)
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 10, ledger, pads, len(ids))
        self.assertEqual(len(plan.cand_ranks), 56)
        self.assertEqual(apply_drop(plan, torch.empty(0, dtype=torch.long), ledger, pads), 56)
        self.assertEqual(len(ledger), 56)
        n_kept, bits = kept_per_tubelet(plan, pads)
        self.assertEqual((n_kept, len(bits), sum(bits)), ([0, 0], 56, 0))
        self.assertNotEqual(pack_bits(bits), "")
        hidden = set(torch.nonzero(plan.hide).flatten().tolist())
        self.assertEqual(hidden, set(pads[:56]))  # the current pair's 112 positions stay visible
        self.assertEqual(int(plan.hide.sum()), 56)
        # next same-parity step: the recorded tokens are hidden, only the new pair is a candidate
        tubs = self.layout(12, [2, 4, 6])
        ids, pads = prompt_for(tubs)
        plan = plan_step(tubs, 12, ledger, pads, len(ids))
        self.assertEqual((len(plan.cand_ranks), int(plan.hide.sum())), (28, 56))


@unittest.skipUnless(HAVE_LIGHTNAV, "needs lightnav (run with PYTHONPATH=baselines/LightNav-0/src)")
class SamplerParityTest(unittest.TestCase):
    """The claims the ledger rests on, checked on upstream's own layout function."""

    def pairs(self, segs, tier_pool):
        return [(ids[i], ids[i + 1]) for s in segs if s["pool_spatial"] == tier_pool
                for ids in [s["frame_ids"]] for i in range(0, len(ids), 2)]

    def test_fast_tubelets_recur_every_other_step_and_keep_identity_into_mid(self):
        from lightnav.slowfast import slowfast_segments
        T = tiers()
        fast = {t: set(self.pairs(slowfast_segments(t, t + 1, T), 2)) for t in range(60, 70)}
        for t in range(60, 68):
            common_next = fast[t] & fast[t + 1]
            common_two = fast[t] & fast[t + 2]
            self.assertEqual(common_next, set(), f"step {t}: a pool-2 tubelet shared with step {t + 1}")
            self.assertGreater(len(common_two), 10, f"step {t}: pool-2 tubelets do not recur at {t + 2}")
        # the pair that ages out of the fast tier (age 34, 35) is the mid tier's newest burst pair
        segs = slowfast_segments(100, 101, T)
        pool2 = self.pairs(segs, 2)
        self.assertIn((100 - 35, 100 - 34), pool2)
        self.assertIn((100 - 33, 100 - 32), pool2)

    def test_token_counts_at_plateau(self):
        from lightnav.slowfast import slowfast_segments
        segs = slowfast_segments(499, 500, tiers())
        layout = [(s["frame_ids"], s["pool_spatial"]) for s in segs]
        grids = [grid_for(s["pool_spatial"], len(s["frame_ids"]) // 2) for s in segs]
        tubs = tubelet_layout(layout, grids)
        hist, cur = split_history(tubs, 499)
        self.assertEqual(sum(t.n_tokens for t in cur), 112)
        self.assertEqual(sum(t.n_tokens for t in hist), 952 - 112)
        self.assertEqual(sum(t.n_tokens for t in hist if t.pool == 2), 16 * 28 + 10 * 28)
        self.assertEqual(TOKENS_PER_FRAME * 32, 16 * 28)  # k=32 = the whole fast tier


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class SelectorTest(unittest.TestCase):
    def test_selection_seed_matches_the_shared_formula(self):
        self.assertEqual(selection_seed(17, "episode_003", 42), (17 * 1_000_003 + 42) ^ zlib.crc32(b"episode_003"))
        try:
            from baselines.uninavid_hm3d.prune import selection_seed as uninavid_seed
        except Exception:  # the Uni-NaVid module needs its own runtime
            return
        self.assertEqual(selection_seed(17, "x_1", 5), uninavid_seed(17, "x_1", 5))

    def test_random_is_seeded_and_sized(self):
        g = torch.Generator().manual_seed(selection_seed(17, "episode_000", 7))
        a = random_select(100, 30, g)
        g = torch.Generator().manual_seed(selection_seed(17, "episode_000", 7))
        b = random_select(100, 30, g)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual((a.numel(), len(set(a.tolist())), bool((a[1:] > a[:-1]).all())), (30, 30, True))
        g = torch.Generator().manual_seed(selection_seed(17, "episode_000", 8))
        self.assertFalse(torch.equal(random_select(100, 30, g), a))

    def test_divprune_matches_naive_max_min(self):
        g = torch.Generator().manual_seed(3)
        rows = torch.randn(40, 16, generator=g)
        got = divprune_select(rows, 12).tolist()
        x = torch.nn.functional.normalize(rows.double(), dim=-1)
        dist = 1.0 - x @ x.T
        nn = dist.clone()
        nn.fill_diagonal_(float("inf"))
        chosen = [int(nn.min(dim=1).values.argmax())]
        while len(chosen) < 12:
            md = dist[:, chosen].min(dim=1).values
            md[chosen] = -float("inf")
            chosen.append(int(md.argmax()))
        self.assertEqual(got, sorted(chosen))
        self.assertEqual(divprune_select(rows, 40).tolist(), list(range(40)))
        self.assertEqual(divprune_select(rows, 0).numel(), 0)

    def test_dart_matches_naive_pivot_quotas(self):
        g = torch.Generator().manual_seed(5)
        n, cap, P = 50, 20, 8
        norms = torch.rand(n, generator=g)
        hidden = torch.randn(n, 16, generator=g)
        got = dart_select(norms, hidden, cap, P).tolist()
        pivots = sorted(range(n), key=lambda i: -float(norms[i]))[:P]
        h = torch.nn.functional.normalize(hidden.double(), dim=-1)
        available = set(range(n)) - set(pivots)
        kept = list(pivots)
        quota, extra = divmod(cap - P, P)
        for rank, p in enumerate(pivots):
            q = quota + (1 if rank < extra else 0)
            sims = sorted(available, key=lambda i: float(h[i] @ h[p]))[:q]
            kept += sims
            available -= set(sims)
        self.assertEqual(got, sorted(kept))
        self.assertEqual(len(got), cap)
        self.assertTrue(set(pivots) <= set(got))
        self.assertEqual(dart_select(norms, hidden, 5, P).numel(), 5)  # pivots clipped to the capacity

    def test_recency_keeps_newest_tubelets_whole(self):
        # three history tubelets, prompt order oldest first: (10,11) x 8 tokens, (20,21) x 28, (22,23) x 28
        keys = [(10, 11, 4, c) for c in range(8)] + [(20, 21, 2, c) for c in range(28)] + [(22, 23, 2, c) for c in range(28)]
        keep = recency_select(keys, 28)
        self.assertEqual(keep.tolist(), list(range(36, 64)))  # exactly the newest tubelet
        keep = recency_select(keys, 30)
        self.assertEqual(keep.tolist(), [8, 9] + list(range(36, 64)))  # then the next-newest, in prompt order
        self.assertEqual(recency_select(keys, 0).numel(), 0)
        self.assertEqual(recency_select(keys, 100).tolist(), list(range(64)))

    def test_single_current_doubles_the_current_frame_only(self):
        frames = torch.arange(12).view(12, 1)  # 12 "frames"
        segs = [{"frame_indices": [4, 5, 8, 9], "pool_spatial": 2, "video": frames[[4, 5, 8, 9]]},
                {"frame_indices": [10, 11], "pool_spatial": 1, "video": frames[[10, 11]]}]
        out = single_current(segs, 11, frames)
        self.assertEqual([s["frame_indices"] for s in out], [[4, 5, 8, 9], [11, 11]])
        self.assertEqual(out[-1]["video"].flatten().tolist(), [11, 11])
        self.assertEqual(out[-1]["pool_spatial"], 1)
        self.assertIs(out[0], segs[0])
        step0 = [{"frame_indices": [0, 0], "pool_spatial": 1, "video": frames[[0, 0]]}]
        self.assertIs(single_current(step0, 0, frames), step0)
        with self.assertRaises(RuntimeError):
            single_current([{"frame_indices": [9, 10], "pool_spatial": 1, "video": frames[[9, 10]]}], 11, frames)
        with self.assertRaises(ValueError):
            PruneState("recency", 0, current_frames=3)
        self.assertEqual(PruneState("recency", 0, current_frames=1).describe()["current_frames"], 1)

    def test_fastv_top_scores(self):
        scores = torch.tensor([0.1, 0.5, 0.5, 0.05, 0.9])
        self.assertEqual(fastv_select(scores, 3).tolist(), [1, 2, 4])
        self.assertEqual(fastv_select(scores, 9).tolist(), [0, 1, 2, 3, 4])

    def test_pack_bits_round_trip(self):
        bits = [True, False, False, True, True, False, True, True, True, False]
        packed = pack_bits(bits)
        back = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))[:len(bits)].astype(bool)
        self.assertEqual(back.tolist(), bits)
        self.assertEqual(pack_bits([]), "")


def naive_attention(query, key, value, scale, allowed):
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
    g = torch.Generator().manual_seed(seed)
    n_blocks = -(-seq_total // BLOCK)
    perm = torch.randperm(n_blocks + 3, generator=g)[:n_blocks]
    cache = torch.randn(2, n_blocks + 3, BLOCK, KVH, D, generator=g)
    key_seq = cache[0, perm].reshape(-1, KVH, D)[:seq_total]
    value_seq = cache[1, perm].reshape(-1, KVH, D)[:seq_total]
    return cache, perm, key_seq, value_seq


class FakeImpl:
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
class HookTest(unittest.TestCase):
    def setUp(self):
        self.prompt = [TEXT, TEXT, VISION_START] + [VIDEO_PAD_ID] * 6 + [VISION_END, TEXT, TEXT]  # 12 tokens
        self.original_calls = []

    def original(self, impl, layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
        self.original_calls.append(layer.layer_name)
        return output.fill_(7.0)

    def state(self, method="random"):
        st = PruneState(method, 2, tokens_per_frame=1)
        st.n_layers = N_LAYERS
        st.strict = True
        return st

    def run_forward(self, state, index, seq_len, n_query, seed=1):
        forward = make_prune_forward(state, self.original)
        cache, row, key, value = paged_cache(seq_len, seed)
        query = torch.randn(n_query, H, D, generator=torch.Generator().manual_seed(seed + 7))
        keys_in = torch.randn(n_query, KVH, D, generator=torch.Generator().manual_seed(seed + 9))
        output = torch.zeros(n_query + 2, H, D)
        layer = SimpleNamespace(layer_name=f"language_model.model.layers.{index}.self_attn.attn")
        got = forward(FakeImpl(), layer, query, keys_in, keys_in, cache, metadata(seq_len, n_query, row), output=output)
        self.assertIs(got, output)
        return output[:n_query], query, key, value, keys_in

    def expect(self, out, query, key, value, hide):
        mask = attention_mask(query.shape[0], key.shape[0], hide)
        torch.testing.assert_close(out.double(), naive_attention(query, key, value, 1 / math.sqrt(D), mask), rtol=1e-5, atol=1e-5)

    def test_hidden_set_applies_on_every_layer(self):
        state = self.state()
        hide = torch.zeros(12, dtype=torch.bool)
        hide[4:7] = True
        state.set_hide(hide)
        for index in (0, 1, 17, 35):
            out, query, key, value, _ = self.run_forward(state, index, seq_len=12, n_query=12)
            self.expect(out, query, key, value, hide)
            out, query, key, value, _ = self.run_forward(state, index, seq_len=15, n_query=1)  # decode
            self.expect(out, query, key, value, hide)
        self.assertEqual((state.calls, self.original_calls), (8, []))
        self.assertIsNone(state.row)  # no capture outside a pre-pass

    def test_prepass_captures_layer_zero_prefill_only(self):
        state = self.state("fastv")
        hide = torch.zeros(12, dtype=torch.bool)
        hide[3] = True
        state.set_hide(hide)
        state.prompt_len = 12
        state.capture = True
        out, query, key, value, keys_in = self.run_forward(state, 1, seq_len=12, n_query=12)  # layer 1: no capture
        self.assertIsNone(state.row)
        out, query, key, value, keys_in = self.run_forward(state, 0, seq_len=12, n_query=10)  # prefix-cached 2 rows
        self.expect(out, query, key, value, hide)
        self.assertEqual(state.row.shape, (12,))
        self.assertEqual(state.offset, 2)
        self.assertTrue(torch.equal(state.keys0, keys_in))
        mask = attention_mask(10, 12, hide)
        self.assertAlmostEqual(float(state.row.sum()), 1.0, places=5)
        self.assertEqual(float(state.row[3]), 0.0)  # hidden column gets no mass
        ref = decision_row(query[-1], key, 1 / math.sqrt(D), mask[-1])
        torch.testing.assert_close(state.row, ref)
        # by hand: mean over heads of the softmax over the allowed columns
        rep = H // KVH
        rows = []
        for h in range(H):
            logits = (key[:, h // rep] @ query[-1, h]) / math.sqrt(D)
            logits = logits.masked_fill(~mask[-1], float("-inf"))
            rows.append(torch.softmax(logits, dim=0))
        torch.testing.assert_close(state.row, torch.stack(rows).mean(0), rtol=1e-5, atol=1e-5)
        self.run_forward(state, 0, seq_len=13, n_query=1)  # a decode forward is not captured
        with self.assertRaises(RuntimeError):
            self.run_forward(state, 0, seq_len=12, n_query=12)  # a second prefill: chunked prefill

    def test_layer1_prehook(self):
        state = self.state("dart")
        hook = make_layer1_prehook(state)
        hidden = torch.randn(5, 3)
        residual = torch.randn(5, 3)
        hook(None, (None, hidden, residual))  # not capturing
        self.assertIsNone(state.h1)
        state.capture = True
        hook(None, (None, hidden[:1], residual[:1]))  # decode
        self.assertIsNone(state.h1)
        hook(None, (None, hidden, residual))
        torch.testing.assert_close(state.h1, hidden + residual)
        with self.assertRaises(RuntimeError):
            hook(None, (None, hidden, residual))
        state.h1 = None
        with self.assertRaises(RuntimeError):
            hook(None, (None, hidden, None))

    def test_warmup_falls_through_and_strict_refuses(self):
        state = PruneState("random", 4)
        forward = make_prune_forward(state, self.original)
        layer = SimpleNamespace(layer_name="language_model.model.layers.3.self_attn.attn")
        out = torch.zeros(2, H, D)
        forward(FakeImpl(), layer, None, None, None, None, None, output=out)
        cache, row, _, _ = paged_cache(8, 0)
        forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)
        self.assertEqual(len(self.original_calls), 2)
        state.strict = True
        with self.assertRaises(RuntimeError):
            forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)

    def test_generate_counts_and_state(self):
        state = self.state()
        state.calls = 36 * 7
        self.assertEqual(check_generate_counts(state, 12, 6), 7)
        state.calls = 35
        with self.assertRaises(RuntimeError):
            check_generate_counts(state, 12, 6)
        state.calls = 36
        with self.assertRaises(RuntimeError):
            check_generate_counts(state, 12, 0)
        with self.assertRaises(ValueError):
            PruneState("pyramiddrop", 4)
        with self.assertRaises(ValueError):
            PruneState("random", -1)
        st = PruneState("dart", 8)
        self.assertEqual((st.budget, st.describe()["pivots"], st.describe()["prepass"]), (112, 8, True))
        st.reset_episode()
        st.ledger.add((1, 2, 2, 0))
        st.reset_episode()
        self.assertEqual((st.episode, st.ledger), (1, set()))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class CliTest(unittest.TestCase):
    def test_required_flags_and_passthrough(self):
        args, rest = build_parser().parse_known_args(["--prune-method", "dart", "--prune-k", "8", "--episodes", "-1",
                                                      "--output_dir", "/tmp/x/raw"])
        self.assertEqual((args.prune_method, args.prune_k, args.prune_tokens_per_frame, args.prune_pivots), ("dart", 8, 14, 8))
        self.assertEqual(rest, ["--episodes", "-1", "--output_dir", "/tmp/x/raw"])
        self.assertEqual(output_dir_from(rest), "/tmp/x/raw")
        self.assertEqual(output_dir_from(["--output_dir=/a/b"]), "/a/b")
        self.assertIsNone(output_dir_from(["--episodes", "-1"]))
        args, _ = build_parser().parse_known_args(["--prune-method", "random", "--prune-k", "0"])
        self.assertEqual(args.prune_k, 0)
        args, _ = build_parser().parse_known_args(["--prune-method", "recency", "--prune-k", "2"])
        self.assertEqual((args.prune_method, args.prune_k, args.prune_current_frames), ("recency", 2, 2))
        args, _ = build_parser().parse_known_args(["--prune-method", "recency", "--prune-k", "0", "--prune-current-frames", "1"])
        self.assertEqual(args.prune_current_frames, 1)
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--prune-method", "recency", "--prune-k", "0", "--prune-current-frames", "3"])
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--prune-method", "fastv"])
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--prune-method", "pyramiddrop", "--prune-k", "4"])


if __name__ == "__main__":
    unittest.main()
