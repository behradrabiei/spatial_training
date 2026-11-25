"""Unit tests for the LightNav-0 protocol wrapper (baselines/lightnav_hm3d/eval_protocol.py).

No weights, no GPU, no vLLM: the whole-model budget solver (against Uni-NaVid's), the nested stage
ledger, the window scope, the per-stage hidden sets and the staged attention hook are checked on
synthetic tensors and stub layers. Needs torch, so run it with the client runtime:

    PYTHONPATH=baselines/LightNav-0/src baselines/.envs/lightnav/bin/python \\
        -m unittest baselines.tests.test_lightnav_protocol
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from types import SimpleNamespace

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch

    from baselines.lightnav_hm3d.eval_blind import attention_mask
    from baselines.lightnav_hm3d.eval_prune import VIDEO_PAD_ID, decision_row, random_select, tubelet_layout
    from baselines.lightnav_hm3d.eval_protocol import (
        PYRAMIDDROP_STARTS,
        UNIFORM_BLIND_LAYER,
        ProtocolState,
        StageLedger,
        build_parser,
        in_scope,
        kept_per_tubelet,
        make_protocol_forward,
        model_stage_budgets,
        stage_hides,
        stage_pool,
        step_layout,
    )
    from baselines.tests.test_lightnav_prune import (
        D, H, FakeImpl, grid_for, metadata, naive_attention, paged_cache, prompt_for,
    )

N_LAYERS = 36


def fast_layout(current, fast_pairs, window):
    """Fast-tier tubelets (a, a+1) at pool 2 (28 tokens each) and the current pair at pool 1."""
    segs = [(sum(([a, a + 1] for a in fast_pairs), []), 2), ([current - 1, current], 1)]
    tubs = tubelet_layout(segs, [grid_for(2, len(fast_pairs)), grid_for(1, 1)])
    ids, pads = prompt_for(tubs)
    return tubs, ids, pads, step_layout(tubs, current, pads, len(ids), window)


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class BudgetTest(unittest.TestCase):
    def test_solver_matches_uninavid(self):
        try:
            from baselines.uninavid_hm3d.prune import model_stage_budgets as reference
        except Exception as exc:  # pragma: no cover - the module needs only numpy + torch
            self.skipTest(f"uninavid_hm3d.prune not importable here: {exc}")
        for v0 in (0, 20, 112, 420, 840, 1300):
            for budget in (28, 56, 112, 224):
                for widths, tail in (([9, 9, 9, 9], None), ([26], (10, 0)), ([36], None), ([8, 8, 8], (8, 4))):
                    self.assertEqual(model_stage_budgets(v0, budget, widths, tail), reference(v0, budget, widths, tail))

    def test_uniform_layer0_budget_is_36_over_26_of_the_whole_model_budget(self):
        for k, expect in ((16, 310), (8, 155), (4, 78), (2, 39)):
            state = ProtocolState("uniform", k)
            state.set_layers(N_LAYERS)
            self.assertEqual((state.starts, state.widths, state.window), ([0, 26], [26, 10], 32))
            self.assertEqual(state.solve_budgets(420), [expect, 0])
            self.assertLess(abs(26 * expect / 36 - 14 * k), 0.5)  # the layer-weighted mean is the budget
            under = (36 * 14 * k) // 26  # 26 layers x v0 <= 36 x B: nothing to prune
            self.assertEqual(state.solve_budgets(under), [under, 0])

    def test_pyramiddrop_budgets_are_nested_and_meet_the_whole_model_budget(self):
        for k in (16, 8, 4, 2):
            state = ProtocolState("pyramiddrop", k)
            state.set_layers(N_LAYERS)
            self.assertEqual((state.starts, state.widths, state.window), (list(PYRAMIDDROP_STARTS), [9, 9, 9, 9], None))
            self.assertEqual(state.score_layer, {8: 1, 17: 2, 26: 3})
            budgets = state.solve_budgets(840)
            self.assertEqual(budgets, sorted(budgets, reverse=True))
            self.assertLess(abs(sum(9 * b for b in budgets) / 36 - 14 * k), 0.75)  # rounding of four stages
            self.assertEqual(state.solve_budgets(14 * k), [14 * k] * 4)  # under budget

    def test_blind_layer_at_the_model_depth_is_plain_random(self):
        state = ProtocolState("uniform", 4, blind_layer=N_LAYERS, window_frames=0)
        state.set_layers(N_LAYERS)
        self.assertEqual((state.widths, state.window), ([36, 0], None))
        self.assertEqual(state.solve_budgets(500), [56, 0])
        self.assertEqual({state.stage_of(i) for i in range(N_LAYERS)}, {0})

    def test_state_validation(self):
        with self.assertRaises(ValueError):
            ProtocolState("hybrid", 4)
        with self.assertRaises(ValueError):
            ProtocolState("uniform", 0)
        with self.assertRaises(ValueError):
            ProtocolState("uniform", 4, stage_starts=[0, 9])
        with self.assertRaises(ValueError):
            ProtocolState("pyramiddrop", 4, stage_starts=[1, 9])
        with self.assertRaises(ValueError):
            ProtocolState("pyramiddrop", 4, stage_starts=[0, 9, 9])
        with self.assertRaises(ValueError):
            ProtocolState("uniform", 4, window_frames=1)
        with self.assertRaises(ValueError):
            ProtocolState("uniform", 4, blind_layer=40).set_layers(N_LAYERS)
        with self.assertRaises(ValueError):
            ProtocolState("pyramiddrop", 4, stage_starts=[0, 36]).set_layers(N_LAYERS)
        d = ProtocolState("uniform", 8).describe()
        self.assertEqual((d["budget"], d["blind_layer"], d["window_frames"], d["stage_selectors"], d["prepass"]),
                         (112, UNIFORM_BLIND_LAYER, 32, ["random", "blind"], False))
        d = ProtocolState("pyramiddrop", 8).describe()
        self.assertEqual((d["stage_starts"], d["blind_layer"], d["window_frames"], d["prepass"]), ([0, 9, 18, 27], None, None, True))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class LedgerScopeTest(unittest.TestCase):
    def test_deleting_at_a_stage_deletes_at_every_deeper_stage(self):
        ledger = StageLedger(3)
        ledger.delete(1, [(2, 3, 2, 0)])
        ledger.delete(0, [(4, 5, 2, 1)])
        self.assertEqual([sorted(d) for d in ledger.deleted],
                         [[(4, 5, 2, 1)], [(2, 3, 2, 0), (4, 5, 2, 1)], [(2, 3, 2, 0), (4, 5, 2, 1)]])
        with self.assertRaises(ValueError):
            StageLedger(0)

    def test_window_scope_is_the_newest_frames(self):
        tubs, ids, pads, layout = fast_layout(40, [2, 8, 9, 20, 36], window=32)  # frames 9..40 are in scope
        self.assertEqual(layout.scope, [False, False, True, True, True])  # (8,9) straddles the edge: out
        self.assertEqual((len(layout.ranks), len(layout.out_ranks)), (3 * 28, 2 * 28))
        self.assertTrue(in_scope(tubs[2], 40, 32) and not in_scope(tubs[1], 40, 32) and in_scope(tubs[0], 40, None))
        whole = step_layout(tubs, 40, pads, len(ids), None)
        self.assertEqual((whole.scope, len(whole.ranks), whole.out_ranks), ([True] * 5, 5 * 28, []))

    def test_stage_hides_cover_scope_ledger_and_blind(self):
        tubs, ids, pads, layout = fast_layout(40, [2, 20, 36], window=32)
        visual = torch.tensor([x == VIDEO_PAD_ID for x in ids])
        ledger = StageLedger(2)
        ledger.delete(0, [layout.keys[0], layout.keys[30]])
        ledger.delete(1, [layout.keys[5]])
        self.assertEqual(len(stage_pool(layout, ledger, 0)), 54)
        self.assertEqual(len(stage_pool(layout, ledger, 1)), 53)
        hides = stage_hides(layout, ledger, blind=visual)
        out_cols = [pads[r] for r in layout.out_ranks]
        self.assertEqual(int(hides[0].sum()), 28 + 2)  # the out-of-scope tubelet + stage 0's two deletions
        self.assertTrue(bool(hides[0][out_cols].all()))
        self.assertFalse(bool(hides[0][pads[-112:]].any()))  # the current pair is visible below the cut
        self.assertTrue(torch.equal(hides[1], visual))  # the blind stage hides every visual column, nothing else
        n_kept, bits = kept_per_tubelet(layout, hides[0])
        self.assertEqual((n_kept, len(bits), sum(bits)), ([0, 27, 27], 84, 54))
        plain = stage_hides(layout, ledger)
        self.assertEqual(int(plain[1].sum()), 28 + 3)


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class SelectionTest(unittest.TestCase):
    def begin(self, state, layout, visual=None):
        state.layout = layout
        state.blind_cols = visual if state.blind else None
        state.budgets = state.solve_budgets(len(stage_pool(layout, state.ledger, 0)))
        state.selected = [False] * state.n_stages
        state.refresh()

    def test_uniform_draw_is_seeded_nested_and_permanent(self):
        tubs, ids, pads, layout = fast_layout(40, [20, 26, 30, 36], window=32)
        visual = torch.tensor([x == VIDEO_PAD_ID for x in ids])
        state = ProtocolState("uniform", 2)
        state.set_layers(N_LAYERS)
        self.begin(state, layout, visual)
        self.assertEqual(state.budgets, [39, 0])
        g = torch.Generator().manual_seed(5)
        dropped = state.select(0, lambda n, b: random_select(n, b, g))
        self.assertEqual((dropped, state.selected[0]), (112 - 39, True))
        self.assertEqual(len(stage_pool(layout, state.ledger, 0)), 39)
        self.assertEqual(state.ledger.deleted[0], state.ledger.deleted[1])  # nested
        self.assertEqual(int((~state.hides[0][pads[:112]]).sum()), 39)
        self.assertTrue(torch.equal(state.hides[1], visual))
        # next inference of the same tubelets: the survivors are under budget, nothing more is dropped
        self.begin(state, layout, visual)
        self.assertEqual(state.select(0, lambda n, b: random_select(n, b, g)), 0)
        state.reset_episode()
        self.assertEqual((state.episode, state.ledger.deleted), (0, [set(), set()]))

    def test_attention_stage_keeps_the_top_scores_of_its_pool(self):
        tubs, ids, pads, layout = fast_layout(10, [2, 6], window=None)
        state = ProtocolState("pyramiddrop", 1)
        state.set_layers(N_LAYERS)
        self.begin(state, layout)
        budgets = state.budgets
        self.assertEqual(len(budgets), 4)
        row = torch.zeros(len(ids))
        row[torch.tensor(pads[:56])] = torch.arange(56, dtype=torch.float)  # later history tokens score higher
        state.select_attention(0, row)
        self.assertEqual(stage_pool(layout, state.ledger, 0), list(range(56 - budgets[0], 56)))
        row2 = torch.zeros(len(ids))
        row2[torch.tensor(pads[:56])] = -torch.arange(56, dtype=torch.float)  # now the earliest survivors win
        state.select_attention(2, row2)  # deleting at stage 2 leaves stage 1 alone
        self.assertEqual(stage_pool(layout, state.ledger, 1), stage_pool(layout, state.ledger, 0))
        first = 56 - budgets[0]
        self.assertEqual(stage_pool(layout, state.ledger, 2), list(range(first, first + budgets[2])))
        self.assertEqual(stage_pool(layout, state.ledger, 3), stage_pool(layout, state.ledger, 2))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class HookTest(unittest.TestCase):
    def setUp(self):
        self.original_calls = []
        self.tubs, self.ids, self.pads, _ = fast_layout(10, [2, 6], window=None)

    def original(self, impl, layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
        self.original_calls.append(layer.layer_name)
        return output.fill_(7.0)

    def state(self, protocol="pyramiddrop", k=1, **kw):
        st = ProtocolState(protocol, k, **kw)
        st.set_layers(N_LAYERS)
        st.strict = True
        st.layout = step_layout(self.tubs, 10, self.pads, len(self.ids), st.window)
        st.blind_cols = torch.tensor([x == VIDEO_PAD_ID for x in self.ids]) if st.blind else None
        st.budgets = st.solve_budgets(len(stage_pool(st.layout, st.ledger, 0)))
        st.selected = [False] * st.n_stages
        st.refresh()
        return st

    def run_forward(self, state, index, seq_len, n_query, seed=1):
        forward = make_protocol_forward(state, self.original)
        cache, row, key, value = paged_cache(seq_len, seed)
        query = torch.randn(n_query, H, D, generator=torch.Generator().manual_seed(seed + 7))
        output = torch.zeros(n_query + 2, H, D)
        layer = SimpleNamespace(layer_name=f"language_model.model.layers.{index}.self_attn.attn")
        got = forward(FakeImpl(), layer, query, None, None, cache, metadata(seq_len, n_query, row), output=output)
        self.assertIs(got, output)
        return output[:n_query], query, key, value

    def expect(self, out, query, key, value, hide):
        mask = attention_mask(query.shape[0], key.shape[0], hide)
        torch.testing.assert_close(out.double(), naive_attention(query, key, value, 1 / math.sqrt(D), mask), rtol=1e-5, atol=1e-5)

    def test_each_layer_hides_its_stage_and_scoring_layers_select(self):
        state = self.state()
        n = len(self.ids)
        budgets = state.budgets
        state.select_attention(0, torch.rand(n, generator=torch.Generator().manual_seed(3)))  # the pre-pass result
        for index in range(N_LAYERS):
            stage = state.stage_of(index)
            self.assertEqual(stage, index // 9)
            hide_before = state.hides[stage].clone()
            pool_next = stage_pool(state.layout, state.ledger, stage + 1) if index in state.score_layer else None
            out, query, key, value = self.run_forward(state, index, seq_len=n, n_query=n - 2, seed=index)
            self.expect(out, query, key, value, hide_before)  # scoring does not change this layer's output
            if index in state.score_layer:
                s = state.score_layer[index]
                self.assertTrue(state.selected[s])
                mask = attention_mask(n - 2, n, hide_before)
                row = decision_row(query[-1], key, 1 / math.sqrt(D), mask[-1])
                cols = torch.tensor([self.pads[state.layout.ranks[i]] for i in pool_next])
                top = sorted(torch.argsort(row[cols], descending=True, stable=True)[:budgets[s]].tolist())
                self.assertEqual(stage_pool(state.layout, state.ledger, s), [pool_next[j] for j in top])
        self.assertTrue(all(state.selected))
        self.assertEqual([len(stage_pool(state.layout, state.ledger, s)) for s in range(4)], budgets)
        self.assertEqual((state.calls, self.original_calls), (N_LAYERS, []))
        # decode forwards keep the final hides and select nothing
        frozen = [h.clone() for h in state.hides]
        for index in (0, 8, 17, 26, 35):
            out, query, key, value = self.run_forward(state, index, seq_len=n + 3, n_query=1, seed=100 + index)
            self.expect(out, query, key, value, frozen[state.stage_of(index)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(frozen, state.hides)))

    def test_prepass_records_the_layer_zero_row_and_selects_nothing(self):
        state = self.state()
        n = len(self.ids)
        state.capture = True
        out, query, key, value = self.run_forward(state, 0, seq_len=n, n_query=n)
        mask = attention_mask(n, n, state.hides[0])
        torch.testing.assert_close(state.row, decision_row(query[-1], key, 1 / math.sqrt(D), mask[-1]))
        self.run_forward(state, 8, seq_len=n, n_query=n)  # a scoring layer in the pre-pass: no selection
        self.assertEqual(state.selected, [False] * 4)
        with self.assertRaises(RuntimeError):
            self.run_forward(state, 0, seq_len=n, n_query=n)  # a second layer-0 prefill

    def test_scoring_needs_the_stages_above_and_the_whole_prompt(self):
        state = self.state()
        n = len(self.ids)
        with self.assertRaises(RuntimeError):
            self.run_forward(state, 8, seq_len=n, n_query=n)  # stage 0 has not selected
        state.selected[0] = True
        with self.assertRaises(RuntimeError):
            self.run_forward(state, 8, seq_len=n + 1, n_query=n)  # not the prompt's prefill

    def test_uniform_is_blind_from_the_cut(self):
        state = self.state("uniform", 1, window_frames=0)
        n = len(self.ids)
        g = torch.Generator().manual_seed(2)
        state.select(0, lambda count, b: random_select(count, b, g))
        self.assertEqual(int((~state.hides[0][torch.tensor(self.pads[:56])]).sum()), state.budgets[0])
        for index in (0, 25):
            out, query, key, value = self.run_forward(state, index, seq_len=n, n_query=n, seed=index)
            self.expect(out, query, key, value, state.hides[0])
            self.assertFalse(bool(state.hides[0][torch.tensor(self.pads[56:])].any()))
        visual = torch.tensor([x == VIDEO_PAD_ID for x in self.ids])
        for index in (26, 35):
            out, query, key, value = self.run_forward(state, index, seq_len=n, n_query=n, seed=index)
            self.expect(out, query, key, value, visual)

    def test_warmup_falls_through_and_strict_refuses(self):
        state = ProtocolState("uniform", 4)
        forward = make_protocol_forward(state, self.original)
        layer = SimpleNamespace(layer_name="language_model.model.layers.3.self_attn.attn")
        out = torch.zeros(2, H, D)
        forward(FakeImpl(), layer, None, None, None, None, None, output=out)
        cache, row, _, _ = paged_cache(8, 0)
        forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)
        self.assertEqual(len(self.original_calls), 2)
        state.strict = True
        with self.assertRaises(RuntimeError):
            forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class CliTest(unittest.TestCase):
    def test_flags_and_passthrough(self):
        args, rest = build_parser().parse_known_args(["--protocol", "uniform", "--prune-k", "8", "--episodes", "-1"])
        self.assertEqual((args.protocol, args.prune_k, args.blind_layer, args.window_frames, args.stage_starts),
                         ("uniform", 8, 26, None, None))
        self.assertEqual(rest, ["--episodes", "-1"])
        args, _ = build_parser().parse_known_args(["--protocol", "pyramiddrop", "--prune-k", "2", "--stage-starts", "0",
                                                   "--window-frames", "0"])
        self.assertEqual((args.stage_starts, args.window_frames), ("0", 0))
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--protocol", "hybrid", "--prune-k", "4"])
        with self.assertRaises(SystemExit):
            build_parser().parse_known_args(["--protocol", "uniform"])


if __name__ == "__main__":
    unittest.main()
