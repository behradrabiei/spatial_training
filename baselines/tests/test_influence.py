"""Unit tests for uninavid_hm3d.influence (no model weights, no Habitat).

Run from the repo root with the baseline runtime:
    PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python \
        -m unittest baselines.tests.test_influence
"""
import math
import unittest
from types import SimpleNamespace

import torch

from uninavid_hm3d.influence import (
    ACTION_WORDS,
    FrameInfluenceProbe,
    HistoryLayout,
    LongTermTracker,
    Probe,
    action_logp,
    build_probes,
    drop_spans,
    kl_divergence,
    member_mean,
    visual_column_masks,
)

IMAGE = -200
SEP, VSTART, VEND, ISTART, IEND, NAV = 32005, 32000, 32001, 32002, 32003, 32004
C = 6  # feature width used by the synthetic layouts


def synthetic_ids(n_text=3, n_tail=2):
    text = list(range(100, 100 + n_text))
    tail = list(range(200, 200 + n_tail))
    return torch.tensor(text + [VSTART, SEP, IMAGE, VEND, ISTART, IEND, NAV] + tail)


def table():
    """Deterministic 'embedding table' over the 32006-id vocabulary."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(32006, C, generator=g)


def assemble(ids, groups, nav, tab):
    """Build the prompt embeddings the way uninavid_arch does from explicit history groups."""
    s = int(torch.nonzero(ids == IMAGE).flatten()[0])
    parts = [tab[ids[:s]]]
    for j, g in enumerate(groups):
        parts.append(g)
        if j < len(groups) - 1:
            parts.append(tab[ids[s - 1:s]])
    parts += [tab[ids[s + 1:s + 3]], nav, tab[ids[s + 3:]]]
    return torch.cat(parts, dim=0)


def upstream_online(state, nav_size=4):
    """Call the real Uni-NaVid merge rule on a fake model state."""
    from uninavid.model.uninavid_arch import UniNaVIDMetaForCausalLM

    fake = SimpleNamespace(get_model=lambda: state)
    return UniNaVIDMetaForCausalLM.online_process_tensor(fake, nav_size)


class TrackerTests(unittest.TestCase):
    def test_tracker_matches_upstream_with_forced_merges(self):
        g = torch.Generator().manual_seed(1)
        rows = torch.randn(91, 4, C, generator=g)
        base = torch.randn(4, C, generator=g)
        rows[10:31] = base + 1e-3 * torch.randn(21, 4, C, generator=g)  # near-identical run
        state = SimpleNamespace(feat_cache=None, long_feat_cache=None, weight=1, new_frames=0)
        tracker = LongTermTracker()
        k, new = 1, 1
        while k <= 91:
            state.feat_cache = rows[:k]
            state.new_frames = new
            tensor, lengths = upstream_online(state)
            tracker.step(state.feat_cache, new)
            tracker.verify(state)  # raises on divergence
            n_long = len(tracker.slots)
            self.assertEqual(lengths, [1] * n_long + [4] * min(k, 64))
            self.assertEqual(tensor.shape[0], sum(lengths))
            k, new = k + 2, 2
        flat = [f for slot in tracker.slots for f in slot]
        self.assertEqual(flat, list(range(91 - 64)))  # partition of the frames that left, in order
        self.assertTrue(any(len(s) > 1 for s in tracker.slots), "no merge happened")
        self.assertEqual(state.weight, tracker.weight)

    def test_tracker_is_inert_inside_the_window(self):
        tracker = LongTermTracker()
        tracker.step(torch.randn(64, 4, C), 2)
        self.assertIsNone(tracker.long)
        self.assertEqual(tracker.slots, [])
        self.assertEqual(tracker.weight, 1)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tab = table()
        self.ids = synthetic_ids()
        g = torch.Generator().manual_seed(2)
        self.nav = torch.randn(64, C, generator=g)

    def short_case(self, k=5):
        g = torch.Generator().manual_seed(3)
        feat = torch.randn(k, 4, C, generator=g)
        embeds = assemble(self.ids, [feat[i] for i in range(k)], self.nav, self.tab)
        layout = HistoryLayout.build(self.ids, [4] * k, k, [], IMAGE)
        return feat, embeds, layout

    def long_case(self):
        """70 frames: 6 long-term slots with members [0],[1,2,3],[4],[5],[6] (5 slots cover 0..6)."""
        g = torch.Generator().manual_seed(4)
        feat = torch.randn(70, 4, C, generator=g)
        members = [[0], [1, 2, 3], [4], [5]]
        # 70 frames, 64 short => frames 0..5 are long-term; group sizes [1]*4 + [4]*64
        longs = [member_mean(feat, m, feat.dtype)[None] for m in members]
        groups = longs + [feat[i] for i in range(6, 70)]
        embeds = assemble(self.ids, groups, self.nav, self.tab)
        layout = HistoryLayout.build(self.ids, [1] * 4 + [4] * 64, 70, members, IMAGE)
        return feat, embeds, layout, members

    def test_spans_and_total_length(self):
        feat, embeds, layout = self.short_case()
        self.assertEqual(layout.total_length, embeds.shape[0])
        self.assertEqual(embeds.shape[0], self.ids.numel() + 4 * 5 + 5 + 62)
        for j, (a, b) in enumerate(layout.groups):
            self.assertTrue(torch.equal(embeds[a:b], feat[j]))
        layout.verify(embeds, feat, lambda t: self.tab[t], self.ids)  # must not raise
        self.assertEqual(layout.frame_of_group, [0, 1, 2, 3, 4])

    def test_verify_catches_drift(self):
        feat, embeds, layout = self.short_case()
        bad = embeds.clone()
        bad[layout.groups[2][0]] += 1.0
        with self.assertRaises(RuntimeError):
            layout.verify(bad, feat, lambda t: self.tab[t], self.ids)

    def test_remove_short_frame_drops_group_and_separator(self):
        feat, embeds, layout = self.short_case()
        probes = build_probes(layout, embeds, feat, horizons=())
        p = next(p for p in probes if p.kind == "frame" and p.frame == 2)
        ref = assemble(self.ids, [feat[i] for i in (0, 1, 3, 4)], self.nav, self.tab)
        self.assertEqual(p.embeds.shape[0], embeds.shape[0] - 5)
        self.assertTrue(torch.equal(p.embeds, ref))
        self.assertEqual(p.tier, "short")

    def test_age0_removes_current_group_and_fine_tokens(self):
        feat, embeds, layout = self.short_case()
        probes = build_probes(layout, embeds, feat, horizons=())
        p = next(p for p in probes if p.kind == "age0")
        ref = assemble(self.ids, [feat[i] for i in range(4)], torch.zeros(0, C), self.tab)
        self.assertEqual(p.embeds.shape[0], embeds.shape[0] - 5 - 64)
        self.assertTrue(torch.equal(p.embeds, ref))

    def test_history_all_and_visual_all(self):
        feat, embeds, layout = self.short_case()
        probes = build_probes(layout, embeds, feat, horizons=())
        hist = next(p for p in probes if p.kind == "history_all")
        self.assertTrue(torch.equal(hist.embeds, assemble(self.ids, [feat[4]], self.nav, self.tab)))
        vis = next(p for p in probes if p.kind == "visual_all")
        no_visual_prompt = self.tab[self.ids[self.ids != IMAGE]]  # what --no-visual-tokens feeds
        self.assertTrue(torch.equal(vis.embeds, no_visual_prompt))

    def test_keep_one_keeps_only_that_group(self):
        feat, embeds, layout = self.short_case()
        probes = build_probes(layout, embeds, feat, horizons=(), measures=("keep_one",))
        self.assertFalse([p for p in probes if p.kind == "frame"])  # leave-one not requested
        keep = {p.frame: p for p in probes if p.kind == "keep_one"}
        self.assertEqual(sorted(keep), [0, 1, 2, 3])  # history frames only; age 0 is history_all
        for i, p in keep.items():
            ref = assemble(self.ids, [feat[i]], torch.zeros(0, C), self.tab)
            self.assertTrue(torch.equal(p.embeds, ref), f"frame {i}")
            self.assertEqual(p.embeds.shape[0], self.ids.numel() - 1 + 4)  # ids minus the placeholder, plus one group
            self.assertEqual(p.tier, "short")
        both = build_probes(layout, embeds, feat, horizons=(), measures=("leave_one", "keep_one"))
        self.assertEqual(len([p for p in both if p.kind == "frame"]), 4)
        self.assertEqual(len([p for p in both if p.kind == "keep_one"]), 4)

    def test_keep_one_long_tier_rebuilds_own_slot(self):
        feat, embeds, layout, members = self.long_case()
        probes = build_probes(layout, embeds, feat, horizons=(), measures=("keep_one",))
        keep = {p.frame: p for p in probes if p.kind == "keep_one"}
        self.assertEqual(len(keep), 69)
        ref = assemble(self.ids, [member_mean(feat, [2], feat.dtype)[None]], torch.zeros(0, C), self.tab)
        self.assertTrue(torch.equal(keep[2].embeds, ref))
        self.assertEqual((keep[2].tier, keep[2].weight, keep[2].slot), ("long", 3, 1))
        ref0 = assemble(self.ids, [member_mean(feat, [0], feat.dtype)[None]], torch.zeros(0, C), self.tab)
        self.assertTrue(torch.equal(keep[0].embeds, ref0))
        ref_short = assemble(self.ids, [feat[10]], torch.zeros(0, C), self.tab)
        self.assertTrue(torch.equal(keep[10].embeds, ref_short))

    def test_keep_one_mask_probes_keep_positions(self):
        feat, embeds, layout = self.short_case()
        all_visual, current = visual_column_masks(layout)
        probes = build_probes(layout, embeds, feat, horizons=(), measures=("keep_one_mask",))
        keep = {p.frame: p for p in probes if p.kind == "keep_one"}
        self.assertEqual(sorted(keep), [0, 1, 2, 3])
        for i, p in keep.items():
            self.assertTrue(torch.equal(p.embeds, embeds))  # the prompt itself is untouched
            want = all_visual.clone(); a, b = layout.groups[i]; want[a:b] = False
            self.assertTrue(torch.equal(p.hide, want), f"frame {i}")
            self.assertEqual(int(p.hide.sum()), 4 * 4 + 64)  # four other groups + the fine tokens
        vis = next(p for p in probes if p.kind == "visual_all_mask")
        self.assertTrue(torch.equal(vis.hide, all_visual) and torch.equal(vis.embeds, embeds))
        cur = next(p for p in probes if p.kind == "only_current_mask")
        self.assertTrue(torch.equal(cur.hide, all_visual & ~current))
        self.assertEqual(cur.frame, 4)
        # long tier: the slot column is rebuilt as the frame's own token and stays open
        feat, embeds, layout, members = self.long_case()
        all_visual, _ = visual_column_masks(layout)
        probes = build_probes(layout, embeds, feat, horizons=(), measures=("keep_one_mask",))
        p = next(p for p in probes if p.kind == "keep_one" and p.frame == 2)
        a = layout.groups[1][0]
        self.assertTrue(torch.allclose(p.embeds[a], member_mean(feat, [2], feat.dtype)))
        self.assertEqual((p.embeds != embeds).any(dim=1).nonzero().flatten().tolist(), [a])
        self.assertFalse(bool(p.hide[a]))
        self.assertEqual(int(p.hide.sum()), int(all_visual.sum()) - 1)

    def test_horizon_bands_keep_k_most_recent_frames(self):
        feat, embeds, layout = self.short_case(k=9)
        probes = build_probes(layout, embeds, feat, horizons=(2, 4, 16))
        bands = {p.K: p for p in probes if p.kind == "horizon"}
        self.assertEqual(sorted(bands), [2, 4])  # K=16 > t=8 is absent
        ref4 = assemble(self.ids, [feat[i] for i in range(5, 9)], self.nav, self.tab)
        self.assertTrue(torch.equal(bands[4].embeds, ref4))
        ref2 = assemble(self.ids, [feat[i] for i in range(7, 9)], self.nav, self.tab)
        self.assertTrue(torch.equal(bands[2].embeds, ref2))

    def test_long_tier_probes(self):
        feat, embeds, layout, members = self.long_case()
        layout.verify(embeds, feat, lambda t: self.tab[t], self.ids)
        probes = build_probes(layout, embeds, feat, horizons=(64,))
        by_frame = {p.frame: p for p in probes if p.kind == "frame"}
        self.assertEqual(len(by_frame), 69)
        # singleton slot: dropped with its separator
        self.assertEqual(by_frame[0].embeds.shape[0], embeds.shape[0] - 2)
        self.assertEqual(by_frame[0].tier, "long")
        self.assertEqual(by_frame[0].weight, 1)
        # merged slot: same length, slot rebuilt from the surviving members in fp32
        p = by_frame[2]
        self.assertEqual(p.embeds.shape[0], embeds.shape[0])
        self.assertEqual(p.weight, 3)
        a = layout.groups[1][0]
        self.assertTrue(torch.allclose(p.embeds[a], member_mean(feat, [1, 3], feat.dtype)))
        diff = (p.embeds != embeds).any(dim=1).nonzero().flatten().tolist()
        self.assertEqual(diff, [a])
        # horizon K=64 at t=69 removes frames 0..5 => every long slot dropped, no partial slot
        band = next(p for p in probes if p.kind == "horizon")
        self.assertEqual(band.partial_slots, 0)
        self.assertEqual(band.embeds.shape[0], embeds.shape[0] - 4 * 2)
        # merged_control: one per merged slot, rebuilt from all members
        ctrl = [p for p in probes if p.kind == "merged_control"]
        self.assertEqual([(c.slot, c.weight) for c in ctrl], [(1, 3)])

    def test_partial_slot_in_horizon(self):
        feat, embeds, layout, members = self.long_case()
        # K=67 at t=69 removes frames 0..2: slot [1,2,3] keeps frame 3 => rebuilt, partial
        probes = build_probes(layout, embeds, feat, horizons=(67,))
        band = next(p for p in probes if p.kind == "horizon")
        self.assertEqual(band.partial_slots, 1)
        self.assertEqual(band.embeds.shape[0], embeds.shape[0] - 2)  # only slot 0 dropped
        # the rebuilt slot is now the first group and equals frame 3's mean token
        self.assertTrue(torch.allclose(band.embeds[layout.groups[0][0]], member_mean(feat, [3], feat.dtype)))

    def test_fp16_closed_form_is_worse_than_member_mean(self):
        g = torch.Generator().manual_seed(5)
        feat = (torch.randn(64, 4, 4096, generator=g) * 3).half()
        # running fp16 mean as upstream accumulates it
        slot = feat[0].mean(0)
        for w in range(1, 64):
            slot = (slot * w + feat[w].mean(0)) / (w + 1)
        exact = feat[1:].float().mean(dim=(0, 1))
        closed = ((slot.float() * 64 - feat[0].float().mean(0)) / 63)
        ours = member_mean(feat, list(range(1, 64)), torch.float32)
        self.assertLess((ours - exact).abs().max().item(), 1e-6)
        self.assertGreater((closed - exact).abs().max().item(), 1e-3)


class StatisticsTests(unittest.TestCase):
    def test_kl_helper(self):
        p = torch.log_softmax(torch.tensor([1.0, 0.0, 0.0, 0.0]), 0).tolist()
        q = torch.log_softmax(torch.tensor([0.0, 1.0, 0.0, 0.0]), 0).tolist()
        self.assertEqual(kl_divergence(p, p), 0.0)
        expected = sum(math.exp(a) * (a - b) for a, b in zip(p, q))
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=6)

    def test_action_logp_renormalises_and_reports_mass(self):
        logits = torch.zeros(10)
        logits[[2, 5]] = 3.0
        logp, mass = action_logp(logits, [2, 5, 7, 8])
        self.assertAlmostEqual(sum(math.exp(x) for x in logp), 1.0, places=6)
        full = torch.softmax(logits, 0)
        self.assertAlmostEqual(mass, float(full[[2, 5, 7, 8]].sum()), places=6)
        self.assertGreater(math.exp(logp[0]), math.exp(logp[2]))

    def test_drop_spans_merges_overlaps(self):
        x = torch.arange(10).unsqueeze(1)
        self.assertEqual(drop_spans(x, [(2, 5), (4, 6), (8, 9)]).flatten().tolist(), [0, 1, 6, 7, 9])


class FakeTokenizer:
    bos_token_id = 1
    _ids = {"stop": 5, "forward": 6, "left": 7, "right": 8}

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self._ids[text]])

    def convert_tokens_to_ids(self, word):
        return 10 + self._ids[word]


class FakeCausalModel:
    """Causal stub: hidden = cumulative mean over positions, logits = linear head."""

    def __init__(self):
        g = torch.Generator().manual_seed(6)
        self.head = torch.randn(C, 12, generator=g)
        self.lm_head = lambda h: h @ self.head

    def model(self, inputs_embeds, use_cache=False):
        return (inputs_embeds.cumsum(dim=1) / torch.arange(1, inputs_embeds.shape[1] + 1).view(1, -1, 1),)

    def get_model(self):
        return SimpleNamespace()


class FakeLayer:
    """Attention stand-in: each position is the mean of the keys its 4-D additive mask leaves open."""

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False):
        open_ = (attention_mask > torch.finfo(attention_mask.dtype).min / 2).to(hidden_states.dtype)[:, 0]
        return ((open_ @ hidden_states) / open_.sum(-1, keepdim=True).clamp(min=1),)


class FakeLayeredModel(FakeCausalModel):
    """One masked-mean layer behind a causal 4-D mask, with the hooks ColumnBlinder installs."""

    def __init__(self, feat, tab):
        super().__init__()
        self.state = SimpleNamespace(layers=[FakeLayer()], feat_cache=feat, embed_tokens=lambda t: tab[t])

    def get_model(self):
        return self.state

    def online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985):
        raise AssertionError("not called during scoring")

    def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
        raise AssertionError("not called during scoring")

    def model(self, inputs_embeds, use_cache=False):
        B, L, _ = inputs_embeds.shape
        mask = torch.full((L, L), torch.finfo(torch.float32).min).triu(1)[None, None].expand(B, 1, L, L)
        hidden = inputs_embeds
        for layer in self.state.layers:
            hidden = layer.forward(hidden, attention_mask=mask, position_ids=None, past_key_value=None,
                                   output_attentions=False, use_cache=use_cache)[0]
        return (hidden,)


class BatchingTests(unittest.TestCase):
    def test_batched_scores_match_single(self):
        model = FakeCausalModel()
        probes = [Probe("base", torch.randn(7 if i % 2 else 9, C)) for i in range(7)]
        one = FrameInfluenceProbe(model, FakeTokenizer(), IMAGE, probe_batch=1)
        many = FrameInfluenceProbe(model, FakeTokenizer(), IMAGE, probe_batch=3)
        a, b = one.score(probes), many.score(probes)
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        self.assertEqual(one.action_ids, [5, 6, 7, 8])
        self.assertEqual(many.describe()["action_words"], list(ACTION_WORDS))

    def test_rejects_bad_horizons(self):
        with self.assertRaises(ValueError):
            FrameInfluenceProbe(FakeCausalModel(), FakeTokenizer(), IMAGE, horizons=(1, 4))

    def test_rejects_bad_measures(self):
        for bad in ((), ("keep_all",)):
            with self.assertRaises(ValueError):
                FrameInfluenceProbe(FakeCausalModel(), FakeTokenizer(), IMAGE, measures=bad)
        self.assertEqual(FrameInfluenceProbe(FakeCausalModel(), FakeTokenizer(), IMAGE,
                                             measures=("keep_one",)).describe()["measures"], ["keep_one"])


class RecordTests(unittest.TestCase):
    """run() on the fake model: keep-one entries are scored against the no-visual prompt."""

    def _run(self, measures, layered=False, probe_batch=8):
        tab = table()
        ids = synthetic_ids()
        g = torch.Generator().manual_seed(7)
        feat = torch.randn(3, 4, C, generator=g)
        nav = torch.randn(64, C, generator=g)
        embeds = assemble(ids, [feat[i] for i in range(3)], nav, tab)
        if layered:
            model = FakeLayeredModel(feat, tab)
        else:
            model = FakeCausalModel()
            model.get_model = lambda: SimpleNamespace(feat_cache=feat, embed_tokens=lambda t: tab[t])
        probe = FrameInfluenceProbe(model, FakeTokenizer(), IMAGE, horizons=(2,), measures=measures,
                                    probe_batch=probe_batch)
        if layered:
            probe.install()  # installs the column blinder's layer hooks
        probe.tracker.slots = []
        probe._embeds, probe._input_ids, probe._result_list = embeds, ids, [4, 4, 4]
        out = torch.cat([ids, torch.tensor([6])])[None]
        return probe.run(ids[None], out)

    def test_keep_one_entries_reference_no_visual(self):
        rec = self._run(("keep_one",))
        self.assertEqual(rec["frames"], [])
        self.assertEqual([e["frame"] for e in rec["keep_one"]], [0, 1])
        vis = rec["visual_all"]["logp"]
        for e in rec["keep_one"]:
            self.assertAlmostEqual(e["kl"], kl_divergence(e["logp"], vis), places=6)
        self.assertAlmostEqual(rec["keep_one_age0"]["kl"], kl_divergence(rec["history_all"]["logp"], vis), places=6)
        self.assertEqual(rec["keep_one_age0"]["frame"], 2)
        self.assertEqual(rec["n_probes"], 1 + 2 + 1 + 1 + 1 + 1)  # base, 2 keep-one, age0, history_all, K=2, visual_all

    def test_masked_keep_one_matches_reference_splice_on_mean_model(self):
        """On a masked-mean model the last position sees exactly the open tokens, so closing the
        columns must equal splicing exactly those tokens out. The masked form leaves the separators
        open (only visual columns close, as in the layer-blind arms and LongNav-R1), so the reference
        drops the other groups and the fine tokens but keeps the separators -- unlike the splice
        form, which the same model must therefore NOT reproduce."""
        tab, ids = table(), synthetic_ids()
        g = torch.Generator().manual_seed(7)
        feat = torch.randn(3, 4, C, generator=g)
        nav = torch.randn(64, C, generator=g)
        embeds = assemble(ids, [feat[i] for i in range(3)], nav, tab)
        layout = HistoryLayout.build(ids, [4, 4, 4], 3, [], IMAGE)
        plain = FakeLayeredModel(feat, tab)  # no hooks: an ordinary forward over spliced prompts

        def expect(keep_groups, keep_nav):
            drops = [layout.groups[j] for j in range(3) if j not in keep_groups]
            if not keep_nav:
                drops.append(layout.nav_span)
            lg = plain.lm_head(plain.model(drop_spans(embeds, drops)[None])[0][:, -1])[0]
            return action_logp(lg, [5, 6, 7, 8])[0]

        for batch in (8, 1, 2):
            rec = self._run(("keep_one_mask",), layered=True, probe_batch=batch)
            self.assertEqual(rec["keep_one_mode"], "mask")
            self.assertEqual([e["frame"] for e in rec["keep_one"]], [0, 1])
            vis = rec["visual_all_mask"]["logp"]
            self.assertTrue(torch.allclose(torch.tensor(vis), torch.tensor(expect((), False)), atol=1e-5))
            for e in rec["keep_one"]:
                self.assertTrue(torch.allclose(torch.tensor(e["logp"]), torch.tensor(expect((e["frame"],), False)), atol=1e-5))
                self.assertAlmostEqual(e["kl"], kl_divergence(e["logp"], vis), places=6)
            age0 = rec["keep_one_age0"]
            self.assertTrue(torch.allclose(torch.tensor(age0["logp"]), torch.tensor(expect((2,), True)), atol=1e-5))
            self.assertAlmostEqual(age0["kl"], kl_divergence(age0["logp"], vis), places=6)
            self.assertEqual(age0["frame"], 2)
            self.assertTrue(torch.allclose(torch.tensor(rec["logp_full"]), torch.tensor(expect((0, 1, 2), True)), atol=1e-5))
            self.assertEqual(rec["n_probes"], 1 + 2 + 1 + 1 + 1 + 1 + 2)  # + visual_all_mask, only_current_mask
        splice = self._run(("keep_one",), layered=True)
        self.assertFalse(torch.allclose(torch.tensor(splice["visual_all"]["logp"]), torch.tensor(vis), atol=1e-5))
        with self.assertRaises(ValueError):
            FrameInfluenceProbe(FakeCausalModel(), FakeTokenizer(), IMAGE, measures=("keep_one", "keep_one_mask"))

    def test_leave_one_record_unchanged(self):
        rec = self._run(("leave_one",))
        self.assertEqual([e["frame"] for e in rec["frames"]], [0, 1])
        self.assertEqual(rec["keep_one"], [])
        self.assertIsNone(rec["keep_one_age0"])
        for e in rec["frames"]:
            self.assertAlmostEqual(e["kl"], kl_divergence(rec["logp_full"], e["logp"]), places=6)


if __name__ == "__main__":
    unittest.main()
