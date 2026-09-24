"""Unit tests for uninavid_hm3d.layer_blind (no model weights, no Habitat).

Run from the repo root with the baseline runtime:
    PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python \
        -m unittest baselines.tests.test_layer_blind
"""
import unittest
from types import SimpleNamespace

import torch

from uninavid_hm3d.eval import parse_args
from uninavid_hm3d.influence import HistoryLayout
from uninavid_hm3d.layer_blind import ColumnBlinder, LayerProbe, hidden_mask, visual_columns
from uninavid_hm3d.prune import N_PER_FRAME, NAV_TOKENS, prompt_length

from baselines.tests.test_prune import C, IMAGE, FakeLayer, causal_mask, synthetic_ids

N_LAYERS = 32


class FakeModel:
    """Model stub with an explicit history layout (long-term slots allowed)."""

    def __init__(self, result_list, n_layers=N_LAYERS):
        self.result_list = list(result_list)
        n_frames = sum(1 for g in self.result_list if g == N_PER_FRAME) + sum(1 for g in self.result_list if g == 1)
        self.state = SimpleNamespace(feat_cache=torch.zeros(n_frames, N_PER_FRAME, C),
                                     layers=[FakeLayer(i) for i in range(n_layers)])

    def get_model(self):
        return self.state

    def online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985):
        return torch.zeros(sum(self.result_list), C), list(self.result_list)


def hidden_columns(mask):
    return set(torch.nonzero((mask[0, 0] == torch.finfo(mask.dtype).min).all(dim=0)).flatten().tolist())


def run_prefill(model, blinder, ids, start, kind="all"):
    """One inference through the hooks: prefill then a decode step, as generate() would."""
    blinder.begin_inference(ids)
    if kind == "all":
        blinder.hide_all_from(start)
    tokens, result_list = model.online_process_tensor(N_PER_FRAME)
    length = prompt_length(ids.numel(), result_list)
    mask = causal_mask(length)
    hidden = torch.zeros(1, length, C)
    for layer in model.state.layers:
        hidden = layer.forward(hidden, attention_mask=mask, position_ids=None, past_key_value=None,
                               output_attentions=False, use_cache=True)[0]
    decode_mask = torch.zeros(1, 1, 1, length + 3, dtype=torch.float16)
    for layer in model.state.layers:
        layer.forward(torch.zeros(1, 1, C), attention_mask=decode_mask, position_ids=None,
                      past_key_value=("k", "v"), output_attentions=False, use_cache=True)
    blinder.finish()
    return mask, decode_mask


class ColumnTests(unittest.TestCase):
    def test_columns_match_history_layout(self):
        ids = synthetic_ids()
        result_list = [1, 1, 4, 4, 4]  # two long-term slots, three short-term frames
        placeholder = int(torch.nonzero(ids == IMAGE)[0])
        all_visual, current, total = visual_columns(placeholder, result_list, ids.numel())
        layout = HistoryLayout.build(ids, result_list, n_frames=3 + 5, slot_members=[[0, 1, 2], [3, 4]],
                                     image_token_index=IMAGE, length_threshold=3)  # 3 short-term frames
        self.assertEqual(total, layout.total_length)
        expected = set()
        for a, b in layout.groups:
            expected |= set(range(a, b))
        expected |= set(range(*layout.nav_span))
        self.assertEqual(set(torch.nonzero(all_visual).flatten().tolist()), expected)
        a, b = layout.groups[-1]
        self.assertEqual(set(torch.nonzero(current).flatten().tolist()),
                         set(range(a, b)) | set(range(*layout.nav_span)))
        # separators and the special tokens stay visible
        for j in range(len(layout.groups) - 1):
            self.assertFalse(bool(all_visual[layout.groups[j][1]]))
        self.assertFalse(bool(all_visual[layout.special_span[0]:layout.special_span[1]].any()))
        self.assertEqual(int(all_visual.sum()), sum(result_list) + NAV_TOKENS)

    def test_hidden_mask_shapes(self):
        mask = causal_mask(10)
        hide = torch.zeros(10, dtype=torch.bool)
        hide[2:5] = True
        out = hidden_mask(mask, hide)
        self.assertIsNot(out, mask)
        self.assertEqual(hidden_columns(out), {2, 3, 4})
        wide = torch.zeros(1, 1, 1, 13, dtype=torch.float16)
        self.assertEqual(set(torch.nonzero(hidden_mask(wide, hide)[0, 0, 0] == torch.finfo(torch.float16).min)
                             .flatten().tolist()), {2, 3, 4})
        batched = hidden_mask(torch.zeros(2, 1, 10, 10, dtype=torch.float16), torch.stack([hide, ~hide]))
        self.assertEqual(hidden_columns(batched[:1]), {2, 3, 4})
        self.assertEqual(hidden_columns(batched[1:]), set(range(10)) - {2, 3, 4})
        with self.assertRaises(RuntimeError):
            hidden_mask(torch.zeros(1, 1, 1, 8, dtype=torch.float16), hide)
        with self.assertRaises(RuntimeError):
            hidden_mask(None, hide)


class BlinderTests(unittest.TestCase):
    def setUp(self):
        self.ids = synthetic_ids()
        self.model = FakeModel([1, 4, 4, 4])
        self.blinder = ColumnBlinder(IMAGE)
        self.blinder.install(self.model)

    def test_identity_at_depth(self):
        mask, decode_mask = run_prefill(self.model, self.blinder, self.ids, start=N_LAYERS)
        for layer in self.model.state.layers:
            self.assertIs(layer.calls[0][1], mask)
            self.assertIs(layer.calls[1][1], decode_mask)

    def test_hides_all_visual_from_layer_on_prefill_and_decode(self):
        mask, decode_mask = run_prefill(self.model, self.blinder, self.ids, start=8)
        placeholder = int(torch.nonzero(self.ids == IMAGE)[0])
        all_visual, _, _ = visual_columns(placeholder, self.model.result_list, self.ids.numel())
        cols = set(torch.nonzero(all_visual).flatten().tolist())
        layers = self.model.state.layers
        for i in range(8):
            self.assertIs(layers[i].calls[0][1], mask)
            self.assertIs(layers[i].calls[1][1], decode_mask)
        for i in range(8, N_LAYERS):
            got = layers[i].calls[0][1]
            self.assertIsNot(got, mask)
            self.assertEqual(hidden_columns(got), cols)
            keep = torch.ones(mask.shape[-1], dtype=torch.bool)
            keep[list(cols)] = False
            self.assertTrue(torch.equal(got[..., keep], mask[..., keep]))
            dec = layers[i].calls[1][1]
            self.assertEqual(set(torch.nonzero(dec[0, 0, 0] == torch.finfo(dec.dtype).min).flatten().tolist()), cols)
        self.assertTrue(torch.equal(mask, causal_mask(mask.shape[-1])))  # shared mask never mutated

    def test_inert_outside_an_inference(self):
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.float16)
        self.model.state.layers[5].forward(torch.zeros(1, 3, C), attention_mask=mask, use_cache=True)
        self.assertIs(self.model.state.layers[5].calls[0][1], mask)
        run_prefill(self.model, self.blinder, self.ids, start=0)
        self.model.state.layers[5].forward(torch.zeros(1, 3, C), attention_mask=mask, use_cache=True)
        self.assertIs(self.model.state.layers[5].calls[-1][1], mask)  # finish() cleared the hide


class ProbeTests(unittest.TestCase):
    """The probe's arithmetic on a stub model whose last-position logits depend on the mask."""

    def make_probe(self, layers):
        model = FakeModel([4, 4, 4])
        model.model = None  # replaced below
        probe = LayerProbe.__new__(LayerProbe)
        probe.model = model
        probe.blinder = ColumnBlinder(IMAGE)
        probe.layers = tuple(sorted(set(layers)))
        probe.action_ids = [10, 11, 12, 13]
        probe._installed = False
        probe._embeds = probe._input_ids = None
        probe.blinder.install(model)
        probe._installed = True
        return model, probe

    def test_kl_zero_at_depth_and_forward_count(self):
        model, probe = self.make_probe([0, 8, N_LAYERS])
        ids = synthetic_ids()
        forwards = []

        def fake_logits(batch, hide, start=0):
            forwards.append((batch.shape[0], None if hide is None else hide.shape, start))
            logits = torch.zeros(batch.shape[0], 20)
            if hide is not None:  # a masked forward tilts the distribution by how much is hidden
                logits[:, 10] = hide.float().sum(dim=1) / 10.0
            return logits

        probe._last_logits = fake_logits
        probe.begin_inference(ids)
        model.online_process_tensor(N_PER_FRAME)
        length = prompt_length(ids.numel(), model.result_list)
        probe._embeds = torch.zeros(length, C)
        probe._input_ids = ids.clone()
        record = probe.run(ids)
        self.assertEqual(record["layers"], [0, 8, N_LAYERS])
        self.assertEqual(record["keep_one"][2], 0.0)
        self.assertEqual(record["leave_one"][2], 0.0)
        self.assertGreater(record["leave_one"][0], 0.0)
        self.assertGreater(record["keep_one"][0], 0.0)
        self.assertEqual(record["n_probes"], 1 + 3 * 2)
        self.assertEqual(forwards[0], (1, None, 0))
        self.assertEqual(forwards[1][0], 3)
        self.assertEqual(forwards[1][1], (3, length))
        self.assertEqual([f[2] for f in forwards[1:]], [0, 8])
        self.assertEqual(record["n_visual_tokens"], 12 + NAV_TOKENS)
        self.assertEqual(record["n_current_tokens"], 4 + NAV_TOKENS)


class CliTests(unittest.TestCase):
    def test_parse_args_guards(self):
        args = parse_args(["--blind-layer", "16"])
        self.assertEqual(args.blind_layer, 16)
        args = parse_args(["--layer-probe", "--probe-layers", "4,0,32"])
        self.assertEqual(args.probe_layers, [0, 4, 32])
        self.assertIsNone(parse_args([]).blind_layer)
        for argv in (
            ["--blind-layer", "16", "--layer-probe"],
            ["--blind-layer", "16", "--context-frames", "32"],
            ["--blind-layer", "16", "--no-visual-tokens"],
            ["--blind-layer", "16", "--frame-influence"],
            ["--blind-layer", "16", "--prune-method", "random", "--prune-k", "4", "--context-frames", "32"],
            ["--layer-probe", "--context-window", "8"],
            ["--layer-probe", "--current-frame-only-64"],
            ["--layer-probe", "--frame-influence"],
            ["--layer-probe", "--probe-layers", "a,b"],
        ):
            with self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
