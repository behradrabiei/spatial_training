"""Unit tests for the LightNav-0 keep-one influence wrapper (baselines/lightnav_hm3d/eval_influence.py)
and its loader (src/longnav/analysis/influence.py).

No weights, no GPU, no vLLM: the bin rule, the distributions, the masks, the attention hook's
no-cached-prefix guard, the logits capture and the record -> row loader are checked on synthetic
tensors and stub layers. Needs torch (and lightnav + the checkpoint's RVQ bundle for the bin table),
so run it with the client runtime:

    PYTHONPATH=src:baselines/LightNav-0/src baselines/.envs/lightnav/bin/python \\
        -m unittest baselines.tests.test_lightnav_influence
"""

from __future__ import annotations

import importlib.util
import base64
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_LIGHTNAV = HAVE_TORCH and importlib.util.find_spec("lightnav") is not None
HAVE_LONGNAV = importlib.util.find_spec("longnav") is not None
if HAVE_TORCH:
    import numpy as np
    import torch

    from baselines.lightnav_hm3d.eval_blind import VIDEO_PAD_ID, attention_mask
    from baselines.lightnav_hm3d.eval_influence import (
        BINS,
        N_CODES,
        InfluenceState,
        action_logp,
        arg_from,
        bin_matrix,
        build_parser,
        check_generate_counts,
        entropy,
        fold_logp,
        kl,
        make_influence_forward,
        make_logits_hook,
        only_mask,
        pack_f16,
        probe_fields,
        unpack_f16,
        velocity_bin,
    )
    from baselines.lightnav_hm3d.eval_prune import split_history, tubelet_layout

H, KVH, D = 4, 2, 8
BLOCK = 4
N_LAYERS = 36
VISION_START, VISION_END, TEXT = 151652, 151653, 1000
MERGE = 2
BUNDLE = Path(__file__).resolve().parents[1] / "LightNav-0/checkpoints/LightNav-0/action_tokenizer"


def grid_for(pool: int, n_tubelets: int):
    h, w = math.ceil(8 / pool) * MERGE, math.ceil(14 / pool) * MERGE
    return [n_tubelets, h, w]


def prompt_for(tubelets):
    ids = [TEXT, TEXT, TEXT]
    for t in tubelets:
        ids += [TEXT, VISION_START] + [VIDEO_PAD_ID] * t.n_tokens + [VISION_END]
    ids += [TEXT, TEXT]
    pads = [i for i, x in enumerate(ids) if x == VIDEO_PAD_ID]
    return ids, pads


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


def fake_state(vocab=600, act0=300):
    """A state with synthetic token ids (256 contiguous act_l0 rows from ``act0``) and a bin table."""
    st = InfluenceState(max_age=33)
    st.n_layers = N_LAYERS
    st.strict = True
    st.act_ids = torch.arange(act0, act0 + N_CODES)
    st.apos0, st.opos0, st.eos_id = 10, 20, 5
    st.bins = [BINS[i % len(BINS)] for i in range(N_CODES)]
    st.fold = bin_matrix(st.bins)
    return st, vocab


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class BinRuleTest(unittest.TestCase):
    def test_velocity_bins(self):
        self.assertEqual(velocity_bin(-1.0, 0.0), "stop")  # 0 m/s, 0 deg/s
        self.assertEqual(velocity_bin(-1.0, 0.02), "stop")  # 0.6 deg/s < min_abs_ang_speed
        self.assertEqual(velocity_bin(-1.0, 0.6), "rot_L")  # 18 deg/s in place
        self.assertEqual(velocity_bin(-1.0, -0.6), "rot_R")
        self.assertEqual(velocity_bin(0.2, 0.0), "fwd")  # 0.15 m/s straight
        self.assertEqual(velocity_bin(0.2, 0.05), "fwd")  # 1.5 deg/s still straight
        self.assertEqual(velocity_bin(0.2, 0.2), "fwd_L")  # 6 deg/s arc
        self.assertEqual(velocity_bin(0.2, -0.5), "fwd_R")
        self.assertEqual(velocity_bin(-0.9, 0.0), "stop")  # 0.0125 m/s < min_abs_lin_speed

    @unittest.skipUnless(HAVE_LIGHTNAV and BUNDLE.is_dir(), "needs lightnav and the checkpoint's RVQ bundle")
    def test_code_bins_of_the_checkpoint(self):
        from lightnav.traj_vocab import load_rvq_bundle

        from baselines.lightnav_hm3d.eval_influence import code_bins, waypoints_bin

        bundle = load_rvq_bundle(BUNDLE, 10)
        bins = code_bins(bundle)
        self.assertEqual(len(bins), N_CODES)
        self.assertEqual(bins[bundle.stop_l0], "stop")
        counts = {b: bins.count(b) for b in BINS}
        self.assertEqual(counts, {"stop": 8, "fwd": 80, "fwd_L": 39, "fwd_R": 42, "rot_L": 43, "rot_R": 44})
        self.assertEqual(waypoints_bin(np.zeros((10, 3), dtype=np.float32)), "stop")
        self.assertEqual(waypoints_bin(bundle.decode_waypoints([6, 122, 174])), "stop")  # the stop tuple decodes to ~0


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class DistributionTest(unittest.TestCase):
    def test_action_logp_renormalises_and_reports_mass(self):
        st, vocab = fake_state()
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(vocab, generator=g)
        logp, p_mass = action_logp(logits, st.act_ids)
        self.assertEqual(logp.shape, (N_CODES,))
        self.assertAlmostEqual(float(logp.exp().sum()), 1.0, places=5)
        full = torch.softmax(logits, dim=-1)
        self.assertAlmostEqual(p_mass, float(full[st.act_ids].sum()), places=6)
        torch.testing.assert_close(logp, torch.log(full[st.act_ids] / full[st.act_ids].sum()), rtol=1e-5, atol=1e-6)

    def test_kl_entropy_fold_and_f16(self):
        st, _ = fake_state()
        g = torch.Generator().manual_seed(1)
        p = torch.log_softmax(torch.randn(N_CODES, generator=g), dim=-1)
        q = torch.log_softmax(torch.randn(N_CODES, generator=g), dim=-1)
        self.assertAlmostEqual(kl(p, p), 0.0, places=6)
        self.assertGreater(kl(p, q), 0.0)
        self.assertAlmostEqual(kl(p, q), float(np.sum(np.exp(p.numpy()) * (p.numpy() - q.numpy()))), places=5)
        self.assertAlmostEqual(entropy(p), float(-np.sum(np.exp(p.numpy()) * p.numpy())), places=5)
        self.assertIsNone(kl(None, q))
        pb = fold_logp(p, st.fold)
        self.assertEqual(pb.shape, (len(BINS),))
        self.assertAlmostEqual(float(pb.exp().sum()), 1.0, places=5)
        # fold = sum of the code probabilities per bin
        for i, b in enumerate(BINS):
            mass = sum(float(p.exp()[c]) for c in range(N_CODES) if st.bins[c] == b)
            self.assertAlmostEqual(float(pb.exp()[i]), mass, places=5)
        # coarsening never increases the KL (data processing)
        self.assertLessEqual(kl(pb, fold_logp(q, st.fold)), kl(p, q) + 1e-6)
        back = unpack_f16(pack_f16(p))
        self.assertEqual(back.shape, (N_CODES,))
        self.assertLess(float(np.abs(back - p.numpy()).max()), 2e-2)

    def test_probe_fields(self):
        st, vocab = fake_state()
        g = torch.Generator().manual_seed(2)
        rows = [torch.randn(vocab, generator=g) for _ in range(3)]
        rows[0][st.apos0 + 7] += 20.0  # confident apos 7
        rows[1][st.opos0 + 3] += 20.0
        rows[2][st.act_ids[9]] += 20.0  # code 9
        tokens = [st.apos0 + 7, st.opos0 + 3, int(st.act_ids[9])]
        fields, logp, logp_bin = probe_fields(st, rows, tokens)
        self.assertEqual((fields["apos"], fields["opos"], fields["code"], fields["bin"]), (7, 3, 9, st.bins[9]))
        self.assertGreater(fields["p_apos"], 0.99)
        self.assertTrue(fields["in_set"])
        self.assertEqual(fields["bin_argmax"], BINS[int(logp_bin.argmax())])
        self.assertEqual(len(fields["logp_bin"]), len(BINS))
        np.testing.assert_allclose(unpack_f16(fields["logp_f16"]), logp.numpy(), atol=2e-2)
        # a generate that ended before the action position carries no distribution
        short, lp, lb = probe_fields(st, rows[:1], [st.eos_id])
        self.assertTrue(short["truncated"])
        self.assertIsNone(lp)
        self.assertIsNone(kl(lp, logp))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class MaskTest(unittest.TestCase):
    def layout(self):
        segs = [([0, 1], 4), ([4, 5, 6, 7], 2), ([9, 10], 1)]  # anchor, two fast tubelets, current
        tubs = tubelet_layout(segs, [grid_for(4, 1), grid_for(2, 2), grid_for(1, 1)])
        ids, pads = prompt_for(tubs)
        return tubs, ids, pads

    def test_only_masks(self):
        tubs, ids, pads = self.layout()
        visual = torch.as_tensor(ids) == VIDEO_PAD_ID
        hist, cur = split_history(tubs, 10)
        self.assertEqual([t.f1 for t in hist], [1, 5, 7])
        no_visual = visual.clone()
        self.assertEqual(int(no_visual.sum()), len(pads))
        only_cur = only_mask(visual, pads, cur)
        self.assertEqual(int((~only_cur & visual).sum()), cur[0].n_tokens)  # 112 open
        only_f = only_mask(visual, pads, [hist[1]])
        open_pos = torch.nonzero(visual & ~only_f).flatten().tolist()
        self.assertEqual(open_pos, [pads[hist[1].start + c] for c in range(hist[1].n_tokens)])
        self.assertTrue(all(only_f[pads[cur[0].start + c]] for c in range(cur[0].n_tokens)))  # current hidden too
        self.assertFalse(bool(only_f[~visual].any()))  # text never hidden

    def test_age_cutoff_selects_recent_tubelets(self):
        st = InfluenceState(max_age=4)
        tubs, _, _ = self.layout()
        hist, _ = split_history(tubs, 10)
        probed = [t for t in hist if 10 - t.f1 <= st.max_age]
        self.assertEqual([t.f1 for t in probed], [7])
        with self.assertRaises(ValueError):
            InfluenceState(max_age=0)


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class HookTest(unittest.TestCase):
    def setUp(self):
        self.original_calls = []

    def original(self, impl, layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
        self.original_calls.append(layer.layer_name)
        return output.fill_(7.0)

    def run_forward(self, state, index, seq_len, n_query, seed=1):
        forward = make_influence_forward(state, self.original)
        cache, row, key, value = paged_cache(seq_len, seed)
        query = torch.randn(n_query, H, D, generator=torch.Generator().manual_seed(seed + 7))
        output = torch.zeros(n_query + 2, H, D)
        layer = SimpleNamespace(layer_name=f"language_model.model.layers.{index}.self_attn.attn")
        got = forward(FakeImpl(), layer, query, key[:n_query], value[:n_query], cache, metadata(seq_len, n_query, row), output=output)
        self.assertIs(got, output)
        return output[:n_query], query, key, value

    def test_hidden_set_on_every_layer_prefill_and_decode(self):
        st, _ = fake_state()
        hide = torch.zeros(12, dtype=torch.bool)
        hide[4:7] = True
        st.set_hide(hide)
        for index in (0, 17, 35):
            out, query, key, value = self.run_forward(st, index, seq_len=12, n_query=12)
            mask = attention_mask(12, 12, hide)
            torch.testing.assert_close(out.double(), naive_attention(query, key, value, 1 / math.sqrt(D), mask), rtol=1e-5, atol=1e-5)
            out, query, key, value = self.run_forward(st, index, seq_len=15, n_query=1)
            mask = attention_mask(1, 15, hide)
            torch.testing.assert_close(out.double(), naive_attention(query, key, value, 1 / math.sqrt(D), mask), rtol=1e-5, atol=1e-5)
        self.assertEqual(st.calls, 6)
        self.assertEqual(self.original_calls, [])

    def test_cached_prefix_is_refused(self):
        st, _ = fake_state()
        st.set_hide(torch.zeros(12, dtype=torch.bool))
        with self.assertRaises(RuntimeError):
            self.run_forward(st, 0, seq_len=12, n_query=8)  # 4 prompt rows came from the prefix cache

    def test_warmup_falls_through_and_strict_refuses(self):
        st = InfluenceState()
        forward = make_influence_forward(st, self.original)
        layer = SimpleNamespace(layer_name="language_model.model.layers.3.self_attn.attn")
        out = torch.zeros(2, H, D)
        forward(FakeImpl(), layer, None, None, None, None, None, output=out)
        cache, row, _, _ = paged_cache(8, 0)
        forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)
        self.assertEqual(len(self.original_calls), 2)
        st.strict = True
        with self.assertRaises(RuntimeError):
            forward(FakeImpl(), layer, torch.zeros(2, H, D), None, None, cache, metadata(8, 2, row), output=out)

    def test_logits_hook_captures_per_call_and_counts(self):
        st, vocab = fake_state()
        calls = []

        def original(hidden, *a, **k):
            calls.append(hidden.shape)
            return (torch.arange(vocab, dtype=torch.float32)[None] * float(len(calls))).expand(hidden.shape[0], -1)

        hook = make_logits_hook(st, original)
        hook(torch.zeros(1, 4))  # no generate in flight: not captured
        st.set_hide(torch.zeros(3, dtype=torch.bool))
        for _ in range(3):
            hook(torch.zeros(1, 4))
        self.assertEqual(len(st.logits), 3)
        self.assertEqual(float(st.logits[2][1]), 4.0)
        with self.assertRaises(RuntimeError):
            hook(torch.zeros(2, 4))
        st.calls = N_LAYERS * 3
        self.assertEqual(check_generate_counts(st, 12, 6, 3), 3)
        with self.assertRaises(RuntimeError):
            check_generate_counts(st, 12, 6, 4)  # fewer logits rows than tokens
        st.calls = N_LAYERS - 1
        with self.assertRaises(RuntimeError):
            check_generate_counts(st, 12, 6, 3)
        st.calls = N_LAYERS
        with self.assertRaises(RuntimeError):
            check_generate_counts(st, 12, 0, 1)

    def test_episode_reset_and_record_log(self):
        st = InfluenceState()
        with tempfile.TemporaryDirectory() as d:
            st.log_dir = Path(d)
            st.reset_episode()
            st.log_record({"step": 0})
            st.reset_episode()
            st.log_record({"step": 0})
            st.log_record({"step": 1})
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["episode_000.jsonl", "episode_001.jsonl"])
            self.assertEqual(len(Path(d, "episode_001.jsonl").read_text().splitlines()), 2)
        self.assertEqual(st.describe()["max_age"], 33)
        self.assertEqual(st.describe()["bins"], list(BINS))


@unittest.skipUnless(HAVE_TORCH, "needs torch (run with baselines/.envs/lightnav)")
class CliTest(unittest.TestCase):
    def test_flags_and_passthrough(self):
        args, rest = build_parser().parse_known_args(["--max-age", "20", "--episodes", "-1", "--output_dir", "/tmp/x/raw",
                                                      "--model_path", "/ckpt"])
        self.assertEqual((args.max_age, args.influence_debug_steps), (20, 0))
        self.assertEqual(rest, ["--episodes", "-1", "--output_dir", "/tmp/x/raw", "--model_path", "/ckpt"])
        self.assertEqual(arg_from(rest, "--model_path"), "/ckpt")
        self.assertEqual(arg_from(["--model_path=/a"], "--model_path"), "/a")
        self.assertIsNone(arg_from(rest, "--nope"))
        self.assertEqual(build_parser().parse_known_args([])[0].max_age, 33)


def write_fake_run(root: Path) -> None:
    """A 2-step LightNav influence run in the converted layout."""
    (root / "episodes").mkdir(parents=True)
    (root / "influence").mkdir()
    (root / "episodes" / "ep_a.json").write_text(json.dumps({
        "episode_label": "ep_a", "metrics": {"success": 1.0, "spl": 0.5, "distance_to_goal": 0.3}, "steps": 2,
        "forced_stop": False}))
    lb = [math.log(x) for x in (0.1, 0.5, 0.1, 0.1, 0.1, 0.1)]
    lb_ref = [math.log(x) for x in (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)]
    # 256-way log-probs: code 12 (bin fwd) 0.7 and code 3 (bin stop) 0.3 for the full decision; the
    # reference puts 0.9 on code 3; a probe peaked on code 12 reproduces the decision, one on code 5 does not
    bins = ["stop", "fwd", "fwd_L", "fwd_R", "rot_L", "rot_R"]
    code_bins = ["fwd"] * 256
    code_bins[3] = "stop"; code_bins[5] = "rot_L"
    (root / "influence" / "meta.json").write_text(json.dumps({"bins": bins, "code_bins": code_bins}))

    def f16(peaks):
        p = np.full(256, 1e-6); p[list(peaks)] = list(peaks.values()); p /= p.sum()
        return base64.b64encode(np.log(p).astype(np.float16).tobytes()).decode("ascii")

    full = {"exec_bin": "fwd", "code": 12, "bin": "fwd", "bin_argmax": "fwd", "p_mass": 0.98, "entropy": 1.2,
            "logp_bin": lb, "p_apos": 0.9, "p_opos": 0.8, "logp_f16": f16({12: 0.7, 3: 0.3})}
    ref = {"code": 3, "bin": "stop", "bin_argmax": "stop", "logp_bin": lb_ref, "logp_f16": f16({3: 0.9})}
    recs = [
        {"step": 0, "full": dict(full, exec_bin="rot_L"), "visual_all_mask": ref, "replay": {"kl": 1e-6, "kl_bin": 1e-7},
         "keep_one_age0": {"kl": 0.5, "kl_bin": 0.4, "logp_f16": f16({3: 0.95})}, "keep_one": [], "frames_unprobed": [],
         "probe_seconds": 0.5},
        {"step": 2, "full": full, "visual_all_mask": ref, "replay": {"kl": 2e-6, "kl_bin": 1e-7},
         "keep_one_age0": {"kl": 0.6, "kl_bin": 0.3, "logp_f16": f16({12: 0.95})},
         "keep_one": [{"frame": 1, "f0": 0, "pool": 2, "n_tokens": 28, "age": 1, "kl": 0.2, "kl_bin": 0.1,
                       "code": 12, "bin": "fwd", "bin_argmax": "fwd", "logp_bin": lb, "logp_f16": f16({5: 0.95})},
                      {"frame": 0, "f0": 0, "pool": 4, "n_tokens": 8, "age": 2, "kl": None, "kl_bin": None,
                       "code": None, "bin": None, "truncated": True}],
         "frames_unprobed": [], "probe_seconds": 1.0},
    ]
    (root / "influence" / "ep_a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


@unittest.skipUnless(HAVE_LONGNAV, "needs longnav on PYTHONPATH (PYTHONPATH=src)")
class LoaderTest(unittest.TestCase):
    def test_rows_and_steps_in_both_spaces(self):
        from longnav.analysis import influence as inf

        with tempfile.TemporaryDirectory() as d:
            write_fake_run(Path(d))
            rows, steps = inf.load_lightnav_influence(d, "LightNav-0")
            self.assertEqual(len(steps), 2)
            self.assertEqual(len(rows), 2)
            r = rows[0]
            self.assertEqual((r["frame"], r["age"], r["infl"], r["tier"], r["n_slots"], r["weight"]), (1, 1, 0.2, "p2", 28, 2))
            self.assertEqual((r["argmax_action"], r["argmax_removed"], r["flip"]), ("12", "12", 1))  # ref argmax code 3
            self.assertEqual((r["chosen_action"], r["action_at_frame"]), ("fwd", ""))  # step 1 was no inference step
            self.assertEqual(rows[1]["action_at_frame"], "rot_L")  # frame 0 = the step-0 decision
            self.assertTrue(math.isnan(rows[1]["infl"]))
            self.assertEqual(rows[1]["flip"], "")
            self.assertEqual(r["logp_full"], "")
            s = steps[1]
            self.assertEqual((s["age0"], s["kl_live_vs_replay"], s["n_scored"], s["success"]), (0.6, 2e-6, 2, 1.0))
            self.assertTrue(math.isnan(s["history_all"]) and s["horizon"] == {})
            self.assertAlmostEqual(s["entropy_t"], 1.2)
            brows, bsteps = inf.load_lightnav_influence(d, "LightNav-0 (6-bin)", space="bins")
            self.assertEqual(brows[0]["infl"], 0.1)
            self.assertEqual((brows[0]["argmax_action"], brows[0]["argmax_removed"], brows[0]["flip"]), ("fwd", "fwd", 1))
            self.assertEqual(len(brows[0]["logp_full"]), 6)
            self.assertEqual(bsteps[1]["age0"], 0.3)
            self.assertAlmostEqual(bsteps[1]["p_chosen"], 0.5, places=6)
            for fields in (inf.ROW_FIELDS,):
                self.assertTrue(set(fields) <= set(rows[0]))
            with self.assertRaises(ValueError):
                inf.load_lightnav_influence(d, measure="leave_one")
            with self.assertRaises(ValueError):
                inf.load_lightnav_influence(d, space="nope")

    def test_full_agreement_by_age(self):
        from longnav.analysis import influence as inf

        with tempfile.TemporaryDirectory() as d:
            write_fake_run(Path(d))
            rows, summ, eps = inf.lightnav_full_agreement(d)
            self.assertEqual(eps, [{"age": 0, "episode": "ep_a", "n": 2, "agree_bin": 0.5},
                                   {"age": 1, "episode": "ep_a", "n": 1, "agree_bin": 0.0}])
            by_age = {r["age"]: r for r in rows}
            self.assertEqual(sorted(by_age), [0, 1])
            # age 0: step 0's current-tubelet probe picks code 3 (stop) against a full decision of code 12
            # (fwd); step 2's picks code 12 -> one agreement of two, in both spaces
            self.assertEqual((by_age[0]["n"], by_age[0]["agree_bin"], by_age[0]["agree_code"]), (2, 0.5, 0.5))
            # age 1: the only probe is peaked on code 5 (rot_L): wrong code and wrong bin
            self.assertEqual((by_age[1]["n"], by_age[1]["agree_bin"], by_age[1]["agree_code"]), (1, 0.0, 0.0))
            self.assertLess(by_age[1]["p_full_bin"], 1e-3)
            self.assertGreater(by_age[1]["kl_full_only"], by_age[0]["kl_full_only"])
            self.assertAlmostEqual(by_age[0]["agree_bin_balanced"], 0.5)  # one episode: balanced == pooled
            self.assertTrue(math.isnan(by_age[0]["agree_bin_sem"]))
            self.assertEqual((summ["n_decisions"], summ["majority_class"], summ["majority_class_rate"]), (2, "fwd", 1.0))
            self.assertEqual(summ["reference_agree_bin"], 0.0)  # the no-visual prior says stop at both decisions

    def test_matrix_uses_age0_on_the_diagonal(self):
        from longnav.analysis import influence as inf

        with tempfile.TemporaryDirectory() as d:
            write_fake_run(Path(d))
            rows, steps = inf.load_lightnav_influence(d, "LightNav-0")
            M = inf.influence_matrix(rows, steps, "ep_a")
            self.assertEqual(M.shape, (3, 3))
            self.assertEqual((M[0, 0], M[2, 2], M[2, 1]), (0.5, 0.6, 0.2))
            self.assertTrue(math.isnan(M[2, 0]))


if __name__ == "__main__":
    unittest.main()
