"""Shared synthetic cache fixtures and fixed image inputs for inference tests."""
import gc
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.kv_prune import KVPruneState
from longnav.utils.pre_rope import ReindexState
from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
WIDE_BUDGET = 10 ** 9
NARROW_BUDGET = 280       # ~70 slots/turn on 240x320 noise frames -> prunes from ~step 4
FIDELITY_TOL = 2e-2
DIVERGENCE_MARGIN = 3.0

START = [
    {"role": "user", "content": [{"type": "text", "text": "Find the chair."}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
TURN = [
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]


def make_images(n, h=240, w=320):
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def build_prune(budget, window=None, importance="attn"):
    return VLMWorker(
        model_id=MODEL_ID,
        attn_impl="sdpa",
        dtype="bfloat16",
        use_sparse=True,
        context_window=window,
        context_window_mode="prune",
        kv_budget=budget,
        kv_prune_importance=importance,
    )


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def run_probs(worker, images):
    out = []
    for idx, image in enumerate(images):
        probs, _, _ = worker.infer_probs(START if idx == 0 else TURN, [image])
        out.append(np.asarray(probs))
    return out


# --- checks 1-2: bookkeeping on synthetic turns, CPU only ----------------------------
# Every token is its own global id AND its own mRoPE position, so cache slices, the
# position table, and the embed db can all be compared as plain lists of integers.
PARITY_PREFIX = 5
PARITY_HEADER = 3
PARITY_WINDOW = 2
PARITY_TURNS = [11, 7, 9, 6, 8, 7, 10]


class FakeLayer:
    def __init__(self, ids):
        self.keys = ids.reshape(1, 1, -1, 1)
        self.values = self.keys.clone()

    def ids(self):
        return self.keys.reshape(-1).tolist()


class FakeCache:
    """One FakeLayer per decoder layer; `ids` is one tensor (shared by n_layers copies) or a
    list with one tensor per layer (layer-selective pruning leaves layers ragged)."""

    def __init__(self, ids, n_layers=1):
        per_layer = list(ids) if isinstance(ids, (list, tuple)) else [ids] * n_layers
        self.layers = [FakeLayer(torch.as_tensor(x).clone()) for x in per_layer]

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].keys.shape[-2]


def bare_worker(mode, window, budget=WIDE_BUDGET, importance="random", recent_turns=2,
                n_layers=1, layer_range=None, score_layers=None):
    """A VLMWorker with just enough state to drive the eviction/prune paths, no model.

    `layer_range` / `score_layers` configure layer-selective pruning over `n_layers` fake
    decoder layers (None = uniform, the default the other checks rely on)."""
    worker = VLMWorker.__new__(VLMWorker)
    worker.kv_prune_layer_start = 0 if layer_range is None else layer_range.start
    worker.kv_prune_layer_end = None if layer_range is None else layer_range.stop
    worker.kv_prune_score_layers = "pruned"
    worker.kv_prune_log_influence = False
    worker.kv_prune_influence_stride = 1
    worker.kv_prune_log_layer_influence = False
    worker.kv_prune_layer_influence_starts = ()
    worker.kv_prune_visual_blind = False
    worker._blind_stage = False
    worker.kv_prune_log_keep_one = False
    worker._n_fake_layers = n_layers
    worker.context_window = window
    worker.context_window_mode = mode
    worker.prefix_ids = list(range(PARITY_HEADER))
    worker._prefix_len = PARITY_PREFIX
    worker.use_sparse = False
    worker.past_image_embeds = None
    worker.attn_probe = None
    worker.past_key_values = None
    worker._frame_records = []
    worker._abs_bounds = []
    worker._vis_counts = []
    worker._dropped = 0
    worker._n_evicted = 0
    worker.context_cut = "header"
    worker._turn_leads = []
    worker._strip_next_lead = False
    worker._window_recs = []
    worker._prefix_rec = None
    worker._boundary_rec = None
    worker._inference_ctx = torch.no_grad
    worker._reindex_state = None
    if mode in ("reindex", "prune"):
        worker._reindex_state = ReindexState()
        worker._reindex_state.configure(pruned_layers=layer_range, score_layers=score_layers,
                                        n_layers=n_layers)
    worker._kv_prune_state = KVPruneState() if mode == "prune" else None
    worker.kv_budget = budget
    worker.kv_prune_recent_turns = recent_turns
    worker.kv_prune_ema_beta = 0.0
    worker.kv_prune_importance = importance
    worker.kv_prune_candidate_scope = "all"
    worker.kv_prune_seed = 17
    worker.kv_prune_budget_frames = None
    worker.kv_prune_stages = None
    worker.kv_prune_stage_importance = None
    worker.kv_prune_tail_budget_frames = None
    worker.kv_prune_budget_scope = "last_stage"
    worker.kv_prune_schedule_anchor = "original"
    worker.kv_prune_dart_pivots = 8
    worker.kv_prune_seed_per_step = False
    worker.kv_prune_log_masks = False
    worker._hidden_db = None
    worker._episode_label = ""
    worker._vtp_log = {}
    worker.offset = 0
    return worker


def feed_turn(worker, ids):
    """Append one synthetic pure-text turn (global positions == token ids) and prune."""
    live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
    worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
    if worker._reindex_state is not None:
        pos = torch.arange(worker.offset, worker.offset + len(ids))
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += len(ids)
    if worker.context_window_mode == "prune":
        worker._apply_kv_prune()
    else:
        worker._apply_context_window()
    return worker.past_key_values.layers[0].ids()


