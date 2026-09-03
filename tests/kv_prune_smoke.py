"""Smoke test for context_window_mode='prune' (importance-based KV-slot pruning).

'prune' enforces a hard KV token budget on the reindex (pre-rotation) substrate: each
step the lowest-importance unprotected slots are sliced out of the cache, the per-slot
position table, the slot metadata, and the sparse embed db together. Importance is an EMA
of the decision token's attention row, captured inside pre_rope_attention_forward.

Six things have to hold:

  1. Forced-schedule parity -- with an effectively infinite budget, prune's window drop
     (turn-id based, since budget drops break evict's contiguous-cut arithmetic) must
     retire exactly the same tokens on exactly the same steps as evict, and survivors
     must keep their ORIGINAL positions (holes expected -- prune never renumbers) with
     worker.offset never rebased. CPU only, no model.

  2. Budget bookkeeping -- with an active budget and injected scores: the cache, position
     table, metadata, and embed db stay 1:1; the prefix and the last kv_prune_recent_turns
     turns always survive; every kept unprotected slot outranks every dropped one; the db
     rows track the visual survivors across repeated prunes. CPU only, no model.

  3. Merge math -- merge_dropped_visual folds a dropped slot into its nearest kept slot
     as an importance-weighted average, moves its importance mass, and updates the sparse
     embedding centroid. CPU only.

  4. Sparse-window alignment -- the current turn's dedup filter sees only visual history
     that will survive this step's context-window eviction, without prematurely mutating
     the canonical embedding DB. CPU only, no model.

  5. Fidelity -- with a budget nothing ever exceeds, prune must reproduce the stock
     forward (capture armed but inert), same tolerance as the reindex smoke.

  6. Non-vacuity -- with a budget that actually prunes, the cache must stay at the
     budget and the decision must move well beyond the arithmetic floor.

Checks 1-4 run anywhere; 5 and 6 need the GPU and the HF model. Run inside longnav_vlm:
    python tests/kv_prune_smoke.py
"""
import gc
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.kv_prune import KVPruneState, merge_dropped_visual
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


def build_prune(budget, window=None, merge=False, importance="attn"):
    return VLMWorker(
        model_id=MODEL_ID,
        attn_impl="sdpa",
        dtype="bfloat16",
        use_sparse=True,
        context_window=window,
        context_window_mode="prune",
        kv_budget=budget,
        kv_prune_merge=merge,
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
                granularity="slot", n_layers=1, layer_range=None, score_layers=None):
    """A VLMWorker with just enough state to drive the eviction/prune paths, no model.

    `layer_range` / `score_layers` configure layer-selective pruning over `n_layers` fake
    decoder layers (None = uniform, the default the other checks rely on)."""
    worker = VLMWorker.__new__(VLMWorker)
    worker.kv_prune_granularity = granularity
    worker.kv_prune_layer_start = 0 if layer_range is None else layer_range.start
    worker.kv_prune_layer_end = None if layer_range is None else layer_range.stop
    worker.kv_prune_score_layers = "pruned"
    worker.kv_prune_log_influence = False
    worker.kv_prune_influence_stride = 1
    worker.kv_prune_log_layer_influence = False
    worker.kv_prune_layer_influence_starts = ()
    worker.kv_prune_visual_blind = False
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
    worker.kv_prune_merge = False
    worker.kv_prune_importance = importance
    worker.kv_prune_candidate_scope = "all"
    worker.kv_prune_seed = 17
    worker.kv_prune_fisher_pool_factor = 2.0
    worker.kv_prune_voxel_cap = 1
    worker.kv_prune_voxel_scale = 1
    worker.kv_prune_voxel_2d = True
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


def check_forced_schedule_parity():
    """Prune with an infinite budget must reproduce evict's window schedule exactly."""
    evict = bare_worker("evict", PARITY_WINDOW)
    prune = bare_worker("prune", PARITY_WINDOW)
    failures = []

    next_id, total = 0, 0
    for step, length in enumerate(PARITY_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        total += length
        a = feed_turn(evict, ids)
        b = feed_turn(prune, ids)
        print(f"  step {step}: {len(a)} tokens retained ({'match' if a == b else 'MISMATCH'})")
        if a != b:
            failures.append(f"step {step}: evict retains {a} but prune retains {b}")

        st = prune._reindex_state
        positions = st.pos_table[0, 0].tolist()
        if positions != b:
            failures.append(f"step {step}: survivors' positions {positions} are not their "
                            f"original ids {b}; prune must never renumber")
        if prune.offset != total:
            failures.append(f"step {step}: offset {prune.offset} was rebased (expected {total})")
        ps = prune._kv_prune_state
        if not (len(b) == st.pos_table.shape[-1] == ps.turn_id.numel()
                == ps.importance.numel() == ps.is_visual.numel()):
            failures.append(f"step {step}: cache({len(b)}) / table({st.pos_table.shape[-1]}) / "
                            f"metadata({ps.turn_id.numel()}) lengths desynced")

    if not evict._n_evicted:
        failures.append("the window never evicted, so nothing was tested")
    for field in ("_dropped", "_n_evicted"):
        got = {m: getattr(w, field) for m, w in (("evict", evict), ("prune", prune))}
        if got["evict"] != got["prune"]:
            failures.append(f"{field} diverged: {got}")
    if prune._kv_prune_state.n_budget_dropped:
        failures.append("an infinite budget dropped slots; forced/budget accounting is mixed up")
    return failures


BUDGET_TURNS = [12, 9, 10, 8, 11]
BUDGET = 22
RECENT = 1


def check_budget_bookkeeping():
    """Injected scores: protection, ranking, and db/table/metadata sync under real prunes."""
    torch.manual_seed(0)
    worker = bare_worker("prune", window=None, budget=BUDGET, importance="attn",
                         recent_turns=RECENT)
    worker.use_sparse = True
    worker.language_model = SimpleNamespace(visual_pos_masks=None)
    worker.past_image_embeds = None
    failures = []

    next_id = 0
    vis_by_id = {}
    prev_kept = []
    for step, length in enumerate(BUDGET_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        # Odd token ids are 'visual'; their db row carries the id so drift is visible.
        vis_row = (ids % 2).bool()
        for i, v in zip(ids.tolist(), vis_row.tolist()):
            vis_by_id[i] = v
        worker.language_model.visual_pos_masks = vis_row.reshape(1, -1)
        db_new = ids[vis_row].float().reshape(-1, 1)
        if worker.past_image_embeds is None:
            worker.past_image_embeds = [db_new]
        else:
            worker.past_image_embeds[0] = torch.cat([worker.past_image_embeds[0], db_new])

        live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
        worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
        pos = torch.arange(worker.offset, worker.offset + length)
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += length
        # Injected decision row: importance == a fixed pseudo-random score per token id,
        # so ranking is deterministic and checkable across steps (beta=0 -> no EMA mixing).
        kv_len = worker.past_key_values.get_seq_length()
        all_ids = torch.tensor(worker.past_key_values.layers[0].ids())
        score = torch.sin(all_ids.float() * 12.9898) * 0.5 + 0.5
        worker._reindex_state.score_sum = score
        worker._reindex_state.score_layers = 1
        worker._apply_kv_prune()

        kept = worker.past_key_values.layers[0].ids()
        ps = worker._kv_prune_state
        st = worker._reindex_state
        n_turns = step + 1
        # 1:1 sync
        if not (len(kept) == st.pos_table.shape[-1] == ps.turn_id.numel()):
            failures.append(f"step {step}: cache/table/metadata desynced")
        if st.pos_table[0, 0].tolist() != kept:
            failures.append(f"step {step}: positions no longer match survivors' original ids")
        # budget respected (floor is far below BUDGET here)
        if len(kept) > max(BUDGET, PARITY_PREFIX):
            failures.append(f"step {step}: cache holds {len(kept)} slots over budget {BUDGET}")
        # prefix + recent turns survive
        prefix_ids_expected = list(range(PARITY_PREFIX))
        if kept[:PARITY_PREFIX] != prefix_ids_expected:
            failures.append(f"step {step}: prefix slots {prefix_ids_expected} not intact: {kept[:8]}")
        recent_lo = sum(BUDGET_TURNS[:n_turns - RECENT])
        recent_expected = list(range(recent_lo, next_id))
        if [i for i in kept if i >= recent_lo] != recent_expected:
            failures.append(f"step {step}: recent-turn slots incomplete")
        # ranking: THIS step's drops must all rank below every kept unprotected slot
        # (a slot protected at drop time can outlive a higher-scored slot dropped earlier,
        # so the greedy property only holds per step)
        dropped_now = sorted(set(prev_kept) | set(ids.tolist()))
        dropped_now = [i for i in dropped_now if i not in set(kept)]
        unprotected_kept = [i for i in kept if i >= PARITY_PREFIX and i < recent_lo]
        if dropped_now and unprotected_kept:
            def score_of(i):
                return float(torch.sin(torch.tensor(float(i)) * 12.9898) * 0.5 + 0.5)
            worst_kept = min(score_of(i) for i in unprotected_kept)
            best_dropped = max(score_of(i) for i in dropped_now)
            if best_dropped > worst_kept + 1e-6:
                failures.append(f"step {step}: dropped a higher-importance slot "
                                f"({best_dropped:.4f}) while keeping {worst_kept:.4f}")
        prev_kept = kept
        # db rows track visual survivors exactly
        db = worker.past_image_embeds[0].reshape(-1).long().tolist()
        vis_kept = [i for i in kept if vis_by_id[i]]
        if db != vis_kept:
            failures.append(f"step {step}: embed db {db} != visual survivors {vis_kept}")
        print(f"  step {step}: {len(kept)} slots kept, db rows {len(db)}, "
              f"budget_dropped so far {ps.n_budget_dropped}")

    if not worker._kv_prune_state.n_budget_dropped:
        failures.append("the budget never dropped anything, so nothing was tested")
    return failures


def check_turn_granularity():
    """granularity='turn': whole turns drop as units; survivors' successors keep a header."""
    torch.manual_seed(0)
    worker = bare_worker("prune", window=None, budget=BUDGET, importance="attn",
                         recent_turns=RECENT, granularity="turn")
    failures = []

    next_id = 0
    turn_slots = {}
    for step, length in enumerate(BUDGET_TURNS):
        ids = torch.arange(next_id, next_id + length)
        turn_slots[step] = ids.tolist()
        next_id += length
        live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
        worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
        pos = torch.arange(worker.offset, worker.offset + length)
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += length
        all_ids = torch.tensor(worker.past_key_values.layers[0].ids())
        worker._reindex_state.score_sum = torch.sin(all_ids.float() * 12.9898) * 0.5 + 0.5
        worker._reindex_state.score_layers = 1
        worker._apply_kv_prune()

        kept = worker.past_key_values.layers[0].ids()
        kept_set = set(kept)
        n_turns = step + 1
        # every turn is either intact, absent, or exactly a trailing header stub
        stubs = 0
        state_by_turn = {}
        for t in range(n_turns):
            slots = turn_slots[t]
            body = [i for i in slots if i in kept_set and i >= PARITY_PREFIX] if t == 0 else \
                   [i for i in slots if i in kept_set]
            if t == 0:
                if [i for i in slots if i in kept_set and i < PARITY_PREFIX] != list(range(PARITY_PREFIX)):
                    failures.append(f"step {step}: prefix broken")
            full = [i for i in slots if i >= PARITY_PREFIX] if t == 0 else slots
            if body == full:
                state_by_turn[t] = "intact"
            elif not body:
                state_by_turn[t] = "absent"
            elif body == full[-PARITY_HEADER:]:
                state_by_turn[t] = "stub"
                stubs += 1
            else:
                failures.append(f"step {step}: turn {t} partially pruned in turn mode: {body}")
        # a stub is only legitimate right before a surviving turn
        for t, s in state_by_turn.items():
            if s == "stub" and t + 1 < n_turns and state_by_turn.get(t + 1) == "absent":
                failures.append(f"step {step}: stale header stub for turn {t} "
                                f"(successor absent)")
        # recent turns intact
        for t in range(max(0, n_turns - RECENT), n_turns):
            if state_by_turn.get(t) != "intact":
                failures.append(f"step {step}: recent turn {t} is {state_by_turn.get(t)}")
        # budget within one turn's slack + header stubs
        if len(kept) > BUDGET + max(BUDGET_TURNS) + stubs * PARITY_HEADER:
            failures.append(f"step {step}: {len(kept)} kept far exceeds budget {BUDGET}")
        print(f"  step {step}: {len(kept)} slots kept, turn states "
              f"{[state_by_turn[t] for t in range(n_turns)]}")

    if not worker._kv_prune_state.n_budget_dropped:
        failures.append("turn-granular budget never dropped anything")
    return failures


def check_sparse_history_window():
    """Pre-forward sparse history excludes turns due to leave the cache this step."""
    failures = []

    worker = VLMWorker.__new__(VLMWorker)
    worker.context_window = 2
    worker.context_window_mode = "evict"
    worker.use_sparse = True
    worker.past_image_embeds = [torch.tensor([[10.0], [20.0], [21.0]])]
    worker._abs_bounds = [10, 20, 30]
    worker._vis_counts = [[2], [1], [2]]
    worker._n_evicted = 1
    view = worker._windowed_sparse_history()
    if not torch.equal(view[0], torch.tensor([[20.0], [21.0]])):
        failures.append(f"evict sparse view kept out-of-window rows: {view[0].flatten().tolist()}")
    if not torch.equal(worker.past_image_embeds[0], torch.tensor([[10.0], [20.0], [21.0]])):
        failures.append("windowed sparse view mutated the canonical embedding DB")

    ps = KVPruneState()
    ps.n_turns = 3
    ps.turn_id = torch.tensor([1, 1, 2, 2, 2])
    ps.is_visual = torch.tensor([True, False, True, True, False])
    worker.context_window_mode = "prune"
    worker._kv_prune_state = ps
    prune_view = worker._windowed_sparse_history()
    if not torch.equal(prune_view[0], torch.tensor([[20.0], [21.0]])):
        failures.append(f"prune sparse view kept out-of-window rows: {prune_view[0].flatten().tolist()}")
    if not torch.equal(worker.past_image_embeds[0], torch.tensor([[10.0], [20.0], [21.0]])):
        failures.append("prune sparse view mutated the canonical embedding DB")
    return failures


def check_merge_math():
    """One dropped slot folds into its nearest kept K/V and embedding as a weighted average."""
    failures = []
    state = KVPruneState()
    state.turn_id = torch.zeros(4, dtype=torch.long)
    state.is_visual = torch.tensor([True, True, True, True])
    state.importance = torch.tensor([1.0, 3.0, 1.0, 1.0])
    keep = torch.tensor([True, False, True, True])
    budget_dropped = torch.tensor([False, True, False, False])
    # db: slot 1 is nearest to slot 2, but not identical, so the centroid must move.
    db = [torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]])]
    layer = FakeLayer(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    merge_dropped_visual([layer], keep, budget_dropped, state, db, torch.device("cpu"))

    got = layer.keys.reshape(-1).tolist()
    # slot 2 <- (1*30 + 3*20) / 4 = 22.5; others untouched
    expected = [10.0, 20.0, 22.5, 40.0]
    if not np.allclose(got, expected):
        failures.append(f"merged keys {got} != expected {expected}")
    if not np.allclose(layer.values.reshape(-1).tolist(), expected):
        failures.append("values did not follow the same merge")
    if abs(float(state.importance[2]) - 4.0) > 1e-6:
        failures.append(f"target importance {float(state.importance[2])} != 4.0 "
                        "(should absorb the dropped mass)")
    # DB row 2 <- normalize((1*[0, 1] + 3*[0.6, 0.8]) / 4).
    expected_center = torch.nn.functional.normalize(torch.tensor([0.45, 0.85]), dim=0)
    if not torch.allclose(db[0][2], expected_center, atol=1e-6):
        failures.append(f"merged embedding {db[0][2].tolist()} != normalized centroid "
                        f"{expected_center.tolist()}")
    if not torch.equal(db[0][[0, 1, 3]],
                       torch.tensor([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])):
        failures.append("embedding merge changed non-target DB rows")
    return failures


# --- voxel tags: per-slot world voxel ids from the sim's patch grid ------------------

VOX_GH, VOX_GW, VOX_TEXT = 3, 4, 2
VOX_TURNS = 5
VOX_BUDGET = 30


def _rope_table(base, gh, gw, keep):
    """get_rope_index-style (3, 1, n) positions for one image's kept tokens."""
    idx = torch.nonzero(keep).flatten()
    r, c = idx // gw, idx % gw
    return torch.stack([torch.full_like(r, base), base + r, base + c]).reshape(3, 1, -1)


def feed_image_turn(worker, next_id, keep, patch_coords):
    """Append text + image(keep) + text with a real mRoPE table and a voxel grid."""
    n_vis = int(keep.sum())
    n_new = VOX_TEXT + n_vis + VOX_TEXT
    ids = torch.arange(next_id, next_id + n_new)
    vis_row = torch.zeros(n_new, dtype=torch.bool)
    vis_row[VOX_TEXT:VOX_TEXT + n_vis] = True
    worker.language_model.visual_pos_masks = vis_row.reshape(1, -1)
    worker.language_model.vis_keep_mask = keep
    worker.cumulative_inputs = {"image_grid_thw": torch.tensor([[1, 2 * VOX_GH, 2 * VOX_GW]])}
    worker._turn_patch_coords = patch_coords
    db_new = ids[vis_row].float().reshape(-1, 1)
    if worker.past_image_embeds is None:
        worker.past_image_embeds = [db_new]
    else:
        worker.past_image_embeds[0] = torch.cat([worker.past_image_embeds[0], db_new])
    live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
    worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
    off = worker.offset
    text_a = torch.arange(off, off + VOX_TEXT)
    base = off + VOX_TEXT
    text_b = torch.arange(base + max(VOX_GH, VOX_GW), base + max(VOX_GH, VOX_GW) + VOX_TEXT)
    table = torch.cat([text_a.reshape(1, 1, -1).expand(3, 1, -1),
                       _rope_table(base, VOX_GH, VOX_GW, keep),
                       text_b.reshape(1, 1, -1).expand(3, 1, -1)], dim=-1)
    worker._reindex_state.append(table.clone())
    worker.offset = int(text_b[-1]) + 1
    worker._apply_kv_prune()
    return ids, vis_row


def check_voxel_bookkeeping():
    """Voxel tags stay 1:1 with the cache through real prunes, and misuse fails loudly."""
    from longnav.utils.voxel_utils import VOXEL_NONE
    torch.manual_seed(0)
    worker = bare_worker("prune", window=None, budget=VOX_BUDGET, importance="voxel_dedup",
                         recent_turns=RECENT)
    worker.use_sparse = True
    worker.language_model = SimpleNamespace(visual_pos_masks=None, vis_keep_mask=None)
    worker.past_image_embeds = None
    failures = []
    rng = np.random.RandomState(0)
    vox_by_id = {}
    next_id = 0
    for step in range(VOX_TURNS):
        keep = torch.rand(VOX_GH * VOX_GW) < 0.7
        keep[-1] = True
        # A random walk so cells repeat across frames; one patch without depth per frame.
        pc = rng.randint(step, step + 3, size=(VOX_GH, VOX_GW, 3)).astype(np.int32)
        pc[0, 0] = VOXEL_NONE
        ids, vis_row = feed_image_turn(worker, next_id, keep, pc)
        flat = pc.reshape(-1, 3)[torch.nonzero(keep).flatten().numpy()]
        for tok, v in zip(ids[vis_row].tolist(), flat.tolist()):
            vox_by_id[tok] = v
        for tok in ids[~vis_row].tolist():
            vox_by_id[tok] = [VOXEL_NONE] * 3
        next_id += len(ids)

        ps, st = worker._kv_prune_state, worker._reindex_state
        kept = worker.past_key_values.layers[0].ids()
        if not (len(kept) == st.pos_table.shape[-1] == ps.turn_id.numel() == ps.voxel.shape[0]):
            failures.append(f"step {step}: cache/table/metadata/voxel lengths desynced")
        expected = torch.tensor([vox_by_id[i] for i in kept], dtype=torch.long)
        if not torch.equal(ps.voxel, expected):
            failures.append(f"step {step}: voxel tags drifted from the survivors")
        if len(kept) > max(VOX_BUDGET, PARITY_PREFIX):
            failures.append(f"step {step}: {len(kept)} slots over budget {VOX_BUDGET}")
        print(f"  step {step}: {len(kept)} slots kept, budget_dropped so far {ps.n_budget_dropped}")
    if not worker._kv_prune_state.n_budget_dropped:
        failures.append("the budget never dropped anything, so nothing was tested")

    # misuse: wrong grid shape, and no coords at all under a voxel selector
    for bad, what in ((np.zeros((VOX_GH + 1, VOX_GW, 3), dtype=np.int32), "wrong-shaped"),
                      (None, "missing")):
        probe = bare_worker("prune", window=None, budget=VOX_BUDGET, importance="voxel_dedup",
                            recent_turns=RECENT)
        probe.use_sparse = True
        probe.language_model = SimpleNamespace(visual_pos_masks=None, vis_keep_mask=None)
        probe.past_image_embeds = None
        try:
            feed_image_turn(probe, 0, torch.ones(VOX_GH * VOX_GW, dtype=torch.bool), bad)
        except RuntimeError as exc:
            if what == "missing" and "needs per-turn patch coords" not in str(exc):
                failures.append(f"missing coords raised an unhelpful error: {exc}")
        else:
            failures.append(f"{what} patch coords did not raise")
    return failures


# --- checks 4-5: real model, GPU ------------------------------------------------------

def check_fidelity(images):
    """With a budget nothing exceeds, prune (capture armed) must not move the decision."""
    plain = VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True)
    baseline = run_probs(plain, images)
    release(plain)

    worker = build_prune(WIDE_BUDGET)
    pruned = run_probs(worker, images)
    ps = worker._kv_prune_state
    table_len = worker._reindex_state.pos_table.shape[-1]
    kv_len = worker.past_key_values.get_seq_length()
    n_dropped = ps.n_budget_dropped
    release(worker)

    failures = []
    if n_dropped:
        failures.append(f"an infinite budget dropped {n_dropped} slots")
    if table_len != kv_len:
        failures.append(f"position table ({table_len}) desynced from cache ({kv_len})")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, pruned)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_stock - p_prune| = {delta:.5f}")
    floor = max(deltas)
    if floor > FIDELITY_TOL:
        failures.append(f"with nothing pruned the capture/pre-rotation path moved the "
                        f"action distribution by {floor:.5f} (> {FIDELITY_TOL})")
    return failures, floor, baseline


def check_divergence(images, floor, baseline, merge):
    tag = "prune+merge" if merge else "prune"
    worker = build_prune(NARROW_BUDGET, merge=merge)
    probs = run_probs(worker, images)
    ps = worker._kv_prune_state
    kv_len = worker.past_key_values.get_seq_length()
    table_len = worker._reindex_state.pos_table.shape[-1]
    db_rows = worker.past_image_embeds[0].shape[0]
    n_vis = int(ps.is_visual.sum())
    release(worker)

    failures = []
    if not ps.n_budget_dropped:
        failures.append(f"[{tag}] budget {NARROW_BUDGET} never pruned; raise N_STEPS")
    if kv_len > NARROW_BUDGET:
        failures.append(f"[{tag}] cache holds {kv_len} slots over budget {NARROW_BUDGET}")
    if table_len != kv_len:
        failures.append(f"[{tag}] position table ({table_len}) desynced from cache ({kv_len})")
    if db_rows != n_vis:
        failures.append(f"[{tag}] embed db ({db_rows}) desynced from visual slots ({n_vis})")
    if any(not np.all(np.isfinite(p)) for p in probs):
        failures.append(f"[{tag}] non-finite action probabilities")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, probs)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_stock - p_{tag}| = {delta:.5f}")
    signal = max(deltas)
    print(f"  [{tag}] signal={signal:.5f} vs floor={floor:.5f} "
          f"({signal / max(floor, 1e-9):.1f}x), final kv_len={kv_len}, "
          f"budget_dropped={ps.n_budget_dropped}")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"[{tag}] differs from stock by only {signal:.5f}, within "
                        f"{DIVERGENCE_MARGIN}x of the {floor:.5f} floor -- pruning is inert")
    return failures


def run_probs_voxel(worker, images):
    """run_probs with a synthetic world-voxel grid attached to every turn (random walk)."""
    from longnav.utils.voxel_utils import VOXEL_NONE
    rng = np.random.RandomState(1)
    out = []
    for idx, image in enumerate(images):
        thw = worker.processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        merge = worker.processor.image_processor.merge_size
        gh, gw = int(thw[1]) // merge, int(thw[2]) // merge
        pc = rng.randint(idx, idx + 3, size=(gh, gw, 3)).astype(np.int32)
        pc[0, :2] = VOXEL_NONE
        probs, _, _ = worker.infer_probs(START if idx == 0 else TURN, [image],
                                         pos_id_kwargs={"mode": "standard", "patch_coords": pc})
        out.append(np.asarray(probs))
    return out


def check_voxel_gpu(images, floor, baseline):
    """voxel_dedup on the real model: inert under a wide budget, active under a narrow one."""
    failures = []
    worker = build_prune(WIDE_BUDGET, importance="voxel_dedup")
    probs = run_probs_voxel(worker, images)
    ps = worker._kv_prune_state
    kv_len = worker.past_key_values.get_seq_length()
    table_len = worker._reindex_state.pos_table.shape[-1]
    vox_len = ps.voxel.shape[0]
    n_tagged = int((ps.voxel[:, 0] != -(2 ** 31))[ps.is_visual].sum())
    n_dropped = ps.n_budget_dropped
    release(worker)
    if n_dropped:
        failures.append(f"[voxel wide] an infinite budget dropped {n_dropped} slots")
    if not (kv_len == table_len == vox_len):
        failures.append(f"[voxel wide] cache {kv_len} / table {table_len} / voxel {vox_len} desynced")
    if n_tagged == 0:
        failures.append("[voxel wide] no visual slot received a voxel tag")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, probs)]
    print(f"  [voxel wide] max |p_stock - p_voxel| = {max(deltas):.5f}, "
          f"{n_tagged}/{int(ps.is_visual.sum())} visual slots tagged")
    if max(deltas) > FIDELITY_TOL:
        failures.append(f"[voxel wide] tagging moved the decision by {max(deltas):.5f}")

    worker = build_prune(NARROW_BUDGET, importance="voxel_dedup")
    probs = run_probs_voxel(worker, images)
    ps = worker._kv_prune_state
    kv_len = worker.past_key_values.get_seq_length()
    table_len = worker._reindex_state.pos_table.shape[-1]
    vox_len = ps.voxel.shape[0]
    db_rows = worker.past_image_embeds[0].shape[0]
    n_vis = int(ps.is_visual.sum())
    release(worker)
    if not ps.n_budget_dropped:
        failures.append(f"[voxel narrow] budget {NARROW_BUDGET} never pruned")
    if kv_len > NARROW_BUDGET:
        failures.append(f"[voxel narrow] {kv_len} slots over budget {NARROW_BUDGET}")
    if not (kv_len == table_len == vox_len) or db_rows != n_vis:
        failures.append(f"[voxel narrow] cache {kv_len} / table {table_len} / voxel {vox_len} "
                        f"/ db {db_rows} vs visual {n_vis} desynced")
    signal = max(float(np.abs(a - b).max()) for a, b in zip(baseline, probs))
    print(f"  [voxel narrow] signal={signal:.5f} vs floor={floor:.5f}, final kv_len={kv_len}, "
          f"budget_dropped={ps.n_budget_dropped}")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"[voxel narrow] differs from stock by only {signal:.5f} -- inert")
    return failures


def main():
    failures = []

    print(f"=== forced-schedule parity: window={PARITY_WINDOW}, budget=inf (no model) ===")
    failures += check_forced_schedule_parity()

    print(f"=== budget bookkeeping: budget={BUDGET}, recent={RECENT} (no model) ===")
    failures += check_budget_bookkeeping()

    print(f"=== turn granularity: budget={BUDGET}, recent={RECENT} (no model) ===")
    failures += check_turn_granularity()

    print("=== sparse history window alignment (no model) ===")
    failures += check_sparse_history_window()

    print("=== merge math (no model) ===")
    failures += check_merge_math()

    print(f"=== voxel bookkeeping: budget={VOX_BUDGET}, {VOX_TURNS} image turns (no model) ===")
    failures += check_voxel_bookkeeping()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    images = make_images(N_STEPS)
    print(f"=== fidelity: budget={WIDE_BUDGET} (never prunes), {N_STEPS} steps ===")
    fidelity_failures, floor, baseline = check_fidelity(images)
    failures += fidelity_failures

    print(f"=== divergence: budget={NARROW_BUDGET}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor, baseline, merge=False)

    print(f"=== divergence (merge arm): budget={NARROW_BUDGET}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor, baseline, merge=True)

    print(f"=== voxel_dedup on the model: budgets {WIDE_BUDGET} / {NARROW_BUDGET} ===")
    failures += check_voxel_gpu(images, floor, baseline)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"\nPASS: prune reproduces evict's window schedule, keeps cache/table/metadata/db "
          f"in sync under real prunes, the merge math is exact, the capture path is inert "
          f"(within {floor:.5f}), and an active budget both holds the cache at the budget "
          f"and moves the decision.")


if __name__ == "__main__":
    main()
