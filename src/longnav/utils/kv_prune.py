"""Importance-based KV-slot pruning under a hard token budget (context_window_mode='prune').

The context-window ablation showed the last ~32 frames carry nearly all the information
the agent uses, but a pure-recency window of 4 frames collapses (SR 0.88 -> 0.70). This
mode keeps the 32-frame *pool* while enforcing a win4-sized KV *budget*: every step, the
lowest-importance unprotected cache slots are dropped until the cache fits. Importance is
the decision token's attention to each slot (TOVA/H2O-style), captured for free inside the
pre-rotation attention forward and folded into a per-slot EMA across steps.

Built on the 'reindex' substrate (longnav.utils.pre_rope): keys are cached unrotated with
a per-slot mRoPE table, so arbitrary slots can be sliced out of the cache with the table
following along. Survivors keep their ORIGINAL positions -- the evict-vs-reindex parity
result showed positional holes are inert, and not renumbering sidesteps the question of
what a scalar shift means for a visual token's (t, h, w) triple.

Layer-selective pruning (kv_prune_layer_start / kv_prune_layer_end): the budget drop can be
confined to a range of decoder layers, following "When Token Pruning is Worse than Random"
(Wang et al., CVPR'26, arXiv 2512.07580), whose unit of ablation is "prune at layer L":
layers below L see every token, layers from L on only the retained set. Here the slot
metadata, position table and embed DB describe the MASTER cache (every slot held by any
layer, i.e. the window-only cache of the unpruned layers); `ReindexState.alive` marks the
master slots the pruned layers still hold, selectors draw candidates from `alive` only, and
`slice_cache_layers` applies the budget drop to the pruned layers and the window drop to all.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class KVPruneState:
    """Per-episode slot metadata, 1:1 with the KV cache (CPU side).

    turn_id: (kv_len,) long -- which infer_step appended the slot. The prompt prefix is
        part of turn 0; it is protected positionally (slot < prefix_len), not by id.
    is_visual: (kv_len,) bool -- post-sparse-filter visual slots, in cache order. Visual
        slots are 1:1 with the rows of the sparse embed DB (same append order), which is
        what lets the DB be trimmed alongside the cache.
    importance: (kv_len,) fp32 -- EMA of the decision row's attention to each slot.
    is_lead: (kv_len,) bool -- the leading slots of a turn's chunk (the previous action and
        its postfix) that a context_cut='frame' window retires together with the older turns.
    raster: (kv_len,) long -- for visual slots, the patch's index in its frame's merged
        token grid (row-major); -1 for text. Lets the retention log name surviving patches.
    visual_per_turn / raster_per_turn: per-turn ORIGINAL visual count and raster list at the
        time the turn entered (never sliced) -- the denominator of frame-relative budgets and
        the reference axis of the per-frame keep masks.
    """

    def __init__(self, seed=17):
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.turn_id = None
        self.is_visual = None
        self.importance = None
        self.is_lead = None
        self.raster = None
        self.visual_per_turn = []
        self.raster_per_turn = []
        self.n_turns = 0
        self.n_budget_dropped = 0  # cumulative slots dropped by the budget (not the window)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def append(self, n_new, visual_row, lead=0, raster=None):
        """Register one turn's `n_new` cache slots. `lead` = number of leading chunk slots
        that belong to the previous reply (context_cut='frame'); `raster` = the kept patches'
        grid indices in cache order (one per visual slot), or None for a raster of arange."""
        assert n_new == visual_row.numel(), \
            f"turn added {n_new} cache slots but the visual mask covers {visual_row.numel()}"
        tid = torch.full((n_new,), self.n_turns, dtype=torch.long)
        vis = visual_row.bool().cpu()
        n_vis = int(vis.sum())
        lead_row = torch.zeros(n_new, dtype=torch.bool)
        lead_row[:int(lead)] = True
        raster_row = torch.full((n_new,), -1, dtype=torch.long)
        raster = torch.arange(n_vis) if raster is None else torch.as_tensor(raster, dtype=torch.long).cpu()
        assert raster.numel() == n_vis, f"{raster.numel()} raster indices for {n_vis} visual slots"
        raster_row[vis] = raster
        self.visual_per_turn.append(n_vis)
        self.raster_per_turn.append(raster.tolist())
        if self.turn_id is None:
            self.turn_id, self.is_visual = tid, vis
            self.is_lead, self.raster = lead_row, raster_row
        else:
            self.turn_id = torch.cat([self.turn_id, tid])
            self.is_visual = torch.cat([self.is_visual, vis])
            self.is_lead = torch.cat([self.is_lead, lead_row])
            self.raster = torch.cat([self.raster, raster_row])
        self.n_turns += 1

    def fold_scores(self, scores, beta):
        """EMA the full-cache score row into per-slot importance; new slots take their
        first score as-is. `scores` covers the CURRENT cache (old + this turn's slots)."""
        scores = scores.float().cpu()
        assert scores.numel() == self.turn_id.numel(), \
            f"score row covers {scores.numel()} slots but metadata covers {self.turn_id.numel()}"
        if self.importance is None:
            self.importance = scores.clone()
            return
        n_old = self.importance.numel()
        self.importance = torch.cat([
            beta * self.importance + (1.0 - beta) * scores[:n_old],
            scores[n_old:],
        ])

    def slice(self, keep):
        self.turn_id = self.turn_id[keep]
        self.is_visual = self.is_visual[keep]
        self.importance = self.importance[keep]
        self.is_lead = self.is_lead[keep]
        self.raster = self.raster[keep]


@dataclass
class PrunePartition:
    """Disjoint masks used by every exact-budget pruning selector."""

    base_keep: torch.Tensor
    protected: torch.Tensor
    candidates: torch.Tensor
    forced: torch.Tensor
    first_keep: int
    capacity: int
    alive: torch.Tensor = None  # layer-selective only: master slots the pruned layers hold


def build_prune_partition(state, prefix_len, context_window, holdback_n, budget,
                          recent_turns, candidate_scope="all", alive=None, frame_cut=False,
                          capacity=None):
    """Apply the frame window, then split survivors into protected and candidates.

    `alive` (layer-selective pruning) marks the master slots the pruned layers still hold:
    slots already budget-dropped from them are neither protected nor candidates, and a
    protected slot must always be alive (prefix and recency protection are never revoked,
    so a slot cannot be dropped first and protected later). `budget=None` disables the
    budget -- every candidate is kept -- for the arm where no layer is pruned.

    `frame_cut` reproduces context_cut='frame': no held-back header, and the oldest
    retained turn's lead (previous action + postfix, `state.is_lead`) is retired with the
    older turns so the window opens on its '<|im_start|>user'. `capacity` is a
    candidate-relative budget (exactly that many candidates survive, however many slots are
    protected); it replaces the total-cache `budget`.
    """
    if candidate_scope not in ("all", "visual"):
        raise ValueError(f"candidate_scope must be 'all' or 'visual', got {candidate_scope!r}")
    if budget is not None and budget < 0:
        raise ValueError(f"kv budget must be nonnegative, got {budget}")
    if capacity is not None and budget is not None:
        raise ValueError("pass either a total-cache budget or a candidate capacity, not both")

    kv_len = state.turn_id.numel()
    slot_idx = torch.arange(kv_len)
    prefix = slot_idx < prefix_len
    recent = state.turn_id > (state.n_turns - 1 - recent_turns)
    forced = torch.zeros(kv_len, dtype=torch.bool)
    first_keep = 0
    if context_window is not None:
        first_keep = state.n_turns - context_window
        if first_keep > 0:
            forced = (state.turn_id < first_keep) & ~prefix
            if frame_cut:
                forced |= state.is_lead & (state.turn_id == first_keep)
            else:
                boundary = torch.nonzero(state.turn_id == first_keep - 1).flatten()
                if boundary.numel():
                    forced[boundary[-holdback_n:]] = False

    base_keep = ~forced
    protected = base_keep & (prefix | recent)
    if candidate_scope == "visual":
        protected |= base_keep & ~state.is_visual
    candidates = base_keep & ~protected
    if candidate_scope == "visual":
        candidates &= state.is_visual
    if alive is not None:
        alive = alive.bool().cpu()
        if alive.numel() != kv_len:
            raise ValueError(f"alive mask covers {alive.numel()} slots but metadata covers {kv_len}")
        stranded = protected & ~alive
        if bool(stranded.any()):
            raise RuntimeError(f"{int(stranded.sum())} protected slots are absent from the pruned "
                               "layers; protection must never be granted after a budget drop")
        candidates &= alive

    n_protected = int(protected.sum())
    if capacity is not None:
        if capacity < 0:
            raise ValueError(f"candidate capacity must be nonnegative, got {capacity}")
        capacity = min(int(capacity), int(candidates.sum()))
    elif budget is None:
        capacity = int(candidates.sum())
    else:
        capacity = int(budget) - n_protected
        if capacity < 0:
            n_visual = int((protected & state.is_visual).sum())
            n_text = n_protected - n_visual
            raise RuntimeError(
                "KV budget is below the protected cache floor: "
                f"budget={budget}, protected={n_protected} "
                f"(visual={n_visual}, nonvisual={n_text}), kv_len={kv_len}, "
                f"scope={candidate_scope!r}, recent_turns={recent_turns}."
            )
        capacity = min(capacity, int(candidates.sum()))
    return PrunePartition(base_keep, protected, candidates, forced, first_keep, capacity, alive)


def random_select(candidate_idx, capacity, generator):
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    capacity = min(int(capacity), candidate_idx.numel())
    if capacity == 0:
        return candidate_idx[:0]
    return candidate_idx[torch.randperm(candidate_idx.numel(), generator=generator)[:capacity]]


def visual_slot_embeddings(state, embed_db):
    """Return the sparse visual embedding DB after checking cache alignment."""
    if embed_db is None or len(embed_db) != 1:
        raise RuntimeError("diversity pruning requires one sparse visual embedding database")
    db = embed_db[0]
    n_visual = int(state.is_visual.sum())
    if db.shape[0] != n_visual:
        raise RuntimeError(f"embed db has {db.shape[0]} rows for {n_visual} visual slots")
    return db


def build_selected_keep(partition, selected):
    """Construct an exact keep mask from protected slots plus selected candidates."""
    keep = partition.protected.clone()
    selected = torch.as_tensor(selected, dtype=torch.long).cpu()
    if selected.numel():
        if not bool(partition.candidates[selected].all()):
            raise ValueError("selector returned a slot outside the candidate partition")
        keep[selected] = True
    expected = int(partition.protected.sum()) + partition.capacity
    if int(keep.sum()) != expected:
        raise RuntimeError(f"selector retained {int(keep.sum())} slots, expected {expected}")
    if partition.alive is not None and bool((keep & ~partition.alive).any()):
        raise RuntimeError("selector retained a slot the pruned layers no longer hold")
    return keep


def accumulate_decision_row(state, query_states, key_states, scaling, layer_idx=None):
    """Fold one layer's decision-row attention into state.score_sum.

    Called from pre_rope_attention_forward with the ROTATED q/k of the current forward.
    The chunk's last query row is always the decision position (infer_step crops every
    turn to end on the assistant header, whose next-token logits are the action). The row
    is a plain fp32 softmax over all kv slots -- the last row needs no causal mask -- and
    costs one (n_heads, kv_len) matvec per layer, so sdpa stays usable for the attention
    itself.

    Under layer-selective pruning the row covers only the slots this layer holds, so it is
    scattered into master coordinates through the layer's index and a per-slot fold count
    (state.score_count) replaces the scalar layer count.
    """
    q_last = query_states[:, :, -1, :].float()  # (1, n_q_heads, head_dim)
    k = key_states.float()                      # (1, n_kv_heads, kv_len, head_dim)
    n_kv = k.shape[1]
    n_rep = q_last.shape[1] // n_kv
    q_last = q_last.view(1, n_kv, n_rep, -1)
    logits = torch.einsum("bgrd,bgkd->bgrk", q_last, k) * scaling
    row = torch.softmax(logits, dim=-1).mean(dim=(1, 2))[0]  # (kv_len,) mean over heads
    fold_decision_row(state, row, layer_idx)


def fold_decision_row(state, row, layer_idx=None):
    """Add one layer's (kv_len_layer,) decision row to the running per-slot score. Layers in
    `state.row_layers` also keep their own row, scattered to master coordinates."""
    if layer_idx is not None and layer_idx in state.row_layers:
        idx = state.layer_index(layer_idx)
        if idx is None:
            state.layer_rows[layer_idx] = row
        else:
            full = torch.zeros(state.master_len(), dtype=row.dtype, device=row.device)
            full.index_copy_(0, idx, row)
            state.layer_rows[layer_idx] = full
    if not state.layer_selective:
        if state.score_sum is None:
            state.score_sum = row
        else:
            state.score_sum += row
        state.score_layers += 1
        return
    n = state.master_len()
    if state.score_sum is None:
        state.score_sum = torch.zeros(n, dtype=row.dtype, device=row.device)
        state.score_count = torch.zeros(n, dtype=torch.long, device=row.device)
    idx = state.layer_index(layer_idx)
    if idx is None:
        state.score_sum += row
        state.score_count += 1
    else:
        state.score_sum.index_add_(0, idx, row)
        state.score_count.index_add_(0, idx, torch.ones_like(idx))
    state.score_layers += 1


def slice_cache_layers(cache_layers, master_keep, keep, alive, pruned_layers, stage_keeps=None):
    """Drop slots from every layer's K/V.

    Uniform pruning (alive is None): `keep` == `master_keep`, one gather for every layer.
    Layer-selective: layers in `pruned_layers` hold the master slots flagged `alive`, in
    master order, and are sliced by `keep[alive]` (positions in their own axis); the other
    layers hold the whole master cache and shrink only by `master_keep` (the window drop).
    Staged: `stage_keeps` = [(layer range, keep_s, alive_s), ...] gives each stage its own
    pair, `keep`/`alive` are ignored. Layers that lose nothing are left untouched.
    """
    if stage_keeps is None:
        stage_keeps = [] if alive is None else [(pruned_layers, keep, alive)]
    master_idx = torch.nonzero(master_keep).flatten()
    stage_idx = [(rng, torch.nonzero(k[a]).flatten()) for rng, k, a in stage_keeps]
    for layer_idx, layer in enumerate(cache_layers):
        idx = master_idx
        for rng, pruned_idx in stage_idx:
            if rng is not None and layer_idx in rng:
                idx = pruned_idx
                break
        if idx.numel() == layer.keys.shape[-2]:
            continue
        dev_idx = idx.to(layer.keys.device)
        layer.keys = layer.keys.index_select(-2, dev_idx)
        layer.values = layer.values.index_select(-2, dev_idx)


# --- visual-token pruning methods (paper/token_prune) ------------------------------------
#
# Training-free selectors from the VLM token-pruning literature, ported to the incremental
# cache: each returns the exact `capacity` candidate slots to KEEP out of `candidate_idx`.
# They see only the alive pool (survivors of earlier steps plus the newest history frame),
# so a slot dropped once never returns -- the permanent-deletion protocol.


def frame_budget(state, first_keep, frames):
    """Candidate capacity worth `frames` history frames: frames x the mean ORIGINAL post-filter
    visual count of the window turns [first_keep, n_turns). None (no pruning) while the window
    holds at most `frames` turns, so the arm coincides with the plain window until then."""
    counts = state.visual_per_turn[first_keep:]
    if len(counts) <= frames:
        return None
    return int(round(frames * sum(counts) / len(counts)))


def pyramid_budgets(total, capacity, n_stages):
    """Nested stage budgets V0*lambda^s (s = 1..S) with lambda = (capacity / total)^(1/S), the
    last stage pinned to exactly `capacity` (PyramidDrop's geometric schedule)."""
    total = max(int(total), 1)
    capacity = min(int(capacity), total)
    if n_stages < 1:
        raise ValueError("need at least one stage")
    lam = (capacity / total) ** (1.0 / n_stages) if capacity > 0 else 0.0
    budgets = [int(round(total * lam ** s)) for s in range(1, n_stages)] + [capacity]
    for i in range(len(budgets) - 2, -1, -1):  # keep the schedule monotone after rounding
        budgets[i] = max(budgets[i], budgets[i + 1])
    return budgets


def model_budgets(total, capacity, widths, n_unpruned=0, tail=None, unpruned_total=None):
    """Nested stage budgets total*lambda^s (s = 1..S) whose LAYER-WEIGHTED MEAN over the whole model
    equals `capacity` (paper/hybrid_prune, kv_prune_budget_scope='model'): the constant-ratio
    PyramidDrop schedule with lambda solved for a whole-model budget instead of being fixed.
    `total` anchors the schedule: the slots the first stage draws from (kv_prune_schedule_anchor:
    the original count of the frames in scope, or the slots the cache holds now).
    `widths` = decoder layers per scheduled stage; `n_unpruned` layers hold `unpruned_total` slots
    (default `total`); `tail` = (layers, slots) of a last stage with its own fixed budget. lambda is clamped to
    [0, 1]: 1 when the model is under budget unpruned, 0 when the unpruned layers and the tail
    alone exceed it (the budget is then unreachable and the stages hold nothing)."""
    total = max(int(total), 1)
    tail_w, tail_n = (0, 0) if tail is None else (int(tail[0]), min(int(tail[1]), total))
    n_layers = int(n_unpruned) + sum(int(w) for w in widths) + tail_w
    target = int(capacity) * n_layers - int(n_unpruned) * int(total if unpruned_total is None else unpruned_total)

    def held(lam):
        # The tail is nested in the last scheduled stage: it holds min(its budget, that stage).
        return total * sum(int(w) * lam ** (s + 1) for s, w in enumerate(widths)) \
            + tail_w * min(tail_n, total * lam ** len(widths))

    if target <= 0:
        lam = 0.0
    elif held(1.0) <= target:
        lam = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            lam = 0.5 * (lo + hi)
            lo, hi = (lam, hi) if held(lam) < target else (lo, lam)
        lam = 0.5 * (lo + hi)
    budgets = [int(round(total * lam ** (s + 1))) for s in range(len(widths))]
    for i in range(len(budgets) - 2, -1, -1):  # keep the schedule monotone after rounding
        budgets[i] = max(budgets[i], budgets[i + 1])
    return budgets


def step_seed(seed, label, step):
    """Deterministic per-(episode, step) seed: reproducible regardless of the trajectory prefix."""
    import zlib

    return (int(seed) * 1_000_003 + int(step)) ^ zlib.crc32(str(label).encode())


def divprune_select(rows, candidate_idx, capacity, block=2048):
    """DivPrune (CVPR'25): greedy Max-Min Diversity under cosine distance on the tokens' LLM
    input embeddings. Init = the token whose nearest neighbour is farthest; then repeatedly add
    the token farthest from the selected set. `rows` (N, d) are the candidates' embeddings in
    candidate order; returns the kept `candidate_idx` entries. GPU-resident, no host sync
    inside the loop."""
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    n = candidate_idx.numel()
    capacity = min(int(capacity), n)
    if capacity == 0:
        return candidate_idx[:0]
    if capacity == n:
        return candidate_idx
    x = F.normalize(rows.float(), dim=-1)
    device = x.device
    nn_dist = torch.empty(n, device=device)
    for start in range(0, n, block):  # nearest-neighbour distance, blockwise to bound memory
        sim = x[start:start + block] @ x.T
        ar = torch.arange(start, min(start + block, n), device=device)
        sim[ar - start, ar] = -torch.inf
        nn_dist[start:start + block] = 1.0 - sim.amax(dim=1)
    chosen = torch.empty(capacity, dtype=torch.long, device=device)
    pick = nn_dist.argmax()
    chosen[0] = pick
    min_dist = 1.0 - x @ x[pick]
    for i in range(1, capacity):
        min_dist.index_fill_(0, pick, -torch.inf)
        pick = min_dist.argmax()
        chosen[i] = pick
        min_dist = torch.minimum(min_dist, 1.0 - x @ x[pick])
    return candidate_idx[chosen.cpu()]


def dart_select(key_norms, hidden, candidate_idx, capacity, n_pivots=8):
    """DART (arXiv 2502.11494): pick pivots by key norm, then RETAIN the tokens least similar
    to each pivot (duplication, not importance). `key_norms` (N,) = L1 norm of the candidates'
    keys at the layer below the cut (heads concatenated); `hidden` (N, d) = their hidden states
    entering the cut layer. Pivots in descending norm order each claim q = (capacity - P) // P
    of the remaining candidates with the lowest cosine similarity to that pivot, the remainder
    going one extra to the first pivots; pivots are kept. Returns kept `candidate_idx` entries."""
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    n = candidate_idx.numel()
    capacity = min(int(capacity), n)
    if capacity == 0:
        return candidate_idx[:0]
    if capacity == n:
        return candidate_idx
    device = hidden.device
    n_piv = max(1, min(int(n_pivots), capacity))
    pivots = torch.argsort(key_norms.float().to(device), descending=True, stable=True)[:n_piv]
    h = F.normalize(hidden.float(), dim=-1)
    available = torch.ones(n, dtype=torch.bool, device=device)
    available[pivots] = False
    quota, extra = divmod(capacity - n_piv, n_piv)
    kept = [pivots]
    for rank, pivot in enumerate(pivots):
        q = quota + (1 if rank < extra else 0)
        if q == 0:
            continue
        sim = (h @ h[pivot]).masked_fill(~available, torch.inf)
        picks = torch.topk(sim, q, largest=False).indices
        available[picks] = False
        kept.append(picks)
    kept = torch.cat(kept)
    assert kept.numel() == capacity, f"DART retained {kept.numel()} of {capacity}"
    return candidate_idx[kept.cpu()]


def pack_keep_bits(state, alive, turns):
    """Per-turn keep masks over each turn's ORIGINAL raster list, packed to one base64 string
    (turn order, then raster order), plus the surviving visual count per turn. `alive` is the
    (kv_len,) mask of the current metadata slots that the deepest pruned stage still holds."""
    import base64

    import numpy as np

    bits, kept = [], []
    for t in turns:
        live = state.raster[(state.turn_id == t) & state.is_visual & alive].numpy()
        row = np.isin(np.asarray(state.raster_per_turn[t], dtype=np.int64), live)
        bits.append(row)
        kept.append(int(row.sum()))
    packed = np.packbits(np.concatenate(bits)) if bits else np.zeros(0, dtype=np.uint8)
    return base64.b64encode(packed.tobytes()).decode("ascii"), kept


def grid_coords_from_pos_table(pos_table):
    """(kv_len, 2) long, CPU: each slot's (row, col) in its image's merged token grid.

    Qwen3-VL's get_rope_index gives an image token (t, h, w) = base + (0, row, col) with the
    same base for every token of that image, so row = h - t and col = w - t survive both
    the scalar chunk offset (_pos_id_fast) and prune's no-renumbering policy. Text slots
    have t == h == w and map to (0, 0). `h - min(h)` would be wrong whenever the sparse
    filter drops a frame's whole first row or column.
    """
    p = pos_table[:, 0].cpu()
    return torch.stack([p[1] - p[0], p[2] - p[0]], dim=1)
