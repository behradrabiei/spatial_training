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

Optional merge arm (kv_prune_merge): budget-dropped visual slots are not discarded but
merged into their nearest kept visual slot (nearest by sparse-embed-DB cosine, which is
layer-agnostic), as an importance-weighted average of pre-rotation K and V -- CaM-style
cache clustering rather than pure eviction.

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

from longnav.utils.voxel_utils import VOXEL_NONE


class KVPruneState:
    """Per-episode slot metadata, 1:1 with the KV cache (CPU side).

    turn_id: (kv_len,) long -- which infer_step appended the slot. The prompt prefix is
        part of turn 0; it is protected positionally (slot < prefix_len), not by id.
    is_visual: (kv_len,) bool -- post-sparse-filter visual slots, in cache order. Visual
        slots are 1:1 with the rows of the sparse embed DB (same append order), which is
        what lets the DB be trimmed alongside the cache.
    importance: (kv_len,) fp32 -- EMA of the decision row's attention to each slot.
    voxel: (kv_len, 3) long or None -- world voxel id of the patch behind each visual slot
        (VOXEL_NONE for text slots and patches without usable depth); only populated when
        the sim attaches patch coords (see VLMWorker._turn_voxel_rows).
    """

    def __init__(self, seed=17):
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.turn_id = None
        self.is_visual = None
        self.importance = None
        self.voxel = None
        self.n_turns = 0
        self.n_budget_dropped = 0  # cumulative slots dropped by the budget (not the window)
        self.dropped_turns = set()  # turn-granularity only: turns whose body was dropped
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        # Side stream for selectors that need extra randomness (e.g. FPS starts) on top of
        # the random arm's draw. Keeping it separate leaves `generator` in lockstep with the
        # random arm, so per-frame quotas are identical between the two.
        self.aux_generator = torch.Generator(device="cpu")
        self.aux_generator.manual_seed(self.seed + 1_000_003)

    def append(self, n_new, visual_row, voxel_rows=None):
        assert n_new == visual_row.numel(), \
            f"turn added {n_new} cache slots but the visual mask covers {visual_row.numel()}"
        tid = torch.full((n_new,), self.n_turns, dtype=torch.long)
        vis = visual_row.bool().cpu()
        if voxel_rows is not None:
            voxel_rows = torch.as_tensor(voxel_rows, dtype=torch.long).cpu()
            assert tuple(voxel_rows.shape) == (n_new, 3), \
                f"voxel rows {tuple(voxel_rows.shape)} for a turn of {n_new} slots"
            if self.voxel is None and self.turn_id is not None:
                # Earlier turns carried no coords: backfill so the tensor stays 1:1.
                self.voxel = torch.full((self.turn_id.numel(), 3), VOXEL_NONE, dtype=torch.long)
        elif self.voxel is not None:
            voxel_rows = torch.full((n_new, 3), VOXEL_NONE, dtype=torch.long)
        if self.turn_id is None:
            self.turn_id, self.is_visual = tid, vis
        else:
            self.turn_id = torch.cat([self.turn_id, tid])
            self.is_visual = torch.cat([self.is_visual, vis])
        if voxel_rows is not None:
            self.voxel = voxel_rows if self.voxel is None else torch.cat([self.voxel, voxel_rows])
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
        if self.voxel is not None:
            self.voxel = self.voxel[keep]


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
                          recent_turns, candidate_scope="all", alive=None):
    """Apply the frame window, then split survivors into protected and candidates.

    `alive` (layer-selective pruning) marks the master slots the pruned layers still hold:
    slots already budget-dropped from them are neither protected nor candidates, and a
    protected slot must always be alive (prefix and recency protection are never revoked,
    so a slot cannot be dropped first and protected later). `budget=None` disables the
    budget -- every candidate is kept -- for the arm where no layer is pruned.
    """
    if candidate_scope not in ("all", "visual"):
        raise ValueError(f"candidate_scope must be 'all' or 'visual', got {candidate_scope!r}")
    if budget is not None and budget < 0:
        raise ValueError(f"kv budget must be nonnegative, got {budget}")

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
    if budget is None:
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


def _capped_equal_quotas(counts, capacity, generator=None):
    """Allocate nearly equal integer quotas, redistributing saturated shares."""
    counts = torch.as_tensor(counts, dtype=torch.long)
    quotas = torch.zeros_like(counts)
    remaining = min(int(capacity), int(counts.sum()))
    active = torch.nonzero(counts > 0).flatten()
    while remaining and active.numel():
        share, extra = divmod(remaining, active.numel())
        if share == 0:
            order = active
            if generator is not None and active.numel() > 1:
                order = active[torch.randperm(active.numel(), generator=generator)]
            quotas[order[:extra]] += 1
            break
        room = counts[active] - quotas[active]
        add = torch.minimum(room, torch.full_like(room, share))
        quotas[active] += add
        used = int(add.sum())
        remaining -= used
        active = active[quotas[active] < counts[active]]
        if used == 0:
            break
    return quotas


def capped_largest_remainder(scores, counts, capacity):
    """Proportional integer allocation with per-group caps and exact redistribution."""
    scores = torch.as_tensor(scores, dtype=torch.float64).clamp_min(0)
    counts = torch.as_tensor(counts, dtype=torch.long)
    target = min(int(capacity), int(counts.sum()))
    if target == 0:
        return torch.zeros_like(counts)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("quota scores must be finite")
    if float(scores.sum()) == 0.0:
        return _capped_equal_quotas(counts, target)

    quotas = torch.zeros_like(counts)
    remaining = target
    active = counts > 0
    while remaining and bool(active.any()):
        active_idx = torch.nonzero(active).flatten()
        weights = scores[active_idx]
        if float(weights.sum()) == 0.0:
            weights = torch.ones_like(weights)
        raw = weights / weights.sum() * remaining
        floor = torch.floor(raw).long()
        room = counts[active_idx] - quotas[active_idx]
        add = torch.minimum(floor, room)
        quotas[active_idx] += add
        remaining -= int(add.sum())
        active = quotas < counts
        if remaining == 0:
            break
        fractions = raw - floor
        ranked = active_idx[torch.argsort(fractions, descending=True, stable=True)]
        progressed = False
        for idx in ranked.tolist():
            if remaining == 0:
                break
            if quotas[idx] < counts[idx]:
                quotas[idx] += 1
                remaining -= 1
                progressed = True
        if not progressed and int(add.sum()) == 0:
            break
    return quotas


def random_select(candidate_idx, capacity, generator):
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    capacity = min(int(capacity), candidate_idx.numel())
    if capacity == 0:
        return candidate_idx[:0]
    return candidate_idx[torch.randperm(candidate_idx.numel(), generator=generator)[:capacity]]


def stratified_select(state, candidate_mask, capacity):
    """Seeded within-turn sampling with an approximately equal turn allocation."""
    turns = torch.unique(state.turn_id[candidate_mask], sorted=True)
    groups = [torch.nonzero(candidate_mask & (state.turn_id == t)).flatten() for t in turns]
    quotas = _capped_equal_quotas([g.numel() for g in groups], capacity, state.generator)
    selected = [random_select(group, int(quota), state.generator)
                for group, quota in zip(groups, quotas)]
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)


def proportional_group_select(state, candidate_mask, capacity, group_scores):
    """Allocate candidates across turns by score, then sample within each turn."""
    turns = torch.unique(state.turn_id[candidate_mask], sorted=True)
    groups = [torch.nonzero(candidate_mask & (state.turn_id == t)).flatten() for t in turns]
    scores = torch.tensor([float(group_scores.get(int(t), 0.0)) for t in turns])
    quotas = capped_largest_remainder(scores, [g.numel() for g in groups], capacity)
    selected = [random_select(group, int(quota), state.generator)
                for group, quota in zip(groups, quotas)]
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)


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


def fps_2d_batched(coord_groups, quotas, generator):
    """Greedy farthest-point subsets for several frames at once.

    coord_groups: list of (n_f, 2) grid coordinates; quotas: per-frame pick counts. Each
    frame gets a seeded random start (one draw per frame, in list order), then repeatedly
    the point farthest (Euclidean) from its chosen set, vectorised across frames; ties go
    to the lowest index. Frames whose quota covers every point are returned whole without
    consuming randomness. Returns per-frame row indices into the group's coordinates.
    """
    picks = [None] * len(coord_groups)
    active = []
    for f, (coords, quota) in enumerate(zip(coord_groups, quotas)):
        n, quota = int(coords.shape[0]), min(int(quota), int(coords.shape[0]))
        if quota <= 0:
            picks[f] = torch.empty(0, dtype=torch.long)
        elif quota == n:
            picks[f] = torch.arange(n)
        else:
            active.append((f, quota))
    if not active:
        return picks

    n_max = max(coord_groups[f].shape[0] for f, _ in active)
    pts = torch.zeros(len(active), n_max, 2)
    valid = torch.zeros(len(active), n_max, dtype=torch.bool)
    for i, (f, _) in enumerate(active):
        n = coord_groups[f].shape[0]
        pts[i, :n] = coord_groups[f].float()
        valid[i, :n] = True
    dist = torch.cdist(pts, pts)  # (F, n_max, n_max)
    dist.masked_fill_(~valid.unsqueeze(1), -torch.inf)  # padding is never the farthest point

    rows = torch.arange(len(active))
    start = torch.stack([
        torch.randint(coord_groups[f].shape[0], (1,), generator=generator)[0] for f, _ in active])
    chosen = [start]
    min_dist = dist[rows, start].clone()
    min_dist[rows, start] = -torch.inf
    for _ in range(max(q for _, q in active) - 1):
        pick = min_dist.argmax(dim=1)
        chosen.append(pick)
        min_dist = torch.minimum(min_dist, dist[rows, pick])
        min_dist[rows, pick] = -torch.inf
    chosen = torch.stack(chosen, dim=1)  # (F, max quota); exhausted frames repeat harmlessly
    for i, (f, quota) in enumerate(active):
        picks[f] = chosen[i, :quota]
    return picks


def grid_stratified_select(state, pos_table, candidate_mask, capacity, generator,
                           aux_generator):
    """Random's per-frame quotas, filled with spatially even picks on the token grid.

    The first step IS the random arm's draw on the same generator stream, so the per-frame
    visual quotas and the text picks are exactly what `random` would keep. Each frame's
    visual picks are then replaced by a farthest-point subset of that frame's visual
    candidates in (row, col) space: coverage within a frame becomes even instead of an
    i.i.d. sample, and nothing else changes. FPS starts draw from `aux_generator`.
    """
    candidate_idx = torch.nonzero(candidate_mask).flatten()
    draw = random_select(candidate_idx, capacity, generator)
    if draw.numel() == 0:
        return draw
    vis_draw = draw[state.is_visual[draw]]
    text_draw = draw[~state.is_visual[draw]]
    quotas = torch.bincount(state.turn_id[vis_draw], minlength=state.n_turns)
    coords = grid_coords_from_pos_table(pos_table)
    turns = torch.nonzero(quotas).flatten().tolist()
    pools = [torch.nonzero(candidate_mask & state.is_visual & (state.turn_id == t)).flatten()
             for t in turns]
    picks = fps_2d_batched([coords[pool] for pool in pools],
                           [int(quotas[t]) for t in turns], aux_generator)
    return torch.cat([text_draw] + [pool[pick] for pool, pick in zip(pools, picks)])


def visual_slot_embeddings(state, embed_db):
    """Return the sparse visual embedding DB after checking cache alignment."""
    if embed_db is None or len(embed_db) != 1:
        raise RuntimeError("diversity pruning requires one sparse visual embedding database")
    db = embed_db[0]
    n_visual = int(state.is_visual.sum())
    if db.shape[0] != n_visual:
        raise RuntimeError(f"embed db has {db.shape[0]} rows for {n_visual} visual slots")
    return db


def farthest_point_select(state, embed_db, candidate_idx, capacity, anchor_mask=None):
    """Cosine-distance FPS over normalized sparse embeddings, returning cache indices."""
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    capacity = min(int(capacity), candidate_idx.numel())
    if capacity == 0:
        return candidate_idx[:0]
    db = visual_slot_embeddings(state, embed_db)
    db_rank = torch.cumsum(state.is_visual.long(), 0) - 1
    device = db.device
    cand = F.normalize(db[db_rank[candidate_idx].to(db.device)].float(), dim=-1)

    anchors = None
    if anchor_mask is not None:
        anchor_idx = torch.nonzero(anchor_mask & state.is_visual).flatten()
        if anchor_idx.numel():
            anchors = F.normalize(db[db_rank[anchor_idx].to(db.device)].float(), dim=-1)
    if anchors is not None:
        min_dist = (1.0 - cand @ anchors.T).amin(dim=1)
    else:
        centroid = F.normalize(cand.mean(dim=0, keepdim=True), dim=-1)
        min_dist = (1.0 - cand @ centroid.T).squeeze(1)

    chosen = []
    available = torch.ones(candidate_idx.numel(), dtype=torch.bool, device=device)
    for _ in range(capacity):
        scores = min_dist.masked_fill(~available, -torch.inf)
        pick = int(scores.argmax())
        chosen.append(pick)
        available[pick] = False
        min_dist = torch.minimum(min_dist, 1.0 - cand @ cand[pick])
    return candidate_idx[torch.tensor(chosen, dtype=torch.long)]


def mixed_diversity_select(state, embed_db, candidate_idx, capacity, anchor_mask=None,
                            nonvisual_scores=None):
    """FPS visual slots while retaining a proportional share of nonvisual slots.

    Sparse embeddings exist only for visual slots. Split the final capacity between
    modalities in proportion to their candidate counts, use FPS for the visual share,
    and use seeded random selection (or supplied Fisher scores) for the text share.
    """
    visual, nonvisual, n_visual, n_nonvisual = split_modalities(state, candidate_idx, capacity)
    selected_visual = farthest_point_select(
        state, embed_db, visual, n_visual, anchor_mask=anchor_mask)
    if n_nonvisual == 0:
        selected_nonvisual = nonvisual[:0]
    elif nonvisual_scores is None:
        selected_nonvisual = random_select(nonvisual, n_nonvisual, state.generator)
    else:
        nonvisual_scores = torch.as_tensor(nonvisual_scores).cpu()
        selected_nonvisual = nonvisual[torch.argsort(
            nonvisual_scores[nonvisual], descending=True, stable=True)[:n_nonvisual]]
    return torch.cat([selected_visual, selected_nonvisual])

def split_modalities(state, candidate_idx, capacity):
    """(visual_idx, nonvisual_idx, n_visual, n_nonvisual): count-proportional quotas."""
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long).cpu()
    visual = candidate_idx[state.is_visual[candidate_idx]]
    nonvisual = candidate_idx[~state.is_visual[candidate_idx]]
    quotas = capped_largest_remainder(
        [float(visual.numel()), float(nonvisual.numel())],
        [visual.numel(), nonvisual.numel()], capacity)
    n_visual, n_nonvisual = (int(x) for x in quotas.tolist())
    return visual, nonvisual, n_visual, n_nonvisual


def voxel_cell_ids(state, idx, scale=1, two_d=True):
    """(n,) long: world-cell id of each slot in `idx`, -1 for text or no usable depth.

    Cells are the sim's voxel ids integer-divided by `scale` (0.15 m * scale per cell);
    `two_d` drops habitat's y (up) axis so floor and wall patches above one another share a
    cell.
    """
    idx = torch.as_tensor(idx, dtype=torch.long).cpu()
    cells = torch.full((idx.numel(),), -1, dtype=torch.long)
    if state.voxel is None or idx.numel() == 0:
        return cells
    v = state.voxel[idx]
    valid = state.is_visual[idx] & (v[:, 0] != VOXEL_NONE)
    if not bool(valid.any()):
        return cells
    key = v[valid][:, [0, 2]] if two_d else v[valid]
    key = torch.div(key, int(scale), rounding_mode="floor")
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    cells[valid] = inverse
    return cells


def _rank_newest_first(state, idx, cells):
    """Rank of each slot within its cell: 0 = newest turn (ties: highest slot), 1, ..."""
    idx = torch.as_tensor(idx, dtype=torch.long).cpu()
    if idx.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    recency = state.turn_id[idx] * (1 << 24) + idx  # larger = newer
    r_max = int(recency.max()) + 1
    key = (cells + 1) * r_max + (r_max - 1 - recency)  # cell asc, then newest first
    order = torch.argsort(key)
    sorted_cells = cells[order]
    first = torch.searchsorted(sorted_cells, sorted_cells)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(idx.numel()) - first
    return rank


def voxel_dedup_select(state, candidate_idx, capacity, cap=1, scale=1, two_d=True,
                       generator=None):
    """Cap visual slots per world cell (newest kept), then sample the survivors at random.

    The appearance filter dedupes patches whose embeddings match; this dedupes patches that
    look at the same piece of the world from different viewpoints or times. Slots without
    a cell are never deduped. Capped-out slots top up the visual quota only if the survivors
    alone cannot fill it, so the cache never falls below budget. Text candidates take a
    count-proportional share, sampled at random (zero under candidate_scope='visual').
    """
    visual, nonvisual, n_visual, n_nonvisual = split_modalities(state, candidate_idx, capacity)
    cells = voxel_cell_ids(state, visual, scale, two_d)
    rank = _rank_newest_first(state, visual, cells)
    survive = (cells < 0) | (rank < int(cap))
    keep_pool, drop_pool = visual[survive], visual[~survive]
    if keep_pool.numel() >= n_visual:
        selected_visual = random_select(keep_pool, n_visual, generator)
    else:
        selected_visual = torch.cat([
            keep_pool, random_select(drop_pool, n_visual - keep_pool.numel(), generator)])
    return torch.cat([selected_visual, random_select(nonvisual, n_nonvisual, generator)])


def voxel_stratified_select(state, candidate_idx, capacity, scale=1, two_d=True,
                            generator=None):
    """Equal-share visual quotas across occupied world cells, newest-first within each.

    Spreads the budget over the explored volume -- no cell gets a second slot before every
    cell has one -- i.e. the KV cache as a sparse map. Slots without a cell form singleton
    cells. Text candidates take a count-proportional share, sampled at random.
    """
    visual, nonvisual, n_visual, n_nonvisual = split_modalities(state, candidate_idx, capacity)
    cells = voxel_cell_ids(state, visual, scale, two_d).clone()
    n_cells = int(cells.max()) + 1 if cells.numel() else 0
    lone = torch.nonzero(cells < 0).flatten()
    cells[lone] = n_cells + torch.arange(lone.numel())
    n_cells += lone.numel()
    counts = torch.bincount(cells, minlength=n_cells)
    quotas = _capped_equal_quotas(counts, n_visual, generator)
    rank = _rank_newest_first(state, visual, cells)
    selected_visual = visual[rank < quotas[cells]]
    return torch.cat([selected_visual, random_select(nonvisual, n_nonvisual, generator)])


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
    """Add one layer's (kv_len_layer,) decision row to the running per-slot score."""
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


def slice_cache_layers(cache_layers, master_keep, keep, alive, pruned_layers):
    """Drop slots from every layer's K/V.

    Uniform pruning (alive is None): `keep` == `master_keep`, one gather for every layer.
    Layer-selective: layers in `pruned_layers` hold the master slots flagged `alive`, in
    master order, and are sliced by `keep[alive]` (positions in their own axis); the other
    layers hold the whole master cache and shrink only by `master_keep` (the window drop).
    Layers that lose nothing are left untouched.
    """
    master_idx = torch.nonzero(master_keep).flatten()
    pruned_idx = None if alive is None else torch.nonzero(keep[alive]).flatten()
    for layer_idx, layer in enumerate(cache_layers):
        pruned = pruned_idx is not None and pruned_layers is not None and layer_idx in pruned_layers
        idx = pruned_idx if pruned else master_idx
        if idx.numel() == layer.keys.shape[-2]:
            continue
        dev_idx = idx.to(layer.keys.device)
        layer.keys = layer.keys.index_select(-2, dev_idx)
        layer.values = layer.values.index_select(-2, dev_idx)


def build_keep_index(state, prefix_len, context_window, holdback_n, budget, recent_turns,
                     granularity="slot"):
    """One boolean keep mask over the current cache: forced window drop, then budget drop.

    Forced drop reproduces the evict schedule exactly in slot coordinates: every slot of a
    turn older than the window goes, except the prompt prefix and the boundary turn's
    trailing assistant header (its last `holdback_n` surviving slots -- the turn may have
    been partially budget-pruned already), so the retained context still opens on valid
    chat markup. A previous boundary's held-back header carries its own old turn_id, so it
    is retired by the next forced drop -- the held-back tokens do not accumulate, same as
    evict.

    Budget drop then removes the lowest-importance survivors until the cache fits, never
    touching the prefix or the last `recent_turns` turns. granularity='slot' drops
    individual slots; granularity='turn' drops whole turns (keyframe selection, ranked by
    mean slot importance), stopping at the last turn that fits -- frames stay intact at
    the cost of undershooting the budget by up to one turn. In turn mode, every dropped
    turn whose successor survives holds back its trailing `holdback_n` slots so the
    successor still opens on the assistant header (turn chunks start one token inside the
    previous reply); a held-back header whose successor is dropped later loses its
    protection and is retired then.

    Returns (keep, forced, first_keep): boolean masks over pre-prune slot coordinates and
    the first retained turn index (0 when no window drop applies).
    """
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
            boundary = torch.nonzero(state.turn_id == first_keep - 1).flatten()
            if boundary.numel():
                forced[boundary[-holdback_n:]] = False

    keep = ~forced
    n_over = int(keep.sum()) - budget
    if n_over > 0:
        cand = keep & ~prefix & ~recent
        if granularity == "turn":
            # A dropped turn's body never comes back, so clear its lingering header stub
            # first and re-derive which stubs are still needed from scratch below.
            for t in state.dropped_turns:
                keep &= ~(cand & (state.turn_id == t))
            n_over = int(keep.sum()) - budget
            pool = [t for t in torch.unique(state.turn_id[cand]).tolist()
                    if t not in state.dropped_turns]
            means = {t: float(state.importance[cand & (state.turn_id == t)].mean())
                     for t in pool}
            for t in sorted(pool, key=means.get):
                if n_over <= 0:
                    break
                t_mask = cand & (state.turn_id == t)
                keep[t_mask] = False
                state.dropped_turns.add(t)
                n_over -= int(t_mask.sum())
            # Re-open each surviving turn on its assistant header: the header lives at the
            # END of the previous turn's chunk, so a dropped turn whose successor is still
            # content keeps its trailing holdback_n slots. Turns outside the window are
            # force-dropped wholesale (the boundary holdback above covers that cut).
            for t in state.dropped_turns:
                succ_content = (t + 1 < state.n_turns) and (t + 1) not in state.dropped_turns
                if succ_content and t >= first_keep:
                    t_slots = torch.nonzero(state.turn_id == t).flatten()
                    keep[t_slots[-holdback_n:]] = True
        else:
            cand_idx = torch.nonzero(cand).flatten()
            n_drop = min(n_over, cand_idx.numel())  # budget below protected floor: best effort
            if n_drop > 0:
                drop = cand_idx[torch.argsort(state.importance[cand_idx])[:n_drop]]
                keep[drop] = False
    return keep, forced, first_keep


def merge_dropped_visual(cache_layers, keep, budget_dropped, state, embed_db, device, eps=1e-6):
    """Merge budget-dropped visual slots into their nearest kept visual slot.

    Nearest by cosine over the sparse embed DB (post-ViT hidden states, already
    L2-normalized, layer-agnostic -- one assignment shared by every layer). K and V are
    pre-rotation under this mode, so averaging across positions mixes content without
    mixing RoPE phase. The kept slot keeps its own position and absorbs the dropped
    slots' importance mass. The sparse embedding DB follows the same weighted merge and
    is re-normalized so later cosine-based deduplication and cluster assignments see the
    content represented by the merged K/V slot. Mutates the cache tensors, embedding DB,
    and state.importance in place; call BEFORE slicing with `keep`.
    """
    drop = torch.nonzero(budget_dropped & state.is_visual).flatten()
    kept_v = torch.nonzero(keep & state.is_visual).flatten()
    if drop.numel() == 0 or kept_v.numel() == 0:
        return
    db_rank = torch.cumsum(state.is_visual.long(), dim=0) - 1  # slot -> embed DB row
    emb = embed_db[0].float().to(device)
    sim = emb[db_rank[drop]] @ emb[db_rank[kept_v]].T
    tgt = kept_v[sim.argmax(dim=1).cpu()]  # per dropped slot, the kept cache slot to merge into

    w = state.importance.clamp_min(eps).to(device)
    drop_dev, tgt_dev = drop.to(device), tgt.to(device)
    uniq_tgt = torch.unique(tgt_dev)
    den = w.clone()
    den.index_add_(0, tgt_dev, w[drop_dev])

    # Keep the layer-agnostic cluster centers in sync with the K/V clusters. DB rows are
    # indexed only by visual slots, so translate the cache-slot assignments through
    # db_rank before accumulating. Normalize after the weighted average because the DB is
    # consumed as cosine-normalized input by filter_embeds and by the next merge.
    drop_db = db_rank[drop].to(device)
    tgt_db = db_rank[tgt].to(device)
    visual_slots = torch.nonzero(state.is_visual).flatten().to(device)
    emb_w = w[visual_slots]
    emb_den = emb_w.clone()
    emb_den.index_add_(0, tgt_db, w[drop_dev])
    emb_num = emb * emb_w.unsqueeze(-1)
    emb_num.index_add_(0, tgt_db, emb.index_select(0, drop_db) * w[drop_dev].unsqueeze(-1))
    merged_emb = emb_num / emb_den.clamp_min(eps).unsqueeze(-1)
    merged_emb = torch.nn.functional.normalize(merged_emb, p=2, dim=-1, eps=eps)
    uniq_tgt_db = torch.unique(tgt_db)
    db = embed_db[0]
    db[uniq_tgt_db.to(db.device)] = merged_emb[uniq_tgt_db].to(device=db.device,
                                                                dtype=db.dtype)

    for layer in cache_layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr)  # (1, n_kv_heads, kv_len, head_dim)
            tf = t.float()
            num = tf * w.view(1, 1, -1, 1)
            num.index_add_(2, tgt_dev, tf.index_select(2, drop_dev) * w[drop_dev].view(1, 1, -1, 1))
            merged = num / den.view(1, 1, -1, 1)
            t[:, :, uniq_tgt, :] = merged[:, :, uniq_tgt, :].to(t.dtype)
    state.importance.index_add_(0, tgt.cpu(), state.importance[drop.cpu()])
