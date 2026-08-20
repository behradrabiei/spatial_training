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
"""
import torch


class KVPruneState:
    """Per-episode slot metadata, 1:1 with the KV cache (CPU side).

    turn_id: (kv_len,) long -- which infer_step appended the slot. The prompt prefix is
        part of turn 0; it is protected positionally (slot < prefix_len), not by id.
    is_visual: (kv_len,) bool -- post-sparse-filter visual slots, in cache order. Visual
        slots are 1:1 with the rows of the sparse embed DB (same append order), which is
        what lets the DB be trimmed alongside the cache.
    importance: (kv_len,) fp32 -- EMA of the decision row's attention to each slot.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.turn_id = None
        self.is_visual = None
        self.importance = None
        self.n_turns = 0
        self.n_budget_dropped = 0  # cumulative slots dropped by the budget (not the window)
        self.dropped_turns = set()  # turn-granularity only: turns whose body was dropped

    def append(self, n_new, visual_row):
        assert n_new == visual_row.numel(), \
            f"turn added {n_new} cache slots but the visual mask covers {visual_row.numel()}"
        tid = torch.full((n_new,), self.n_turns, dtype=torch.long)
        vis = visual_row.bool().cpu()
        if self.turn_id is None:
            self.turn_id, self.is_visual = tid, vis
        else:
            self.turn_id = torch.cat([self.turn_id, tid])
            self.is_visual = torch.cat([self.is_visual, vis])
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


def accumulate_decision_row(state, query_states, key_states, scaling):
    """Fold one layer's decision-row attention into state.score_sum.

    Called from pre_rope_attention_forward with the ROTATED q/k of the current forward.
    The chunk's last query row is always the decision position (infer_step crops every
    turn to end on the assistant header, whose next-token logits are the action). The row
    is a plain fp32 softmax over all kv slots -- the last row needs no causal mask -- and
    costs one (n_heads, kv_len) matvec per layer, so sdpa stays usable for the attention
    itself.
    """
    q_last = query_states[:, :, -1, :].float()  # (1, n_q_heads, head_dim)
    k = key_states.float()                      # (1, n_kv_heads, kv_len, head_dim)
    n_kv = k.shape[1]
    n_rep = q_last.shape[1] // n_kv
    q_last = q_last.view(1, n_kv, n_rep, -1)
    logits = torch.einsum("bgrd,bgkd->bgrk", q_last, k) * scaling
    row = torch.softmax(logits, dim=-1).mean(dim=(1, 2))[0]  # (kv_len,) mean over heads
    if state.score_sum is None:
        state.score_sum = row
    else:
        state.score_sum += row
    state.score_layers += 1


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
    slots' importance mass. Mutates the cache tensors and state.importance in place;
    call BEFORE slicing with `keep`.
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
    for layer in cache_layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr)  # (1, n_kv_heads, kv_len, head_dim)
            tf = t.float()
            num = tf * w.view(1, 1, -1, 1)
            num.index_add_(2, tgt_dev, tf.index_select(2, drop_dev) * w[drop_dev].view(1, 1, -1, 1))
            merged = num / den.view(1, 1, -1, 1)
            t[:, :, uniq_tgt, :] = merged[:, :, uniq_tgt, :].to(t.dtype)
    state.importance.index_add_(0, tgt.cpu(), state.importance[drop.cpu()])
