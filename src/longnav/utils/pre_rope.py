"""StreamingLLM-style pre-rotation KV cache for Qwen3-VL (context_window_mode='reindex').

Stock Qwen3VLTextAttention rotates keys with their mRoPE phase BEFORE caching, so evicting
turns from the middle of the cache leaves a positional hole: relative distances straddling
the cut stay episode-sized no matter how small the window is. Following StreamingLLM
(arXiv:2309.17453), this module caches keys UNROTATED and applies RoPE at attention time
from a per-slot position table, which makes the survivors' phases a free variable the
worker renumbers to contiguous positions on eviction (VLMWorker._apply_context_window).

The per-slot table is also the substrate any future cache-pruning policy needs: prune
arbitrary slots, slice the table alongside, renumber however the experiment demands.

Layer-selective pruning (context_window_mode='prune' with kv_prune_layer_start/_end): the
budget may apply to a contiguous range of decoder layers only, so layers no longer share one
cache length. The position table stays a single MASTER table (one row per slot held by any
layer); each pruned layer holds the subset flagged by `alive`, and the attention forward
gathers that layer's rotation phases and attention-mask columns through `alive_index`.
Unpruned layers keep every master slot and use the table as-is, bit-identical to uniform mode.
"""
import types

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    eager_attention_forward,
    rotate_half,
)


class ReindexState:
    """Per-episode mRoPE bookkeeping shared by every decoder layer's attention.

    pos_table: (3, 1, kv_len) long -- one (t, h, w) triple per MASTER cache slot, appended
        post-sparse-filter so it stays 1:1 with the cache (see TextMixin.forward).
    cos/sin: (1, kv_len, head_dim) -- rotation phases for the whole master cache, computed
        once per forward at the text-model level; they are layer-independent, as in stock.

    Layer-selective pruning (pruned_layers is a range, set once by `configure`):
    alive: (kv_len,) CPU bool -- master slots the pruned layers still hold. Unpruned layers
        hold every master slot. `alive_index` is its nonzero index on the table's device.
    score_count: (kv_len,) long -- per-slot count of layers folded into `score_sum`, used
        instead of the scalar `score_layers` because a slot absent from a pruned layer
        receives no contribution from it.
    replay_extra / replay_extra_layers: an additive (1, 1, 1, kv_len) attention term applied
        only on the given layers during a decision replay (leave-one-frame-out ablations,
        Fisher gates, the layer-influence diagnostic). The base replay bias travels on the
        model-level attention_mask so window-forced slots are hidden from every layer.
    """

    def __init__(self):
        self.pruned_layers = None  # range of layers the KV budget applies to; None = every layer
        self.score_layer_set = None  # range of layers folded into the attn score; None = every layer
        self.blind_layers = None  # layers on which no query may attend to a visual slot; None = off
        self.n_layers = None
        self.reset()

    # --- configuration (once per model) ------------------------------------------------

    def configure(self, pruned_layers=None, score_layers=None, n_layers=None, blind_layers=None):
        """Fix which layers are budget-pruned / scored / visual-blind. A pruned range covering
        every layer is the uniform mode and is stored as None so the fast paths stay
        byte-identical; `blind_layers` is kept as given (it is a mask, not a slot axis)."""
        if n_layers is not None:
            self.n_layers = int(n_layers)
        for name, rng in (("pruned_layers", pruned_layers), ("score_layers", score_layers),
                          ("blind_layers", blind_layers)):
            if rng is None:
                continue
            if not isinstance(rng, range) or rng.step != 1:
                raise ValueError(f"{name} must be a step-1 range, got {rng!r}")
            if rng.start < 0 or (self.n_layers is not None and rng.stop > self.n_layers):
                raise ValueError(f"{name}={rng!r} is outside the {self.n_layers} decoder layers")
        if pruned_layers is not None and self.n_layers is not None \
                and pruned_layers == range(0, self.n_layers):
            pruned_layers = None
        self.pruned_layers = pruned_layers
        if pruned_layers is None:
            score_layers = None
        elif score_layers is not None and self.n_layers is not None \
                and score_layers == range(0, self.n_layers):
            score_layers = None
        self.score_layer_set = score_layers
        self.blind_layers = blind_layers
        self.reset()

    @property
    def layer_selective(self):
        return self.pruned_layers is not None

    @property
    def no_pruned_layers(self):
        """Layer-selective with an empty range: the budget never applies (a control arm)."""
        return self.pruned_layers is not None and len(self.pruned_layers) == 0

    @property
    def all_layers_pruned(self):
        """No layer holds more than the pruned set, so the master cache compacts with it."""
        if self.pruned_layers is None:
            return True
        return self.n_layers is not None and len(self.pruned_layers) >= self.n_layers

    # --- per-episode state ---------------------------------------------------------------

    def reset(self):
        self.pos_table = None
        self.cos = None
        self.sin = None
        # Decision-row attention capture for context_window_mode='prune' (see
        # longnav.utils.kv_prune). Armed by the worker around the decision forward only;
        # plain reindex leaves capture_scores False and pays one getattr per layer.
        self.capture_scores = False
        self.score_sum = None  # (kv_len,) fp32 on device, summed over layers
        self.score_layers = 0
        self.score_count = None  # (kv_len,) long on device, layer-selective only
        self.alive = None
        self.alive_index = None
        self.visual = None  # (kv_len,) device bool per master slot; tracked only for blind_layers
        self.replay_extra = None
        self.replay_extra_layers = None
        self._table_cache = None

    def master_len(self):
        return 0 if self.pos_table is None else int(self.pos_table.shape[-1])

    def append(self, position_ids, visual_row=None):
        p = position_ids.detach()
        n_old = self.master_len()
        self.pos_table = p if self.pos_table is None else torch.cat([self.pos_table, p], dim=-1)
        if self.blind_layers is not None:
            n_new = p.shape[-1]
            if visual_row is None:
                row = torch.zeros(n_new, dtype=torch.bool, device=p.device)
            else:
                row = visual_row.detach().reshape(-1).bool().to(p.device)
                if row.numel() != n_new:
                    raise ValueError(f"visual row covers {row.numel()} slots for a {n_new}-token turn")
            self.visual = row if self.visual is None else torch.cat([self.visual, row])
        if self.layer_selective:
            # New slots enter every layer, so they are alive by construction.
            n_new = p.shape[-1]
            ones = torch.ones(n_new, dtype=torch.bool)
            new_idx = torch.arange(n_old, n_old + n_new, device=p.device)
            if self.alive is None:
                self.alive, self.alive_index = ones, new_idx
            else:
                self.alive = torch.cat([self.alive, ones])
                self.alive_index = torch.cat([self.alive_index, new_idx])

    def set_alive(self, alive):
        """Replace the pruned layers' membership mask (over the current master slots)."""
        alive = alive.bool().cpu()
        if alive.numel() != self.master_len():
            raise ValueError(f"alive mask covers {alive.numel()} slots but the master table "
                             f"holds {self.master_len()}")
        self.alive = alive
        self.alive_index = torch.nonzero(alive).flatten().to(self.pos_table.device)
        self._table_cache = None

    def slice_master(self, master_idx):
        """Keep only the master slots in `master_idx` (table, visual flags); phases go stale."""
        master_idx = master_idx.to(self.pos_table.device)
        self.pos_table = self.pos_table.index_select(-1, master_idx)
        if self.visual is not None:
            self.visual = self.visual.index_select(0, master_idx)
        self.invalidate_rotation()

    def invalidate_rotation(self):
        """Drop the cached cos/sin (and per-layer gathers) after the table changed."""
        self.cos = self.sin = None
        self._table_cache = None

    def arm_capture(self):
        self.capture_scores = True
        self.score_sum = None
        self.score_layers = 0
        self.score_count = None

    # --- per-layer views -----------------------------------------------------------------

    def is_pruned(self, layer_idx):
        return self.pruned_layers is not None and layer_idx in self.pruned_layers

    def scores_layer(self, layer_idx):
        return self.score_layer_set is None or layer_idx in self.score_layer_set

    def layer_index(self, layer_idx):
        """Master indices held by this layer, or None when it holds the whole master table."""
        if self.is_pruned(layer_idx):
            return self.alive_index
        return None

    def layer_len(self, layer_idx):
        idx = self.layer_index(layer_idx)
        return self.master_len() if idx is None else int(idx.numel())

    def layer_tables(self, layer_idx):
        """(cos, sin) restricted to the slots this layer holds; memoised per forward."""
        idx = self.layer_index(layer_idx)
        if idx is None:
            return self.cos, self.sin
        cache = self._table_cache
        if cache is None or cache[0] is not self.cos or cache[1] is not idx:
            cache = (self.cos, idx, self.cos.index_select(1, idx), self.sin.index_select(1, idx))
            self._table_cache = cache
        return cache[2], cache[3]

    def mean_scores(self):
        """(kv_len,) per-slot mean of the captured decision rows over the folded layers.
        Slots absent from every scored layer (budget-dropped from the pruned layers) score 0."""
        n = self.master_len()
        if self.score_sum is None:
            return torch.zeros(n, dtype=torch.float32)
        if self.score_count is not None:
            return self.score_sum / self.score_count.clamp_min(1).to(self.score_sum.dtype)
        return self.score_sum / max(self.score_layers, 1)

    # --- decision replay support ---------------------------------------------------------

    def snapshot(self):
        names = ("pos_table", "cos", "sin", "capture_scores", "score_sum", "score_layers",
                 "score_count", "alive", "alive_index", "visual", "replay_extra",
                 "replay_extra_layers", "_table_cache")
        return {name: getattr(self, name) for name in names}

    def restore(self, snap):
        for name, value in snap.items():
            setattr(self, name, value)

    def pop_last(self):
        """Drop the last master slot (the cached decision token) for a one-token replay.
        Out-of-place, so `restore` puts the originals back by reference."""
        self.pos_table = self.pos_table[..., :-1]
        if self.visual is not None:
            self.visual = self.visual[:-1]
        self.invalidate_rotation()
        self.capture_scores = False
        if self.alive is not None:
            if not bool(self.alive[-1]):
                raise RuntimeError("the decision slot is not alive in the pruned layers")
            self.alive = self.alive[:-1]
            self.alive_index = self.alive_index[:-1]


def _rotate(x, cos, sin):
    # Out-of-place on purpose: the key tensor handed in may be the cache's own storage.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def mask_for_layer(mask, idx_l, n_keys, q, master_len, attn_impl="sdpa"):
    """Resize a model-level 4-D attention mask to one layer's key length.

    transformers builds one causal mask from layer 0's cache length and hands it to every
    layer; sdpa and eager then silently truncate a too-wide mask from the right (which hides
    the newest keys and leaks intra-chunk future tokens) and a too-short one raises. Under
    layer-selective pruning the layers differ in length, so every layer's mask must have
    exactly `n_keys` columns:
      * width == n_keys: returned as-is (the same object -- uniform mode stays bit-identical);
      * width == master_len: the mask indexes master slots (a replay bias, or the causal mask
        when layer 0 is unpruned) -> gather this layer's columns through `idx_l`;
      * otherwise: a causal mask built from a shorter layer (layer 0 pruned, this one not) ->
        rebuild structurally as an all-visible past block plus the mask's own causal tail,
        after checking the past block really is all-visible.
    `None` stays None only when it is safe: q == 1 (nothing to mask), no past (n_keys == q, where
    sdpa's top-left is_causal coincides with the bottom-right alignment) or FlashAttention, whose
    mask-free path is bottom-right causal; otherwise sdpa's is_causal would be wrong.
    """
    if mask is None:
        if q > 1 and n_keys != q and attn_impl != "flash_attention_2":
            raise RuntimeError("a multi-token forward reached a decoder layer without a mask; "
                               f"{attn_impl} would apply a top-left causal mask over the cache")
        return None
    width = mask.shape[-1]
    if width == n_keys:
        return mask
    if width == master_len and idx_l is not None:
        out = mask.index_select(-1, idx_l.to(mask.device))
    else:
        if mask.shape[-2] != q:
            raise RuntimeError(f"mask has {mask.shape[-2]} query rows for a {q}-token chunk")
        past = mask[..., :width - q]
        if mask.dtype == torch.bool:
            visible = bool(past.all())
            fill = torch.ones(*mask.shape[:-1], n_keys - q, dtype=torch.bool, device=mask.device)
        else:
            visible = bool((past == 0).all())
            fill = torch.zeros(*mask.shape[:-1], n_keys - q, dtype=mask.dtype, device=mask.device)
        if not visible:
            raise RuntimeError(f"cannot resize a {width}-column mask to {n_keys} keys: its past "
                               f"block is not all-visible (master_len={master_len})")
        out = torch.cat([fill, mask[..., width - q:]], dim=-1)
    if out.shape[-1] != n_keys:
        raise RuntimeError(f"mask resize produced {out.shape[-1]} columns for {n_keys} keys")
    return out


def hide_columns(mask, hide, q, n_keys):
    """Forbid every query row from attending to the key columns flagged in `hide` (n_keys,).

    Works on the mask a layer is about to use: a bool sdpa mask (cleared), a float eager /
    replay mask (filled with the dtype minimum), or None (a fresh past-visible + causal-tail
    bool mask is built first, so sdpa's top-left is_causal is never relied on).
    """
    hide = hide.to(torch.bool).view(1, 1, 1, n_keys)
    if mask is None:
        past = torch.ones(1, 1, q, n_keys - q, dtype=torch.bool, device=hide.device)
        tail = torch.ones(q, q, dtype=torch.bool, device=hide.device).tril().view(1, 1, q, q)
        mask = torch.cat([past, tail], dim=-1)
    if mask.shape[-1] != n_keys:
        raise RuntimeError(f"cannot hide columns: mask has {mask.shape[-1]} columns for {n_keys} keys")
    if mask.dtype == torch.bool:
        return mask & ~hide
    return mask.masked_fill(hide, torch.finfo(mask.dtype).min)


def pre_rope_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                               past_key_values=None, cache_position=None, **kwargs):
    """Qwen3VLTextAttention.forward (transformers 4.57.6) with rotation moved to read time.

    Differences from stock: keys enter the cache unrotated; the full post-update key tensor
    is rotated with the ReindexState's whole-cache cos/sin; queries take the tail of that
    same table -- bit-identical to the chunk-local `position_embeddings`, which are ignored
    so the positions have a single source of truth. Under layer-selective pruning the
    cos/sin and the attention mask are restricted to the slots this layer holds.
    """
    state = self._reindex_state
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    seq_len = query_states.shape[-2]
    cos_q, sin_q = state.cos[:, -seq_len:], state.sin[:, -seq_len:]
    query_states = _rotate(query_states, cos_q, sin_q)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin_q, "cos": cos_q, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    layer_idx = self.layer_idx
    expected = state.layer_len(layer_idx)
    if key_states.shape[-2] != expected:
        raise RuntimeError(
            f"reindex position table covers {expected} slots for layer {layer_idx} but the "
            f"cache holds {key_states.shape[-2]}; a forward reached the cache without updating "
            "the table")
    cos_k, sin_k = state.layer_tables(layer_idx)
    key_states = _rotate(key_states, cos_k, sin_k)

    if getattr(state, "capture_scores", False) and state.scores_layer(layer_idx):
        from longnav.utils.kv_prune import accumulate_decision_row
        accumulate_decision_row(state, query_states, key_states, self.scaling, layer_idx)

    attn_impl = self.config._attn_implementation
    idx_l = state.layer_index(layer_idx)
    if state.layer_selective:
        attention_mask = mask_for_layer(attention_mask, idx_l, key_states.shape[-2], seq_len,
                                        state.master_len(), attn_impl)
    if state.replay_extra is not None and (
            state.replay_extra_layers is None or layer_idx in state.replay_extra_layers):
        extra = state.replay_extra if idx_l is None else state.replay_extra.index_select(-1, idx_l)
        attention_mask = extra if attention_mask is None else attention_mask + extra
    if state.blind_layers is not None and layer_idx in state.blind_layers:
        # Visual-blind layers: no query (text or visual, cached or current chunk) may read a
        # visual slot here -- the paper's "drop all visual tokens at layer L".
        visual = state.visual if idx_l is None else state.visual.index_select(0, idx_l)
        attention_mask = hide_columns(attention_mask, visual, seq_len, key_states.shape[-2])

    attention_interface = eager_attention_forward
    if attn_impl != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def install_pre_rope(text_model, pruned_layers=None, score_layers=None, blind_layers=None):
    """Bind the pre-rotation forward onto every decoder layer's attention.

    Returns the shared ReindexState. TextMixin.forward feeds it (table append + whole-cache
    cos/sin) whenever `_reindex_state` is present on the text model; instance binding
    bypasses the @deprecate_kwarg wrapper, which is fine -- every caller already passes
    the new `past_key_values` name. `pruned_layers` / `score_layers` select the decoder
    layers the KV budget and the attn importance score apply to (None = all); `blind_layers`
    are the layers on which visual slots are hidden from every query.
    """
    state = ReindexState()
    state.configure(pruned_layers=pruned_layers, score_layers=score_layers,
                    n_layers=len(text_model.layers), blind_layers=blind_layers)
    text_model._reindex_state = state
    for layer in text_model.layers:
        layer.self_attn._reindex_state = state
        layer.self_attn.forward = types.MethodType(pre_rope_attention_forward, layer.self_attn)
    return state
