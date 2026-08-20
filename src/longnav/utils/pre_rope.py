"""StreamingLLM-style pre-rotation KV cache for Qwen3-VL (context_window_mode='reindex').

Stock Qwen3VLTextAttention rotates keys with their mRoPE phase BEFORE caching, so evicting
turns from the middle of the cache leaves a positional hole: relative distances straddling
the cut stay episode-sized no matter how small the window is. Following StreamingLLM
(arXiv:2309.17453), this module caches keys UNROTATED and applies RoPE at attention time
from a per-slot position table, which makes the survivors' phases a free variable the
worker renumbers to contiguous positions on eviction (VLMWorker._apply_context_window).

The per-slot table is also the substrate any future cache-pruning policy needs: prune
arbitrary slots, slice the table alongside, renumber however the experiment demands.
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

    pos_table: (3, 1, kv_len) long -- one (t, h, w) triple per cache slot, appended
        post-sparse-filter so it stays 1:1 with the cache (see TextMixin.forward).
    cos/sin: (1, kv_len, head_dim) -- rotation phases for the whole cache, computed once
        per forward at the text-model level; they are layer-independent, as in stock.
    """

    def __init__(self):
        self.reset()

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

    def append(self, position_ids):
        p = position_ids.detach()
        self.pos_table = p if self.pos_table is None else torch.cat([self.pos_table, p], dim=-1)


def _rotate(x, cos, sin):
    # Out-of-place on purpose: the key tensor handed in may be the cache's own storage.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def pre_rope_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                               past_key_values=None, cache_position=None, **kwargs):
    """Qwen3VLTextAttention.forward (transformers 4.57.6) with rotation moved to read time.

    Differences from stock: keys enter the cache unrotated; the full post-update key tensor
    is rotated with the ReindexState's whole-cache cos/sin; queries take the tail of that
    same table -- bit-identical to the chunk-local `position_embeddings`, which are ignored
    so the positions have a single source of truth.
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

    if key_states.shape[-2] != state.cos.shape[1]:
        raise RuntimeError(
            f"reindex position table covers {state.cos.shape[1]} slots but the cache holds "
            f"{key_states.shape[-2]}; a forward reached the cache without updating the table")
    key_states = _rotate(key_states, state.cos, state.sin)

    if getattr(state, "capture_scores", False):
        from longnav.utils.kv_prune import accumulate_decision_row
        accumulate_decision_row(state, query_states, key_states, self.scaling)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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


def install_pre_rope(text_model):
    """Bind the pre-rotation forward onto every decoder layer's attention.

    Returns the shared ReindexState. TextMixin.forward feeds it (table append + whole-cache
    cos/sin) whenever `_reindex_state` is present on the text model; instance binding
    bypasses the @deprecate_kwarg wrapper, which is fine -- every caller already passes
    the new `past_key_values` name.
    """
    state = ReindexState()
    text_model._reindex_state = state
    for layer in text_model.layers:
        layer.self_attn._reindex_state = state
        layer.self_attn.forward = types.MethodType(pre_rope_attention_forward, layer.self_attn)
    return state
