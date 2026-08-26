"""HAMLET for longnav-r1: moment tokens + a memory module (arXiv 2510.00695).

HAMLET turns a single-frame VLA into a history-aware policy with two additions:

* **moment tokens** -- ``n_moment`` learnable tokens appended to every step's VLM
  input; their last-layer hidden states ``m'_t`` are compact per-step summaries;
* a **memory module** -- a shallow block-causal transformer over the stacked
  ``m'`` of the past steps whose output conditions the action expert.

longnav-r1 differs from the diffusion-head VLAs in the paper in two ways that
shape this port. The policy reads its action from one token's logits over a
growing KV cache, and RL training replays the whole episode as ONE causal
forward over the cached input embeds (``VLMWorker._pack_embeds``). Feeding the
memory back into the LM *input* at step t would depend on that same forward's
outputs at earlier steps -- a two-pass forward -- and would break the
rollout/``old_logprobs`` equivalence the PPO ratio relies on. So the memory is
fused *after* the LM, at the decision token's final hidden state, with a
zero-initialised projection::

    h      = last_hidden_state                       # post final RMSNorm
    m'_t   = h[moment positions]                     # (n_moment, H)
    delta  = HamletModule.fuse(h[-1], [m'_0..m'_t])  # memory transformer read-out
    logits = lm_head(h[-1] + delta)                  # delta == 0 at init

The same function runs on the incremental rollout chunk (read-out for the last
block only) and on the training replay (read-outs for every block at once);
per block the computation is identical, so rollout, ``old_logprobs``, and the
training forward stay commensurable.

Moment tokens are real placeholder ids spliced into ``input_ids`` right after
each frame's ``<|vision_end|>`` (ids ``len(tokenizer)+i``; Qwen3-VL's embedding
matrix has spare rows, so neither the tokenizer nor the tied embedding matrix
is touched). Their input embeddings live in this module and are swapped in by
an ``embed_tokens`` forward hook (rollout) or an out-of-place ``torch.where``
on the replayed embeds (training), so gradient reaches them.

Deliberate deviations from the paper, each a stated design decision:

* moment tokens persist in the KV cache and are attended by later tokens (the
  paper re-runs its frozen VLM per frame with ``use_cache=False``);
* the memory window defaults to the whole episode (the paper's T=4 blocks at
  a 16-step stride is sized for chunked manipulation, ObjectNav episodes run
  to 350 decisions);
* the decision token's hidden state joins the memory transformer as a read-out
  row that attends over the past moment blocks (the analogue of the paper's
  action expert attending over ``[h_t; m~'_t]``);
* no time-contrastive warm-start; moment tokens train end-to-end through the
  policy loss (supported by the official code, ``--hamlet-mode finetune``).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attribute name of the module on the base HF model. PEFT matches
# ``modules_to_save`` by module name, and VLMWrapper._freeze_vision_tower
# freezes anything called ``visual``/``vision_*``, so keep this plain.
HAMLET_MODULE_NAME = "hamlet"


# --------------------------------------------------------------------------- #
# token bookkeeping
# --------------------------------------------------------------------------- #
def moment_token_ids(tokenizer, n_moment: int, vocab_rows: int) -> list:
    """Placeholder ids for the moment tokens: the first ``n_moment`` unused
    embedding rows above the tokenizer's vocabulary."""
    base = len(tokenizer)
    ids = list(range(base, base + n_moment))
    if ids and ids[-1] >= vocab_rows:
        raise ValueError(
            f"need {n_moment} spare embedding rows above id {base} but the embedding "
            f"matrix only has {vocab_rows} rows"
        )
    return ids


def splice_moment_tokens(turn_inputs, moment_ids: Sequence[int], vision_end_id: int, prefix_len: int):
    """Insert the moment placeholder ids into one cropped turn chunk, in place.

    They go right after the last ``<|vision_end|>`` so, under causal attention,
    they see the frame and everything before it (the paper appends them at the
    tail of the frame's sequence). A chunk without an image gets them right
    before the trailing assistant header, keeping the invariant that every
    decision is preceded by exactly one moment block.
    """
    if not moment_ids:
        return turn_inputs
    ids = turn_inputs["input_ids"]
    seq = ids[0]
    hits = torch.nonzero(seq == vision_end_id, as_tuple=False)
    if hits.numel():
        at = int(hits[-1]) + 1
    else:
        at = int(seq.shape[0]) - int(prefix_len)
        if at < 0:
            raise ValueError("turn chunk shorter than the assistant header; cannot place moment tokens")
    ins = torch.tensor(list(moment_ids), dtype=ids.dtype, device=ids.device)[None]
    turn_inputs["input_ids"] = torch.cat([ids[:, :at], ins, ids[:, at:]], dim=1)
    mask = turn_inputs.get("attention_mask", None)
    if mask is not None:
        turn_inputs["attention_mask"] = torch.cat(
            [mask[:, :at], torch.ones_like(ins, dtype=mask.dtype), mask[:, at:]], dim=1
        )
    mm = turn_inputs.get("mm_token_type_ids", None)
    if mm is not None:
        turn_inputs["mm_token_type_ids"] = torch.cat(
            [mm[:, :at], torch.zeros_like(ins, dtype=mm.dtype), mm[:, at:]], dim=1
        )
    return turn_inputs


def moment_positions(input_ids: torch.Tensor, moment_ids: Sequence[int]) -> torch.Tensor:
    """``(T, n_moment)`` positions of the moment slots in a 1-D id sequence.

    Raises if the count is not a whole number of blocks or the slots are out
    of order -- a chunk that lost a placeholder would otherwise silently shift
    every later block.
    """
    seq = input_ids.reshape(-1)
    n = len(moment_ids)
    lo, hi = int(moment_ids[0]), int(moment_ids[-1])
    pos = torch.nonzero((seq >= lo) & (seq <= hi), as_tuple=False).reshape(-1)
    if pos.numel() % n:
        raise ValueError(f"found {pos.numel()} moment tokens, not a multiple of n_moment={n}")
    pos = pos.view(-1, n)
    slots = seq[pos] - lo
    expected = torch.arange(n, device=seq.device, dtype=slots.dtype).expand_as(slots)
    if not torch.equal(slots, expected):
        raise ValueError("moment token slots are out of order")
    return pos


def replace_moment_rows(embeds: torch.Tensor, input_ids: torch.Tensor, moment_ids: Sequence[int],
                        rows: torch.Tensor) -> torch.Tensor:
    """Out-of-place swap of the placeholder rows of ``embeds`` (``(B, S, H)``)
    for the learnable moment embeddings ``rows`` (``(n_moment, H)``).

    Out of place on purpose: under gradient checkpointing the replayed embeds
    are a leaf that requires grad, which forbids in-place writes.
    """
    lo = int(moment_ids[0])
    ids = input_ids.to(embeds.device)
    mask = (ids >= lo) & (ids <= lo + rows.shape[0] - 1)
    if not bool(mask.any()):
        return embeds
    slot = (ids - lo).clamp_(0, rows.shape[0] - 1)
    return torch.where(mask[..., None], rows.to(embeds.dtype)[slot], embeds)


def make_moment_embed_hook(get_rows: Callable[[], torch.Tensor], moment_ids: Sequence[int]):
    """Forward hook for ``embed_tokens`` that swaps in the moment embeddings
    whenever the looked-up ids contain placeholders. Only the incremental
    rollout path goes through ``embed_tokens``; training replays embeds."""

    def hook(module, inputs, output):
        input_ids = inputs[0]
        lo = int(moment_ids[0])
        if not bool(((input_ids >= lo) & (input_ids <= lo + len(moment_ids) - 1)).any()):
            return None
        return replace_moment_rows(output, input_ids, moment_ids, get_rows())

    return hook


# --------------------------------------------------------------------------- #
# memory transformer (port of HAMLET's gr00t/model/modules/memory.py)
# --------------------------------------------------------------------------- #
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(x.dtype)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """RoPE phases for integer positions (``(L,)``) -> ``cos, sin`` of ``(L, head_dim)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"d_mem={dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, attn_bias):
        L = x.shape[0]
        q = self.q(x).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k(x).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v(x).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        q = q * cos[None] + _rotate_half(q) * sin[None]
        k = k * cos[None] + _rotate_half(k) * sin[None]
        out = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=attn_bias[None, None])
        return self.o(out[0].transpose(0, 1).reshape(L, -1))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_mult: int):
        super().__init__()
        inner = ffn_mult * dim
        self.gate = nn.Linear(dim, inner, bias=False)
        self.up = nn.Linear(dim, inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult, rms_eps):
        super().__init__()
        self.attn_norm = _RMSNorm(dim, rms_eps)
        self.attn = _Attention(dim, n_heads)
        self.ffn_norm = _RMSNorm(dim, rms_eps)
        self.ffn = _SwiGLU(dim, ffn_mult)

    def forward(self, x, cos, sin, attn_bias):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_bias)
        return x + self.ffn(self.ffn_norm(x))


class MemoryTransformer(nn.Module):
    """LLaMA-style pre-norm transformer over a flat token sequence whose RoPE
    positions are *step indices* (all tokens of one step share a position) and
    whose attention pattern is an explicit boolean ``allowed`` matrix."""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 8, ffn_mult: int = 4,
                 rms_eps: float = 1e-5, init_range: float = 0.02, rope_theta: float = 10000.0):
        super().__init__()
        self.head_dim = dim // n_heads
        self.rope_theta = rope_theta
        self.blocks = nn.ModuleList([_Block(dim, n_heads, ffn_mult, rms_eps) for _ in range(n_layers)])
        self.final_norm = _RMSNorm(dim, rms_eps)
        self._init_range = init_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self._init_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, block_idx: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        cos, sin = _rope_cos_sin(block_idx, self.head_dim, self.rope_theta)
        attn_bias = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device)
        attn_bias.masked_fill_(~allowed, float("-inf"))
        for block in self.blocks:
            x = block(x, cos, sin, attn_bias)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# the HAMLET module attached to the policy
# --------------------------------------------------------------------------- #
class HamletModule(nn.Module):
    """Moment-token embeddings + memory transformer + zero-init read-out.

    Every entry point is a ``mode`` of :meth:`forward` so that calls made
    through PEFT's ``ModulesToSaveWrapper`` (which only proxies ``forward``)
    honour ``disable_adapter()``: the reference-policy pass then sees this
    module's untrained copy, whose zero-initialised ``out_proj`` makes it an
    exact no-op.
    """

    def __init__(self, hidden_size: int, n_moment: int = 4, d_mem: int = 512, n_layers: int = 2,
                 n_heads: int = 8, ffn_mult: int = 4, memory_window: Optional[int] = None,
                 max_blocks: int = 1024, init_range: float = 0.02):
        super().__init__()
        if n_moment < 1:
            raise ValueError("n_moment must be >= 1")
        self.hidden_size = hidden_size
        self.n_moment = n_moment
        self.d_mem = d_mem
        self.memory_window = memory_window
        self.max_blocks = max_blocks
        self.moment_embed = nn.Embedding(n_moment, hidden_size)
        self.in_proj = nn.Linear(hidden_size, d_mem)
        self.dec_proj = nn.Linear(hidden_size, d_mem)
        self.slot_embed = nn.Embedding(n_moment + 1, d_mem)  # n_moment moment slots + the read-out slot
        self.memory = MemoryTransformer(d_mem, n_layers, n_heads, ffn_mult, init_range=init_range)
        self.out_norm = _RMSNorm(d_mem)
        self.out_proj = nn.Linear(d_mem, hidden_size)
        for lin in (self.in_proj, self.dec_proj):
            nn.init.normal_(lin.weight, mean=0.0, std=init_range)
            nn.init.zeros_(lin.bias)
        nn.init.normal_(self.slot_embed.weight, mean=0.0, std=init_range)
        # Exact no-op at init: the policy starts as the checkpoint it was given,
        # and the reference policy (this module's untrained copy) stays one.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def init_moment_embeds(self, embed_weight: torch.Tensor, std: float = 0.02):
        """Start the moment tokens as 'neutral' tokens: the mean embedding row
        plus small noise (what ``resize_token_embeddings`` does for new tokens)."""
        mean = embed_weight.detach().float().mean(dim=0, keepdim=True)
        noise = torch.randn(self.n_moment, self.hidden_size, device=mean.device) * std
        self.moment_embed.weight.copy_((mean + noise).to(self.moment_embed.weight.dtype))

    # -- modes ------------------------------------------------------------- #
    def forward(self, mode: str = "fuse", **kwargs):
        if mode == "moment_embeds":
            return self.moment_embed.weight
        if mode == "fuse":
            return self.fuse(**kwargs)
        if mode == "out_proj_norm":
            return self.out_proj.weight.detach().float().norm()
        raise ValueError(f"unknown HamletModule mode {mode!r}")

    def fuse(self, h_dec: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        """Memory read-out to add to the decision hidden state(s).

        ``moments``: ``(T, n_moment, H)`` moment hidden states of blocks 0..T-1
        (oldest first). ``h_dec``: ``(T, H)`` -- one read-out per block (training
        replay) -- or ``(1, H)`` -- read-out for the last block only (rollout).
        Returns ``delta`` of the same leading shape as ``h_dec`` and its dtype.

        The read-out row of block t attends to the moment rows of blocks
        ``(t - memory_window, t]`` and to itself; moment rows attend to moment
        rows of blocks ``<= t`` (bidirectional inside a block); read-out rows
        are never keys. Per block this is the same computation whether one or
        all read-outs are requested.

        ``memory_window`` is a per-layer attention window shared by every row
        (moment rows of block t are restricted to ``(t - memory_window, t]``
        too), so the read-out's receptive field grows to about
        ``n_layers * memory_window`` blocks through the moment rows -- it is not
        the hard input cut of HAMLET's fixed-T window. ``None`` (default, whole
        episode) is unaffected.
        """
        if moments.dim() != 3 or moments.shape[1] != self.n_moment:
            raise ValueError(f"moments must be (T, {self.n_moment}, H), got {tuple(moments.shape)}")
        T, n, _ = moments.shape
        Tq = h_dec.shape[0]
        if Tq not in (T, 1):
            raise ValueError(f"h_dec has {Tq} rows but there are {T} moment blocks")
        if T > self.max_blocks:
            raise ValueError(f"episode has {T} blocks, above max_blocks={self.max_blocks}")
        device = moments.device
        ar_n = torch.arange(n, device=device)
        m = self.in_proj(moments.float()) + self.slot_embed(ar_n)[None]  # (T, n, d)
        r = self.dec_proj(h_dec.float()) + self.slot_embed(ar_n.new_full((1,), n))  # (Tq, d)
        x = torch.cat([m.reshape(T * n, self.d_mem), r], dim=0)
        m_blocks = torch.arange(T, device=device).repeat_interleave(n)
        r_blocks = torch.arange(T, device=device) if Tq == T else torch.full((1,), T - 1, device=device)
        block_idx = torch.cat([m_blocks, r_blocks])
        L = block_idx.shape[0]
        is_read = torch.zeros(L, dtype=torch.bool, device=device)
        is_read[T * n:] = True
        bi, bj = block_idx[:, None], block_idx[None, :]
        allowed = (bj <= bi) & ~is_read[None, :]
        if self.memory_window is not None:
            allowed &= bj > (bi - int(self.memory_window))
        allowed |= torch.eye(L, dtype=torch.bool, device=device)
        y = self.memory(x, block_idx, allowed)
        delta = self.out_proj(self.out_norm(y[T * n:]))  # (Tq, H)
        return delta.to(h_dec.dtype)


def build_hamlet_module(hidden_size: int, cfg: Dict) -> HamletModule:
    return HamletModule(
        hidden_size=hidden_size,
        n_moment=int(cfg.get("n_moment", 4)),
        d_mem=int(cfg.get("d_mem", 512)),
        n_layers=int(cfg.get("n_layers", 2)),
        n_heads=int(cfg.get("n_heads", 8)),
        ffn_mult=int(cfg.get("ffn_mult", 4)),
        memory_window=cfg.get("memory_window", None),
        max_blocks=int(cfg.get("max_blocks", 1024)),
    )


def attach_hamlet(model, cfg: Dict, tokenizer) -> list:
    """Create the HAMLET module on the *base* HF model (before PEFT wrapping so
    ``modules_to_save`` can pick it up) and return the moment placeholder ids."""
    text_cfg = model.config.text_config
    ids = moment_token_ids(tokenizer, int(cfg.get("n_moment", 4)), int(text_cfg.vocab_size))
    module = build_hamlet_module(int(text_cfg.hidden_size), cfg)
    embed_weight = model.get_input_embeddings().weight
    module.to(device=embed_weight.device)  # fp32 parameters on the model's device
    module.init_moment_embeds(embed_weight, std=float(cfg.get("moment_init_std", 0.02)))
    setattr(model, HAMLET_MODULE_NAME, module)
    return ids


# --------------------------------------------------------------------------- #
# the shared training / post-processing replay forward
# --------------------------------------------------------------------------- #
def forward_embeds_core(embeds_inputs: Dict, *, language_model, lm_head, dtype, training: bool,
                        hamlet=None, moment_ids: Optional[Sequence[int]] = None,
                        compute_values: bool = False, value_head=None,
                        value_grad_scale: Optional[float] = None):
    """Replay a packed episode (``VLMWorker._pack_embeds`` output) through the
    language model and read the action logits at the decision positions.

    This is the single implementation behind ``VLMWrapper._forward_embeds``
    (DDP training) and ``VLMTrainingMixin._forward_embeds`` (``old``/``ref``
    log-probs); the two used to be duplicated, and any drift between them
    silently biases the PPO ratio.

    Returns ``(logits, values, stats)`` with ``logits`` of shape ``(1, T, V)``.
    """
    device = lm_head.weight.device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in embeds_inputs.items()}
    logits_to_keep = inputs.pop("logits_to_keep")
    ids_ref = inputs.pop("input_ids_reference")
    emb = inputs["inputs_embeds"].to(dtype)
    if training:
        # gradient checkpointing needs an input that carries grad
        emb.requires_grad_(True)
    if hamlet is not None:
        emb = replace_moment_rows(emb, ids_ref, moment_ids, hamlet(mode="moment_embeds"))
    inputs["inputs_embeds"] = emb
    inputs["deepstack_visual_embeds"] = [v.to(dtype) for v in inputs["deepstack_visual_embeds"]]
    inputs["seq_keep_mask"] = "everything"  # the packed sequence is already sparse
    # No KV cache (nothing to reuse, and a DynamicCache over a whole episode is
    # pure memory) and an explicit all-ones attention mask. The mask is
    # load-bearing: when `create_causal_mask` gets neither a mask nor a cache --
    # exactly what gradient checkpointing forces in train mode (`use_cache=False`)
    # -- transformers >= 4.53 scans `position_ids` for packed-sequence boundaries.
    # mRoPE temporal positions are non-monotonic across image patches, so every
    # image became a separate "sequence", the mask went block-diagonal, and the
    # training log-probs disagreed with the eval-mode replay by >1 nat
    # (ppo_kl ~ 1 at the very first optimizer step).
    inputs["attention_mask"] = torch.ones((1, emb.shape[1]), dtype=torch.long, device=device)
    inputs["use_cache"] = False
    hidden = language_model(**inputs).last_hidden_state  # (1, S, H)
    h_dec = hidden[:, logits_to_keep]  # (1, T, H)
    stats = {}
    if hamlet is not None:
        pos = moment_positions(ids_ref[0], moment_ids).to(hidden.device)  # (T, n)
        if pos.shape[0] != h_dec.shape[1]:
            raise ValueError(f"{pos.shape[0]} moment blocks but {h_dec.shape[1]} decisions in the packed episode")
        moments = hidden[0, pos]  # (T, n, H)
        delta = hamlet(mode="fuse", h_dec=h_dec[0], moments=moments)  # (T, H)
        with torch.no_grad():  # ||delta|| / ||h_dec|| pre-fusion, same as the rollout's mean/hamlet_delta_ratio
            ratio = delta.float().norm(dim=-1) / h_dec[0].float().norm(dim=-1).clamp_min(1e-6)
            stats["hamlet_delta_ratio"] = float(ratio.mean())
        h_dec = h_dec + delta[None]
    values = None
    if compute_values:
        value_hidden = h_dec.to(value_head.dtype)
        if value_grad_scale is not None:
            if value_grad_scale <= 0:
                value_hidden = value_hidden.detach()
            else:
                # forward: identity; backward: gradient * scale
                value_hidden = value_hidden * value_grad_scale + value_hidden.detach() * (1 - value_grad_scale)
        values = value_head(value_hidden).squeeze(-1)
    logits = lm_head(h_dec)
    return logits, values, stats
