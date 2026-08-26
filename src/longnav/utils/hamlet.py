"""HAMLET for longnav-r1: moment tokens + a memory module (arXiv 2510.00695),
with the memory read-out fed back into the LM *input* as a memory token.

HAMLET turns a single-frame VLA into a history-aware policy with two additions:

* **moment tokens** -- ``n_moment`` learnable tokens appended to every step's VLM
  input; their last-layer hidden states ``m'_t`` are compact per-step summaries;
* a **memory module** -- a shallow block-causal transformer over the stacked
  ``m'`` of the past steps whose output conditions the policy.

v2 (this file): the memory conditions the policy through the *context*. Every
turn carries ``n_mem`` **memory tokens** (default 1) whose input embedding is the
memory transformer's read-out over the moment blocks of all *previous* steps::

    turn t:  ... <|im_start|>user\n [MEM_t] <|vision_start|> frame_t <|vision_end|>
             [M_0 .. M_{n_moment-1}] <|im_end|>\n<|im_start|>assistant\n**   <- decision
    MEM_t    = mem_embed + out_proj(memory([m'_0 .. m'_{t-1}]))        (out_proj == 0 at init)
    m'_t     = last_hidden_state[moment positions of turn t]           (n_moment, H)

so the frame, this turn's moment tokens and the decision token all attend to
memory. (v1 added the read-out to the decision token's final hidden state
before ``lm_head`` -- post-LM fusion -- which is what the "delta" naming in
older run logs refers to.)

longnav-r1 replays the whole episode for training as ONE causal forward over the
cached input embeds (``VLMWorker._pack_embeds``). Feeding an output of that
forward (``m'_{<t}``) back into its own input at position t is inherently
sequential, so the replay does not recompute the moment hidden states it feeds
to the memory: the rollout stores every step's ``m'_t`` (``moment_history`` in
the packed episode) and the replay rebuilds ``MEM_t`` from those stored moments
with the *current* module parameters. Gradient therefore reaches every HAMLET
parameter and, through the memory token's input row, the LoRA; it does not flow
back through the moment hidden states themselves (they are constants of the
replay -- the moment *embeddings* still train through the LM's attention over
them). At the first optimizer step the parameters equal the rollout's, so the
replayed log-probs equal ``old_logprobs`` and the PPO ratio starts at 1; the
staleness of the approximation is logged as ``hamlet_moment_drift``.

Moment and memory tokens are real placeholder ids spliced into ``input_ids``
(ids ``len(tokenizer)+i``; Qwen3-VL's embedding matrix has spare rows, so
neither the tokenizer nor the tied embedding matrix is touched). Their input
embeddings live in this module and are swapped in by an ``embed_tokens``
forward hook (rollout) or an out-of-place swap on the replayed embeds
(training), so gradient reaches them.

Deliberate deviations from the paper, each a stated design decision:

* moment tokens persist in the KV cache and are attended by later tokens (the
  paper re-runs its frozen VLM per frame with ``use_cache=False``);
* the memory window defaults to the whole episode (the paper's T=4 blocks at
  a 16-step stride is sized for chunked manipulation, ObjectNav episodes run
  to 350 decisions);
* the read-out is a learned query row that attends over the past moment blocks
  and is written back into the LM input (the paper's action expert attends over
  ``[h_t; m~'_t]``);
* no time-contrastive warm-start; moment tokens train end-to-end through the
  policy loss (supported by the official code, ``--hamlet-mode finetune``).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

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
def moment_token_ids(tokenizer, n_ids: int, vocab_rows: int) -> list:
    """Placeholder ids: the first ``n_ids`` unused embedding rows above the
    tokenizer's vocabulary (moment ids first, then memory ids)."""
    base = len(tokenizer)
    ids = list(range(base, base + n_ids))
    if ids and ids[-1] >= vocab_rows:
        raise ValueError(
            f"need {n_ids} spare embedding rows above id {base} but the embedding "
            f"matrix only has {vocab_rows} rows"
        )
    return ids


def _insert_ids(turn_inputs, inserts: Sequence[Tuple[int, Sequence[int]]]):
    """Insert id blocks into one turn chunk at the given positions of the
    *original* sequence (positions may repeat; earlier entries land first).
    Extends ``attention_mask`` (ones) and ``mm_token_type_ids`` (zeros)."""
    ids = turn_inputs["input_ids"]
    mask = turn_inputs.get("attention_mask", None)
    mm = turn_inputs.get("mm_token_type_ids", None)
    pieces_ids, pieces_mask, pieces_mm = [], [], []
    cur = 0
    for at, block in sorted(inserts, key=lambda x: x[0]):
        if not block:
            continue
        ins = torch.tensor(list(block), dtype=ids.dtype, device=ids.device)[None]
        pieces_ids += [ids[:, cur:at], ins]
        if mask is not None:
            pieces_mask += [mask[:, cur:at], torch.ones_like(ins, dtype=mask.dtype)]
        if mm is not None:
            pieces_mm += [mm[:, cur:at], torch.zeros_like(ins, dtype=mm.dtype)]
        cur = at
    pieces_ids.append(ids[:, cur:])
    turn_inputs["input_ids"] = torch.cat(pieces_ids, dim=1)
    if mask is not None:
        pieces_mask.append(mask[:, cur:])
        turn_inputs["attention_mask"] = torch.cat(pieces_mask, dim=1)
    if mm is not None:
        pieces_mm.append(mm[:, cur:])
        turn_inputs["mm_token_type_ids"] = torch.cat(pieces_mm, dim=1)
    return turn_inputs


def splice_moment_tokens(turn_inputs, moment_ids: Sequence[int], vision_end_id: int, prefix_len: int,
                         mem_ids: Sequence[int] = (), vision_start_id: Optional[int] = None):
    """Insert the placeholder ids into one cropped turn chunk.

    Moment ids go right after the last ``<|vision_end|>`` so, under causal
    attention, they see the frame and everything before it (the paper appends
    them at the tail of the frame's sequence). Memory ids go right before the
    last ``<|vision_start|>`` so the frame, the moment tokens and the decision
    all attend to memory. A chunk without an image gets both blocks (memory
    first) right before the trailing assistant header, keeping the invariant
    that every decision is preceded by exactly one memory block and one moment
    block, in that order.
    """
    if not moment_ids and not mem_ids:
        return turn_inputs
    seq = turn_inputs["input_ids"][0]
    hits = torch.nonzero(seq == vision_end_id, as_tuple=False)
    inserts = []
    if hits.numel():
        moment_at = int(hits[-1]) + 1
        starts = torch.nonzero(seq == vision_start_id, as_tuple=False) if vision_start_id is not None else hits[:0]
        mem_at = int(starts[-1]) if starts.numel() else moment_at
    else:
        moment_at = int(seq.shape[0]) - int(prefix_len)
        if moment_at < 0:
            raise ValueError("turn chunk shorter than the assistant header; cannot place moment tokens")
        mem_at = moment_at
    inserts.append((mem_at, list(mem_ids)))
    inserts.append((moment_at, list(moment_ids)))
    return _insert_ids(turn_inputs, inserts)


def moment_positions(input_ids: torch.Tensor, block_ids: Sequence[int]) -> torch.Tensor:
    """``(T, len(block_ids))`` positions of one kind of placeholder block
    (moment or memory) in a 1-D id sequence.

    Raises if the count is not a whole number of blocks or the slots are out
    of order -- a chunk that lost a placeholder would otherwise silently shift
    every later block.
    """
    seq = input_ids.reshape(-1)
    n = len(block_ids)
    lo, hi = int(block_ids[0]), int(block_ids[-1])
    pos = torch.nonzero((seq >= lo) & (seq <= hi), as_tuple=False).reshape(-1)
    if pos.numel() % n:
        raise ValueError(f"found {pos.numel()} placeholder tokens, not a multiple of the block size {n}")
    pos = pos.view(-1, n)
    slots = seq[pos] - lo
    expected = torch.arange(n, device=seq.device, dtype=slots.dtype).expand_as(slots)
    if not torch.equal(slots, expected):
        raise ValueError("placeholder token slots are out of order")
    return pos


def replace_moment_rows(embeds: torch.Tensor, input_ids: torch.Tensor, placeholder_ids: Sequence[int],
                        rows: torch.Tensor) -> torch.Tensor:
    """Out-of-place swap of the placeholder rows of ``embeds`` (``(B, S, H)``)
    for the rows of the table ``rows`` (``(len(placeholder_ids), H)``), indexed
    by ``id - placeholder_ids[0]`` (the ids are contiguous).

    Out of place on purpose: under gradient checkpointing the replayed embeds
    are a leaf that requires grad, which forbids in-place writes.
    """
    lo = int(placeholder_ids[0])
    ids = input_ids.to(embeds.device)
    mask = (ids >= lo) & (ids <= lo + rows.shape[0] - 1)
    if not bool(mask.any()):
        return embeds
    slot = (ids - lo).clamp_(0, rows.shape[0] - 1)
    return torch.where(mask[..., None], rows.to(embeds.dtype)[slot], embeds)


def replace_rows_at(embeds: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Out-of-place write of per-block rows: ``embeds`` ``(1, S, H)``,
    ``positions`` ``(T, k)`` sequence positions, ``rows`` ``(T, k, H)``."""
    flat = positions.reshape(-1).to(embeds.device)
    values = rows.reshape(-1, rows.shape[-1]).to(embeds.dtype)
    return embeds.index_put((torch.zeros_like(flat), flat), values)


def make_moment_embed_hook(get_rows: Callable[[], torch.Tensor], placeholder_ids: Sequence[int]):
    """Forward hook for ``embed_tokens`` that swaps in the placeholder rows
    (``get_rows()`` -> ``(len(placeholder_ids), H)``: moment embeddings followed
    by the current turn's memory rows) whenever the looked-up ids contain
    placeholders. Only the incremental rollout path goes through
    ``embed_tokens``; training replays embeds."""

    def hook(module, inputs, output):
        input_ids = inputs[0]
        lo = int(placeholder_ids[0])
        if not bool(((input_ids >= lo) & (input_ids <= lo + len(placeholder_ids) - 1)).any()):
            return None
        return replace_moment_rows(output, input_ids, placeholder_ids, get_rows())

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
    """Moment-token embeddings + memory-token base embeddings + memory
    transformer + zero-init read-out projection.

    Every entry point is a ``mode`` of :meth:`forward` so that calls made
    through PEFT's ``ModulesToSaveWrapper`` (which only proxies ``forward``)
    honour ``disable_adapter()``: the reference-policy pass then sees this
    module's untrained copy, whose zero-initialised ``out_proj`` makes every
    memory token the constant ``mem_embed`` row.
    """

    def __init__(self, hidden_size: int, n_moment: int = 8, n_mem: int = 1, d_mem: int = 512,
                 n_layers: int = 2, n_heads: int = 8, ffn_mult: int = 4,
                 memory_window: Optional[int] = None, max_blocks: int = 1024, init_range: float = 0.02):
        super().__init__()
        if n_moment < 1 or n_mem < 1:
            raise ValueError("n_moment and n_mem must be >= 1")
        self.hidden_size = hidden_size
        self.n_moment = n_moment
        self.n_mem = n_mem
        self.d_mem = d_mem
        self.memory_window = memory_window
        self.max_blocks = max_blocks
        self.moment_embed = nn.Embedding(n_moment, hidden_size)
        self.mem_embed = nn.Embedding(n_mem, hidden_size)  # the memory token at zero read-out
        self.in_proj = nn.Linear(hidden_size, d_mem)
        self.slot_embed = nn.Embedding(n_moment + n_mem, d_mem)  # moment slots, then read-out (query) slots
        self.memory = MemoryTransformer(d_mem, n_layers, n_heads, ffn_mult, init_range=init_range)
        self.out_norm = _RMSNorm(d_mem)
        self.out_proj = nn.Linear(d_mem, hidden_size)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=init_range)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.slot_embed.weight, mean=0.0, std=init_range)
        # Exact no-op read-out at init: the memory token is a fixed 'neutral' token,
        # the policy starts as the checkpoint it was given plus that token, and the
        # reference policy (this module's untrained copy) stays exactly that.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def init_placeholder_embeds(self, embed_weight: torch.Tensor, std: float = 0.02):
        """Start the moment and memory tokens as 'neutral' tokens: the mean
        embedding row plus small noise (what ``resize_token_embeddings`` does
        for new tokens)."""
        mean = embed_weight.detach().float().mean(dim=0, keepdim=True)
        for emb in (self.moment_embed, self.mem_embed):
            noise = torch.randn(emb.weight.shape[0], self.hidden_size, device=mean.device) * std
            emb.weight.copy_((mean + noise).to(emb.weight.dtype))

    # -- modes ------------------------------------------------------------- #
    def forward(self, mode: str = "readout", **kwargs):
        if mode == "moment_embeds":
            return self.moment_embed.weight
        if mode == "mem_embeds":
            return self.mem_embed.weight
        if mode == "readout":
            return self.readout(**kwargs)
        if mode == "out_proj_norm":
            return self.out_proj.weight.detach().float().norm()
        raise ValueError(f"unknown HamletModule mode {mode!r}")

    def readout(self, moments: torch.Tensor, query_blocks: torch.Tensor) -> torch.Tensor:
        """Memory-token input rows for the given query blocks.

        ``moments``: ``(T, n_moment, H)`` moment hidden states of blocks 0..T-1
        (oldest first; ``T == 0`` allowed). ``query_blocks``: ``(Q,)`` block
        indices; the read-out for block ``q`` attends to the moment rows of
        blocks ``< q`` -- at turn t only ``m'_{<t}`` exist -- and to itself.
        Rollout asks for ``query_blocks=[T]``; the training replay for
        ``arange(T)``. Per block the computation is identical.

        Moment rows attend to moment rows of blocks ``<= own`` (bidirectional
        inside a block); read-out rows are never keys. ``memory_window`` limits
        both to the last ``memory_window`` blocks (``None`` = whole episode).

        Returns ``(Q, n_mem, H)`` float32 rows: ``mem_embed + out_proj(...)``.
        """
        if moments.dim() != 3 or moments.shape[1] != self.n_moment:
            raise ValueError(f"moments must be (T, {self.n_moment}, H), got {tuple(moments.shape)}")
        T, n, _ = moments.shape
        device = moments.device
        query_blocks = query_blocks.to(device=device, dtype=torch.long).reshape(-1)
        Q = query_blocks.shape[0]
        if T > self.max_blocks or (Q and int(query_blocks.max()) >= self.max_blocks):
            raise ValueError(f"episode exceeds max_blocks={self.max_blocks} (T={T}, queries={query_blocks.tolist()[:3]}...)")
        k = self.n_mem
        ar_n = torch.arange(n, device=device)
        m = self.in_proj(moments.float()) + self.slot_embed(ar_n)[None]  # (T, n, d)
        r = self.slot_embed(n + torch.arange(k, device=device))[None].expand(Q, k, self.d_mem)  # (Q, k, d)
        x = torch.cat([m.reshape(T * n, self.d_mem), r.reshape(Q * k, self.d_mem)], dim=0)
        m_blocks = torch.arange(T, device=device).repeat_interleave(n)
        r_blocks = query_blocks.repeat_interleave(k)
        block_idx = torch.cat([m_blocks, r_blocks])
        L = block_idx.shape[0]
        is_read = torch.zeros(L, dtype=torch.bool, device=device)
        is_read[T * n:] = True
        bi, bj = block_idx[:, None], block_idx[None, :]
        # moment rows: keys are moment rows of blocks <= own; read rows: blocks < own
        allowed = ~is_read[None, :] & torch.where(is_read[:, None], bj < bi, bj <= bi)
        if self.memory_window is not None:
            w = int(self.memory_window)
            allowed &= torch.where(is_read[:, None], bj >= bi - w, bj > bi - w)
        allowed |= torch.eye(L, dtype=torch.bool, device=device)
        y = self.memory(x, block_idx, allowed)
        y_read = y[T * n:].view(Q, k, self.d_mem)
        return self.mem_embed.weight.float()[None] + self.out_proj(self.out_norm(y_read))  # (Q, k, H)


def mem_ratio(rows: torch.Tensor, base: torch.Tensor) -> float:
    """Mean ||read-out|| / ||mem_embed|| over ``rows`` ``(..., n_mem, H)``: 0 at
    init, the rollout/training health signal for the memory pathway."""
    with torch.no_grad():
        r = (rows.float() - base.float()).norm(dim=-1) / base.float().norm(dim=-1).clamp_min(1e-6)
        return float(r.mean())


def build_hamlet_module(hidden_size: int, cfg: Dict) -> HamletModule:
    return HamletModule(
        hidden_size=hidden_size,
        n_moment=int(cfg.get("n_moment", 8)),
        n_mem=int(cfg.get("n_mem", 1)),
        d_mem=int(cfg.get("d_mem", 512)),
        n_layers=int(cfg.get("n_layers", 2)),
        n_heads=int(cfg.get("n_heads", 8)),
        ffn_mult=int(cfg.get("ffn_mult", 4)),
        memory_window=cfg.get("memory_window", None),
        max_blocks=int(cfg.get("max_blocks", 1024)),
    )


def attach_hamlet(model, cfg: Dict, tokenizer) -> Tuple[list, list]:
    """Create the HAMLET module on the *base* HF model (before PEFT wrapping so
    ``modules_to_save`` can pick it up) and return ``(moment_ids, mem_ids)``."""
    text_cfg = model.config.text_config
    n_moment, n_mem = int(cfg.get("n_moment", 8)), int(cfg.get("n_mem", 1))
    ids = moment_token_ids(tokenizer, n_moment + n_mem, int(text_cfg.vocab_size))
    module = build_hamlet_module(int(text_cfg.hidden_size), cfg)
    embed_weight = model.get_input_embeddings().weight
    module.to(device=embed_weight.device)  # fp32 parameters on the model's device
    module.init_placeholder_embeds(embed_weight, std=float(cfg.get("moment_init_std", 0.02)))
    setattr(model, HAMLET_MODULE_NAME, module)
    return ids[:n_moment], ids[n_moment:]


# --------------------------------------------------------------------------- #
# the shared training / post-processing replay forward
# --------------------------------------------------------------------------- #
def forward_embeds_core(embeds_inputs: Dict, *, language_model, lm_head, dtype, training: bool,
                        hamlet=None, moment_ids: Optional[Sequence[int]] = None,
                        mem_ids: Optional[Sequence[int]] = None,
                        compute_values: bool = False, value_head=None,
                        value_grad_scale: Optional[float] = None):
    """Replay a packed episode (``VLMWorker._pack_embeds`` output) through the
    language model and read the action logits at the decision positions.

    This is the single implementation behind ``VLMWrapper._forward_embeds``
    (DDP training) and ``VLMTrainingMixin._forward_embeds`` (``old``/``ref``
    log-probs); the two used to be duplicated, and any drift between them
    silently biases the PPO ratio.

    With HAMLET on, the memory-token rows of the replayed embeds are rebuilt
    from the packed ``moment_history`` (the rollout's stored moment hidden
    states) with the current module parameters -- see the module docstring.

    Returns ``(logits, values, stats)`` with ``logits`` of shape ``(1, T, V)``.
    """
    device = lm_head.weight.device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in embeds_inputs.items()}
    logits_to_keep = inputs.pop("logits_to_keep")
    ids_ref = inputs.pop("input_ids_reference")
    history = inputs.pop("moment_history", None)
    emb = inputs["inputs_embeds"].to(dtype)
    if training:
        # gradient checkpointing needs an input that carries grad
        emb.requires_grad_(True)
    stats = {}
    T = int(logits_to_keep.numel())
    if hamlet is not None:
        emb = replace_moment_rows(emb, ids_ref, moment_ids, hamlet(mode="moment_embeds"))
        if history is None:
            raise ValueError("packed episode has no moment_history; was it produced with HAMLET on?")
        history = history.to(device)
        mem_pos = moment_positions(ids_ref[0], mem_ids).to(device)  # (T, n_mem)
        if history.shape[0] != T or mem_pos.shape[0] != T:
            raise ValueError(f"{history.shape[0]} stored moment blocks, {mem_pos.shape[0]} memory blocks, "
                             f"{T} decisions in the packed episode")
        mem_rows = hamlet(mode="readout", moments=history, query_blocks=torch.arange(T, device=device))  # (T, n_mem, H)
        stats["hamlet_mem_ratio"] = mem_ratio(mem_rows, hamlet(mode="mem_embeds"))
        emb = replace_rows_at(emb, mem_pos, mem_rows)
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
    if hamlet is not None:
        pos = moment_positions(ids_ref[0], moment_ids).to(hidden.device)  # (T, n)
        if pos.shape[0] != T:
            raise ValueError(f"{pos.shape[0]} moment blocks but {T} decisions in the packed episode")
        with torch.no_grad():  # staleness of the stored moments the memory read: replay m' vs rollout m'
            replayed = hidden[0, pos].float()
            drift = (replayed - history.float()).norm(dim=-1) / history.float().norm(dim=-1).clamp_min(1e-6)
            stats["hamlet_moment_drift"] = float(drift.mean())
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
