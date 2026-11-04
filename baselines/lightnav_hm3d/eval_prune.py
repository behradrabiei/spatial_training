"""Run ``lightnav.cli.eval_habitat`` with the visual history pruned to a token budget.

The flat token-prune study (paper/token_prune, "Flat variant") keeps ``B = k x (tokens per
history frame)`` visual tokens of the whole history under one selection rule -- random, FastV,
DivPrune or DART -- scored and deleted at decoder layer 0, deletions permanent. LongNav-R1 does
it on its KV cache, Uni-NaVid by row removal in its re-assembled prompt. LightNav-0 rebuilds its
prompt every step (anchor / span / burst / fast tubelets, oldest first, then the current pair)
and prefills it inside vLLM, so this wrapper, without touching the upstream clone:

* forces eager vLLM and replaces ``FlashAttentionImpl.forward`` on every decoder layer with the
  exact masked attention of ``eval_blind.py`` (the same kernel path as the information-horizon
  reference ``lightnav_ref``), hiding the **pruned** ``<|video_pad|>`` columns from every query
  on every layer. A hidden token influences nothing downstream, so this equals removing it with
  its mRoPE position kept (what LongNav-R1's cache deletion does); timestamps, vision markers
  and the task text stay visible;
* keeps a per-episode **ledger** of deleted tokens keyed by tubelet identity
  ``(frame0, frame1, pool, cell)``: each step the candidates are the history tokens not in the
  ledger (the current 2-frame tubelet is exempt), and when they exceed ``B`` the rule keeps
  ``B`` and the rest join the ledger. The fast / mid tiers pair consecutive frames, so the
  pairing flips parity every step: a tubelet recurs every other step (each parity class is
  pruned on its own), fast->mid keeps the pair and pool, mid->long re-pools and the span tier
  re-places its pairs, so those tubelets re-enter as fresh candidates;
* budget ``B = tokens_per_frame x k`` with ``tokens_per_frame = 14`` by default: a fast-tier
  tubelet (2 frames, pool 2) holds 28 tokens, what this model spends on a recent history frame
  (Uni-NaVid: ``k x 4``, LongNav-R1: ``k x`` its mean post-filter count); ``k = 0`` hides every
  history token (the current tubelet and the whole text skeleton -- timestamps, vision markers,
  task -- stay visible), the no-visual-history control of the layout check;
* ``--prune-current-frames 1`` (2026-10-27, the ``K = 1`` point of the old-blocks-kept control) makes
  the sampler hand the encoder the current tubelet ``(t, t)`` -- the current frame doubled, what
  upstream itself feeds at step 0 and what the ``K = 1`` context window (``eval_ctx.py``) feeds at
  every step -- instead of ``(t-1, t)``; with ``k = 0`` the model then sees one frame plus the
  text skeleton. The tubelet's 112 tokens cover both of its frames, so a column mask alone cannot
  drop frame ``t-1``; this is the one place the wrapper changes the sampled layout;
* ``recency`` keeps the newest candidates by tubelet age (the window-content control of the layout
  check, 2026-10-25): at ``k = 2`` exactly the newest history tubelet, frames ``(t-3, t-2)`` at
  pool 2, which is the pair the K = 4 context window (``eval_ctx.py``) shows -- same visual content
  as that window, but every older block's timestamp and vision markers stay in the prompt;
* FastV and DART need layer-0 quantities of the current prompt: a **scoring pre-pass**
  (``vllm.LLM.generate`` with ``max_tokens=1``, ledger columns hidden) captures, at layer 0 of
  the prefill, the post-softmax attention row of the last prompt token (the ``\\n`` after
  ``assistant``, whose logits emit the first action token; mean over the 32 query heads, all
  visible causal keys) and the rotated keys, and at the input of layer 1 the residual stream
  (deepstack-0 injection included); the real generate then hides the dropped columns from
  layer 0 on -- Uni-NaVid's pre-pass semantics. Random needs nothing; DivPrune reads the LLM
  input embeddings of the visual tokens (the main channels of ``video_embeds``).

Selection rules are copies of LongNav-R1's (``src/longnav/utils/kv_prune.py``): random =
``randperm`` seeded per (episode, step); DivPrune = greedy max-min cosine distance on the input
embeddings; DART = 8 pivots by L1 key norm, each claiming its quota of the least-similar
candidates (cosine on the layer-1 input), pivots kept; FastV = top-B attention.

Per inference one record goes to ``<output_dir>/prune/episode_NNN.jsonl`` (Uni-NaVid's schema:
``step, frames`` = newer frame of each history tubelet, ``n_tokens, n_kept, bits`` = packed keep
mask over the tubelets in prompt order, ``budget, n_pool`` = candidates, ``prompt_tokens,
seconds`` plus ``tubelets``/``pools``); ``convert_results.py`` renames them to
``prune/<label>.jsonl``.

Runs in the LightNav client runtime (``baselines/.envs/lightnav``) with
``PYTHONPATH=baselines/LightNav-0/src``; ``baselines/run_lightnav.sh`` uses it when
``PRUNE_METHOD`` / ``PRUNE_K`` are set:

    python baselines/lightnav_hm3d/eval_prune.py --prune-method fastv --prune-k 8 <eval_habitat args...>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from baselines.lightnav_hm3d.eval_blind import (
        VIDEO_PAD_ID, _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )
except ImportError:  # run as a script: sys.path[0] is this directory
    from eval_blind import (  # type: ignore
        VIDEO_PAD_ID, _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )

METHODS = ("random", "fastv", "divprune", "dart", "recency")
PREPASS_METHODS = ("fastv", "dart")  # need layer-0 quantities of the current prompt
TOKENS_PER_FRAME = 14  # a fast-tier tubelet = 2 frames at pool 2 = 28 tokens
N_PIVOTS = 8
SEED = 17


# --- selection rules (copies of src/longnav/utils/kv_prune.py) ------------------------------
def selection_seed(seed: int, label: str, step: int) -> int:
    """Per-(episode, step) seed of the random selector (LongNav-R1's ``step_seed`` / Uni-NaVid's)."""
    return (int(seed) * 1_000_003 + int(step)) ^ zlib.crc32(str(label).encode("utf-8"))


def random_select(n: int, capacity: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform sample of ``capacity`` candidate indices (sorted)."""
    return torch.randperm(n, generator=generator)[:capacity].sort().values


def divprune_select(rows: torch.Tensor, capacity: int, block: int = 2048) -> torch.Tensor:
    """DivPrune (CVPR'25): greedy Max-Min Diversity under cosine distance on the tokens' LLM
    input embeddings. Init = the token whose nearest neighbour is farthest; then repeatedly add
    the token farthest from the selected set. ``rows`` (N, d) in candidate order; returns the
    kept candidate indices (sorted)."""
    n = int(rows.shape[0])
    capacity = min(int(capacity), n)
    if capacity == 0:
        return torch.empty(0, dtype=torch.long)
    if capacity == n:
        return torch.arange(n)
    x = F.normalize(rows.float(), dim=-1)
    device = x.device
    nn_dist = torch.empty(n, device=device)
    for start in range(0, n, block):
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
    return chosen.cpu().sort().values


def dart_select(key_norms: torch.Tensor, hidden: torch.Tensor, capacity: int, n_pivots: int = N_PIVOTS) -> torch.Tensor:
    """DART (arXiv 2502.11494): pivots = the ``n_pivots`` largest L1 key norms; in descending norm
    order each pivot claims ``q = (capacity - P) // P`` of the remaining candidates with the lowest
    cosine similarity to it (the remainder one extra to the first pivots); pivots are kept.
    ``key_norms`` (N,), ``hidden`` (N, d) in candidate order; returns kept indices (sorted)."""
    n = int(key_norms.numel())
    capacity = min(int(capacity), n)
    if capacity == 0:
        return torch.empty(0, dtype=torch.long)
    if capacity == n:
        return torch.arange(n)
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
    return kept.cpu().sort().values


def recency_select(keys: Sequence[Tuple[int, int, int, int]], capacity: int) -> torch.Tensor:
    """The ``capacity`` newest candidates: by tubelet frames ``(f1, f0)`` descending, ties in prompt
    order (so a tubelet is kept whole before an older one is touched). Sorted candidate positions."""
    order = sorted(range(len(keys)), key=lambda i: (-keys[i][1], -keys[i][0], i))
    return torch.as_tensor(sorted(order[:capacity]), dtype=torch.long)


def fastv_select(scores: torch.Tensor, capacity: int) -> torch.Tensor:
    """Top-``capacity`` attention scores (stable, descending); returns kept indices (sorted)."""
    n = int(scores.numel())
    capacity = min(int(capacity), n)
    order = torch.argsort(scores.float().cpu(), descending=True, stable=True)[:capacity]
    return order.sort().values


def pack_bits(bits: Sequence[bool]) -> str:
    """base64 of ``numpy.packbits`` over a boolean keep mask (Uni-NaVid's ``prune.pack_bits``)."""
    return base64.b64encode(np.packbits(np.asarray(list(bits), dtype=bool))).decode("ascii")


# --- prompt layout ------------------------------------------------------------------------
class Tubelet(NamedTuple):
    segment: int
    f0: int
    f1: int
    pool: int
    n_tokens: int
    start: int  # visual rank (index among the prompt's <|video_pad|> tokens) of its first cell


def tubelet_layout(segments: Sequence[Tuple[Sequence[int], int]], grids: Sequence[Sequence[int]],
                   merge_size: int = 2) -> List[Tubelet]:
    """The prompt's tubelets in order. ``segments`` = ``(frame_ids, pool_spatial)`` per video
    segment as the SlowFast sampler emitted them (oldest first, even frame counts), ``grids`` =
    the post-pool ``video_grid_thw`` rows ``(T, H, W)`` of the same videos (pre-merge space):
    ``T`` tubelets of ``(H / merge) x (W / merge)`` tokens each, raster-ordered."""
    if len(segments) != len(grids):
        raise ValueError(f"{len(segments)} segments but {len(grids)} video grids")
    out: List[Tubelet] = []
    rank = 0
    for si, ((ids, pool), (t, h, w)) in enumerate(zip(segments, grids)):
        t, h, w = int(t), int(h), int(w)
        ids = [int(i) for i in ids]
        if len(ids) != 2 * t:
            raise ValueError(f"segment {si}: {len(ids)} frame ids for {t} tubelets")
        if h % merge_size or w % merge_size:
            raise ValueError(f"segment {si}: grid {h}x{w} not a multiple of merge size {merge_size}")
        per = (h // merge_size) * (w // merge_size)
        for j in range(t):
            out.append(Tubelet(si, ids[2 * j], ids[2 * j + 1], int(pool), per, rank))
            rank += per
    return out


def token_key(tub: Tubelet, cell: int) -> Tuple[int, int, int, int]:
    return (tub.f0, tub.f1, tub.pool, cell)


def single_current(segs: List[Dict[str, Any]], current_abs: int, episode_frames: Any) -> List[Dict[str, Any]]:
    """Replace the current segment's pair ``(t-1, t)`` by ``(t, t)`` (upstream's own step-0 / K = 1
    layout); every other segment is untouched. The current segment must be the last one and hold
    exactly one tubelet ending in ``current_abs``."""
    cur = segs[-1]
    ids = [int(i) for i in cur["frame_indices"]]
    if len(ids) != 2 or ids[1] != current_abs or any(current_abs in [int(i) for i in s["frame_indices"]] for s in segs[:-1]):
        raise RuntimeError(f"unexpected current segment {ids} for frame {current_abs}")
    if ids[0] == current_abs:
        return segs  # step 0: already doubled
    new = dict(cur)
    new["frame_indices"] = [current_abs, current_abs]
    new["video"] = episode_frames[[current_abs, current_abs]]
    return segs[:-1] + [new]


def split_history(tubelets: Sequence[Tubelet], current_abs: int) -> Tuple[List[Tubelet], List[Tubelet]]:
    """(history tubelets, current tubelets): a tubelet holding the current frame is exempt."""
    cur = [t for t in tubelets if current_abs in (t.f0, t.f1)]
    hist = [t for t in tubelets if current_abs not in (t.f0, t.f1)]
    if not cur:
        raise RuntimeError(f"no tubelet holds the current frame {current_abs}")
    return hist, cur


class StepPlan(NamedTuple):
    hide: torch.Tensor  # bool [prompt_len]: ledger tokens of this prompt
    history: List[Tubelet]
    cand_ranks: List[int]  # visual ranks of the candidates (history tokens not in the ledger)
    cand_keys: List[Tuple[int, int, int, int]]


def plan_step(tubelets: Sequence[Tubelet], current_abs: int, ledger: set, pad_pos: Sequence[int],
              prompt_len: int) -> StepPlan:
    """Hidden set and candidates of one prompt: every history token already in the ledger is hidden,
    the rest are candidates; the current tubelet is exempt (never hidden, never a candidate)."""
    hist, _ = split_history(tubelets, current_abs)
    hide = torch.zeros(int(prompt_len), dtype=torch.bool)
    cand_ranks: List[int] = []
    cand_keys: List[Tuple[int, int, int, int]] = []
    for tub in hist:
        for c in range(tub.n_tokens):
            key = token_key(tub, c)
            if key in ledger:
                hide[pad_pos[tub.start + c]] = True
            else:
                cand_ranks.append(tub.start + c)
                cand_keys.append(key)
    return StepPlan(hide, hist, cand_ranks, cand_keys)


def apply_drop(plan: StepPlan, keep: torch.Tensor, ledger: set, pad_pos: Sequence[int]) -> int:
    """Hide every candidate not in ``keep`` (positions into ``plan.cand_ranks``) and record it in
    the ledger; returns how many were dropped."""
    keep_set = set(int(i) for i in keep.tolist())
    n = 0
    for i, (rank, key) in enumerate(zip(plan.cand_ranks, plan.cand_keys)):
        if i not in keep_set:
            plan.hide[pad_pos[rank]] = True
            ledger.add(key)
            n += 1
    return n


def kept_per_tubelet(plan: StepPlan, pad_pos: Sequence[int]) -> Tuple[List[int], List[bool]]:
    """(tokens visible per history tubelet, keep bits over the history tokens in prompt order)."""
    hidden = plan.hide.tolist()
    bits: List[bool] = []
    n_kept: List[int] = []
    for tub in plan.history:
        b = [not hidden[pad_pos[tub.start + c]] for c in range(tub.n_tokens)]
        bits.extend(b)
        n_kept.append(int(sum(b)))
    return n_kept, bits


# --- shared state -------------------------------------------------------------------------
class PruneState:
    def __init__(self, method: str, k: int, tokens_per_frame: int = TOKENS_PER_FRAME,
                 n_pivots: int = N_PIVOTS, seed: int = SEED, debug_steps: int = 0, current_frames: int = 2):
        if current_frames not in (1, 2):
            raise ValueError(f"current frames must be 1 or 2, got {current_frames}")
        self.current_frames = int(current_frames)
        if method not in METHODS:
            raise ValueError(f"unknown prune method {method!r}; expected one of {METHODS}")
        if k < 0 or tokens_per_frame < 1:
            raise ValueError(f"prune k must be >= 0 and tokens per frame >= 1, got k={k}, tpf={tokens_per_frame}")
        self.method = method
        self.k = int(k)
        self.tokens_per_frame = int(tokens_per_frame)
        self.budget = self.k * self.tokens_per_frame
        self.n_pivots = int(n_pivots)
        self.seed = int(seed)
        self.debug_steps = debug_steps
        self.n_layers: Optional[int] = None
        self.hidden_size: Optional[int] = None
        self.strict = False
        # per generate call
        self.hide: Optional[torch.Tensor] = None  # bool [prompt_len]; None = no generate in flight
        self._hide_dev: Optional[torch.Tensor] = None
        self.prompt_len = 0
        self.capture = False  # pre-pass: record layer-0 row / keys and the layer-1 input
        self.row: Optional[torch.Tensor] = None  # [seq_len] fp32, last query's layer-0 attention
        self.keys0: Optional[torch.Tensor] = None  # [n_query, KVH, D] layer-0 rotated keys (uncached rows)
        self.h1: Optional[torch.Tensor] = None  # [n_query, hidden] residual stream entering layer 1
        self.offset = 0  # seq_len - n_query of the captured prefill (prefix-cached rows)
        self.calls = 0
        self.generates = 0
        # per episode
        self.ledger: set = set()
        self.episode = -1
        self.segments: Optional[List[Tuple[List[int], int]]] = None
        self.current_abs: Optional[int] = None
        self.log_dir: Optional[Path] = None
        self._log = None

    def hide_on(self, device: torch.device) -> torch.Tensor:
        if self._hide_dev is None or self._hide_dev.device != device:
            self._hide_dev = self.hide.to(device)
        return self._hide_dev

    def set_hide(self, hide: Optional[torch.Tensor]) -> None:
        self.hide = hide
        self._hide_dev = None

    def reset_episode(self) -> None:
        self.ledger = set()
        self.episode += 1
        self.segments = None
        self.current_abs = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_record(self, rec: Dict[str, Any]) -> None:
        if self.log_dir is None:
            return
        if self._log is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log = (self.log_dir / f"episode_{self.episode:03d}.jsonl").open("a")
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()

    def describe(self) -> Dict[str, Any]:
        return {"method": self.method, "k": self.k, "tokens_per_frame": self.tokens_per_frame,
                "budget": self.budget, "pivots": self.n_pivots if self.method == "dart" else None,
                "seed": self.seed if self.method == "random" else None, "prepass": self.method in PREPASS_METHODS,
                "ledger": "tubelet (f0, f1, pool, cell)", "current_tubelet_exempt": True,
                "current_frames": self.current_frames}


# --- vLLM hooks ---------------------------------------------------------------------------
def decision_row(query_last: torch.Tensor, k: torch.Tensor, scale: float, allowed: torch.Tensor) -> torch.Tensor:
    """Post-softmax attention of one query (``[H, D]``) over ``k`` (``[S, KVH, D]``, GQA) restricted
    to the ``allowed`` columns, fp32, mean over the heads -> ``[S]``."""
    rep = query_last.shape[0] // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1)  # [S, H, D]
    logits = torch.einsum("hd,shd->hs", query_last.float(), kf) * float(scale)
    logits = logits.masked_fill(~allowed[None, :].to(logits.device), float("-inf"))
    return torch.softmax(logits, dim=-1).mean(dim=0)


def make_prune_forward(state: PruneState, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """``FlashAttentionImpl.forward`` replacement: ``eval_blind``'s masked kernel with the hidden
    set = ``state.hide`` on every layer; during a scoring pre-pass the layer-0 prefill call also
    records the last query's attention row and the rotated keys."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        attn_type = getattr(self, "attn_type", None)
        is_decoder = attn_type is None or str(getattr(attn_type, "name", attn_type)).split(".")[-1] == "DECODER"
        if attn_metadata is None or state.hide is None or not is_decoder:
            if state.strict and attn_metadata is not None and is_decoder:
                raise RuntimeError("decoder attention forward outside a hooked generate call")
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output=output,
                            output_scale=output_scale, output_block_scale=output_block_scale)
        if output is None:
            raise RuntimeError("vLLM must pass an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by the masked kernel")
        if str(getattr(self, "kv_cache_dtype", "auto")).startswith("fp8"):
            raise NotImplementedError("fp8 KV cache is not supported by the masked kernel")
        if tuple(getattr(self, "sliding_window", (-1, -1))) != (-1, -1) or getattr(self, "alibi_slopes", None) is not None \
                or getattr(self, "logits_soft_cap", None):
            raise NotImplementedError("sliding window / alibi / soft cap are not supported by the masked kernel")
        if not getattr(attn_metadata, "causal", True):
            raise RuntimeError("non-causal attention metadata")
        qsl = attn_metadata.query_start_loc
        if int(qsl.numel()) != 2:
            raise RuntimeError(f"expected one request per forward, got {int(qsl.numel()) - 1}")
        n_query = int(attn_metadata.num_actual_tokens)
        if int(qsl[1].item()) != n_query:
            raise RuntimeError(f"query_start_loc {int(qsl[1].item())} != num_actual_tokens {n_query}")
        seq_len = int(attn_metadata.seq_lens[0].item())
        if n_query < 1 or seq_len < n_query:
            raise RuntimeError(f"bad lengths: n_query={n_query} seq_len={seq_len}")
        if seq_len > int(state.hide.numel()) + 4096:
            raise RuntimeError(f"seq_len {seq_len} far beyond the prompt ({int(state.hide.numel())} tokens)")

        hide = state.hide_on(query.device)
        key_cache, value_cache = kv_cache.unbind(0)
        row = attn_metadata.block_table[0]
        k = gather_kv(key_cache, row, seq_len)
        v = gather_kv(value_cache, row, seq_len)
        mask = attention_mask(n_query, seq_len, hide, device=query.device)
        out = masked_attention(query[:n_query], k, v, float(self.scale), mask)
        output[:n_query].copy_(out.to(output.dtype))
        state.calls += 1
        if state.capture and n_query > 1 and layer_index(layer.layer_name) == 0:
            if state.row is not None:
                raise RuntimeError("layer-0 prefill captured twice in one pre-pass (chunked prefill?)")
            if seq_len != state.prompt_len:
                raise RuntimeError(f"pre-pass prefill seq_len {seq_len} != prompt length {state.prompt_len}")
            state.row = decision_row(query[n_query - 1], k, float(self.scale), mask[-1]).cpu()
            state.keys0 = key[:n_query].detach().float().cpu()
            state.offset = seq_len - n_query
        return output

    forward._prune_method = state.method  # type: ignore[attr-defined]
    return forward


def make_layer1_prehook(state: PruneState) -> Callable[..., None]:
    """Forward pre-hook on decoder layer 1: its ``(positions, hidden_states, residual)`` input;
    the residual stream after layer 0 (deepstack-0 added by the model loop) is their sum."""

    def hook(module, args):
        if not state.capture:
            return
        hidden_states, residual = args[1], args[2]
        if hidden_states.shape[0] <= 1:
            return  # decode forward
        if residual is None:
            raise RuntimeError("layer 1 received no residual stream")
        if state.h1 is not None:
            raise RuntimeError("layer-1 input captured twice in one pre-pass")
        state.h1 = (hidden_states + residual).detach().float().cpu()

    return hook


def check_generate_counts(state: PruneState, prompt_len: int, n_visual: int) -> int:
    n = state.n_layers
    if not n:
        raise RuntimeError("engine layer count unknown (engine hook did not run)")
    if state.calls == 0 or state.calls % n:
        raise RuntimeError(f"masked attention ran {state.calls} times, not a multiple of {n} layers "
                           "(compiled / CUDA-graph path?)")
    if n_visual == 0:
        raise RuntimeError(f"prompt of {prompt_len} tokens has no <|video_pad|> token")
    return state.calls // n


# --- one inference ------------------------------------------------------------------------
def select_keep(state: PruneState, items, cand_ranks: List[int], pads: torch.Tensor,
                run_prepass: Callable[[], int], cand_keys: Optional[Sequence[Tuple[int, int, int, int]]] = None,
                ) -> Tuple[torch.Tensor, int]:
    """Which of the candidates (by position in ``cand_ranks``) survive this step: a sorted index
    tensor of length ``budget``. Runs the scoring pre-pass for fastv / dart; returns
    (kept positions, forwards used by the pre-pass)."""
    n, budget = len(cand_ranks), state.budget
    forwards = 0
    if state.method == "random":
        g = torch.Generator().manual_seed(selection_seed(state.seed, f"episode_{state.episode:03d}", state.current_abs))
        return random_select(n, budget, g), 0
    if state.method == "recency":
        if cand_keys is None or len(cand_keys) != n:
            raise RuntimeError("recency needs the candidates' tubelet keys")
        return recency_select(cand_keys, budget), 0
    if state.method == "divprune":
        emb = items[0].video_embeds
        if emb is None or state.hidden_size is None or emb.shape[1] % state.hidden_size:
            raise RuntimeError(f"video_embeds {None if emb is None else tuple(emb.shape)} do not hold the LLM input "
                               f"embeddings (hidden {state.hidden_size})")
        rows = emb[torch.as_tensor(cand_ranks), :state.hidden_size]
        if torch.cuda.is_available():
            rows = rows.cuda()
        return divprune_select(rows, budget), 0
    # fastv / dart: scoring pre-pass with the ledger hidden
    state.capture = True
    state.row = state.keys0 = state.h1 = None
    try:
        forwards = run_prepass()
    finally:
        state.capture = False
    if state.row is None or state.keys0 is None:
        raise RuntimeError("the pre-pass did not capture the layer-0 prefill")
    cand_pos = pads[torch.as_tensor(cand_ranks)]  # prompt positions of the candidates
    if int(cand_pos.min()) < state.offset:
        raise RuntimeError(f"candidate at prompt index {int(cand_pos.min())} inside the prefix-cached "
                           f"{state.offset} tokens")
    if state.method == "fastv":
        return fastv_select(state.row[cand_pos], budget), forwards
    if state.h1 is None:
        raise RuntimeError("the pre-pass did not capture the layer-1 input (DART)")
    rows = cand_pos - state.offset
    norms = state.keys0[rows].abs().sum(dim=(1, 2))
    hidden = state.h1[rows]
    if torch.cuda.is_available():
        hidden = hidden.cuda()
    return dart_select(norms, hidden, budget, state.n_pivots), forwards


def install_visual_prune(state: PruneState) -> None:
    """Patch vLLM's flash-attention kernel, LightNav's engine loader, sampler, generate and reset."""
    os.environ["VLN_VLLM_ENFORCE_EAGER"] = "1"
    import lightnav.inference.engine as engine_mod
    import lightnav.inference.model as model_mod
    import lightnav.inference.samples as samples_mod
    from lightnav.inference.vllm_utils import get_vllm_model
    from vllm.v1.attention.backends import flash_attn as fa

    original_forward = fa.FlashAttentionImpl.forward
    if getattr(original_forward, "_blind_layer", None) is not None or getattr(original_forward, "_prune_method", None) is not None:
        raise RuntimeError("an attention hook is already installed")
    fa.FlashAttentionImpl.forward = make_prune_forward(state, original_forward)

    # The SlowFast sampler: record this step's segment layout (frame ids, pool) and current frame.
    original_segments = samples_mod.slowfast_video_segments
    if getattr(original_segments, "_context_frames", None) is not None:
        raise RuntimeError("the context window wrapper is installed; PRUNE_* and CONTEXT_FRAMES are not combined")

    def recording_segments(episode_frames, current_abs, n_available, tiers):
        total = int(episode_frames.shape[0])
        if current_abs != total - 1 or n_available != total:
            raise AssertionError(f"unexpected SlowFast call convention: total={total} current_abs={current_abs} "
                                 f"n_available={n_available}")
        segs = original_segments(episode_frames, current_abs, n_available, tiers)
        if state.current_frames == 1:
            segs = single_current(segs, int(current_abs), episode_frames)
        state.segments = [([int(i) for i in s["frame_indices"]], int(s["pool_spatial"])) for s in segs]
        state.current_abs = int(current_abs)
        return segs

    samples_mod.slowfast_video_segments = recording_segments

    original_load = model_mod.load_vllm_local

    def load_vllm_local(config, task_type="vlnce"):
        bundle, llm = original_load(config, task_type=task_type)
        cfg = llm.llm_engine.vllm_config
        if not cfg.model_config.enforce_eager:
            raise RuntimeError("vLLM is not in eager mode (VLN_VLLM_ENFORCE_EAGER=1 required)")
        from vllm.config.compilation import CUDAGraphMode
        if cfg.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(f"CUDA graphs enabled ({cfg.compilation_config.cudagraph_mode}); hooks would not run")
        from vllm.config.compilation import CompilationMode
        if cfg.compilation_config.mode != CompilationMode.NONE:
            raise RuntimeError(f"torch.compile enabled ({cfg.compilation_config.mode}); the layer-1 pre-hook would not run")
        if getattr(cfg.scheduler_config, "enable_chunked_prefill", False):
            raise RuntimeError("chunked prefill is on; the pre-pass needs one prefill forward")
        names = [n for n in cfg.compilation_config.static_forward_context if _LAYER_RE.search(n)]
        indices = sorted(layer_index(n) for n in names)
        n_layers = len(indices)
        if indices != list(range(n_layers)) or n_layers < 2:
            raise RuntimeError(f"unexpected decoder attention layers: {names}")
        expected = int(cfg.model_config.hf_text_config.num_hidden_layers)
        if n_layers != expected:
            raise RuntimeError(f"{n_layers} attention layers registered, config says {expected}")
        backend_name = type(getattr(next(iter(cfg.compilation_config.static_forward_context.values())), "impl", None)).__name__
        if backend_name != "FlashAttentionImpl":
            raise RuntimeError(f"attention backend is {backend_name}, the hook patches FlashAttentionImpl")
        state.n_layers = n_layers
        state.hidden_size = int(cfg.model_config.hf_text_config.hidden_size)
        layers = get_vllm_model(llm).language_model.model.layers
        layers[1].register_forward_pre_hook(make_layer1_prehook(state))
        state.strict = True
        print(f"[prune] method={state.method} k={state.k} budget={state.budget} tokens "
              f"(tokens_per_frame={state.tokens_per_frame}) n_layers={n_layers} hidden={state.hidden_size} "
              f"eager={cfg.model_config.enforce_eager} cudagraph={cfg.compilation_config.cudagraph_mode.name}", flush=True)
        return bundle, llm

    model_mod.load_vllm_local = load_vllm_local

    original_reset = engine_mod.VLNInferenceEngine.reset_episode_state

    def reset_episode_state(self):
        state.reset_episode()
        return original_reset(self)

    engine_mod.VLNInferenceEngine.reset_episode_state = reset_episode_state

    original_generate = engine_mod.VLNInferenceEngine.llm_generate_batch

    def llm_generate_batch(self, items, max_new_tokens):
        if len(items) != 1:
            raise RuntimeError(f"the pruning wrapper handles one request per generate call, got {len(items)}")
        if state.segments is None or state.current_abs is None:
            raise RuntimeError("generate called before the SlowFast sampler recorded the segment layout")
        if state.episode < 0:
            raise RuntimeError("generate called before the first episode reset")
        t0 = time.monotonic()
        ids = list(items[0].prompt_ids)
        visual = visual_positions(ids)
        pads = torch.nonzero(visual).flatten()
        pad_pos = pads.tolist()  # prompt index of each visual rank
        grids = items[0].video_grid_thw
        if grids is None:
            raise RuntimeError("prompt carries no video grid")
        tubelets = tubelet_layout(state.segments, [[int(x) for x in row] for row in grids.tolist()])
        n_pads = sum(t.n_tokens for t in tubelets)
        if n_pads != int(pads.numel()):
            raise RuntimeError(f"tubelet layout holds {n_pads} tokens but the prompt has {int(pads.numel())} <|video_pad|>")
        segments, current_abs = state.segments, state.current_abs
        state.segments = None  # one generate per sampled layout
        plan = plan_step(tubelets, current_abs, state.ledger, pad_pos, len(ids))
        hide, hist, cand_ranks = plan.hide, plan.history, plan.cand_ranks
        state.prompt_len = len(ids)
        state.set_hide(hide)
        state.calls = 0
        n_dropped, prepass_forwards = 0, 0
        try:
            if len(cand_ranks) > state.budget:
                def run_prepass() -> int:
                    original_generate(self, items, 1)
                    n_fwd = check_generate_counts(state, len(ids), int(visual.sum()))
                    state.calls = 0
                    return n_fwd
                if state.budget == 0:  # k = 0: nothing to choose, no scoring pre-pass
                    keep, prepass_forwards = torch.empty(0, dtype=torch.long), 0
                else:
                    keep, prepass_forwards = select_keep(state, items, cand_ranks, pads, run_prepass, plan.cand_keys)
                if int(keep.numel()) != state.budget:
                    raise RuntimeError(f"{state.method} kept {int(keep.numel())} of {len(cand_ranks)} candidates, budget {state.budget}")
                n_dropped = apply_drop(plan, keep, state.ledger, pad_pos)
                state.set_hide(hide)
            out = original_generate(self, items, max_new_tokens)
            n_forwards = check_generate_counts(state, len(ids), int(visual.sum()))
        finally:
            state.set_hide(None)
            state.capture = False
        seconds = time.monotonic() - t0
        state.generates += 1

        n_kept, kept_bits = kept_per_tubelet(plan, pad_pos)
        state.log_record({
            "step": current_abs, "frames": [t.f1 for t in hist], "n_tokens": [t.n_tokens for t in hist],
            "n_kept": n_kept, "bits": pack_bits(kept_bits), "budget": state.budget, "n_pool": len(cand_ranks),
            "n_dropped": n_dropped, "ledger": len(state.ledger), "prompt_tokens": len(ids),
            "n_visual": int(visual.sum()), "tubelets": [[t.f0, t.f1] for t in hist], "pools": [t.pool for t in hist],
            "forwards": n_forwards, "prepass_forwards": prepass_forwards, "seconds": seconds,
            "episode_label": f"episode_{state.episode:03d}", "n_segments": len(segments),
        })
        if state.generates <= state.debug_steps:
            print(f"[prune] episode={state.episode} step={current_abs} prompt_len={len(ids)} visual={int(visual.sum())} "
                  f"history_tubelets={len(hist)} candidates={len(cand_ranks)} budget={state.budget} dropped_now={n_dropped} "
                  f"ledger={len(state.ledger)} kept={sum(n_kept)} forwards={n_forwards} prepass={prepass_forwards} "
                  f"{seconds * 1000:.0f}ms", flush=True)
        return out

    engine_mod.VLNInferenceEngine.llm_generate_batch = llm_generate_batch


# --- CLI ----------------------------------------------------------------------------------
def output_dir_from(rest: Sequence[str]) -> Optional[str]:
    """``--output_dir X`` / ``--output_dir=X`` from the pass-through arguments."""
    for i, a in enumerate(rest):
        if a == "--output_dir" and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith("--output_dir="):
            return a.split("=", 1)[1]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_prune",
        description="lightnav.cli.eval_habitat with the visual history pruned to k frames' worth of "
        "tokens under one selection rule (eager vLLM, masked SDPA on every layer). Every other "
        "argument is passed through.",
    )
    p.add_argument("--prune-method", choices=METHODS, required=True)
    p.add_argument("--prune-k", type=int, required=True, help="budget in frames' worth (B = k x tokens per frame; 0 hides the whole visual history)")
    p.add_argument("--prune-tokens-per-frame", type=int, default=TOKENS_PER_FRAME,
                   help="history tokens one frame is worth (default: the fast tier's 14 = 28 per 2-frame tubelet)")
    p.add_argument("--prune-current-frames", type=int, choices=(1, 2), default=2,
                   help="1: feed the current frame doubled as the current tubelet (upstream's K = 1 layout) instead of (t-1, t)")
    p.add_argument("--prune-pivots", type=int, default=N_PIVOTS, help="DART pivots")
    p.add_argument("--prune-seed", type=int, default=SEED, help="random selector seed")
    p.add_argument("--prune-log-dir", default=None, help="per-episode records (default <output_dir>/prune)")
    p.add_argument("--prune-debug-steps", type=int, default=0, help="Print the first N inferences' bookkeeping.")
    return p


def main(argv: List[str] | None = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    state = PruneState(args.prune_method, args.prune_k, args.prune_tokens_per_frame, args.prune_pivots,
                       args.prune_seed, args.prune_debug_steps, args.prune_current_frames)
    log_dir = args.prune_log_dir or (os.path.join(output_dir_from(rest), "prune") if output_dir_from(rest) else None)
    if log_dir:
        state.log_dir = Path(log_dir)
        state.log_dir.mkdir(parents=True, exist_ok=True)
        (state.log_dir / "meta.json").write_text(json.dumps(state.describe(), indent=2) + "\n")
    install_visual_prune(state)
    print(f"[prune] installed: {state.method} k={state.k} budget={state.budget} tokens, current frames={state.current_frames}, ledger per episode, "
          f"{'scoring pre-pass, ' if state.method in PREPASS_METHODS else ''}eager vLLM, masked SDPA on every layer"
          + (f", records -> {state.log_dir}" if state.log_dir else ""), flush=True)
    from lightnav.cli.eval_habitat import main as upstream_main

    return upstream_main(rest)


if __name__ == "__main__":
    sys.exit(main())
