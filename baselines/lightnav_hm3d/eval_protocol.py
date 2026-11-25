"""Run ``lightnav.cli.eval_habitat`` under a staged pruning protocol at a whole-model token budget.

The protocol study (paper/protocol) holds ``k`` frames' worth of history visual tokens
**averaged over all decoder layers** and compares how the layers share them. LongNav-R1 does it on
its KV cache, Uni-NaVid in ``baselines/uninavid_hm3d/prune.py``; this wrapper is LightNav-0's
(36 layers, 14 tokens = one history frame's worth: a fast-tier tubelet is 2 frames in 28 tokens),
for the two protocols run on it:

* ``pyramiddrop`` -- the baseline, no window: the whole history is in scope. Attention stages that
  hide from layers 0 / 9 / 18 / 27 (quarter depths, as 0/8/16/24 of 32 on Uni-NaVid and 0/7/14/21
  of 28 on LongNav-R1), each keeping the tokens the last prompt token attends to most (mean over
  the heads). Stage 0 is ranked by the layer-0 row in a scoring pre-pass (``max_tokens=1``, as the
  flat FastV arm of ``eval_prune.py``); stage ``s >= 1`` by the row of the layer below its first
  layer (8 / 17 / 26), read inside the real prefill.
* ``uniform`` -- the newest 32 frames in scope (the current pair plus 30 history frames; older
  visual tokens are hidden on every layer while their timestamps and vision markers stay in the
  prompt -- the "old blocks kept" window of the context ablation), ONE seeded uniform random draw
  at layer 0 that every layer below the cut holds, and a **visual-blind** last stage: from
  ``--blind-layer`` (26, LightNav-0's information horizon) every ``<|video_pad|>`` column is
  hidden, the current pair included.

Shared mechanics, mirroring Uni-NaVid's ``TokenPruner``:

* **Whole-model budget.** ``B = k x 14`` bounds the history tokens a query attends to averaged
  over the 36 layers. Every inference ``model_stage_budgets`` solves the constant ratio ``lambda``
  of the schedule (stage ``s`` keeps ``lambda`` of stage ``s - 1``) so that the layer-weighted
  mean equals ``B``; the schedule shrinks from what stage 0 can draw from now (its survivors plus
  the tokens that just entered: the ``pool`` anchor of the ``_mp<k>`` arms). Uniform: one
  scheduled stage of 26 layers and a 10-layer tail holding nothing, so layers 0-25 hold
  ``36 B / 26`` tokens. Nothing is pruned while the scope is under budget.
* **Permanent deletion, nested by stage.** A token dropped at stage ``s`` is recorded in the
  ledgers of stage ``s`` and of every deeper stage, keyed by tubelet identity
  ``(frame0, frame1, pool, cell)`` as in ``eval_prune.py`` (the prompt is rebuilt and re-prefilled
  every step; a tubelet recurs every other step). The current 2-frame tubelet (112 tokens) is
  never a candidate and is not counted in the budget.
* **Masked attention.** Eager vLLM with ``FlashAttentionImpl.forward`` replaced on every decoder
  layer by the exact masked kernel of ``eval_blind.py``; a layer hides the columns of the deepest
  stage that starts at or below it, on the prefill and on every decode forward.

Per inference one record goes to ``<output_dir>/prune/episode_NNN.jsonl`` (the schema of
``eval_prune.py`` plus ``stage_budgets``, ``stage_kept`` = history tokens visible per stage,
``held_tokens`` = their layer-weighted mean, ``in_scope``); ``n_kept`` / ``bits`` describe the
deepest stage that holds history (uniform: stage 0).

Runs in the LightNav client runtime (``baselines/.envs/lightnav``) with
``PYTHONPATH=baselines/LightNav-0/src``; ``baselines/run_lightnav.sh`` uses it when ``PROTOCOL``
is set:

    python baselines/lightnav_hm3d/eval_protocol.py --protocol uniform --prune-k 8 <eval_habitat args...>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

try:
    from baselines.lightnav_hm3d.eval_blind import (
        _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )
    from baselines.lightnav_hm3d.eval_prune import (
        SEED, TOKENS_PER_FRAME, Tubelet, check_generate_counts, decision_row, fastv_select, output_dir_from,
        pack_bits, random_select, selection_seed, split_history, token_key, tubelet_layout,
    )
except ImportError:  # run as a script: sys.path[0] is this directory
    from eval_blind import (  # type: ignore
        _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )
    from eval_prune import (  # type: ignore
        SEED, TOKENS_PER_FRAME, Tubelet, check_generate_counts, decision_row, fastv_select, output_dir_from,
        pack_bits, random_select, selection_seed, split_history, token_key, tubelet_layout,
    )

PROTOCOLS = ("pyramiddrop", "uniform")
PYRAMIDDROP_STARTS = (0, 9, 18, 27)  # first layer of each attention stage (36 layers: quarter depths)
UNIFORM_BLIND_LAYER = 26  # first visual-blind layer: LightNav-0's information horizon (paper/info_horizon)
UNIFORM_WINDOW = 32  # frames in scope, the current pair included
Key = Tuple[int, int, int, int]


def model_stage_budgets(v0: int, budget: int, widths: Sequence[int], tail: Optional[Tuple[int, int]] = None) -> List[int]:
    """Nested budgets ``round(V0 * lambda^(s+1))`` whose LAYER-WEIGHTED MEAN over the model is
    ``budget``: ``widths`` = layers per scheduled stage, ``tail`` = (layers, tokens) of a last stage
    with a fixed budget. lambda in [0, 1]: 1 while the unpruned history is under budget. A copy of
    ``baselines/uninavid_hm3d/prune.py::model_stage_budgets`` (checked against it in the tests)."""
    if not widths:
        raise ValueError("need at least one scheduled stage")
    tail_w, tail_n = (0, 0) if tail is None else (int(tail[0]), min(int(tail[1]), int(v0)))
    target = int(budget) * (sum(widths) + tail_w)

    def held(lam: float) -> float:
        # The tail is nested in the last scheduled stage: it holds min(its budget, that stage).
        return v0 * sum(w * lam ** (s + 1) for s, w in enumerate(widths)) + tail_w * min(tail_n, v0 * lam ** len(widths))

    if v0 <= 0 or held(1.0) <= target:
        return [int(v0)] * len(widths)
    lo, hi = 0.0, 1.0
    if target > 0:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if held(mid) < target else (lo, mid)
    lam = 0.5 * (lo + hi) if target > 0 else 0.0
    out = [int(round(v0 * lam ** (s + 1))) for s in range(len(widths))]
    for s in range(len(out) - 2, -1, -1):
        out[s] = max(out[s], out[s + 1])
    return out


# --- ledger and prompt layout -------------------------------------------------------------
class StageLedger:
    """Tokens deleted at each stage, by tubelet identity; the sets are nested (stage ``s`` is a
    subset of stage ``s + 1``)."""

    def __init__(self, n_stages: int):
        if n_stages < 1:
            raise ValueError("need at least one stage")
        self.deleted: List[set] = [set() for _ in range(n_stages)]

    def delete(self, stage: int, keys: Sequence[Key]) -> None:
        for s in range(stage, len(self.deleted)):
            self.deleted[s].update(keys)


def in_scope(tubelet: Tubelet, current_abs: int, window_frames: Optional[int]) -> bool:
    """A history tubelet is in scope iff both of its frames are among the ``window_frames`` newest
    frames of the episode (the current one included); no window = the whole history."""
    return window_frames is None or tubelet.f0 >= current_abs - (window_frames - 1)


class StepLayout(NamedTuple):
    pad_pos: List[int]  # prompt index of each visual rank
    history: List[Tubelet]
    scope: List[bool]  # per history tubelet: in the window
    ranks: List[int]  # visual rank of every in-scope history token, prompt order
    keys: List[Key]  # its ledger key
    out_ranks: List[int]  # visual ranks of the out-of-scope history tokens (hidden on every layer)
    prompt_len: int


def step_layout(tubelets: Sequence[Tubelet], current_abs: int, pad_pos: Sequence[int], prompt_len: int,
                window_frames: Optional[int]) -> StepLayout:
    hist, _ = split_history(tubelets, current_abs)
    scope, ranks, keys, out_ranks = [], [], [], []
    for tub in hist:
        inside = in_scope(tub, current_abs, window_frames)
        scope.append(inside)
        for c in range(tub.n_tokens):
            if inside:
                ranks.append(tub.start + c)
                keys.append(token_key(tub, c))
            else:
                out_ranks.append(tub.start + c)
    return StepLayout(list(pad_pos), list(hist), scope, ranks, keys, out_ranks, int(prompt_len))


def stage_pool(layout: StepLayout, ledger: StageLedger, stage: int) -> List[int]:
    """Positions (into ``layout.ranks``) of the in-scope history tokens alive at ``stage``."""
    gone = ledger.deleted[stage]
    return [i for i, key in enumerate(layout.keys) if key not in gone]


def stage_hides(layout: StepLayout, ledger: StageLedger, blind: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """Per stage, the bool mask over the prompt of the columns it hides: the out-of-scope history,
    the tokens its ledger holds, and for a blind last stage (``blind`` = every visual column) those."""
    out = []
    n_stages = len(ledger.deleted)
    out_cols = [layout.pad_pos[rank] for rank in layout.out_ranks]
    for s in range(n_stages):
        gone = ledger.deleted[s]
        cols = out_cols + [layout.pad_pos[rank] for rank, key in zip(layout.ranks, layout.keys) if key in gone]
        hide = torch.zeros(layout.prompt_len, dtype=torch.bool)
        if cols:
            hide[torch.as_tensor(cols, dtype=torch.long)] = True  # one indexed write, not one per token
        if blind is not None and s == n_stages - 1:
            hide |= blind
        out.append(hide)
    return out


def kept_per_tubelet(layout: StepLayout, hide: torch.Tensor) -> Tuple[List[int], List[bool]]:
    """(tokens visible per history tubelet, keep bits over the history tokens in prompt order)
    under one stage's hide mask."""
    hidden = hide.tolist()
    bits: List[bool] = []
    n_kept: List[int] = []
    for tub in layout.history:
        b = [not hidden[layout.pad_pos[tub.start + c]] for c in range(tub.n_tokens)]
        bits.extend(b)
        n_kept.append(int(sum(b)))
    return n_kept, bits


# --- shared state -------------------------------------------------------------------------
class ProtocolState:
    def __init__(self, protocol: str, k: int, tokens_per_frame: int = TOKENS_PER_FRAME, seed: int = SEED,
                 blind_layer: int = UNIFORM_BLIND_LAYER, window_frames: Optional[int] = None,
                 stage_starts: Optional[Sequence[int]] = None, debug_steps: int = 0):
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
        if k < 1 or tokens_per_frame < 1:
            raise ValueError(f"prune k and tokens per frame must be >= 1, got k={k}, tpf={tokens_per_frame}")
        self.protocol = protocol
        self.k = int(k)
        self.tokens_per_frame = int(tokens_per_frame)
        self.budget = self.k * self.tokens_per_frame
        self.seed = int(seed)
        self.debug_steps = debug_steps
        self.blind = protocol == "uniform"  # the last stage hides every visual column
        if self.blind:
            if stage_starts is not None:
                raise ValueError("uniform has one random stage at layer 0 and its blind cut; no --stage-starts")
            if blind_layer < 1:
                raise ValueError(f"the blind layer must be >= 1, got {blind_layer}")
            self.starts = [0, int(blind_layer)]
            self.window_frames = UNIFORM_WINDOW if window_frames is None else int(window_frames)
        else:
            self.starts = [int(s) for s in (PYRAMIDDROP_STARTS if stage_starts is None else stage_starts)]
            if self.starts[0] != 0 or any(b <= a for a, b in zip(self.starts, self.starts[1:])):
                raise ValueError(f"stage starts must begin at layer 0 and increase, got {self.starts}")
            self.window_frames = 0 if window_frames is None else int(window_frames)
        if self.window_frames < 0 or self.window_frames == 1:
            raise ValueError(f"window frames must be 0 (off) or >= 2, got {self.window_frames}")
        self.window: Optional[int] = self.window_frames or None
        self.n_stages = len(self.starts)
        # Stages 1.. of pyramiddrop are scored by the attention row of the layer below their first layer.
        self.score_layer: Dict[int, int] = {} if self.blind else {start - 1: s for s, start in enumerate(self.starts) if s}
        self.n_layers: Optional[int] = None
        self.widths: Optional[List[int]] = None
        self.strict = False
        # per generate call
        self.layout: Optional[StepLayout] = None
        self.hides: Optional[List[torch.Tensor]] = None  # None = no generate in flight
        self._hides_dev: Dict[int, torch.Tensor] = {}
        self.blind_cols: Optional[torch.Tensor] = None
        self.budgets: List[int] = []
        self.selected: List[bool] = []
        self.capture = False  # scoring pre-pass: record the layer-0 row, select nothing
        self.row: Optional[torch.Tensor] = None
        self.calls = 0
        self.generates = 0
        # per episode
        self.ledger = StageLedger(self.n_stages)
        self.episode = -1
        self.segments: Optional[List[Tuple[List[int], int]]] = None
        self.current_abs: Optional[int] = None
        self.log_dir: Optional[Path] = None
        self._log = None

    def set_layers(self, n_layers: int) -> None:
        if self.starts[-1] > n_layers or (not self.blind and self.starts[-1] >= n_layers):
            raise ValueError(f"stage starts {self.starts} do not fit a {n_layers}-layer decoder")
        self.n_layers = int(n_layers)
        bounds = self.starts + [self.n_layers]
        self.widths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]

    def stage_of(self, layer: int) -> int:
        """The deepest stage that starts at or below ``layer``."""
        return max(s for s, start in enumerate(self.starts) if start <= layer)

    def solve_budgets(self, v0: int) -> List[int]:
        """This inference's stage budgets from the tokens stage 0 can draw from."""
        if self.blind:
            return model_stage_budgets(v0, self.budget, self.widths[:1], tail=(self.widths[1], 0)) + [0]
        return model_stage_budgets(v0, self.budget, self.widths)

    def hide_on(self, stage: int, device: torch.device) -> torch.Tensor:
        cached = self._hides_dev.get(stage)
        if cached is None or cached.device != device:
            cached = self._hides_dev[stage] = self.hides[stage].to(device)
        return cached

    def refresh(self) -> None:
        self.hides = stage_hides(self.layout, self.ledger, self.blind_cols)
        self._hides_dev = {}

    def end_generate(self) -> None:
        self.hides = None
        self._hides_dev = {}
        self.capture = False

    def select(self, stage: int, choose: Callable[[int, int], torch.Tensor]) -> int:
        """Keep ``budgets[stage]`` of the stage's pool (``choose(n_pool, budget)`` -> sorted pool
        positions) and delete the rest there and at every deeper stage; returns how many were deleted."""
        pool = stage_pool(self.layout, self.ledger, stage)
        budget = self.budgets[stage]
        n_dropped = 0
        if len(pool) > budget:
            keep = set(int(i) for i in choose(len(pool), budget).tolist())
            if len(keep) != budget:
                raise RuntimeError(f"stage {stage} kept {len(keep)} of {len(pool)} tokens, budget {budget}")
            drop = [self.layout.keys[i] for j, i in enumerate(pool) if j not in keep]
            self.ledger.delete(stage, drop)
            n_dropped = len(drop)
            self.refresh()
        self.selected[stage] = True
        return n_dropped

    def select_attention(self, stage: int, row: torch.Tensor) -> int:
        """Top-budget of the stage's pool by the attention ``row`` over the prompt."""
        pos = torch.as_tensor(self.layout.pad_pos)

        def choose(n_pool: int, budget: int) -> torch.Tensor:
            pool = stage_pool(self.layout, self.ledger, stage)
            cols = pos[torch.as_tensor([self.layout.ranks[i] for i in pool], dtype=torch.long)]
            return fastv_select(row[cols], budget)

        return self.select(stage, choose)

    def reset_episode(self) -> None:
        self.ledger = StageLedger(self.n_stages)
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
        return {"protocol": self.protocol, "k": self.k, "tokens_per_frame": self.tokens_per_frame,
                "budget": self.budget, "budget_scope": "model", "schedule_anchor": "pool",
                "stage_starts": list(self.starts),
                "stage_selectors": ["random", "blind"] if self.blind else ["attn"] * self.n_stages,
                "blind_layer": self.starts[-1] if self.blind else None,
                "window_frames": self.window, "window": "old blocks kept" if self.window else None,
                "seed": self.seed if self.blind else None, "prepass": not self.blind,
                "query": None if self.blind else "last prompt token, mean over heads",
                "ledger": "per stage, nested, tubelet (f0, f1, pool, cell)", "current_tubelet_exempt": True}


# --- vLLM hook ----------------------------------------------------------------------------
def make_protocol_forward(state: ProtocolState, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """``FlashAttentionImpl.forward`` replacement: ``eval_blind``'s masked kernel with the hidden
    set of the layer's stage. On the prefill a stage's scoring layer also reads the last prompt
    token's attention row and selects that stage; a scoring pre-pass records the layer-0 row."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        attn_type = getattr(self, "attn_type", None)
        is_decoder = attn_type is None or str(getattr(attn_type, "name", attn_type)).split(".")[-1] == "DECODER"
        if attn_metadata is None or state.hides is None or not is_decoder:
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
        prompt_len = state.layout.prompt_len
        if n_query < 1 or seq_len < n_query:
            raise RuntimeError(f"bad lengths: n_query={n_query} seq_len={seq_len}")
        if seq_len > prompt_len + 4096:
            raise RuntimeError(f"seq_len {seq_len} far beyond the prompt ({prompt_len} tokens)")

        index = layer_index(layer.layer_name)
        hide = state.hide_on(state.stage_of(index), query.device)
        key_cache, value_cache = kv_cache.unbind(0)
        row = attn_metadata.block_table[0]
        k = gather_kv(key_cache, row, seq_len)
        v = gather_kv(value_cache, row, seq_len)
        mask = attention_mask(n_query, seq_len, hide, device=query.device)
        out = masked_attention(query[:n_query], k, v, float(self.scale), mask)
        output[:n_query].copy_(out.to(output.dtype))
        state.calls += 1
        if n_query > 1:  # prefill
            score_stage = None
            if state.capture:
                score_stage = 0 if index == 0 else None
            elif index in state.score_layer and not state.selected[state.score_layer[index]]:
                score_stage = state.score_layer[index]
            if score_stage is not None:
                if seq_len != prompt_len:
                    raise RuntimeError(f"scoring prefill seq_len {seq_len} != prompt length {prompt_len}")
                attn_row = decision_row(query[n_query - 1], k, float(self.scale), mask[-1]).cpu()
                if state.capture:
                    if state.row is not None:
                        raise RuntimeError("layer-0 prefill captured twice in one pre-pass (chunked prefill?)")
                    state.row = attn_row
                else:
                    if not all(state.selected[:score_stage]):
                        raise RuntimeError(f"stage {score_stage} reached before the stages above it selected")
                    state.select_attention(score_stage, attn_row)
        return output

    forward._prune_method = f"protocol:{state.protocol}"  # type: ignore[attr-defined]
    return forward


# --- installation -------------------------------------------------------------------------
def install_protocol(state: ProtocolState) -> None:
    """Patch vLLM's flash-attention kernel, LightNav's engine loader, sampler, generate and reset."""
    os.environ["VLN_VLLM_ENFORCE_EAGER"] = "1"
    import lightnav.inference.engine as engine_mod
    import lightnav.inference.model as model_mod
    import lightnav.inference.samples as samples_mod
    from vllm.v1.attention.backends import flash_attn as fa

    original_forward = fa.FlashAttentionImpl.forward
    if getattr(original_forward, "_blind_layer", None) is not None or getattr(original_forward, "_prune_method", None) is not None:
        raise RuntimeError("an attention hook is already installed")
    fa.FlashAttentionImpl.forward = make_protocol_forward(state, original_forward)

    original_segments = samples_mod.slowfast_video_segments
    if getattr(original_segments, "_context_frames", None) is not None:
        raise RuntimeError("the context window wrapper is installed; PROTOCOL and CONTEXT_FRAMES are not combined")

    def recording_segments(episode_frames, current_abs, n_available, tiers):
        total = int(episode_frames.shape[0])
        if current_abs != total - 1 or n_available != total:
            raise AssertionError(f"unexpected SlowFast call convention: total={total} current_abs={current_abs} "
                                 f"n_available={n_available}")
        segs = original_segments(episode_frames, current_abs, n_available, tiers)
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
        from vllm.config.compilation import CompilationMode, CUDAGraphMode
        if cfg.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(f"CUDA graphs enabled ({cfg.compilation_config.cudagraph_mode}); hooks would not run")
        if cfg.compilation_config.mode != CompilationMode.NONE:
            raise RuntimeError(f"torch.compile enabled ({cfg.compilation_config.mode}); the hooks would not run")
        if getattr(cfg.scheduler_config, "enable_chunked_prefill", False):
            raise RuntimeError("chunked prefill is on; the stages need one prefill forward")
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
        state.set_layers(n_layers)
        state.strict = True
        print(f"[protocol] {state.protocol} k={state.k} budget={state.budget} tokens averaged over {n_layers} layers "
              f"stage_starts={state.starts} widths={state.widths} window={state.window} "
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
            raise RuntimeError(f"the protocol wrapper handles one request per generate call, got {len(items)}")
        if state.segments is None or state.current_abs is None:
            raise RuntimeError("generate called before the SlowFast sampler recorded the segment layout")
        if state.episode < 0:
            raise RuntimeError("generate called before the first episode reset")
        t0 = time.monotonic()
        ids = list(items[0].prompt_ids)
        visual = visual_positions(ids)
        pads = torch.nonzero(visual).flatten()
        grids = items[0].video_grid_thw
        if grids is None:
            raise RuntimeError("prompt carries no video grid")
        tubelets = tubelet_layout(state.segments, [[int(x) for x in row] for row in grids.tolist()])
        n_pads = sum(t.n_tokens for t in tubelets)
        if n_pads != int(pads.numel()):
            raise RuntimeError(f"tubelet layout holds {n_pads} tokens but the prompt has {int(pads.numel())} <|video_pad|>")
        segments, current_abs = state.segments, state.current_abs
        state.segments = None  # one generate per sampled layout
        layout = step_layout(tubelets, current_abs, pads.tolist(), len(ids), state.window)
        state.layout = layout
        state.blind_cols = visual.clone() if state.blind else None
        n_pool = len(stage_pool(layout, state.ledger, 0))
        state.budgets = state.solve_budgets(n_pool)
        state.selected = [False] * state.n_stages
        state.refresh()
        state.calls = 0
        n_dropped, prepass_forwards = 0, 0
        try:
            if state.blind:
                g = torch.Generator().manual_seed(selection_seed(state.seed, f"episode_{state.episode:03d}", current_abs))
                n_dropped = state.select(0, lambda n, budget: random_select(n, budget, g))
                state.selected[1] = True  # the blind stage selects nothing: it holds no visual token
            elif n_pool > state.budgets[0]:
                state.capture, state.row = True, None
                try:
                    original_generate(self, items, 1)
                finally:
                    state.capture = False
                if state.row is None:
                    raise RuntimeError("the pre-pass did not capture the layer-0 prefill")
                prepass_forwards = check_generate_counts(state, len(ids), int(visual.sum()))
                state.calls = 0
                n_dropped = state.select_attention(0, state.row)
            else:
                state.selected[0] = True
            out = original_generate(self, items, max_new_tokens)
            n_forwards = check_generate_counts(state, len(ids), int(visual.sum()))
            if not all(state.selected):
                raise RuntimeError(f"stages {[s for s, ok in enumerate(state.selected) if not ok]} did not select "
                                   "(no full prefill forward?)")
            hides = state.hides
        finally:
            state.end_generate()
        seconds = time.monotonic() - t0
        state.generates += 1

        history_cols = torch.as_tensor([layout.pad_pos[t.start + c] for t in layout.history for c in range(t.n_tokens)],
                                       dtype=torch.long)
        stage_kept = [int((~h[history_cols]).sum()) for h in hides]
        held = sum(w * n for w, n in zip(state.widths, stage_kept)) / state.n_layers
        deepest = state.n_stages - (2 if state.blind else 1)
        n_kept, kept_bits = kept_per_tubelet(layout, hides[deepest])
        state.log_record({
            "step": current_abs, "frames": [t.f1 for t in layout.history], "n_tokens": [t.n_tokens for t in layout.history],
            "n_kept": n_kept, "bits": pack_bits(kept_bits), "budget": state.budget, "n_pool": n_pool,
            "in_scope": len(layout.ranks), "stage_budgets": list(state.budgets), "stage_kept": stage_kept,
            "held_tokens": held, "n_dropped": n_dropped, "ledger": [len(d) for d in state.ledger.deleted],
            "prompt_tokens": len(ids), "n_visual": int(visual.sum()),
            "tubelets": [[t.f0, t.f1] for t in layout.history], "pools": [t.pool for t in layout.history],
            "forwards": n_forwards, "prepass_forwards": prepass_forwards, "seconds": seconds,
            "episode_label": f"episode_{state.episode:03d}", "n_segments": len(segments),
        })
        if state.generates <= state.debug_steps:
            print(f"[protocol] episode={state.episode} step={current_abs} prompt_len={len(ids)} visual={int(visual.sum())} "
                  f"history_tubelets={len(layout.history)} in_scope={len(layout.ranks)} pool={n_pool} "
                  f"budgets={state.budgets} stage_kept={stage_kept} held={held:.1f} forwards={n_forwards} "
                  f"prepass={prepass_forwards} {seconds * 1000:.0f}ms", flush=True)
        return out

    engine_mod.VLNInferenceEngine.llm_generate_batch = llm_generate_batch


# --- CLI ----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_protocol",
        description="lightnav.cli.eval_habitat under a staged pruning protocol at a whole-model budget of k "
        "frames' worth of history visual tokens (eager vLLM, masked SDPA on every layer). Every other "
        "argument is passed through.",
    )
    p.add_argument("--protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--prune-k", type=int, required=True, help="whole-model budget in frames' worth (B = k x tokens per frame)")
    p.add_argument("--prune-tokens-per-frame", type=int, default=TOKENS_PER_FRAME,
                   help="history tokens one frame is worth (default: the fast tier's 14 = 28 per 2-frame tubelet)")
    p.add_argument("--blind-layer", type=int, default=UNIFORM_BLIND_LAYER,
                   help="uniform: first visual-blind decoder layer (the model depth = no blind stage)")
    p.add_argument("--window-frames", type=int, default=None,
                   help=f"frames in scope, current pair included; 0 = whole history (default: {UNIFORM_WINDOW} for uniform, 0 for pyramiddrop)")
    p.add_argument("--stage-starts", default=None,
                   help=f"pyramiddrop: comma-separated first layer of each stage (default {','.join(map(str, PYRAMIDDROP_STARTS))})")
    p.add_argument("--prune-seed", type=int, default=SEED, help="seed of uniform's random draw")
    p.add_argument("--prune-log-dir", default=None, help="per-episode records (default <output_dir>/prune)")
    p.add_argument("--prune-debug-steps", type=int, default=0, help="Print the first N inferences' bookkeeping.")
    return p


def main(argv: List[str] | None = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    starts = None if args.stage_starts is None else [int(s) for s in args.stage_starts.split(",")]
    state = ProtocolState(args.protocol, args.prune_k, args.prune_tokens_per_frame, args.prune_seed,
                          args.blind_layer, args.window_frames, starts, args.prune_debug_steps)
    log_dir = args.prune_log_dir or (os.path.join(output_dir_from(rest), "prune") if output_dir_from(rest) else None)
    if log_dir:
        state.log_dir = Path(log_dir)
        state.log_dir.mkdir(parents=True, exist_ok=True)
        (state.log_dir / "meta.json").write_text(json.dumps(state.describe(), indent=2) + "\n")
    install_protocol(state)
    print(f"[protocol] installed: {state.protocol} k={state.k} budget={state.budget} tokens (whole model), "
          f"stage starts {state.starts}, window {state.window or 'off'}, ledger per episode, eager vLLM, masked SDPA on "
          "every layer" + (f", records -> {state.log_dir}" if state.log_dir else ""), flush=True)
    from lightnav.cli.eval_habitat import main as upstream_main

    return upstream_main(rest)


if __name__ == "__main__":
    sys.exit(main())
