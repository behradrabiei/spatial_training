"""Run ``lightnav.cli.eval_habitat`` with every visual token hidden from decoder layer ``L`` on.

The information-horizon experiment (paper/info_horizon) closes the visual tokens' key columns
for *every* query on decoder layers ``>= L`` (history and current frame alike; separators,
timestamps and special tokens stay visible), leaves layers ``< L`` untouched and never moves a
position. LongNav-R1 and Uni-NaVid do this on the additive attention mask of an eager
transformers forward. LightNav-0 decodes inside vLLM (``--backend vllm_local``): its attention
is ``FlashAttentionImpl.forward`` over a paged KV cache, which takes no mask, and the default
engine compiles the model and replays decode as a CUDA graph, so a Python hook would not run.

This wrapper therefore, without touching the upstream clone:

* forces eager vLLM (``VLN_VLLM_ENFORCE_EAGER=1``: no torch.compile, no CUDA graphs);
* replaces ``FlashAttentionImpl.forward`` for **every** decoder layer with an exact masked
  attention (torch SDPA over the K/V gathered from the paged cache, causal mask, GQA): layers
  ``< L`` keep the plain causal mask, layers ``>= L`` additionally drop every ``<|video_pad|>``
  column. One kernel path for every arm, so ``L = n_layers`` (nothing hidden) is the matched
  reference and the only thing that differs between arms is the hidden set;
* records, before each ``vllm.LLM.generate`` call, which prompt positions are
  ``<|video_pad|>`` tokens (one request at a time, so KV position ``j`` is prompt index ``j``;
  generated tokens are never visual) and checks after it that the masked kernel ran on every
  layer of every forward.

Qwen3-VL also injects ViT features ("deepstack") into the visual positions after decoder
layers 0-2; those positions are exactly the hidden columns, so for ``L >= 4`` the semantics match
the other two models. Prefix caching stays on: only the text prefix that precedes every visual
token can hit it, and its K/V do not depend on the hidden set.

Runs in the LightNav client runtime (``baselines/.envs/lightnav``) with
``PYTHONPATH=baselines/LightNav-0/src``; ``baselines/run_lightnav.sh`` uses it when
``BLIND_LAYER`` is set:

    python baselines/lightnav_hm3d/eval_blind.py --blind-layer 16 <eval_habitat args...>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

VIDEO_PAD_ID = 151656  # <|video_pad|> in the LightNav-0 (Qwen3-VL) tokenizer
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class BlindState:
    """What the hooks share: the blind layer, the current prompt's visual mask, call counters."""

    def __init__(self, layer: int, debug_steps: int = 0):
        if layer < 0:
            raise ValueError(f"blind layer must be >= 0, got {layer}")
        self.layer = layer
        self.debug_steps = debug_steps
        self.n_layers: Optional[int] = None  # known once the engine is built
        self.strict = False  # True after the engine is built: every forward must be masked
        self.visual: Optional[torch.Tensor] = None  # bool [prompt_len], set per generate call
        self._visual_dev: Optional[torch.Tensor] = None
        self.calls = 0
        self.hidden_calls = 0
        self.generates = 0

    def visual_on(self, device: torch.device) -> torch.Tensor:
        if self._visual_dev is None or self._visual_dev.device != device:
            self._visual_dev = self.visual.to(device)
        return self._visual_dev

    def set_visual(self, visual: Optional[torch.Tensor]) -> None:
        self.visual = visual
        self._visual_dev = None


def visual_positions(prompt_ids: Sequence[int]) -> torch.Tensor:
    """Bool mask of the ``<|video_pad|>`` positions of a prompt (vision_start/end and the
    ``<t seconds>`` text are not visual)."""
    ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
    return ids == VIDEO_PAD_ID


def layer_index(layer_name: str) -> int:
    """Decoder index from a vLLM attention layer name (``...layers.<i>.self_attn.attn``)."""
    found = _LAYER_RE.findall(layer_name)
    if len(found) != 1:
        raise ValueError(f"cannot read a layer index from {layer_name!r}")
    return int(found[0])


def gather_kv(cache: torch.Tensor, block_table_row: torch.Tensor, seq_len: int) -> torch.Tensor:
    """``[num_blocks, block_size, heads, dim]`` paged cache -> ``[seq_len, heads, dim]``."""
    block_size = int(cache.shape[1])
    n_blocks = -(-seq_len // block_size)
    blocks = block_table_row[:n_blocks].to(torch.long)
    return cache.index_select(0, blocks).reshape(-1, *cache.shape[2:])[:seq_len]


def attention_mask(
    n_query: int, seq_len: int, hide: Optional[torch.Tensor] = None, device: Any = None
) -> torch.Tensor:
    """Bool ``[n_query, seq_len]`` (True = may attend): causal for the last ``n_query``
    positions of a ``seq_len`` sequence, minus the ``hide`` columns (padded with False past
    its length, so generated tokens are never hidden)."""
    q_pos = torch.arange(seq_len - n_query, seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    mask = k_pos[None, :] <= q_pos[:, None]
    if hide is not None:
        cols = torch.zeros(seq_len, dtype=torch.bool, device=device)
        n = min(int(hide.numel()), seq_len)
        cols[:n] = hide[:n].to(device)
        mask = mask & ~cols[None, :]
    return mask


def masked_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, mask: torch.Tensor
) -> torch.Tensor:
    """``query [Q, H, D]``, ``key``/``value [S, KVH, D]`` (GQA), ``mask [Q, S]`` -> ``[Q, H, D]``."""
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[None, None], scale=scale, enable_gqa=q.shape[1] != k.shape[1]
    )
    return out[0].transpose(0, 1)


def make_blind_forward(state: BlindState, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """The ``FlashAttentionImpl.forward`` replacement (same signature as vLLM 0.19's).

    Falls through to ``original`` only for the engine's warm-up forwards (no attention metadata
    or no generate call in flight before the engine is built); afterwards every forward goes
    through the masked kernel or raises.
    """

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        attn_type = getattr(self, "attn_type", None)
        is_decoder = attn_type is None or str(getattr(attn_type, "name", attn_type)).split(".")[-1] == "DECODER"
        if attn_metadata is None or state.visual is None or not is_decoder:
            if state.strict and attn_metadata is not None and is_decoder:
                raise RuntimeError("decoder attention forward outside a hooked generate call")
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output=output,
                            output_scale=output_scale, output_block_scale=output_block_scale)
        if output is None:
            raise RuntimeError("vLLM must pass an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by the blind kernel")
        if str(getattr(self, "kv_cache_dtype", "auto")).startswith("fp8"):
            raise NotImplementedError("fp8 KV cache is not supported by the blind kernel")
        if tuple(getattr(self, "sliding_window", (-1, -1))) != (-1, -1) or getattr(self, "alibi_slopes", None) is not None \
                or getattr(self, "logits_soft_cap", None):
            raise NotImplementedError("sliding window / alibi / soft cap are not supported by the blind kernel")
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
        if seq_len > int(state.visual.numel()) + 4096:
            raise RuntimeError(f"seq_len {seq_len} far beyond the prompt ({int(state.visual.numel())} tokens)")

        hide = state.visual_on(query.device) if layer_index(layer.layer_name) >= state.layer else None
        key_cache, value_cache = kv_cache.unbind(0)
        row = attn_metadata.block_table[0]
        k = gather_kv(key_cache, row, seq_len)
        v = gather_kv(value_cache, row, seq_len)
        mask = attention_mask(n_query, seq_len, hide, device=query.device)
        out = masked_attention(query[:n_query], k, v, float(self.scale), mask)
        output[:n_query].copy_(out.to(output.dtype))
        state.calls += 1
        if hide is not None:
            state.hidden_calls += 1
        return output

    forward._blind_layer = state.layer  # type: ignore[attr-defined]
    return forward


def check_generate_counts(state: BlindState, prompt_len: int, n_visual: int) -> int:
    """After one generate call: the masked kernel ran on every layer of every forward, hiding
    on exactly the layers ``>= L``. Returns the number of model forwards."""
    n = state.n_layers
    if not n:
        raise RuntimeError("engine layer count unknown (engine hook did not run)")
    if state.calls == 0 or state.calls % n:
        raise RuntimeError(f"masked attention ran {state.calls} times, not a multiple of {n} layers "
                           "(compiled / CUDA-graph path?)")
    expect_hidden = state.calls // n * max(0, n - state.layer)
    if state.hidden_calls != expect_hidden:
        raise RuntimeError(f"hidden on {state.hidden_calls} layer calls, expected {expect_hidden}")
    if n_visual == 0:
        raise RuntimeError(f"prompt of {prompt_len} tokens has no <|video_pad|> token")
    return state.calls // n


def install_visual_blind(state: BlindState) -> None:
    """Patch vLLM's flash-attention kernel, LightNav's engine loader and its generate call."""
    os.environ["VLN_VLLM_ENFORCE_EAGER"] = "1"
    import lightnav.inference.engine as engine_mod
    import lightnav.inference.model as model_mod
    from vllm.v1.attention.backends import flash_attn as fa

    original_forward = fa.FlashAttentionImpl.forward
    if getattr(original_forward, "_blind_layer", None) is not None:
        raise RuntimeError("visual blind already installed")
    fa.FlashAttentionImpl.forward = make_blind_forward(state, original_forward)

    original_load = model_mod.load_vllm_local

    def load_vllm_local(config, task_type="vlnce"):
        bundle, llm = original_load(config, task_type=task_type)
        cfg = llm.llm_engine.vllm_config
        if not cfg.model_config.enforce_eager:
            raise RuntimeError("vLLM is not in eager mode (VLN_VLLM_ENFORCE_EAGER=1 required)")
        from vllm.config.compilation import CUDAGraphMode
        if cfg.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(f"CUDA graphs enabled ({cfg.compilation_config.cudagraph_mode}); hooks would not run")
        names = [n for n in cfg.compilation_config.static_forward_context if _LAYER_RE.search(n)]
        indices = sorted(layer_index(n) for n in names)
        n_layers = len(indices)
        if indices != list(range(n_layers)) or n_layers == 0:
            raise RuntimeError(f"unexpected decoder attention layers: {names}")
        expected = int(cfg.model_config.hf_text_config.num_hidden_layers)
        if n_layers != expected:
            raise RuntimeError(f"{n_layers} attention layers registered, config says {expected}")
        if state.layer > n_layers:
            raise SystemExit(f"--blind-layer {state.layer} exceeds the model depth {n_layers}")
        backend_name = type(getattr(next(iter(cfg.compilation_config.static_forward_context.values())), "impl", None)).__name__
        if backend_name != "FlashAttentionImpl":
            raise RuntimeError(f"attention backend is {backend_name}, the hook patches FlashAttentionImpl")
        state.n_layers = n_layers
        state.strict = True
        print(f"[blind] blind_layer={state.layer} n_layers={n_layers} "
              f"({'nothing hidden' if state.layer >= n_layers else f'<|video_pad|> columns hidden on layers {state.layer}..{n_layers - 1}'}) "
              f"eager={cfg.model_config.enforce_eager} cudagraph={cfg.compilation_config.cudagraph_mode.name}", flush=True)
        return bundle, llm

    model_mod.load_vllm_local = load_vllm_local

    original_generate = engine_mod.VLNInferenceEngine.llm_generate_batch

    def llm_generate_batch(self, items, max_new_tokens):
        if len(items) != 1:
            raise RuntimeError(f"the blind kernel handles one request per generate call, got {len(items)}")
        ids = list(items[0].prompt_ids)
        visual = visual_positions(ids)
        state.set_visual(visual)
        state.calls = state.hidden_calls = 0
        try:
            out = original_generate(self, items, max_new_tokens)
            n_forwards = check_generate_counts(state, len(ids), int(visual.sum()))
        finally:
            state.set_visual(None)
        state.generates += 1
        if state.generates <= state.debug_steps:
            print(f"[blind] generate={state.generates} prompt_len={len(ids)} visual={int(visual.sum())} "
                  f"forwards={n_forwards} hidden_layer_calls={state.hidden_calls}", flush=True)
        return out

    engine_mod.VLNInferenceEngine.llm_generate_batch = llm_generate_batch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_blind",
        description="lightnav.cli.eval_habitat with every visual token hidden from decoder layer L "
        "on (L = depth hides nothing but keeps the same eager masked kernel). Every other "
        "argument is passed through.",
    )
    p.add_argument("--blind-layer", type=int, required=True,
                   help="First decoder layer (0-based) without visual tokens; 0 = text only.")
    p.add_argument("--blind-debug-steps", type=int, default=0,
                   help="Print prompt length / visual count / forwards of the first N generate calls.")
    return p


def main(argv: List[str] | None = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    if args.blind_layer < 0:
        raise SystemExit("--blind-layer must be >= 0")
    state = BlindState(args.blind_layer, args.blind_debug_steps)
    install_visual_blind(state)
    print(f"[blind] installed: visual columns hidden from layer {args.blind_layer} on; eager vLLM, masked SDPA on every layer",
          flush=True)
    from lightnav.cli.eval_habitat import main as upstream_main

    return upstream_main(rest)


if __name__ == "__main__":
    sys.exit(main())
