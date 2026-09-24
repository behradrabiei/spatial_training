"""Layer-wise visual blinding and the information-horizon probe for Uni-NaVid (paper/info_horizon).

"Hiding from layer L" closes the key columns of visual prompt tokens for *every* query on
decoder layers ``>= L`` (an additive ``finfo.min`` on the 4-D mask every layer receives, the
mechanism of ``prune.py``'s layer-placed methods); layers below ``L`` see the tokens as usual and
positions never change. Uni-NaVid re-prefills its single-turn prompt on every inference, so
what a layer cannot read is genuinely absent from that layer on -- there is no cache to leak
through.

Visual columns are the history groups (one row per long-term slot, four per short-term frame;
the current frame's own 4-token group is the last of them) and the current frame's 64 fine
tokens. ``<image_sep>``, ``</video_special>`` and ``<image_special>`` stay visible, matching
LongNav-R1's ``kv_prune_visual_blind`` which hides visual slots only.

* ``ColumnBlinder`` -- the live arm (``--blind-layer L``): every visual column hidden from
  layer ``L`` on, for the prefill and every decode step of every inference.
* ``LayerProbe`` -- the information arm (``--layer-probe``): after each (unmodified) inference,
  re-forward the captured prompt once unmasked and, per probe layer ``L``, with three masks --
  the current frame hidden (leave-one-out), every visual token hidden, every visual token but
  the current frame hidden (keep-one-in) -- and score the four action words at the last
  position. ``leave_one = KL(P_full || P_current_hidden)``,
  ``keep_one = KL(P_only_current || P_no_visual)``; ``L = n_layers`` is 0 by construction and
  issues no forward.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from uninavid_hm3d.influence import ACTION_WORDS, action_logp, kl_divergence
from uninavid_hm3d.prune import (
    LENGTH_THRESHOLD,
    NAV_TOKENS,
    SIMILARITY_THRESHOLD,
    history_columns,
    prompt_length,
)

DEFAULT_PROBE_LAYERS = (0, 4, 8, 12, 16, 18, 20, 22, 24, 28)


# ------------------------------------------------------------------------------ columns
def visual_columns(placeholder: int, result_list: Sequence[int], n_ids: int
                   ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(all_visual, current, prompt_len)``: boolean column masks over the assembled prompt.

    ``all_visual`` covers every history group and the 64 fine tokens; ``current`` the last
    history group (the current frame's 4 tokens) and the fine tokens. Mirrors
    ``influence.HistoryLayout.build`` without needing the tensors.
    """
    spans = history_columns(placeholder, result_list)
    total = prompt_length(n_ids, result_list)
    if not spans:
        raise ValueError("a prompt without history groups has no current frame")
    pos = spans[-1][1]  # </video_special>, <image_special>, then the fine tokens
    nav = (pos + 2, pos + 2 + NAV_TOKENS)
    if nav[1] > total:
        raise ValueError("fine-token span overruns the prompt; layout arithmetic is inconsistent")
    all_visual = torch.zeros(total, dtype=torch.bool)
    for a, b in spans:
        all_visual[a:b] = True
    all_visual[nav[0]:nav[1]] = True
    current = torch.zeros(total, dtype=torch.bool)
    a, b = spans[-1]
    current[a:b] = True
    current[nav[0]:nav[1]] = True
    return all_visual, current, total


def hidden_mask(attention_mask: torch.Tensor, hide: torch.Tensor) -> torch.Tensor:
    """Out-of-place copy of a 4-D additive mask with the ``hide`` columns closed for every query.

    ``hide`` is ``(prompt_len,)`` or ``(batch, prompt_len)`` boolean; the mask may be wider than
    the prompt (decode steps append columns, which stay open).
    """
    if attention_mask is None:
        raise RuntimeError("a decoder layer received no attention mask")
    if hide.dim() == 1:
        hide = hide.unsqueeze(0)
    width = attention_mask.shape[-1]
    n = hide.shape[-1]
    if width < n:
        raise RuntimeError(f"attention mask has {width} columns for a {n}-token prompt")
    if hide.shape[0] not in (1, attention_mask.shape[0]):
        raise RuntimeError(f"{hide.shape[0]} hide rows for a batch of {attention_mask.shape[0]}")
    cols = torch.zeros(hide.shape[0], width, dtype=torch.bool, device=attention_mask.device)
    cols[:, :n] = hide.to(attention_mask.device)
    return attention_mask.masked_fill(cols.view(hide.shape[0], 1, 1, width), torch.finfo(attention_mask.dtype).min)


# ------------------------------------------------------------------------------ blinder
class ColumnBlinder:
    """Install the layer hooks and hide a chosen column set from a chosen layer on.

    ``begin_inference(input_ids)`` marks the prompt; the ``online_process_tensor`` wrapper then
    derives the column sets once the history layout is known; ``set_hide`` chooses what the
    hooks close (``None`` = nothing) and from which layer; ``finish`` clears everything. Outside
    an inference the hooks are inert and pass the mask object through untouched.
    """

    def __init__(self, image_token_index: int) -> None:
        self.image_token_index = int(image_token_index)
        self.n_layers: int | None = None
        self._installed = False
        self._clear()

    def _clear(self) -> None:
        self._active = False
        self._placeholder: int | None = None
        self._n_ids = 0
        self._result_list: list[int] | None = None
        self.all_visual: torch.Tensor | None = None
        self.current: torch.Tensor | None = None
        self.prompt_len: int | None = None
        self._hide: torch.Tensor | None = None
        self._kind: str | None = None  # "all": hide every visual column once the layout is known
        self._start = 0

    # -- per inference -----------------------------------------------------------------
    def begin_inference(self, input_ids: torch.Tensor) -> None:
        ids = input_ids.reshape(-1)
        where = torch.nonzero(ids == self.image_token_index).flatten().tolist()
        if len(where) != 1:
            raise ValueError(f"expected exactly one image placeholder, found {len(where)}")
        self._clear()
        self._active = True
        self._placeholder = int(where[0])
        self._n_ids = int(ids.numel())

    def set_hide(self, hide: torch.Tensor | None, start: int = 0) -> None:
        """Close ``hide`` (``(prompt_len,)`` or ``(batch, prompt_len)``) on layers ``>= start``."""
        if hide is not None and self.prompt_len is not None and hide.shape[-1] != self.prompt_len:
            raise ValueError(f"hide has {hide.shape[-1]} columns, the prompt {self.prompt_len}")
        self._hide = hide
        self._start = int(start)

    def hide_all_from(self, start: int) -> None:
        """Live arm: close every visual column on layers ``>= start`` as soon as the history
        layout of this inference is assembled (inside ``generate``)."""
        self._kind = "all"
        self._start = int(start)

    def finish(self) -> None:
        self._clear()

    @property
    def result_list(self) -> list[int] | None:
        return self._result_list

    # -- installation ------------------------------------------------------------------
    def install(self, model: Any) -> None:
        if self._installed:
            return
        layers = model.get_model().layers
        self.n_layers = len(layers)
        orig = model.online_process_tensor
        blinder = self

        def online_process_tensor(nav_size, length_threshold=LENGTH_THRESHOLD,
                                  similarity_threshold=SIMILARITY_THRESHOLD):
            tokens, result_list = orig(nav_size, length_threshold, similarity_threshold)
            if blinder._active:
                blinder._result_list = [int(g) for g in result_list]
                blinder.all_visual, blinder.current, blinder.prompt_len = visual_columns(
                    blinder._placeholder, blinder._result_list, blinder._n_ids)
                if blinder._kind == "all":
                    blinder._hide = blinder.all_visual
            return tokens, result_list

        model.online_process_tensor = online_process_tensor
        for index, layer in enumerate(layers):
            layer.forward = self._layer_forward(layer, index)
        self._installed = True

    def _layer_forward(self, layer: Any, index: int):
        orig = layer.forward
        blinder = self

        def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False):
            if blinder._active and blinder._hide is not None and index >= blinder._start:
                attention_mask = hidden_mask(attention_mask, blinder._hide)
            return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions,
                        use_cache=use_cache)

        return forward


# -------------------------------------------------------------------------------- probe
class LayerProbe:
    """Capture one prefill during ``generate`` and score the layer-wise ablations afterwards."""

    def __init__(self, model: Any, tokenizer: Any, image_token_index: int,
                 layers: Sequence[int] = DEFAULT_PROBE_LAYERS) -> None:
        from uninavid_hm3d.influence import FrameInfluenceProbe

        self.model = model
        self.blinder = ColumnBlinder(image_token_index)
        self.layers = tuple(sorted({int(l) for l in layers}))
        if any(l < 0 for l in self.layers):
            raise ValueError("probe layers must be non-negative")
        self.action_ids = FrameInfluenceProbe._resolve_action_ids(tokenizer)
        self._installed = False
        self._embeds: torch.Tensor | None = None
        self._input_ids: torch.Tensor | None = None

    @property
    def n_layers(self) -> int:
        if self.blinder.n_layers is None:
            raise RuntimeError("install() has not run")
        return self.blinder.n_layers

    def describe(self) -> dict[str, Any]:
        return {"action_words": list(ACTION_WORDS), "action_token_ids": list(self.action_ids),
                "layers": list(self.layers), "n_layers": self.blinder.n_layers,
                "columns_hidden": "history groups (long-term slots and 4-token frames) + the current "
                                  "frame's 64 fine tokens; separators and special tokens visible",
                "hiding": "additive finfo.min on the key columns for every query on layers >= L",
                "keep_one": "KL(P_only_current_frame || P_no_visual)",
                "leave_one": "KL(P_full || P_current_frame_hidden)",
                "kl_direction": "KL(a || b), expectation under a", "temperature": 1.0}

    def reset(self) -> None:
        self._embeds = self._input_ids = None
        self.blinder.finish()

    # -- capture -----------------------------------------------------------------------
    def install(self) -> None:
        if self._installed:
            return
        model = self.model
        self.blinder.install(model)
        if any(l > self.blinder.n_layers for l in self.layers):
            raise ValueError(f"probe layers {self.layers} exceed the model's {self.blinder.n_layers} layers")
        orig_prepare = model.prepare_inputs_labels_for_multimodal
        probe = self

        def prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels,
                                                 images, prompts=None):
            out = orig_prepare(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)
            if input_ids is not None and input_ids.shape[1] > 1:  # the prefill, not a decode step
                probe._embeds = out[3][0].detach()
                probe._input_ids = input_ids[0].detach().clone()
            return out

        model.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
        self._installed = True

    def begin_inference(self, input_ids: torch.Tensor) -> None:
        """Before ``generate``: mark the prompt so the layout is captured; nothing is hidden."""
        self._embeds = self._input_ids = None
        self.blinder.begin_inference(input_ids)
        self.blinder.set_hide(None)

    # -- scoring -----------------------------------------------------------------------
    @torch.inference_mode()
    def _last_logits(self, batch: torch.Tensor, hide: torch.Tensor | None, start: int = 0) -> torch.Tensor:
        self.blinder.set_hide(hide, start)
        try:
            hidden = self.model.model(inputs_embeds=batch, use_cache=False)[0][:, -1]
        finally:
            self.blinder.set_hide(None)
        return self.model.lm_head(hidden).float()

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor) -> dict[str, Any]:
        """Score the prefill captured by the last ``generate`` call; clears the capture."""
        b = self.blinder
        if self._embeds is None or self._input_ids is None or b.result_list is None:
            raise RuntimeError("no prefill captured; was begin_inference() called before generate()?")
        started = time.perf_counter()
        embeds, ids = self._embeds, self._input_ids
        if not torch.equal(ids, input_ids.reshape(-1).to(ids.device)):
            raise RuntimeError("captured prompt ids differ from the generate() prompt")
        if int(embeds.shape[0]) != b.prompt_len:
            raise RuntimeError(f"assembled prompt has {embeds.shape[0]} embeddings, layout expects {b.prompt_len}")
        all_visual, current = b.all_visual, b.current
        n_frames = int(self.model.get_model().feat_cache.shape[0])
        n_long = sum(1 for g in b.result_list if g == 1)

        base = self._last_logits(embeds[None], None)[0]
        base_logp, base_mass = action_logp(base, self.action_ids)
        n_probes = 1
        keep_one, leave_one, rows = [], [], []
        for L in self.layers:
            if L >= self.n_layers:
                keep_one.append(0.0)
                leave_one.append(0.0)
                rows.append({"layer": L, "keep_one": 0.0, "leave_one": 0.0, "logp_current_hidden": base_logp,
                             "logp_no_visual": base_logp, "logp_only_current": base_logp})
                continue
            hide = torch.stack([current, all_visual, all_visual & ~current], dim=0)
            logits = self._last_logits(embeds[None].expand(3, -1, -1), hide, L)
            n_probes += 3
            minus_cur, _ = action_logp(logits[0], self.action_ids)
            text_all, _ = action_logp(logits[1], self.action_ids)
            only_cur, _ = action_logp(logits[2], self.action_ids)
            leave = kl_divergence(base_logp, minus_cur)
            keep = kl_divergence(only_cur, text_all)
            keep_one.append(keep)
            leave_one.append(leave)
            rows.append({"layer": L, "keep_one": keep, "leave_one": leave, "logp_current_hidden": minus_cur,
                         "logp_no_visual": text_all, "logp_only_current": only_cur})
        record = {
            "n_frames": n_frames,
            "n_long_slots": n_long,
            "prompt_tokens": int(embeds.shape[0]),
            "n_visual_tokens": int(all_visual.sum()),
            "n_current_tokens": int(current.sum()),
            "logp_full": base_logp,
            "p_mass": base_mass,
            "argmax_action": ACTION_WORDS[int(torch.tensor(base_logp).argmax())],
            "layers": list(self.layers),
            "keep_one": keep_one,
            "leave_one": leave_one,
            "per_layer": rows,
            "n_probes": n_probes,
            "probe_seconds": time.perf_counter() - started,
        }
        self._embeds = self._input_ids = None
        b.finish()
        return record
