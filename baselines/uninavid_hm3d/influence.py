"""Leave-one-frame-out influence probes for Uni-NaVid (paper/frame_influence).

Uni-NaVid rebuilds its LLM prompt on every inference from a per-frame feature cache: each
observed frame is one post-projector row of ``feat_cache`` (4 tokens, 4096-d). Frames inside
the short-term window (the last 64) enter the prompt as their own 4-token group; older frames
are folded into 1-token long-term slots by a greedy cosine-similarity running mean
(``UniNaVIDMetaForCausalLM.online_process_tensor``). The current frame additionally supplies
64 fine tokens. Because the rows are independent per frame, "remove frame i from the context"
is exact: delete its group (or its contribution to a merged slot) from the assembled
``inputs_embeds`` and re-run the LLM. Everything else -- text, the other frames, the fine
tokens -- is byte-identical to the on-policy prompt.

Design (see ``FrameInfluenceProbe``):

* The probe wraps the model's ``prepare_inputs_labels_for_multimodal`` and
  ``online_process_tensor`` so the prefill of the real ``generate`` call hands over the
  assembled embeddings and the history layout. Probes therefore run *after* the on-policy
  sample has been drawn, consume no RNG, re-run no vision tower and mutate no state: the
  trajectory is untouched by construction.
* ``LongTermTracker`` mirrors the long-term merge rule while recording which frames each slot
  holds, and is verified against the model's own cache after every inference.
* Ablations are pure splices of the captured embeddings; equal-length probes are batched.
  A merged slot minus one member is rebuilt as the fp32 mean of the remaining members (the
  fp16 closed form ``(w*slot - x)/(w-1)`` amplifies rounding ~w-fold); ``merged_control``
  rebuilds a slot from *all* its members to measure that reconstruction's noise floor.
* The action distribution is the softmax over the four single-token action words at the
  last prompt position (temperature 1); KL is ``KL(full || ablated)``.
* ``measures`` selects the per-frame probes: ``leave_one`` (above) and/or ``keep_one``
  (leave-one-IN): the prompt with ONLY frame i's history group -- every other group, the
  separators and the current frame's fine tokens removed, text and specials intact -- scored
  as ``KL(only i || no visual)`` against the ``visual_all`` prompt. Its age-0 value is
  ``KL(history_all || visual_all)`` (the current frame alone vs no visual), so it needs no
  extra prompt. That splice re-encodes the lone group contiguously, so the prompt is the same
  whichever decision it is scored at and carries no age. ``keep_one_mask`` is the
  position-preserving form: the full prompt stays and every visual column except frame i's
  group is closed for every query on every layer (``layer_blind.ColumnBlinder``, the mechanism
  of the layer-blind arms), so the lone group keeps its place in the sequence; the reference is
  the same prompt with every visual column closed (``visual_all_mask``) and age 0 is the
  current frame's columns alone (``only_current_mask``, Exp 6's layer-0 keep-one). Both forms
  write the same ``keep_one`` / ``keep_one_age0`` record fields; ``keep_one_mode`` says which.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

ACTION_WORDS = ("stop", "forward", "left", "right")
LENGTH_THRESHOLD = 64  # frames kept as 4-token groups (Uni-NaVid default)
SIMILARITY_THRESHOLD = 0.985  # long-term merge threshold (Uni-NaVid default)
NAV_TOKENS = 64  # fine tokens of the current frame
DEFAULT_HORIZONS = (4, 8, 16, 32)
MEASURES = ("leave_one", "keep_one", "keep_one_mask")


# ----------------------------------------------------------------------------- statistics
def action_logp(logits: torch.Tensor, action_ids: Sequence[int]) -> tuple[list[float], float]:
    """(log-probs over the action words renormalised, probability mass those words hold)."""
    logits = logits.detach().float()
    full = torch.log_softmax(logits, dim=-1)
    ids = torch.as_tensor(list(action_ids), device=logits.device)
    logp = torch.log_softmax(logits[ids], dim=-1)
    return logp.cpu().tolist(), float(full[ids].exp().sum().cpu())


def kl_divergence(p_logp: Sequence[float], q_logp: Sequence[float]) -> float:
    """KL(p || q) in nats over matching log-prob vectors, clamped at 0 (fp32)."""
    p = torch.as_tensor(list(p_logp), dtype=torch.float32)
    q = torch.as_tensor(list(q_logp), dtype=torch.float32)
    return max(float(torch.sum(p.exp() * (p - q))), 0.0)


# ------------------------------------------------------------------------- long-term slots
class LongTermTracker:
    """Shadow of ``online_process_tensor``'s long-term merge that remembers slot membership.

    ``slots[j]`` lists the frame indices (rows of ``feat_cache``) merged into long-term slot
    ``j``; ``long`` mirrors ``model_state.long_feat_cache`` with the same fp16 arithmetic in
    the same order, so ``verify`` can hold the two to a tight tolerance.
    """

    def __init__(self, length_threshold: int = LENGTH_THRESHOLD,
                 similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.length_threshold = int(length_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.reset()

    def reset(self) -> None:
        self.long: torch.Tensor | None = None
        self.slots: list[list[int]] = []

    @property
    def weight(self) -> int:
        """Run length of the newest slot -- what the model stores as ``weight``."""
        return len(self.slots[-1]) if self.slots else 1

    def step(self, feat_cache: torch.Tensor, new_frames: int) -> None:
        """Fold the frames that just left the short-term window, exactly as upstream does."""
        k = feat_cache.shape[0]
        cos = torch.nn.CosineSimilarity(dim=0)
        for frame_index in range(int(new_frames) - 1, -1, -1):
            f = k - self.length_threshold - frame_index - 1
            if f < 0:
                continue
            token = feat_cache[f].mean(dim=0)
            if self.long is not None:
                if cos(self.long[-1], token) > self.similarity_threshold:
                    w = len(self.slots[-1])
                    self.long[-1] = (self.long[-1] * w + token) / (w + 1)
                    self.slots[-1].append(f)
                else:
                    self.long = torch.cat([self.long, token[None]], dim=0)
                    self.slots.append([f])
            else:
                self.long = token[None].clone()
                self.slots.append([f])

    def verify(self, model_state: Any, rtol: float = 1e-3, atol: float = 1e-4) -> None:
        """Raise unless the shadow matches the model's long-term cache and run weight."""
        actual = getattr(model_state, "long_feat_cache", None)
        if (actual is None) != (self.long is None):
            raise RuntimeError("long-term tracker diverged: one side has no long-term cache")
        if actual is not None:
            if actual.shape != self.long.shape or not torch.allclose(
                    actual.float(), self.long.to(actual.device).float(), rtol=rtol, atol=atol):
                raise RuntimeError(f"long-term tracker diverged from the model cache "
                                   f"(shapes {tuple(actual.shape)} vs {tuple(self.long.shape)})")
            if int(model_state.weight) != self.weight:
                raise RuntimeError(f"long-term tracker weight {self.weight} != model weight {model_state.weight}")


# ------------------------------------------------------------------------------- layout
Span = tuple[int, int]  # [start, end) in the assembled embeddings


@dataclass
class HistoryLayout:
    """Where each frame's tokens sit in the assembled ``inputs_embeds`` of one prefill.

    ``groups[j]`` is the span of history group ``j`` (long-term slots first, then the 4-token
    short-term groups in frame order, the current frame last). ``frame_of_group[j]`` is the
    frame index of a short-term group, ``slot_of_group[j]`` the long-term slot index (each
    ``None`` for the other tier). Consecutive groups are separated by one ``<image_sep>``
    embedding; none follows the last.
    """

    placeholder: int  # index of the image placeholder in the prompt ids (= prefix length)
    n_frames: int
    groups: list[Span]
    frame_of_group: list[int | None]
    slot_of_group: list[int | None]
    slot_members: list[list[int]]
    special_span: Span  # </video_special>, <image_special>
    nav_span: Span  # the 64 fine tokens of the current frame
    tail_start: int  # first embedding after the fine tokens (</image_special> ...)
    total_length: int

    @classmethod
    def build(cls, input_ids: torch.Tensor, result_list: Sequence[int], n_frames: int,
              slot_members: Sequence[Sequence[int]], image_token_index: int,
              length_threshold: int = LENGTH_THRESHOLD) -> "HistoryLayout":
        ids = input_ids.reshape(-1)
        where = torch.nonzero(ids == image_token_index).flatten().tolist()
        if len(where) != 1:
            raise ValueError(f"expected exactly one image placeholder, found {len(where)}")
        s = int(where[0])
        n_short = min(int(n_frames), length_threshold)
        n_long = len(result_list) - n_short
        if n_long != len(slot_members):
            raise ValueError(f"{n_long} long-term groups in the prompt but the tracker holds "
                             f"{len(slot_members)} slots")
        expected = [1] * n_long + [4] * n_short
        if list(result_list) != expected:
            raise ValueError(f"unexpected history group sizes {list(result_list)}")
        groups, frame_of, slot_of = [], [], []
        pos = s
        for j, g in enumerate(result_list):
            groups.append((pos, pos + int(g)))
            pos += int(g)
            if j < len(result_list) - 1:
                pos += 1  # <image_sep>
            if j < n_long:
                frame_of.append(None)
                slot_of.append(j)
            else:
                frame_of.append(n_frames - n_short + (j - n_long))
                slot_of.append(None)
        special = (pos, pos + 2)
        nav = (pos + 2, pos + 2 + NAV_TOKENS)
        tail = nav[1]
        total = ids.numel() - 1 + sum(int(g) for g in result_list) + max(len(result_list) - 1, 0) + NAV_TOKENS
        if tail + (ids.numel() - s - 3) != total:
            raise AssertionError("layout arithmetic is inconsistent")
        return cls(placeholder=s, n_frames=int(n_frames), groups=groups, frame_of_group=frame_of,
                   slot_of_group=slot_of, slot_members=[list(m) for m in slot_members],
                   special_span=special, nav_span=nav, tail_start=tail, total_length=total)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def group_of_frame(self, frame: int) -> tuple[int, str]:
        """(group index, tier) holding ``frame``; tier is 'short' or 'long'."""
        for j, f in enumerate(self.frame_of_group):
            if f == frame:
                return j, "short"
        for j, members in enumerate(self.slot_members):
            if frame in members:
                return j, "long"
        raise KeyError(f"frame {frame} is not in the prompt")

    def removal_span(self, j: int) -> Span:
        """Span deleting group ``j`` together with one adjacent separator."""
        start, end = self.groups[j]
        if self.n_groups == 1:
            return (start, end)
        if j < self.n_groups - 1:
            return (start, end + 1)  # trailing <image_sep>
        return (start - 1, end)  # last group: the separator before it

    def verify(self, embeds: torch.Tensor, feat_cache: torch.Tensor, embed_tokens: Callable,
               input_ids: torch.Tensor) -> None:
        """Anchor the computed spans on exact tensor equality; raise on any drift."""
        if embeds.shape[0] != self.total_length:
            raise RuntimeError(f"assembled prompt has {embeds.shape[0]} embeddings, layout expects {self.total_length}")
        for j, frame in enumerate(self.frame_of_group):
            if frame is None:
                continue
            a, b = self.groups[j]
            if not torch.equal(embeds[a:b], feat_cache[frame].to(embeds.dtype)):
                raise RuntimeError(f"short-term group {j} does not match feat_cache[{frame}]")
        ids = input_ids.reshape(-1)
        s = self.placeholder
        ref = embed_tokens(ids[s + 1:s + 3].to(embeds.device))
        if not torch.equal(embeds[self.special_span[0]:self.special_span[1]], ref.to(embeds.dtype)):
            raise RuntimeError("special-token span does not match </video_special>,<image_special>")
        ref = embed_tokens(ids[s + 3:s + 4].to(embeds.device))
        if not torch.equal(embeds[self.tail_start:self.tail_start + 1], ref.to(embeds.dtype)):
            raise RuntimeError("tail does not start with </image_special>")
        if self.n_groups > 1:
            sep = embed_tokens(ids[s - 1:s].to(embeds.device)).to(embeds.dtype)
            a, b = self.groups[0]
            if not torch.equal(embeds[b:b + 1], sep):
                raise RuntimeError("no <image_sep> after the first history group")


# ------------------------------------------------------------------------------ splices
def drop_spans(embeds: torch.Tensor, spans: Sequence[Span]) -> torch.Tensor:
    """Copy of ``embeds`` (L, C) without the rows in ``spans`` (overlaps allowed)."""
    keep = torch.ones(embeds.shape[0], dtype=torch.bool, device=embeds.device)
    for a, b in spans:
        keep[a:b] = False
    return embeds[keep]


def member_mean(feat_cache: torch.Tensor, members: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """fp32 mean token of the given frames (what a long-term slot ideally holds)."""
    idx = torch.as_tensor(list(members), device=feat_cache.device)
    return feat_cache.index_select(0, idx).float().mean(dim=(0, 1)).to(dtype)


@dataclass
class Probe:
    kind: str  # base | frame | keep_one | age0 | history_all | visual_all | horizon | merged_control
    #            | visual_all_mask | only_current_mask (the masked forms carry ``hide``)
    embeds: torch.Tensor
    hide: torch.Tensor | None = None  # (L,) bool: key columns closed for every query on every layer
    frame: int | None = None
    tier: str | None = None
    slot: int | None = None
    weight: int | None = None
    K: int | None = None
    partial_slots: int = 0


def _ablate_frames(layout: HistoryLayout, embeds: torch.Tensor, feat_cache: torch.Tensor,
                   frames: set[int]) -> tuple[torch.Tensor, int]:
    """Remove a set of frames: drop their groups; rebuild partially covered long-term slots.

    Returns the spliced embeddings and the number of long-term slots that were rebuilt from
    their surviving members instead of dropped.
    """
    out = embeds
    drops: list[Span] = []
    partial = 0
    for j in range(layout.n_groups):
        f = layout.frame_of_group[j]
        if f is not None:
            if f in frames:
                drops.append(layout.removal_span(j))
            continue
        members = layout.slot_members[layout.slot_of_group[j]]
        gone = [m for m in members if m in frames]
        if not gone:
            continue
        rest = [m for m in members if m not in frames]
        if not rest:
            drops.append(layout.removal_span(j))
        else:
            out = out.clone() if out is embeds else out
            out[layout.groups[j][0]] = member_mean(feat_cache, rest, embeds.dtype)
            partial += 1
    return drop_spans(out, drops), partial


def _keep_only_frame(layout: HistoryLayout, embeds: torch.Tensor, feat_cache: torch.Tensor,
                     frame: int) -> torch.Tensor:
    """The prompt with only ``frame``'s history group: every other group and separator and the
    current frame's fine tokens go, so the visual region is the canonical one-group layout
    (``[group]`` with no separator, exactly what a one-frame prompt holds); text and the
    special tokens stay. A long-term slot is rebuilt as the frame's own mean token first."""
    j, tier = layout.group_of_frame(frame)
    out = embeds
    if tier == "long":
        out = embeds.clone()
        out[layout.groups[j][0]] = member_mean(feat_cache, [frame], embeds.dtype)
    a, b = layout.groups[j]
    return drop_spans(out, [(layout.groups[0][0], a), (b, layout.special_span[0]), layout.nav_span])


def visual_column_masks(layout: HistoryLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """``(all_visual, current)`` boolean column masks over the assembled prompt: every history
    group plus the fine tokens, and the current frame's group plus the fine tokens (separators
    and special tokens stay open) -- ``layer_blind.visual_columns`` from the layout."""
    all_visual = torch.zeros(layout.total_length, dtype=torch.bool)
    for a, b in layout.groups:
        all_visual[a:b] = True
    all_visual[layout.nav_span[0]:layout.nav_span[1]] = True
    current = torch.zeros(layout.total_length, dtype=torch.bool)
    a, b = layout.groups[-1]
    current[a:b] = True
    current[layout.nav_span[0]:layout.nav_span[1]] = True
    return all_visual, current


def _keep_only_frame_mask(layout: HistoryLayout, embeds: torch.Tensor, feat_cache: torch.Tensor,
                          frame: int, all_visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(embeds, hide)``: the full prompt with every visual column but ``frame``'s group closed.
    A long-term slot is rebuilt as the frame's own mean token first (its column stays open)."""
    j, tier = layout.group_of_frame(frame)
    out = embeds
    if tier == "long":
        out = embeds.clone()
        out[layout.groups[j][0]] = member_mean(feat_cache, [frame], embeds.dtype)
    hide = all_visual.clone()
    a, b = layout.groups[j]
    hide[a:b] = False
    return out, hide


def build_probes(layout: HistoryLayout, embeds: torch.Tensor, feat_cache: torch.Tensor,
                 horizons: Sequence[int] = DEFAULT_HORIZONS,
                 measures: Sequence[str] = ("leave_one",)) -> list[Probe]:
    """Every ablation of one prefill, the unmodified prompt first."""
    t = layout.n_frames - 1
    probes = [Probe("base", embeds)]
    all_visual, current = visual_column_masks(layout)
    for i in range(t):
        j, tier = layout.group_of_frame(i)
        weight = len(layout.slot_members[layout.slot_of_group[j]]) if tier == "long" else None
        if "leave_one" in measures:
            spliced, _ = _ablate_frames(layout, embeds, feat_cache, {i})
            probes.append(Probe("frame", spliced, frame=i, tier=tier,
                                slot=layout.slot_of_group[j], weight=weight))
        if "keep_one" in measures:
            probes.append(Probe("keep_one", _keep_only_frame(layout, embeds, feat_cache, i), frame=i,
                                tier=tier, slot=layout.slot_of_group[j], weight=weight))
        if "keep_one_mask" in measures:
            kept, hide = _keep_only_frame_mask(layout, embeds, feat_cache, i, all_visual)
            probes.append(Probe("keep_one", kept, hide=hide, frame=i, tier=tier,
                                slot=layout.slot_of_group[j], weight=weight))
    if "keep_one_mask" in measures:
        # References of the masked form: no visual column open / only the current frame's.
        probes.append(Probe("visual_all_mask", embeds, hide=all_visual))
        probes.append(Probe("only_current_mask", embeds, hide=all_visual & ~current, frame=t, tier="short"))
    # Current frame: its history group plus the 64 fine tokens.
    j_cur, _ = layout.group_of_frame(t)
    probes.append(Probe("age0", drop_spans(embeds, [layout.removal_span(j_cur), layout.nav_span]),
                        frame=t, tier="short"))
    hist, partial = _ablate_frames(layout, embeds, feat_cache, set(range(t)))
    probes.append(Probe("history_all", hist, K=1, partial_slots=partial))
    for K in sorted({int(k) for k in horizons}):
        if t - K < 0:
            continue
        band, partial = _ablate_frames(layout, embeds, feat_cache, set(range(t - K + 1)))
        probes.append(Probe("horizon", band, K=K, partial_slots=partial))
    # Text-only prompt: no history groups, no separators, no fine tokens; specials stay.
    visual = drop_spans(embeds, [(layout.groups[0][0], layout.special_span[0]), layout.nav_span])
    probes.append(Probe("visual_all", visual))
    for slot, members in enumerate(layout.slot_members):
        if len(members) < 2:
            continue
        j = layout.slot_of_group.index(slot)
        ctrl = embeds.clone()
        ctrl[layout.groups[j][0]] = member_mean(feat_cache, members, embeds.dtype)
        probes.append(Probe("merged_control", ctrl, slot=slot, weight=len(members)))
    return probes


# ------------------------------------------------------------------------------- engine
class FrameInfluenceProbe:
    """Capture one prefill's embeddings during ``generate`` and score its ablations afterwards."""

    def __init__(self, model: Any, tokenizer: Any, image_token_index: int, probe_batch: int = 8,
                 horizons: Sequence[int] = DEFAULT_HORIZONS,
                 length_threshold: int = LENGTH_THRESHOLD,
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 measures: Sequence[str] = ("leave_one",)) -> None:
        if probe_batch < 1:
            raise ValueError("probe_batch must be at least 1")
        if any(int(k) < 2 for k in horizons):
            raise ValueError("horizons must all be >= 2 (K=1 is the history_all probe)")
        self.measures = tuple(str(m) for m in measures)
        if not self.measures or any(m not in MEASURES for m in self.measures):
            raise ValueError(f"measures must be a non-empty subset of {MEASURES}, got {self.measures}")
        if "keep_one" in self.measures and "keep_one_mask" in self.measures:
            raise ValueError("keep_one and keep_one_mask write the same record fields; pick one")
        self.keep_one_mode = ("mask" if "keep_one_mask" in self.measures
                              else "splice" if "keep_one" in self.measures else None)
        self.blinder = None  # layer_blind.ColumnBlinder, installed for the masked form only
        self.model = model
        self.image_token_index = int(image_token_index)
        self.probe_batch = int(probe_batch)
        self.horizons = tuple(sorted({int(k) for k in horizons}))
        self.length_threshold = int(length_threshold)
        self.tracker = LongTermTracker(length_threshold, similarity_threshold)
        self.action_ids = self._resolve_action_ids(tokenizer)
        # Word-internal forms ('stop' without the leading '▁'); reported so a sampled first
        # token outside the action set can be recognised as a spelling variant.
        self.action_ids_nospace = [int(tokenizer.convert_tokens_to_ids(w)) for w in ACTION_WORDS]
        self._installed = False
        self.reset()

    @staticmethod
    def _resolve_action_ids(tokenizer: Any) -> list[int]:
        """Token id of each action word as it follows "ASSISTANT:" (the sentencepiece
        word-initial form, e.g. '▁stop'); each must be a single token."""
        ids = []
        for word in ACTION_WORDS:
            toks = tokenizer(word, add_special_tokens=False).input_ids
            toks = [t for t in toks if t != getattr(tokenizer, "bos_token_id", None)]
            if len(toks) != 1:
                raise ValueError(f"action word {word!r} is not a single token: {toks}")
            ids.append(int(toks[0]))
        if len(set(ids)) != len(ids):
            raise ValueError(f"action tokens collide: {ids}")
        return ids

    def describe(self) -> dict[str, Any]:
        return {"action_words": list(ACTION_WORDS), "action_token_ids": list(self.action_ids),
                "action_token_ids_word_internal": list(self.action_ids_nospace),
                "probe_batch": self.probe_batch, "horizons": list(self.horizons),
                "length_threshold": self.length_threshold,
                "similarity_threshold": self.tracker.similarity_threshold,
                "measures": list(self.measures), "kl_direction": "KL(full || ablated)",
                "keep_one_kl_direction": "KL(only frame || no visual)", "keep_one_mode": self.keep_one_mode,
                "keep_one_hiding": ("every visual column but the frame's group closed (finfo.min) for every "
                                    "query on every layer; positions unchanged" if self.keep_one_mode == "mask"
                                    else "every other group, the separators and the fine tokens spliced out"
                                    if self.keep_one_mode == "splice" else None),
                "temperature": 1.0}

    def reset(self) -> None:
        """New episode: forget the long-term slots and any captured prefill."""
        self.tracker.reset()
        self.reset_capture()

    def reset_capture(self) -> None:
        """Drop the captured prefill (an inference skipped by the stride)."""
        self._embeds: torch.Tensor | None = None
        self._input_ids: torch.Tensor | None = None
        self._result_list: list[int] | None = None

    # -- capture -----------------------------------------------------------------------
    def install(self) -> None:
        """Wrap the model's prefill so the assembled embeddings and layout are captured."""
        if self._installed:
            return
        model = self.model
        state = model.get_model()
        orig_prepare = model.prepare_inputs_labels_for_multimodal
        orig_online = model.online_process_tensor
        probe = self

        def online_process_tensor(nav_size, length_threshold=LENGTH_THRESHOLD,
                                  similarity_threshold=SIMILARITY_THRESHOLD):
            if (length_threshold, similarity_threshold) != (probe.tracker.length_threshold,
                                                             probe.tracker.similarity_threshold):
                raise RuntimeError("online_process_tensor called with thresholds the tracker does not mirror")
            probe.tracker.step(state.feat_cache, state.new_frames)
            out = orig_online(nav_size, length_threshold, similarity_threshold)
            probe.tracker.verify(state)
            probe._result_list = [int(g) for g in out[1]]
            return out

        def prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels,
                                                 images, prompts=None):
            out = orig_prepare(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)
            if input_ids is not None and input_ids.shape[1] > 1:  # the prefill, not a decode step
                probe._embeds = out[3][0].detach()
                probe._input_ids = input_ids[0].detach().clone()
            return out

        model.online_process_tensor = online_process_tensor
        model.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
        if self.keep_one_mode == "mask":
            from uninavid_hm3d.layer_blind import ColumnBlinder

            # Its layer hooks pass the mask through untouched unless a hide is set inside score().
            self.blinder = ColumnBlinder(self.image_token_index)
            self.blinder.install(model)
        self._installed = True

    # -- scoring -----------------------------------------------------------------------
    @torch.inference_mode()
    def _last_logits(self, batch: torch.Tensor, hide: torch.Tensor | None = None) -> torch.Tensor:
        """Last-position logits; ``hide`` (batch, L) closes key columns on every layer."""
        if hide is not None:
            if self.blinder is None:
                raise RuntimeError("masked probes need the column blinder (measures must include keep_one_mask)")
            self.blinder.begin_inference(self._mask_ids)
            self.blinder.set_hide(hide, 0)
        try:
            hidden = self.model.model(inputs_embeds=batch, use_cache=False)[0][:, -1]
        finally:
            if hide is not None:
                self.blinder.finish()
        return self.model.lm_head(hidden).float()

    def score(self, probes: Sequence[Probe]) -> list[torch.Tensor]:
        """Last-position logits for every probe, batching equal-length prompts (a batch may mix
        masked and unmasked probes: an all-open hide row is the plain prompt)."""
        logits: list[torch.Tensor | None] = [None] * len(probes)
        by_len: dict[int, list[int]] = {}
        for idx, p in enumerate(probes):
            by_len.setdefault(p.embeds.shape[0], []).append(idx)
        for idxs in by_len.values():
            for start in range(0, len(idxs), self.probe_batch):
                chunk = idxs[start:start + self.probe_batch]
                batch = torch.stack([probes[i].embeds for i in chunk], dim=0)
                hide = None
                if any(probes[i].hide is not None for i in chunk):
                    hide = torch.stack([probes[i].hide if probes[i].hide is not None
                                        else torch.zeros(batch.shape[1], dtype=torch.bool) for i in chunk], dim=0)
                out = self._last_logits(batch, hide)
                for row, i in enumerate(chunk):
                    logits[i] = out[row]
        return logits  # type: ignore[return-value]

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor, output_ids: torch.Tensor) -> dict[str, Any]:
        """Score every ablation of the prefill captured by the last ``generate`` call.

        Runs under ``inference_mode`` because the captured tensors were created inside
        ``generate``'s inference context and may not be touched by autograd-tracking ops.
        """
        if self._embeds is None or self._result_list is None or self._input_ids is None:
            raise RuntimeError("no prefill captured; was install() called before generate()?")
        started = time.perf_counter()
        state = self.model.get_model()
        feat_cache = state.feat_cache
        embeds, ids = self._embeds, self._input_ids
        if not torch.equal(ids, input_ids.reshape(-1).to(ids.device)):
            raise RuntimeError("captured prompt ids differ from the generate() prompt")
        layout = HistoryLayout.build(ids, self._result_list, feat_cache.shape[0], self.tracker.slots,
                                     self.image_token_index, self.length_threshold)
        layout.verify(embeds, feat_cache, state.embed_tokens, ids)
        self._mask_ids = ids  # the blinder locates the placeholder from the prompt ids
        probes = build_probes(layout, embeds, feat_cache, self.horizons, self.measures)
        logits = self.score(probes)
        scored = [action_logp(lg, self.action_ids) for lg in logits]

        base_logp, base_mass = scored[0]
        argmax_full = ACTION_WORDS[int(torch.tensor(base_logp).argmax())]
        sampled_id = int(output_ids[0, input_ids.shape[1]]) if output_ids.shape[1] > input_ids.shape[1] else None
        record: dict[str, Any] = {
            "n_frames": layout.n_frames,
            "n_long_slots": len(layout.slot_members),
            "prompt_tokens": int(embeds.shape[0]),
            "logits_full": logits[0][self.action_ids].cpu().tolist(),
            "logp_full": base_logp,
            "p_mass": base_mass,
            "argmax_action": argmax_full,
            "sampled_first_token_id": sampled_id,
            "sampled_first_token_in_set": sampled_id in self.action_ids if sampled_id is not None else None,
            "frames": [], "keep_one": [], "merged_control": [], "horizon": [],
            "history_all": None, "visual_all": None, "age0": None, "keep_one_age0": None,
            "keep_one_mode": self.keep_one_mode,
        }
        # Keep-one probes are scored against the no-visual prompt, not the full one: the spliced
        # one for the splice form, the column-masked one for the masked form.
        vis_logp = next(lp for p, (lp, _) in zip(probes, scored) if p.kind == "visual_all")
        if self.keep_one_mode == "mask":
            vis_logp = next(lp for p, (lp, _) in zip(probes, scored) if p.kind == "visual_all_mask")
        for p, (logp, mass) in zip(probes[1:], scored[1:]):
            ref = vis_logp if p.kind in ("keep_one", "only_current_mask") else base_logp
            entry = {"kl": kl_divergence(logp, ref) if p.kind == "keep_one" else kl_divergence(ref, logp),
                     "logp": logp, "p_mass": mass,
                     "argmax_action": ACTION_WORDS[int(torch.tensor(logp).argmax())]}
            if p.kind in ("frame", "keep_one"):
                entry.update({"frame": p.frame, "tier": p.tier, "slot": p.slot, "weight": p.weight})
                record["frames" if p.kind == "frame" else "keep_one"].append(entry)
            elif p.kind == "merged_control":
                entry.update({"slot": p.slot, "weight": p.weight})
                record["merged_control"].append(entry)
            elif p.kind == "horizon":
                entry.update({"K": p.K, "partial_slots": p.partial_slots})
                record["horizon"].append(entry)
            elif p.kind == "history_all":
                entry["partial_slots"] = p.partial_slots
                record["history_all"] = entry
            elif p.kind == "only_current_mask":
                # Age 0 of the masked keep-one curve: the current frame's columns alone vs none.
                entry["kl"] = kl_divergence(logp, ref)
                entry.update({"frame": p.frame, "tier": p.tier})
                record["keep_one_age0"] = entry
            else:
                record[p.kind] = entry  # visual_all, age0, visual_all_mask
        if "keep_one" in self.measures:
            # Age 0 of the keep-one curve: the current frame alone (the history_all prompt)
            # against no visual at all.
            h = record["history_all"]
            record["keep_one_age0"] = {"kl": kl_divergence(h["logp"], vis_logp), "logp": h["logp"],
                                       "p_mass": h["p_mass"], "argmax_action": h["argmax_action"],
                                       "frame": layout.n_frames - 1, "tier": "short"}
        record["n_probes"] = len(probes)
        record["probe_seconds"] = time.perf_counter() - started
        self._embeds = self._input_ids = self._result_list = None
        return record
