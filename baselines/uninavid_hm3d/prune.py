"""Persistent visual-token pruning for Uni-NaVid (paper/token_prune).

Under a finite context window (``--context-frames K``) Uni-NaVid rebuilds a single-turn
prompt every inference from ``feat_cache``: one post-projector row per observed frame, each
flattened into a 4-token history group (``online_process_tensor`` short-circuits at or below
64 frames, so there is no long-term tier), separated by ``<image_sep>``; the current frame
additionally supplies 64 fine tokens. The pruning hook acts on the *history pool* -- the
4-token groups of every frame except the current one -- and keeps a budget of
``B' = k * 4`` tokens ("what k history frames cost this model").

Protocol -- permanent deletion:

* A history token that is not kept is deleted for good, keyed by ``(frame id, token index)``
  (frame id = env step of the observation). Later inferences select among the survivors
  plus the frames that have since become history; nothing is pruned while the pool holds
  ``<= B'`` tokens.
* Input-level methods (``random``, ``divprune``) remove the deleted rows from the assembled
  prompt (``online_process_tensor`` wrapper); a frame whose four tokens are all gone loses
  its separator with them. Layer-placed methods (``fastv``, ``dart`` at layer 2;
  ``pyramiddrop`` staged at layers 8/16/24) keep every row in the prompt and hide the
  deleted tokens' key columns from that layer onward, on the prefill and on every decode
  step, so layers below the cut still see the whole window -- exactly as the papers
  prescribe. Scores are read inside the same prefill (FastV: the last prompt token's
  attention at layer L-1, mean over heads; DART: L1 norm of the layer-1 keys picks pivots,
  cosine similarity of the layer-1 output hidden states ranks duplicates).
* Flat variant (``layer=0``, ``--prune-layer 0``): every method scores AND deletes at decoder
  layer 0, by the row removal of the input-level methods. For ``fastv`` / ``dart`` a scoring
  pre-pass runs decoder layer 0 alone on the assembled prompt (rows deleted by earlier
  inferences already absent), reads the same quantities there (the last prompt token's
  layer-0 attention; layer-0 keys and output hidden states), selects, and removes the freshly
  deleted rows -- and an emptied frame's separator -- before the real prefill. The four
  methods then differ in the selection rule only. ``random`` / ``divprune`` are unchanged.
* ``hybrid`` (paper/hybrid_prune): PyramidDrop-like attention stages at layers 8/16 whose
  geometric schedule ends at ``B'`` (``lambda = (B'/V0)^(1/2)``), then a RANDOM last stage from
  layer 24 onward that keeps ``TAIL_FRAMES`` frames' worth (4 tokens) of the previous stage's
  survivors whatever ``k`` is -- it binds as soon as the pool exceeds it, before the schedule does.
* Whole-model budget (``scope="model"``, ``--prune-budget-scope model``; staged methods only): ``B'``
  bounds the history tokens a query can attend to AVERAGED OVER ALL 32 LAYERS instead of the last
  stage's count. A stage at layer 0 is prepended -- the flat variant's layer-0 pre-pass, which ranks
  by the layer-0 attention and REMOVES the rows, so layers 0-7 hold its survivors -- and the
  constant ratio ``lambda`` of the schedule is solved each inference so that the layer-weighted
  mean (hybrid: random tail included) equals ``B'`` (``model_stage_budgets``).
* Schedule anchor (``anchor``, ``--prune-schedule-anchor``): the stage budgets shrink from the
  history's ORIGINAL token count (``original``, paper/token_prune) or from the tokens the first stage
  can draw from now -- its survivors plus the frames that just became history (``pool``,
  paper/hybrid_prune: "keep a fraction lambda of what the previous stage holds").
* The selection RNG (random, hybrid) is a private generator reseeded per (seed, episode, step),
  so the official sampled decoding is untouched.

``TokenPruner`` owns the bookkeeping and the model hooks; ``PruneState`` the frame/token
ledger; the selectors are pure functions so they can be tested without weights.
"""
from __future__ import annotations

import base64
import time
import zlib
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

N_PER_FRAME = 4
NAV_TOKENS = 64
LENGTH_THRESHOLD = 64
SIMILARITY_THRESHOLD = 0.985
METHODS = ("identity", "random", "fastv", "pyramiddrop", "divprune", "dart", "hybrid")
INPUT_LEVEL = ("random", "divprune")
# (scoring layer, first hidden layer) per stage; LLaMA-7B has 32 decoder layers. A stage without
# a scoring layer is a random stage, drawn when its first hidden layer runs.
STAGES = {"fastv": ((1, 2),), "dart": ((1, 2),), "pyramiddrop": ((7, 8), (15, 16), (23, 24)),
          "hybrid": ((7, 8), (15, 16), (None, 24))}
STAGED = ("pyramiddrop", "hybrid")  # the stages are the method: no layer-0 variant
TAIL_FRAMES = 1  # hybrid: frames' worth kept by the random last stage
N_LAYERS = 32
SCOPES = ("last_stage", "model")
ANCHORS = ("original", "pool")  # what the stage schedule shrinks from: h x 4 tokens, or the tokens alive now
PREPASS_STAGE = (0, 0)  # scope='model': score at layer 0 in a pre-pass, remove the rows
FLAT_METHODS = ("fastv", "dart")  # layer-placed methods that ``layer=0`` turns into a layer-0 pre-pass


def check_layer(method: str, layer: int | None) -> bool:
    """Validate ``layer`` (None = the method's paper placement, 0 = flat); True for a pre-pass arm."""
    if layer is None:
        return False
    if int(layer) != 0:
        raise ValueError(f"prune layer must be 0 or unset, got {layer!r}")
    if method == "identity" or method in STAGED:
        raise ValueError(f"{method} has no layer-0 variant")
    return method in FLAT_METHODS


# ---------------------------------------------------------------------------- selectors
def selection_seed(seed: int, label: str, step: int) -> int:
    """Per-(episode, step) seed of the random selector."""
    return (int(seed) * 1_000_003 + int(step)) ^ zlib.crc32(label.encode("utf-8"))


def select_random(n_pool: int, budget: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform sample of ``budget`` pool indices (sorted)."""
    return torch.randperm(n_pool, generator=generator)[:budget].sort().values


def select_topk(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Indices of the ``budget`` largest scores (sorted)."""
    return torch.topk(scores.float(), budget).indices.sort().values


def cosine_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    return 1.0 - x @ x.T


def select_divprune(x: torch.Tensor, budget: int) -> torch.Tensor:
    """DivPrune: greedy Max-Min diversity under cosine distance on the input embeddings.

    Starts from the token whose nearest neighbour is farthest away, then repeatedly adds the
    token maximising its minimum distance to the selected set (Alvar et al., CVPR 2025).
    """
    n = x.shape[0]
    if budget >= n:
        return torch.arange(n)
    dist = cosine_distance_matrix(x)
    eye = torch.eye(n, dtype=torch.bool, device=dist.device)
    first = dist.masked_fill(eye, float("inf")).amin(dim=1).argmax()
    chosen = [first]
    min_dist = dist[first].clone()
    min_dist[first] = -float("inf")
    for _ in range(budget - 1):
        pick = min_dist.argmax()
        chosen.append(pick)
        min_dist = torch.minimum(min_dist, dist[pick])
        min_dist[pick] = -float("inf")
    return torch.stack(chosen).cpu().sort().values


def select_dart(keys: torch.Tensor, hidden: torch.Tensor, budget: int, n_pivots: int) -> torch.Tensor:
    """DART: keep pivots (largest key L1 norm) plus, per pivot, the least-duplicated tokens.

    Pivots are visited in descending norm order; each takes ``q = (B'-P)//P`` of the tokens
    with the lowest cosine similarity to it among those not yet retained, the remainder
    ``B'-P-P*q`` going one extra to the first pivots (Wen et al., 2025; reference
    ``get_retained_image_token`` with the pool restricted to the history tokens).
    """
    n = keys.shape[0]
    if budget >= n:
        return torch.arange(n)
    n_pivots = max(1, min(int(n_pivots), budget))
    norms = keys.float().abs().sum(dim=-1)
    pivots = torch.topk(norms, n_pivots).indices
    sim = F.normalize(hidden.float(), dim=-1)
    sim = sim @ sim.T
    remaining = torch.ones(n, dtype=torch.bool, device=sim.device)
    remaining[pivots] = False
    q, r = divmod(budget - n_pivots, n_pivots)
    chosen = pivots.tolist()
    for rank, pivot in enumerate(pivots.tolist()):
        take = q + (1 if rank < r else 0)
        if take == 0:
            continue
        picks = torch.topk(sim[pivot].masked_fill(~remaining, float("inf")), take, largest=False).indices
        remaining[picks] = False
        chosen.extend(picks.tolist())
    return torch.tensor(sorted(chosen), dtype=torch.long)


def stage_budgets(v0: int, budget: int, n_stages: int) -> list[int]:
    """PyramidDrop budgets ``round(V0 * lambda^(s+1))``, lambda = (B'/V0)^(1/S), last = B'."""
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1")
    if v0 <= budget:
        return [int(v0)] * n_stages
    lam = (budget / v0) ** (1.0 / n_stages)
    out = [int(round(v0 * lam ** (s + 1))) for s in range(n_stages)]
    out[-1] = int(budget)
    for s in range(n_stages - 2, -1, -1):
        out[s] = max(out[s], out[s + 1])
    return out


def model_stage_budgets(v0: int, budget: int, widths: Sequence[int], tail: tuple[int, int] | None = None) -> list[int]:
    """Nested budgets ``round(V0 * lambda^(s+1))`` whose LAYER-WEIGHTED MEAN over the model is
    ``budget``: ``widths`` = layers per scheduled stage, ``tail`` = (layers, tokens) of a last stage
    with a fixed budget. lambda in [0, 1]: 1 while the unpruned history is under budget."""
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


def pack_bits(bits: Sequence[bool]) -> str:
    """base64 of ``numpy.packbits`` over a boolean keep mask (big-endian bit order)."""
    return base64.b64encode(np.packbits(np.asarray(list(bits), dtype=bool))).decode("ascii")


def unpack_bits(packed: str, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))[:n].astype(bool)


# ------------------------------------------------------------------------------ ledger
Token = tuple[int, int]  # (frame id, token index)


class PruneState:
    """Frames in ``feat_cache`` order and the tokens deleted at each stage (nested sets)."""

    def __init__(self, n_stages: int, n_per_frame: int = N_PER_FRAME) -> None:
        self.n_stages = int(n_stages)
        self.n_per_frame = int(n_per_frame)
        self.reset()

    def reset(self) -> None:
        self.frame_ids: list[int] = []
        self.deleted: list[set[Token]] = [set() for _ in range(self.n_stages)]

    def advance(self, n_old: int, new_ids: Sequence[int]) -> None:
        """Mirror ``trim_visual_history`` + the append inside ``token_generation``."""
        if n_old > len(self.frame_ids):
            raise RuntimeError(f"{n_old} rows survived the trim but only {len(self.frame_ids)} frames are known")
        self.frame_ids = (self.frame_ids[len(self.frame_ids) - n_old:] if n_old else []) + [int(i) for i in new_ids]
        alive = set(self.frame_ids)
        for gone in self.deleted:
            gone.intersection_update({t for t in gone if t[0] in alive})

    @property
    def history(self) -> list[int]:
        """Frame ids of the pool: everything but the current (last) frame."""
        return self.frame_ids[:-1]

    def pool(self, stage: int) -> list[Token]:
        gone = self.deleted[stage]
        return [(f, t) for f in self.history for t in range(self.n_per_frame) if (f, t) not in gone]

    def delete(self, stage: int, tokens: Sequence[Token]) -> None:
        """Deleting at stage ``s`` deletes at every deeper stage (nested masks)."""
        for s in range(stage, self.n_stages):
            self.deleted[s].update(tokens)

    def kept_per_frame(self, stage: int) -> list[int]:
        gone = self.deleted[stage]
        return [sum((f, t) not in gone for t in range(self.n_per_frame)) for f in self.history]

    def keep_bits(self, stage: int) -> list[bool]:
        gone = self.deleted[stage]
        return [(f, t) not in gone for f in self.history for t in range(self.n_per_frame)]


# ------------------------------------------------------------------------------ hooks
def history_columns(placeholder: int, result_list: Sequence[int]) -> list[tuple[int, int]]:
    """``[start, end)`` of each history group in the assembled prompt (groups separated by
    one ``<image_sep>``), mirroring ``prepare_inputs_labels_for_multimodal``."""
    spans, pos = [], int(placeholder)
    for j, g in enumerate(result_list):
        spans.append((pos, pos + int(g)))
        pos += int(g) + (1 if j < len(result_list) - 1 else 0)
    return spans


def prompt_length(n_ids: int, result_list: Sequence[int]) -> int:
    return int(n_ids) - 1 + sum(int(g) for g in result_list) + max(len(result_list) - 1, 0) + NAV_TOKENS


def check_scope(method: str, scope: str, layer: int | None = None) -> bool:
    """Validate ``scope``; True for a whole-model budget."""
    if scope not in SCOPES:
        raise ValueError(f"budget scope must be one of {SCOPES}, got {scope!r}")
    if scope == "model" and (method not in STAGED or layer is not None):
        raise ValueError("a whole-model budget needs a staged method (pyramiddrop, hybrid) and no prune layer")
    return scope == "model"


def describe(method: str, k: int | None, seed: int, n_pivots: int, layer: int | None = None,
             scope: str = "last_stage", anchor: str = "original") -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown prune method {method!r}; expected one of {METHODS}")
    if anchor not in ANCHORS or (anchor == "pool" and method not in STAGED):
        raise ValueError(f"schedule anchor must be one of {ANCHORS} ('pool' for staged methods only), got {anchor!r}")
    flat = check_layer(method, layer)
    whole = check_scope(method, scope, layer)
    stages = ((0, 0),) if flat else STAGES.get(method)
    if whole:
        stages = (PREPASS_STAGE,) + stages
    return {
        "method": method,
        "k": k,
        "budget_tokens": None if k is None or method == "identity" else int(k) * N_PER_FRAME,
        "n_per_frame": N_PER_FRAME,
        "protocol": "permanent deletion",
        "placement": "input" if method in INPUT_LEVEL else "layer0-prepass" if flat else "layer" if stages else None,
        "first_hidden_layer": None if stages is None else stages[0][1],
        "stages": None if stages is None else [list(s) for s in stages],
        "pivots": n_pivots if method == "dart" else None,
        "seed": seed if method in ("random", "hybrid") else None,
        "query": "last prompt token, mean over heads" if method in ("fastv", "pyramiddrop", "hybrid") else None,
        "stage_selectors": None if method != "hybrid" else ["random" if sc is None else "attn" for sc, _ in stages],
        "tail_budget_tokens": TAIL_FRAMES * N_PER_FRAME if method == "hybrid" else None,
        "prune_layer": layer,
        "removal": None if method == "identity" else "rows" if method in INPUT_LEVEL or flat
        else "rows at layer 0, columns after" if whole else "columns",
        "budget_scope": scope,
        "schedule_anchor": anchor if method in STAGED else None,
    }


class TokenPruner:
    """Install the pruning hook on a loaded Uni-NaVid model and keep its ledger."""

    def __init__(self, method: str, k: int, image_token_index: int, seed: int = 17,
                 n_pivots: int = 8, layer: int | None = None, scope: str = "last_stage",
                 anchor: str = "original") -> None:
        if method not in METHODS or method == "identity":
            raise ValueError(f"TokenPruner needs a pruning method, got {method!r}")
        if int(k) < 1:
            raise ValueError("k must be at least 1")
        self.method = method
        self.k = int(k)
        self.budget = self.k * N_PER_FRAME
        self.image_token_index = int(image_token_index)
        self.seed = int(seed)
        self.n_pivots = int(n_pivots)
        self.layer = layer
        self.flat = check_layer(method, layer)  # layer-0 pre-pass + row removal
        self.scope = scope
        self.anchor = anchor
        describe(method, k, seed, n_pivots, layer, scope, anchor)  # validates the combination
        self.whole = check_scope(method, scope, layer)  # whole-model budget: pre-pass stage + the method's own
        self.prepass = self.flat or self.whole
        self.stages = () if self.flat else STAGES.get(method, ())
        if self.whole:
            self.stages = (PREPASS_STAGE,) + self.stages
        self.n_stages = max(len(self.stages), 1)
        self.state = PruneState(self.n_stages)
        self.label = ""
        self._installed = False
        self.reset()

    def describe(self) -> dict[str, Any]:
        return describe(self.method, self.k, self.seed, self.n_pivots, self.layer, self.scope, self.anchor)

    def reset(self) -> None:
        self.state.reset()
        self._clear_inference()

    def begin_episode(self, label: str) -> None:
        self.label = str(label)

    def _clear_inference(self) -> None:
        self._step: int | None = None
        self._placeholder: int | None = None
        self._n_ids = 0
        self._spans: list[tuple[int, int]] | None = None
        self._col: dict[Token, int] | None = None  # flat: prompt column of every surviving history token
        self._sep: dict[int, int] = {}  # flat: column of the separator closing each history group
        self._prompt_len: int | None = None
        self._budgets: list[int] = []
        self._widths: list[int] | None = None
        self._n_pool = 0
        self._hide: list[torch.Tensor | None] = [None] * self.n_stages
        self._selected = [False] * self.n_stages
        self._seconds = 0.0

    # -- per inference -----------------------------------------------------------------
    def begin_inference(self, input_ids: torch.Tensor, step: int) -> None:
        """Called before ``generate``; the pool is the history of ``state.frame_ids``."""
        ids = input_ids.reshape(-1)
        where = torch.nonzero(ids == self.image_token_index).flatten().tolist()
        if len(where) != 1:
            raise ValueError(f"expected exactly one image placeholder, found {len(where)}")
        self._clear_inference()
        self._step = int(step)
        self._placeholder = int(where[0])
        self._n_ids = int(ids.numel())
        self._n_pool = len(self.state.pool(0))
        # Stage budgets shrink geometrically from the window's ORIGINAL token count (h x 4,
        # the same denominator LongNav-R1 uses), not from the already-pruned pool.
        v0 = len(self.state.history) * N_PER_FRAME
        if self.anchor == "pool":
            v0 = self._n_pool  # what stage 0 can draw from now: its survivors + the new history frames
        tailed = self.method == "hybrid"  # the random stage has its own budget, outside the schedule
        n_sched = self.n_stages - (1 if tailed else 0)
        if self.whole:
            starts = [hide for _, hide in self.stages] + [N_LAYERS]
            widths = [b - a for a, b in zip(starts[:-1], starts[1:])]
            sched = model_stage_budgets(v0, self.budget, widths[:n_sched],
                                        tail=(widths[-1], TAIL_FRAMES * N_PER_FRAME) if tailed else None)
        else:
            sched = stage_budgets(v0, self.budget, n_sched)
        self._budgets = sched + ([min(TAIL_FRAMES * N_PER_FRAME, sched[-1])] if tailed else [])
        self._widths = widths if self.whole else None
        self._refresh_hide()

    def finish(self) -> dict[str, Any]:
        """Record of the inference just decoded (per-frame retention at the deepest stage)."""
        if self._step is None:
            raise RuntimeError("finish() called without begin_inference()")
        if not all(self._selected):
            raise RuntimeError("the pruning hook did not run on every stage; is it installed?")
        deepest = self.n_stages - 1
        record = {
            "step": self._step,
            "frames": list(self.state.history),
            "n_tokens": [N_PER_FRAME] * len(self.state.history),
            "n_kept": self.state.kept_per_frame(deepest),
            "bits": pack_bits(self.state.keep_bits(deepest)),
            "budget": self.budget,
            "n_pool": self._n_pool,
            "stage_kept": [len(self.state.pool(s)) for s in range(self.n_stages)] if self.n_stages > 1 else None,
            "stage_budgets": list(self._budgets) if self.n_stages > 1 else None,
            # whole-model budget: history tokens attended, averaged over all layers
            "held_tokens": None if self._widths is None else sum(
                w * len(self.state.pool(s)) for s, w in enumerate(self._widths)) / N_LAYERS,
            "prompt_tokens": self._prompt_len,
            "seconds": self._seconds,
        }
        self._clear_inference()
        return record

    # -- selection ---------------------------------------------------------------------
    def _select(self, stage: int, choose: Callable[[list[Token]], torch.Tensor]) -> None:
        """Delete every pool token of ``stage`` that ``choose`` does not return."""
        started = time.perf_counter()
        pool = self.state.pool(stage)
        budget = self._budgets[stage]
        if len(pool) > budget:
            keep = choose(pool).tolist()
            if len(set(keep)) != budget:
                raise RuntimeError(f"selector returned {len(set(keep))} tokens for a budget of {budget}")
            kept = set(keep)
            self.state.delete(stage, [tok for i, tok in enumerate(pool) if i not in kept])
        self._selected[stage] = True
        self._refresh_hide()
        self._seconds += time.perf_counter() - started

    def _refresh_hide(self) -> None:
        """Per-stage boolean column mask over the prompt of the deleted, still-present tokens."""
        if self._spans is None and self._col is None:
            return
        rows = {f: i for i, f in enumerate(self.state.history)}
        for s in range(self.n_stages):
            hide = torch.zeros(self._prompt_len, dtype=torch.bool)
            for tok in self.state.deleted[s]:
                if self._col is not None:  # pre-pass arms: rows removed at layer 0 have no column
                    if tok in self._col:
                        hide[self._col[tok]] = True
                elif tok[0] in rows:
                    hide[self._spans[rows[tok[0]]][0] + tok[1]] = True
            self._hide[s] = hide

    def _columns(self, pool: Sequence[Token]) -> torch.Tensor:
        if self._col is not None:
            return torch.tensor([self._col[tok] for tok in pool], dtype=torch.long)
        rows = {f: i for i, f in enumerate(self.state.history)}
        return torch.tensor([self._spans[rows[f]][0] + t for f, t in pool], dtype=torch.long)

    def hidden_mask(self, attention_mask: torch.Tensor, stage: int) -> torch.Tensor:
        """Out-of-place copy of a 4-D additive mask with the stage's deleted columns closed."""
        hide = self._hide[stage]
        if hide is None or not bool(hide.any()):
            return attention_mask
        if attention_mask is None:
            raise RuntimeError("a decoder layer received no attention mask")
        width = attention_mask.shape[-1]
        if width < hide.numel():
            raise RuntimeError(f"attention mask has {width} columns for a {hide.numel()}-token prompt")
        cols = torch.zeros(width, dtype=torch.bool, device=attention_mask.device)
        cols[: hide.numel()] = hide.to(attention_mask.device)
        return attention_mask.masked_fill(cols.view(1, 1, 1, width), torch.finfo(attention_mask.dtype).min)

    # -- installation ------------------------------------------------------------------
    def install(self, model: Any) -> None:
        if self._installed:
            return
        self._wrap_online(model)
        if self.prepass:
            self._wrap_prefill(model)
        hooked = self._hooked_stages()
        if hooked:
            layers = model.get_model().layers
            first = min(hide for _, _, hide in hooked)
            for index, layer in enumerate(layers):
                if index >= first or any(index == score for _, score, _ in hooked):
                    layer.forward = self._layer_forward(layer, index)
        self._installed = True

    def _hooked_stages(self) -> list[tuple[int, int | None, int]]:
        """(stage, score layer, first hidden layer) of the stages that act through the layer hooks --
        every stage of a layer-placed method except a pre-pass stage."""
        if self.flat or self.method in INPUT_LEVEL:
            return []
        return [(s, score, hide) for s, (score, hide) in enumerate(self.stages) if not (self.prepass and s == 0)]

    def _wrap_online(self, model: Any) -> None:
        orig = model.online_process_tensor
        state = model.get_model()
        pruner = self

        def online_process_tensor(nav_size, length_threshold=LENGTH_THRESHOLD,
                                  similarity_threshold=SIMILARITY_THRESHOLD):
            tokens, result_list = orig(nav_size, length_threshold, similarity_threshold)
            if pruner._step is None:
                return tokens, result_list
            n_frames = int(state.feat_cache.shape[0])
            if n_frames != len(pruner.state.frame_ids):
                raise RuntimeError(f"feat_cache holds {n_frames} frames but the ledger {len(pruner.state.frame_ids)}")
            if list(result_list) != [N_PER_FRAME] * n_frames:
                raise RuntimeError(f"unexpected history layout {list(result_list)}; is the context window set?")
            if pruner.method in INPUT_LEVEL or pruner.prepass:
                return pruner._prune_input(tokens, n_frames)
            pruner._spans = history_columns(pruner._placeholder, result_list)
            pruner._prompt_len = prompt_length(pruner._n_ids, result_list)
            pruner._refresh_hide()
            return tokens, result_list

        model.online_process_tensor = online_process_tensor

    def _prune_input(self, tokens: torch.Tensor, n_frames: int) -> tuple[torch.Tensor, list[int]]:
        """Input-level methods: select on the flattened history rows, return the pruned prompt.
        Flat fastv/dart select later (``_prepass``); here only the rows deleted by earlier
        inferences go, and the surviving tokens' prompt columns are recorded."""
        rows = {f: i for i, f in enumerate(self.state.history)}

        def flat(pool: Sequence[Token]) -> torch.Tensor:
            return torch.tensor([rows[f] * N_PER_FRAME + t for f, t in pool], dtype=torch.long)

        if self.method == "random":
            generator = torch.Generator().manual_seed(selection_seed(self.seed, self.label, self._step))
            self._select(0, lambda pool: select_random(len(pool), self._budgets[0], generator))
        elif self.method == "divprune":
            self._select(0, lambda pool: select_divprune(tokens[flat(pool).to(tokens.device)], self._budgets[0]))
        keep = torch.ones(n_frames * N_PER_FRAME, dtype=torch.bool)
        gone = [tok for tok in self.state.deleted[0] if tok[0] in rows]
        if gone:
            keep[flat(gone)] = False
        counts = keep.view(n_frames, N_PER_FRAME).sum(dim=1).tolist()
        result_list = [int(c) for c in counts if c > 0]
        self._prompt_len = prompt_length(self._n_ids, result_list)
        if self.prepass:
            spans = history_columns(self._placeholder, result_list)
            self._col, self._sep, group = {}, {}, 0
            for i, f in enumerate(self.state.frame_ids):
                alive = [t for t in range(N_PER_FRAME) if keep[i * N_PER_FRAME + t]]
                if not alive:
                    continue
                for rank, t in enumerate(alive):
                    self._col[(f, t)] = spans[group][0] + rank
                if group < len(result_list) - 1:
                    self._sep[f] = spans[group][1]
                group += 1
        return tokens[keep.to(tokens.device)], result_list

    def _wrap_prefill(self, model: Any) -> None:
        """Flat fastv/dart: score on decoder layer 0 before the real prefill, then drop the rows."""
        llama = model.get_model()
        orig = llama.forward
        pruner = self

        def forward(*args, **kwargs):
            embeds = kwargs.get("inputs_embeds")
            if (pruner._step is None or pruner._col is None or pruner._selected[0] or embeds is None
                    or kwargs.get("past_key_values") is not None):
                return orig(*args, **kwargs)
            keep = pruner._prepass(llama, embeds, kwargs.get("attention_mask"))
            if not bool(keep.all()):
                kwargs["inputs_embeds"] = embeds[:, keep.to(embeds.device)]
                mask = kwargs.get("attention_mask")
                if mask is not None:
                    kwargs["attention_mask"] = mask[:, keep.to(mask.device)]
            return orig(*args, **kwargs)

        llama.forward = forward

    def _prepass(self, llama: Any, embeds: torch.Tensor, mask2d: torch.Tensor | None) -> torch.Tensor:
        """Run decoder layer 0 on the assembled prompt, select, return the prompt rows to keep."""
        q = int(embeds.shape[1])
        if embeds.shape[0] != 1 or q != self._prompt_len:
            raise RuntimeError(f"prefill has shape {tuple(embeds.shape[:2])}, layout expects (1, {self._prompt_len})")
        keep = torch.ones(q, dtype=torch.bool)
        if len(self.state.pool(0)) <= self._budgets[0]:
            self._selected[0] = True
            return keep
        started = time.perf_counter()
        if mask2d is None:
            mask2d = torch.ones((1, q), dtype=torch.bool, device=embeds.device)
        elif mask2d.shape[-1] != q or not bool(mask2d.bool().all()):
            raise RuntimeError("the layer-0 pre-pass expects an unpadded single-sequence prefill")
        mask = llama._prepare_decoder_attention_mask(mask2d, (1, q), embeds, 0)
        position_ids = torch.arange(q, dtype=torch.long, device=embeds.device).unsqueeze(0)
        out = llama.layers[0](embeds, attention_mask=mask, position_ids=position_ids, past_key_value=None,
                              output_attentions=True, use_cache=True)
        self._seconds += time.perf_counter() - started
        before = set(self.state.deleted[0])
        self._score_and_select(0, out[1], out[0], out[2])
        fresh = self.state.deleted[0] - before
        for tok in fresh:
            keep[self._col[tok]] = False
        for f in {f for f, _ in fresh}:
            if all((f, t) in self.state.deleted[0] for t in range(N_PER_FRAME)):
                keep[self._sep[f]] = False
        n_kept = int(keep.sum())
        expect = prompt_length(self._n_ids, [c for c in self.state.kept_per_frame(0) if c > 0] + [N_PER_FRAME])
        if n_kept != expect:
            raise RuntimeError(f"pre-pass keeps {n_kept} prompt rows, the ledger expects {expect}")
        self._prompt_len = n_kept
        # Later stages hide columns of the SHORTENED prompt: move every survivor to its new column.
        shift = torch.cumsum(keep.long(), 0) - 1
        self._col = {tok: int(shift[c]) for tok, c in self._col.items() if bool(keep[c])}
        self._refresh_hide()
        return keep

    def _layer_forward(self, layer: Any, index: int) -> Callable:
        orig = layer.forward
        hooked = self._hooked_stages()
        score_stage = next((s for s, score, _ in hooked if score == index), None)
        random_stage = next((s for s, score, hide in hooked if score is None and hide == index), None)
        hide_stages = [s for s, _, hide in hooked if hide <= index]
        hide_stage = max(hide_stages) if hide_stages else None
        pruner = self

        def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False):
            if pruner._step is None or (pruner._spans is None and pruner._col is None):
                return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_value=past_key_value, output_attentions=output_attentions,
                            use_cache=use_cache)
            prefill = hidden_states.shape[1] > 1
            if random_stage is not None and prefill and not pruner._selected[random_stage]:
                pruner._select_random_stage(random_stage)
            if hide_stage is not None:
                attention_mask = pruner.hidden_mask(attention_mask, hide_stage)
            if score_stage is None or not prefill:
                return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_value=past_key_value, output_attentions=output_attentions,
                            use_cache=use_cache)
            if hidden_states.shape[1] != pruner._prompt_len:
                raise RuntimeError(f"prefill has {hidden_states.shape[1]} tokens, layout expects {pruner._prompt_len}")
            out = orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                       past_key_value=past_key_value, output_attentions=True, use_cache=use_cache)
            hidden, attn = out[0], out[1]
            present = out[2] if use_cache else None
            pruner._score_and_select(score_stage, attn, hidden, present)
            return (hidden,) + ((attn,) if output_attentions else ()) + ((present,) if use_cache else ())

        return forward

    def _select_random_stage(self, stage: int) -> None:
        """Uniform draw among the previous stage's survivors, just before the stage's first layer."""
        if not all(self._selected[:stage]):
            raise RuntimeError(f"random stage {stage} reached before the stages above it selected")
        generator = torch.Generator().manual_seed(selection_seed(self.seed, self.label, self._step))
        self._select(stage, lambda pool: select_random(len(pool), self._budgets[stage], generator))

    def _score_and_select(self, stage: int, attn: torch.Tensor, hidden: torch.Tensor,
                          present: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        if self.method == "dart":
            if present is None:
                raise RuntimeError("DART needs the scoring layer's key states (use_cache=True)")
            keys = present[0][0].permute(1, 0, 2).reshape(present[0].shape[2], -1)  # (Q, heads*dim)
            self._select(stage, lambda pool: select_dart(
                keys[self._columns(pool).to(keys.device)], hidden[0][self._columns(pool).to(hidden.device)],
                self._budgets[stage], self.n_pivots))
        else:
            row = attn[0, :, -1, :].float().mean(dim=0)  # last prompt token, mean over heads
            self._select(stage, lambda pool: select_topk(row[self._columns(pool).to(row.device)],
                                                         self._budgets[stage]))
