"""Per-visual-token, per-decoder-layer information of the CURRENT frame (paper/layer_token_info).

The layer-wise analogue of Fig. 7 in "When Token Pruning is Worse than Random" (arXiv
2512.07580) for an incremental-cache navigation agent. After a decision step, for every decoder
layer ``i`` and every visual token ``k`` of the frame just decided on, with every hiding applied
on layers ``>= i`` only (layers ``< i`` untouched, as in the paper):

    leave[i, k]    = KL( P_full                         || P_{-k} )
    keep_cur[i, k] = KL( P_{only k; history intact}     || P_{current frame hidden} )
    keep_all[i, k] = KL( P_{only k; every other visual hidden} || P_{no visual} )
    leave_nohist[i, k] = KL( P_{history visuals hidden}  || P_{history visuals hidden, -k} )

``leave_nohist`` is the leave-one-out counterpart of ``keep_all``: both describe the current frame
with no visual history (every history frame's visual tokens hidden on layers ``>= i``; the text of
the history -- the past actions -- stays). ``measures`` selects which are computed.

``P`` is the action distribution at the decision token (``worker.vocab_ids``), KL in nats with
the expectation under the first argument (``VLMWorker._action_kl``). Two arms, one function each:

* :func:`probe_mask_arm` -- the LEAKY arm. Worker in ``context_window_mode='prune'`` with an
  unreachable budget (nothing is ever pruned). Hiding = ``-inf`` on the hidden slots' key
  columns for the replayed decision token only (``VLMWorker._decision_replay`` with
  ``extra_layers``); every other cached token keeps what it absorbed from ``k`` while it was
  encoded. This is the per-slot version of ``_score_layer_frame_influence``.
* :func:`probe_recompute_arm` -- the LEAK-FREE arm. Worker in ``context_window_mode='evict'``
  with the full episode in context and ``token_info_probe=True`` (stored turn records). Every
  probe re-prefills prefix + all turns from the stored post-filter embeddings against an empty
  cache while the hidden columns are invisible to EVERY query on layers ``>= i`` (a forward
  pre-hook on each layer's attention swaps the mask in). Survivors keep their absolute mRoPE
  positions. This is the paper's ``H^i * M`` procedure, with removal instead of zeroing.

Both return the same dict of numpy arrays (see ``_empty_result``) and leave the worker's live
state untouched.

Resolution. The model reads its action from a bf16 ``lm_head``; at the decision's logit
magnitude one bf16 ulp is ~1e-3 nats of KL, and removing a single visual token typically moves
the logits by less than that, so KLs read from the bf16 head collapse onto a few discrete levels.
By default (``fp32_head=True``) every probe instead projects the final-norm hidden state of the
decision token onto the four action rows of the head in float32 (the hidden state itself stays
bf16), which resolves the differences the bf16 rounding hid. The base uses the same reading, so
argmax agrees with the live decision except at near-ties; the smoke pins the bf16 reading against
the frame-level diagnostic and the fp32 reading against the live argmax.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

from longnav.utils.kv_prune import grid_coords_from_pos_table
from longnav.utils.pre_rope import hide_columns

MEASURES = ("leave", "keep_cur", "keep_all")  # the default set
NOHIST_MEASURES = ("keep_all", "leave_nohist")  # the current frame without any visual history
ALL_MEASURES = MEASURES + ("leave_nohist",)


def _check(worker, mode):
    if not getattr(worker, "token_info_probe", False):
        raise RuntimeError("the worker was built without token_info_probe=True")
    if worker.context_window_mode != mode:
        raise RuntimeError(f"this probe needs context_window_mode={mode!r}, "
                           f"worker has {worker.context_window_mode!r}")


def _resolve_layers(worker, layers):
    n_layers = len(worker.language_model.layers)
    if layers is None:
        return list(range(n_layers)), n_layers
    layers = [int(x) for x in layers]
    if any(x < 0 or x >= n_layers for x in layers):
        raise ValueError(f"probe layers must lie in [0, {n_layers}), got {layers}")
    return layers, n_layers


def current_frame_grid(worker):
    """``(t, h, w)`` of the latest frame in vision-patch units (merge 2 -> LLM grid h//2 x w//2)."""
    grid = worker.cumulative_inputs["image_grid_thw"][-1]
    return tuple(int(x) for x in grid.tolist())


def current_frame_tokens(worker):
    """Where the current frame's visual tokens sit.

    Returns ``(cols, raster, grid_thw)``: ``cols`` (N,) long -- master cache slots (prune mode) or
    positions in the concatenated stored records (evict mode) -- and ``raster`` (N,) long, each
    token's row-major index in the frame's merged token grid. Asserts the invariant every
    heatmap relies on: cache order == raster order (ascending), and, in prune mode, that the
    mRoPE table agrees with the raster coordinates.
    """
    grid = current_frame_grid(worker)
    gh, gw = grid[1] // 2, grid[2] // 2
    if worker.context_window_mode == "prune":
        ps, st = worker._kv_prune_state, worker._reindex_state
        mask = ps.is_visual & (ps.turn_id == ps.n_turns - 1)
        cols = torch.nonzero(mask, as_tuple=False).flatten()
        raster = ps.raster[mask].clone()
        rc = grid_coords_from_pos_table(st.pos_table)[cols]
        expect = torch.stack([raster // gw, raster % gw], dim=1)
        if not torch.equal(rc, expect):
            raise RuntimeError("mRoPE (row, col) of the current frame's slots disagree with their raster indices")
    else:
        recs = [worker._prefix_rec] + list(worker._window_recs)
        if worker._prefix_rec is None or not worker._window_recs:
            raise RuntimeError("no stored turn records; was the worker built with token_info_probe=True?")
        offset = sum(int(r["inputs_embeds"].shape[1]) for r in recs[:-1])
        vpm = recs[-1]["visual_pos_masks"]
        if vpm is None:
            raise RuntimeError("the current turn's record has no visual positions")
        cols = torch.nonzero(vpm[0], as_tuple=False).flatten() + offset
        raster = torch.nonzero(worker.vis_keep_masks[-1].reshape(-1), as_tuple=False).flatten()
    if raster.numel() != cols.numel():
        raise RuntimeError(f"{cols.numel()} visual slots but {raster.numel()} kept patches")
    if raster.numel() and not bool((raster[1:] > raster[:-1]).all()):
        raise RuntimeError("raster indices are not ascending: cache order != raster order")
    if raster.numel() and int(raster.max()) >= gh * gw:
        raise RuntimeError("raster index outside the merged grid")
    return cols.cpu(), raster.cpu(), grid


def _resolve_measures(measures):
    measures = tuple(MEASURES if measures is None else measures)
    unknown = [m for m in measures if m not in ALL_MEASURES]
    if unknown or not measures:
        raise ValueError(f"measures must be a non-empty subset of {ALL_MEASURES}, got {measures}")
    return measures


def _empty_result(layers, n_tokens, n_vocab, measures=MEASURES):
    L = len(layers)
    out = {
        "layers": np.asarray(layers, dtype=np.int64),
        "measures": np.asarray(measures),
        "base_logp": np.zeros(n_vocab, dtype=np.float32),
        "text_cur_logp": np.zeros((L, n_vocab), dtype=np.float32),
        "text_all_logp": np.zeros((L, n_vocab), dtype=np.float32),
        # the two joint references relative to the full decision, per layer
        "text_cur_kl": np.zeros(L, dtype=np.float32),
        "text_all_kl": np.zeros(L, dtype=np.float32),
    }
    for m in measures:
        out[m] = np.zeros((L, n_tokens), dtype=np.float32)
    if "leave_nohist" in measures:
        # the reference of leave_nohist: every history frame's visual tokens hidden
        out["text_hist_logp"] = np.zeros((L, n_vocab), dtype=np.float32)
        out["text_hist_kl"] = np.zeros(L, dtype=np.float32)
    return out


def n_forwards(n_layers, n_tokens, measures=MEASURES):
    """Forwards / replays one arm costs: the base, per layer the references, per token one per measure."""
    return 1 + n_layers * (2 + int("leave_nohist" in measures) + len(measures) * n_tokens)


def _run_probes(worker, out, layers, measures, cols, cur, visual_all, forward, full, progress, t0):
    """The probe loop both arms share. ``forward(hide, start)`` -> action log-probs with the bool
    ``hide`` columns hidden on layers ``>= start``; ``cur`` / ``visual_all`` are bool masks in the
    arm's column space (master slots or re-prefill positions) and ``cols`` the current frame's
    columns in raster order."""
    kl = worker._action_kl
    hist = visual_all & ~cur
    none = torch.zeros_like(cur)
    for li, start in enumerate(layers):
        text_cur = forward(cur, start)
        text_all = forward(visual_all, start)
        out["text_cur_logp"][li] = text_cur.float().cpu().numpy()
        out["text_all_logp"][li] = text_all.float().cpu().numpy()
        out["text_cur_kl"][li] = kl(full, text_cur)
        out["text_all_kl"][li] = kl(full, text_all)
        if "leave_nohist" in measures:
            text_hist = forward(hist, start)
            out["text_hist_logp"][li] = text_hist.float().cpu().numpy()
            out["text_hist_kl"][li] = kl(full, text_hist)
        for k in range(cols.numel()):
            c = int(cols[k])
            if "leave" in measures:
                h = none.clone(); h[c] = True
                out["leave"][li, k] = kl(full, forward(h, start))
            if "keep_cur" in measures:
                h = cur.clone(); h[c] = False
                out["keep_cur"][li, k] = kl(forward(h, start), text_cur)
            if "keep_all" in measures:
                h = visual_all.clone(); h[c] = False
                out["keep_all"][li, k] = kl(forward(h, start), text_all)
            if "leave_nohist" in measures:
                h = hist.clone(); h[c] = True
                out["leave_nohist"][li, k] = kl(text_hist, forward(h, start))
        if progress is not None:
            progress(li, start, time.perf_counter() - t0)


class _ActionHead:
    """Log action distribution from the decision token's final-norm hidden state."""

    def __init__(self, worker, fp32):
        self.worker, self.fp32 = worker, bool(fp32)
        head = worker.model.get_output_embeddings()
        self.w = head.weight[worker.vocab_ids].detach().float() if self.fp32 else None

    def from_hidden(self, hidden):
        """``hidden`` (D,) or (1, D): the post-norm hidden state of the decision position."""
        with torch.no_grad():
            h = hidden.detach().reshape(1, -1)
            if self.fp32:
                logits = (h.float() @ self.w.T)[0]
            else:
                logits = self.worker.model.get_output_embeddings()(h)[0, self.worker.vocab_ids].float()
            return torch.log_softmax(logits, dim=-1)


def _finite(logp, what):
    if not bool(torch.isfinite(logp).all()):
        raise RuntimeError(f"non-finite action log-probs from the {what} probe")
    return logp


# --------------------------------------------------------------------------- leaky arm


def replay_action_logp(worker, head, attention_bias, extra=None, extra_layers=None):
    """One decision replay read through ``head`` (see _ActionHead)."""
    out = worker._decision_replay(attention_bias, extra=extra, extra_layers=extra_layers)
    return _finite(head.from_hidden(out["last_hidden_state"][0, -1]), "mask")


def probe_mask_arm(worker, layers=None, progress=None, fp32_head=True, measures=None):
    """Leaky per-slot probes on the prune substrate (see module docstring).

    Cost: ``n_forwards(L, N, measures)`` single-token decision replays (``1 + L * (2 + 3N)`` for
    the default measures).
    """
    _check(worker, "prune")
    measures = _resolve_measures(measures)
    head = _ActionHead(worker, fp32_head)
    layers, n_layers = _resolve_layers(worker, layers)
    if worker.context_window is not None:
        raise RuntimeError("the mask arm assumes nothing is window-forced (context_window=None)")
    ps = worker._kv_prune_state
    cols, raster, grid = current_frame_tokens(worker)
    n = cols.numel()
    kv_len = worker._master_len()
    base_bias = worker._replay_base_bias(torch.zeros(kv_len, dtype=torch.bool))

    def forward(hide=None, start=0):
        if hide is None:
            return replay_action_logp(worker, head, base_bias)
        return replay_action_logp(worker, head, base_bias, extra=worker._replay_ablation_term(hide),
                                  extra_layers=range(start, n_layers))

    t0 = time.perf_counter()
    n0 = worker._prune_replay_count
    full = forward()
    out = _empty_result(layers, n, full.numel(), measures)
    out["base_logp"][:] = full.float().cpu().numpy()
    visual_all = ps.is_visual.clone().cpu()
    cur = visual_all & (ps.turn_id == ps.n_turns - 1).cpu()
    _run_probes(worker, out, layers, measures, cols, cur, visual_all, forward, full, progress, t0)
    out["raster"] = raster.numpy()
    out["grid_thw"] = np.asarray(grid, dtype=np.int64)
    out["fp32_head"] = np.bool_(fp32_head)
    out["n_probes"] = np.int64(worker._prune_replay_count - n0)
    out["seconds"] = np.float64(time.perf_counter() - t0)
    return out


# --------------------------------------------------------------------------- leak-free arm


class _ColumnHider:
    """One forward pre-hook per decoder attention, gated by shared state.

    While ``hide`` is set, layers with index ``>= start`` replace their (None) attention mask
    with a causal mask whose ``hide`` columns no query may read. The scratch prefill has no
    past, so keys == queries == the chunk length.

    Every forward of the leak-free arm goes through here, the base included (with nothing
    hidden): sdpa takes a different kernel with a materialised mask than without one, and the
    bf16 gap between the two paths (~1e-2 in probability) would otherwise sit inside every
    ``leave`` value. Probes therefore only ever differ by the hidden columns.
    """

    def __init__(self, worker):
        self.worker = worker
        self.hide = None
        self._none = None
        self.start = 0
        self.handles = []

    def _hook_for(self, layer_idx):
        def hook(module, args, kwargs):
            if self.hide is None:
                return None
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if kwargs.get("attention_mask") is not None:
                raise RuntimeError("expected attention_mask=None in the scratch prefill")
            q = int(hidden.shape[1])
            if q != self.hide.numel():
                raise RuntimeError(f"hide vector covers {self.hide.numel()} columns, chunk has {q}")
            # Layers below `start` get an all-visible mask rather than none: every layer of every
            # forward then takes the same (masked) kernel path and only the hidden columns differ.
            hide = self.hide if layer_idx >= self.start else self._none
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = hide_columns(None, hide, q, q)
            return args, kwargs
        return hook

    def __enter__(self):
        for j, layer in enumerate(self.worker.language_model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook_for(j), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hide = None
        return False

    @contextlib.contextmanager
    def hiding(self, hide, start):
        self.hide, self.start = hide, int(start)
        self._none = torch.zeros_like(hide)
        try:
            yield
        finally:
            self.hide = None
            self._none = None

    def forward(self, recs, hide=None, start=0, head=None):
        """Action log-probs of ``recs`` re-prefilled with ``hide`` columns invisible on layers
        ``>= start``; ``hide=None`` hides nothing but still takes the masked kernel path.
        ``head`` (an _ActionHead) reads the decision; None = the worker's bf16 head."""
        if hide is None:
            T = sum(int(r["inputs_embeds"].shape[1]) for r in recs)
            hide = torch.zeros(T, dtype=torch.bool, device=recs[0]["inputs_embeds"].device)
        with self.hiding(hide, start):
            if head is None:
                return _finite(self.worker._influence_forward_logp(recs), "recompute")
            return _finite(_forward_last_hidden(self.worker, recs, head), "recompute")


def _forward_last_hidden(worker, recs, head):
    """_influence_forward_logp with the decision read through ``head`` instead of the bf16 lm_head."""
    merged = worker._cat_recs(recs)
    with worker._inference_ctx():
        out = worker.language_model(**merged, attention_mask=None, past_key_values=None,
                                    use_cache=True, seq_keep_mask='everything',
                                    vis_keep_mask='everything')
        logp = head.from_hidden(out.last_hidden_state[0, -1]).cpu()
    del out
    worker._influence_probe_count += 1
    return logp


def forward_logp_masked(worker, recs, hide=None, start=0, fp32_head=False):
    """One hooked re-prefill of ``recs`` (masked kernel path; see _ColumnHider)."""
    head = _ActionHead(worker, True) if fp32_head else None
    with _ColumnHider(worker) as hider:
        return hider.forward(recs, hide, start, head)


def probe_recompute_arm(worker, layers=None, progress=None, fp32_head=True, measures=None):
    """Leak-free per-slot probes by hooked full re-prefill (see module docstring).

    Cost: ``n_forwards(L, N, measures)`` full prefills of the whole context (``1 + L * (2 + 3N)``
    for the default measures).
    """
    _check(worker, "evict")
    measures = _resolve_measures(measures)
    head = _ActionHead(worker, fp32_head)
    layers, n_layers = _resolve_layers(worker, layers)
    cols, raster, grid = current_frame_tokens(worker)
    n = cols.numel()
    prefix, turns = worker._device_recs()
    recs = [prefix] + list(turns)
    T = sum(int(r["inputs_embeds"].shape[1]) for r in recs)
    device = recs[0]["inputs_embeds"].device
    masks = [r["visual_pos_masks"] for r in recs]
    if any(m is None for m in masks):
        raise RuntimeError("a stored record has no visual position mask")
    visual_all = torch.cat(masks, dim=1)[0].to(device=device, dtype=torch.bool)
    cur = torch.zeros(T, dtype=torch.bool, device=device)
    cur[cols.to(device)] = True
    if not bool((visual_all & cur).sum() == n):
        raise RuntimeError("current-frame columns are not all visual positions")

    hider = _ColumnHider(worker)

    def forward(hide=None, start=0):
        return hider.forward(recs, hide, start, head)

    t0 = time.perf_counter()
    n0 = worker._influence_probe_count
    with hider:
        full = forward()
        out = _empty_result(layers, n, full.numel(), measures)
        out["base_logp"][:] = full.float().numpy()
        _run_probes(worker, out, layers, measures, cols, cur, visual_all, forward, full, progress, t0)
    out["raster"] = raster.numpy()
    out["grid_thw"] = np.asarray(grid, dtype=np.int64)
    out["fp32_head"] = np.bool_(fp32_head)
    out["n_probes"] = np.int64(worker._influence_probe_count - n0)
    out["seconds"] = np.float64(time.perf_counter() - t0)
    return out


def joint_current_frame_kl(worker, start=0, fp32_head=True):
    """Leak-free KL(P_full || P with the whole current frame hidden on layers >= start): one
    hooked prefill plus the base. The survey score of the driver."""
    _check(worker, "evict")
    head = _ActionHead(worker, fp32_head)
    cols, _, _ = current_frame_tokens(worker)
    prefix, turns = worker._device_recs()
    recs = [prefix] + list(turns)
    T = sum(int(r["inputs_embeds"].shape[1]) for r in recs)
    device = recs[0]["inputs_embeds"].device
    cur = torch.zeros(T, dtype=torch.bool, device=device)
    cur[cols.to(device)] = True
    hider = _ColumnHider(worker)
    with hider:
        full = hider.forward(recs, head=head)
        hidden = hider.forward(recs, cur, start, head)
    return worker._action_kl(full, hidden), full.float().numpy(), hidden.float().numpy()


# ----------------------------------------------------------------- frame-level horizon probe


def horizon_probe(worker, layers, fp32_head=True):
    """Frame-level information horizon of the current frame (paper/info_horizon).

    Per probe layer ``L`` the current frame's visual tokens are hidden on layers ``>= L``:
    ``leave_one = KL(P_full || P_current_hidden)`` and
    ``keep_one = KL(P_only_current || P_no_visual)``. The worker's mode picks the arm:
    ``evict`` re-prefills prefix + every turn with the columns invisible to every query
    (leak-free, ``_ColumnHider``); ``prune`` replays the decision token against the live cache
    with the columns hidden from that one query only (leaky, ``_decision_replay``). Both cost
    ``1 + 3 * #{L < n_layers}`` forwards/replays; ``L >= n_layers`` is 0 with no forward.
    Returns plain lists so the row can go straight into the step log.
    """
    mode = worker.context_window_mode
    _check(worker, mode)
    n_layers = len(worker.language_model.layers)
    layers = [int(x) for x in layers]
    if any(x < 0 for x in layers):
        raise ValueError(f"probe layers must be non-negative, got {layers}")
    head = _ActionHead(worker, fp32_head)
    t0 = time.perf_counter()
    keep, leave, extra = [], [], {}
    zeros = [0.0] * len(layers)
    if mode == "evict":
        cols, _, _ = current_frame_tokens(worker)
        prefix, turns = worker._device_recs()
        recs = [prefix] + list(turns)
        T = sum(int(r["inputs_embeds"].shape[1]) for r in recs)
        device = recs[0]["inputs_embeds"].device
        masks = [r["visual_pos_masks"] for r in recs]
        if any(m is None for m in masks):
            raise RuntimeError("a stored record has no visual position mask")
        visual_all = torch.cat(masks, dim=1)[0].to(device=device, dtype=torch.bool)
        cur = torch.zeros(T, dtype=torch.bool, device=device)
        cur[cols.to(device)] = True
        n_cur = int(cols.numel())
        n0 = worker._influence_probe_count
        hider = _ColumnHider(worker)
        with hider:
            full = hider.forward(recs, head=head)
            if n_cur:
                for L in layers:
                    if L >= n_layers:
                        keep.append(0.0); leave.append(0.0); continue
                    minus_cur = hider.forward(recs, cur, L, head)
                    text_all = hider.forward(recs, visual_all, L, head)
                    only_cur = hider.forward(recs, visual_all & ~cur, L, head)
                    leave.append(worker._action_kl(full, minus_cur))
                    keep.append(worker._action_kl(only_cur, text_all))
        n_probes = worker._influence_probe_count - n0
    else:
        if worker.context_window is not None:
            raise RuntimeError("the leaky arm assumes nothing is window-forced (context_window=None)")
        ps = worker._kv_prune_state
        kv_len = worker._master_len()
        forced = torch.zeros(kv_len, dtype=torch.bool)
        base_bias = worker._replay_base_bias(forced)
        n0 = worker._prune_replay_count
        full = replay_action_logp(worker, head, base_bias)
        visual_all = ps.is_visual.clone()
        cur = visual_all & (ps.turn_id == ps.n_turns - 1)
        n_cur = int(cur.sum())
        if n_cur:
            term_cur = worker._replay_ablation_term(cur)
            term_all = worker._replay_ablation_term(visual_all)
            term_only = worker._replay_ablation_term(visual_all & ~cur)
            for L in layers:
                if L >= n_layers:
                    keep.append(0.0); leave.append(0.0); continue
                ext = range(L, n_layers)
                minus_cur = replay_action_logp(worker, head, base_bias, extra=term_cur, extra_layers=ext)
                text_all = replay_action_logp(worker, head, base_bias, extra=term_all, extra_layers=ext)
                only_cur = replay_action_logp(worker, head, base_bias, extra=term_only, extra_layers=ext)
                leave.append(worker._action_kl(full, minus_cur))
                keep.append(worker._action_kl(only_cur, text_all))
        n_probes = worker._prune_replay_count - n0
    if not n_cur:  # no visual token survived the sparse filter: nothing to hide
        keep, leave = list(zeros), list(zeros)
    return {"layers": layers, "keep_one": keep, "leave_one": leave,
            "base_logp": [float(x) for x in full.float().cpu().tolist()],
            "n_cur_tokens": n_cur, "n_probes": int(n_probes), "seconds": time.perf_counter() - t0}
