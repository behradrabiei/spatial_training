"""Decision replay and paper information probes; never changes model weights."""
import time
import torch

def _round_list(values, ndigits=6):
    """Tensor/list -> list of floats rounded for compact JSON logging."""
    import torch

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().tolist()
    return [round(float(v), ndigits) for v in values]


class DiagnosticsMixin:
    def _ablation_layers(self):
        """Layers a replay ablation term applies to: where a slot would actually be removed."""
        st = self._reindex_state
        return st.pruned_layers if (st is not None and st.layer_selective) else None


    def _decision_replay(self, attention_bias=None, require_grad=False, extra=None,
                         extra_layers=None):
        """Replay the cached decision token without changing any live episode state.

        The cache stores pre-RoPE keys. The original decision slot and its position are
        temporarily removed, then the same one-token input is replayed; the replayed token
        re-enters the cache as the last slot, so the prepared terms below keep the master
        width. `attention_bias` is a 4-D additive mask over the MASTER cache that every
        decoder layer sees (each layer gathers its own columns); `extra` is a second additive
        term over the master cache applied only on `extra_layers` (None = all layers), which
        is how an ablation is confined to the layers a slot would actually be pruned from.
        All cache references, rotation state, and sparse-model bookkeeping are restored even
        if replay fails.
        """
        import torch

        if self.past_key_values is None or self._decision_inputs is None:
            raise RuntimeError("decision replay requires a completed decision forward")
        if torch.is_inference_mode_enabled() and require_grad:
            raise RuntimeError("Fisher replay cannot run inside torch.inference_mode()")
        kv_total = self._master_len()
        if kv_total < 1:
            raise RuntimeError("decision replay requires a nonempty cache")
        for name, term in (("replay mask", attention_bias), ("replay extra term", extra)):
            if term is not None and term.shape[-1] != kv_total:
                raise ValueError(f"{name} covers {term.shape[-1]} slots, cache has {kv_total}")

        st = self._reindex_state
        saved_layers = [(layer.keys, layer.values) for layer in self.past_key_values.layers]
        saved_reindex = st.snapshot()
        sparse_names = ("visual_pos_masks", "vis_keep_mask", "seq_keep_mask", "inputs_embeds",
                        "deepstack_visual_embeds", "position_ids", "kept_visual_embeds")
        sparse_state = {name: getattr(self.language_model, name, None) for name in sparse_names}
        for layer, (keys, values) in zip(self.past_key_values.layers, saved_layers):
            layer.keys = keys[..., :-1, :]
            layer.values = values[..., :-1, :]
        st.pop_last()
        st.replay_extra = extra
        st.replay_extra_layers = extra_layers

        replay = dict(self._decision_inputs)
        replay["seq_keep_mask"] = "all"
        replay["attention_mask"] = attention_bias
        # FlashAttention's backward rejects some production KV lengths because its
        # saved log-sum-exp buffer is not stride-aligned. Normal inference and KL
        # replays stay on the configured backend; only Fisher's differentiable replay
        # uses eager attention, avoiding fused-kernel alignment constraints.
        saved_attn_impl = self.language_model.config._attn_implementation
        self.language_model.config._attn_implementation = "eager" if require_grad else saved_attn_impl
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        try:
            with ctx:
                outputs = self.model.forward(
                    **replay,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            self._prune_replay_count += 1
            return outputs
        finally:
            for layer, (keys, values) in zip(self.past_key_values.layers, saved_layers):
                layer.keys, layer.values = keys, values
            st.restore(saved_reindex)
            for name, value in sparse_state.items():
                setattr(self.language_model, name, value)
            self.language_model.config._attn_implementation = saved_attn_impl


    def _replay_base_bias(self, forced):
        """Prepared additive mask (master width) that makes replay see the post-window pool."""
        return self._replay_ablation_term(forced)


    def _replay_ablation_term(self, mask):
        """(1, 1, 1, master_len) additive term hiding the master slots flagged in `mask`."""
        import torch

        kv_len = self._master_len()
        device = self.past_key_values.layers[0].keys.device
        dtype = self.past_key_values.layers[0].keys.dtype
        term = torch.zeros((1, 1, 1, kv_len), dtype=dtype, device=device)
        if bool(mask.any()):
            term[..., mask.to(device)] = -torch.inf
        return term


    def _replay_action_logp(self, attention_bias, **replay_kwargs):
        """Log action distribution of the replayed decision under the given masking."""
        import torch

        logits = self._decision_replay(attention_bias, **replay_kwargs).logits[0, -1, self.vocab_ids]
        return torch.log_softmax(logits.float(), dim=-1)


    @staticmethod
    def _action_kl(base_logp, other_logp):
        import torch

        kl = torch.sum(base_logp.exp() * (base_logp - other_logp))
        return max(float(kl.detach().cpu()), 0.0)


    @staticmethod
    def _strip_visual_rec(rec):
        """The record without its visual positions (text, positions and markup survive).

        Returns ``(stripped_rec, n_visual)``. The deepstack embeds are indexed by visual token,
        so removing every visual position leaves them empty.
        """
        mask = rec['visual_pos_masks']
        if mask is None or not bool(mask.any()):
            return rec, 0
        keep = ~mask[0]
        out = {
            'inputs_embeds': rec['inputs_embeds'][:, keep],
            'position_ids': rec['position_ids'][..., keep],
            'visual_pos_masks': mask[:, keep],
            'deepstack_visual_embeds': (None if rec['deepstack_visual_embeds'] is None
                                        else [v[:0] for v in rec['deepstack_visual_embeds']]),
        }
        return out, int(mask.sum())


    def _influence_forward_logp(self, recs):
        """Action log-probs of the decision when ``recs`` (prefix first) are re-prefilled from
        scratch against an empty cache. Survivors keep their absolute mRoPE positions; the
        stored embeds are already sparse-filtered, so the filter is bypassed."""
        import torch

        merged = self._cat_recs(recs)
        with self._inference_ctx():
            # use_cache=True on purpose: it is the path _rebuild_window_cache validated, and with
            # use_cache=False the sparse text model's visual pathway (deepstack injection) is
            # silently skipped, so the rebuilt decision would ignore every frame. The scratch
            # cache is dropped with `out`.
            out = self.language_model(**merged, attention_mask=None, past_key_values=None,
                                      use_cache=True, seq_keep_mask='everything',
                                      vis_keep_mask='everything')
            hidden = out.last_hidden_state[:, -1]
            logits = self.model.get_output_embeddings()(hidden)[0, self.vocab_ids].float()
        del out
        self._influence_probe_count += 1
        return torch.log_softmax(logits, dim=-1).cpu()


    def _device_recs(self):
        """``(prefix_rec, [turn_recs])`` mirrored on the device, one copy per turn, cached across
        calls (records are captured on CPU by _record_window_turn)."""
        while len(self._influence_recs) < 1 + len(self._window_recs):
            src = self._prefix_rec if not self._influence_recs else self._window_recs[len(self._influence_recs) - 1]
            self._influence_recs.append({
                k: ([t.to(self.device) for t in v] if isinstance(v, list)
                    else (v.to(self.device) if v is not None else None))
                for k, v in src.items()})
        return self._influence_recs[0], self._influence_recs[1:]


    def _recompute_influence_probe(self):
        """Leak-free leave-one-frame-out influence (paper/frame_influence, recompute arm).

        Fills the same ``_kl_influence_*`` fields as the prune-mode replay probe, but every
        probe rebuilds the whole context -- prefix + every turn's stored post-filter
        embeddings, with the removed frame's visual tokens deleted -- and reads the decision
        from the last token. Nothing survives from a cache that saw the removed frame, so the
        result is a strict function of what remains; the gap to the replay probe is what the
        frame had already written into later tokens' K/V. One full prefill per probe:
        t + 2 + #horizons(<= t) forwards per scored step, each over the whole context.
        """
        import torch

        for name in ("_kl_influence_row", "_kl_influence_logp", "_kl_influence_base_logp",
                     "_kl_influence_history_all", "_kl_influence_visual_all", "_kl_influence_horizon",
                     "_kl_influence_keep_one", "_kl_influence_keep_one_logp"):
            setattr(self, name, [])
        prefix, turns = self._device_recs()
        cur = len(turns) - 1
        self._prune_visual_new = self._strip_visual_rec(turns[cur])[1]
        if cur % self.kv_prune_influence_stride != 0:
            return
        t0 = time.perf_counter()
        stripped = [self._strip_visual_rec(r) for r in turns]

        def probe(removed):
            recs = [prefix] + [stripped[i][0] if i in removed else turns[i] for i in range(len(turns))]
            return self._influence_forward_logp(recs)

        base_logp = probe(set())
        base_list = _round_list(base_logp)
        self._kl_influence_base_logp = base_list
        if "leave_one" in self.kv_prune_influence_measures:
            rows, logps = [], []
            for i in range(len(turns)):
                if stripped[i][1] == 0:  # no visual tokens survived the filter: nothing to remove
                    rows.append(0.0); logps.append(base_list)
                    continue
                lp = probe({i})
                rows.append(self._action_kl(base_logp, lp)); logps.append(_round_list(lp))
            self._kl_influence_row, self._kl_influence_logp = rows, logps

        def joint(removed):
            removed = {i for i in removed if stripped[i][1] > 0}
            if not removed:
                return [0.0, base_list]
            lp = probe(removed)
            return [self._action_kl(base_logp, lp), _round_list(lp)]

        self._kl_influence_history_all = joint(range(cur))
        everything = {i for i in range(cur + 1) if stripped[i][1] > 0}
        vis_logp = probe(everything) if everything else base_logp
        vis_list = _round_list(vis_logp)
        self._kl_influence_visual_all = [self._action_kl(base_logp, vis_logp) if everything else 0.0, vis_list]
        self._kl_influence_horizon = [[K, *joint(range(cur - K + 1))]
                                      for K in self.kv_prune_influence_horizons if cur - K >= 0]
        if "keep_one" in self.kv_prune_influence_measures:
            # Leave-one-IN, leak-free: rebuild the context with every frame's visual tokens
            # deleted except frame i's; reference = the no-visual rebuild above.
            keep_rows, keep_logps = [], []
            for i in range(len(turns)):
                if stripped[i][1] == 0:
                    keep_rows.append(0.0); keep_logps.append(vis_list)
                    continue
                lp = probe(everything - {i})
                keep_rows.append(self._action_kl(lp, vis_logp)); keep_logps.append(_round_list(lp))
            self._kl_influence_keep_one, self._kl_influence_keep_one_logp = keep_rows, keep_logps
        self._influence_probe_latency += time.perf_counter() - t0


    def _horizon_probe(self):
        """paper/info_horizon: frame-level layer-wise KL of the current frame after a decision
        (longnav.utils.token_info.horizon_probe; the mode picks the leak-free or leaky arm)."""
        from longnav.utils.token_info import horizon_probe

        n_layers = len(self.language_model.layers)
        if any(l > n_layers for l in self.horizon_probe_layers):
            raise ValueError(f"horizon_probe_layers {self.horizon_probe_layers} exceed the model's {n_layers} layers")
        t0 = time.perf_counter()
        self._horizon_row = horizon_probe(self, self.horizon_probe_layers)
        self._horizon_probe_count += int(self._horizon_row["n_probes"])
        self._horizon_probe_latency += time.perf_counter() - t0


    def _selector_probe(self):
        """paper/selector_kl: the flat selectors' layer-0 criteria and the leaky per-token KLs of
        the current frame after a decision (longnav.utils.selector_probe.selector_probe)."""
        from longnav.utils.selector_probe import selector_probe

        t0 = time.perf_counter()
        self._selector_row = selector_probe(self, self.selector_probe_budgets, self.selector_probe_measures)
        self._selector_probe_count += int(self._selector_row["n_probes"])
        self._selector_probe_latency += time.perf_counter() - t0


    def _kl_frame_probe(self, candidate_mask, forced, base_logp=None):
        """Leave-one-frame-out replay over the action distribution.

        Returns ``(base_logp, {turn: (kl, ablated_logp)})``: for every turn holding a
        candidate slot, KL(base || that turn's visual slots hidden) and the hidden-frame
        action log-probs. The ablation applies on the layers the budget applies to (every
        layer in uniform mode), so it measures exactly what pruning those slots would
        remove. Turns with no visual candidate replay nothing: their KL is identically 0
        and their log-probs are the base ones. Log-probs are fp32 CPU tensors.
        """
        import torch

        ps = self._kv_prune_state
        base_bias = self._replay_base_bias(forced)
        if base_logp is None:
            base_logp = self._replay_action_logp(base_bias)
        layers = self._ablation_layers()
        per_turn = {}
        for turn in torch.unique(ps.turn_id[candidate_mask], sorted=True).tolist():
            mask = candidate_mask & ps.is_visual & (ps.turn_id == turn)
            if not bool(mask.any()):
                per_turn[int(turn)] = (0.0, base_logp.detach().cpu())
                continue
            ablated_logp = self._replay_action_logp(
                base_bias, extra=self._replay_ablation_term(mask), extra_layers=layers)
            per_turn[int(turn)] = (self._action_kl(base_logp, ablated_logp), ablated_logp.detach().cpu())
        return base_logp.detach().cpu(), per_turn


    def _kl_joint_probe(self, mask, base_bias, base_logp):
        """[kl, logp] of the decision with every slot flagged in ``mask`` hidden at once.

        An empty mask replays nothing: KL 0 and the base log-probs. ``base_logp`` may live on
        any device; the returned log-probs are a rounded list.
        """
        if not bool(mask.any()):
            return [0.0, _round_list(base_logp)]
        ablated_logp = self._replay_action_logp(
            base_bias, extra=self._replay_ablation_term(mask), extra_layers=self._ablation_layers())
        return [self._action_kl(base_logp.to(ablated_logp.device), ablated_logp), _round_list(ablated_logp)]


    def _score_layer_frame_influence(self, visual_mask, forced):
        """Per probe layer i and cached frame f, with every hiding applied on layers >= i:
        keep-one-in KL(P only f || P no visual) and leave-one-out KL(P full || P without f).

        The two rows side by side separate a frame's STANDALONE information (the paper's
        measure) from its UNIQUE information (what the kl selector ranks). Returns two lists
        over probe layers of lists over frames 0..n_turns-1; frames without visual slots
        score 0.0 in both. Cost: 1 + 2 * n_frames replays per probe layer.
        """
        import torch

        st, ps = self._reindex_state, self._kv_prune_state
        starts = self.kv_prune_layer_influence_starts
        n_turns = ps.n_turns
        if not bool(visual_mask.any()):
            return ([[0.0] * n_turns for _ in starts], [[0.0] * n_turns for _ in starts])
        base_bias = self._replay_base_bias(forced)
        full_logp = self._replay_action_logp(base_bias)
        all_term = self._replay_ablation_term(visual_mask)
        frame_masks = [visual_mask & (ps.turn_id == t) for t in range(n_turns)]
        keep_rows, leave_rows = [], []
        for start in starts:
            layers = range(min(int(start), st.n_layers), st.n_layers)
            text_logp = self._replay_action_logp(base_bias, extra=all_term, extra_layers=layers)
            keep, leave = [], []
            for mine in frame_masks:
                if not bool(mine.any()):
                    keep.append(0.0)
                    leave.append(0.0)
                    continue
                only_logp = self._replay_action_logp(
                    base_bias, extra=self._replay_ablation_term(visual_mask & ~mine), extra_layers=layers)
                minus_logp = self._replay_action_logp(
                    base_bias, extra=self._replay_ablation_term(mine), extra_layers=layers)
                keep.append(self._action_kl(only_logp, text_logp))
                leave.append(self._action_kl(full_logp, minus_logp))
            keep_rows.append(keep)
            leave_rows.append(leave)
        return keep_rows, leave_rows


    def _score_layer_influence(self, visual_mask, forced):
        """KL(P || P with `visual_mask` hidden from layers >= i) for each probe layer i.

        The KV-cache analogue of the CVPR'26 "information horizon" metric (arXiv 2512.07580):
        hiding the slots from layer i onward leaves what they contributed to layers < i in the
        residual stream, so the curve over i shows how deep visual information still reaches
        the decision -- and predicts what pruning those slots at layer i would cost. One
        decision replay per probe layer; the entry for i == n_layers hides nothing and is 0.
        """
        st = self._reindex_state
        starts = self.kv_prune_layer_influence_starts
        if not bool(visual_mask.any()):
            return [0.0 for _ in starts]
        base_bias = self._replay_base_bias(forced)
        base_logp = self._replay_action_logp(base_bias)
        term = self._replay_ablation_term(visual_mask)
        row = []
        for start in starts:
            layers = range(min(int(start), st.n_layers), st.n_layers)
            logp = self._replay_action_logp(base_bias, extra=term, extra_layers=layers)
            row.append(self._action_kl(base_logp, logp))
        return row

