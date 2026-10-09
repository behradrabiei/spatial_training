"""Retained visual-token pruning policies and per-layer cache bookkeeping."""
import time
import torch
from longnav.utils.diagnostics import _round_list


class PruningMixin:
    def _pruned_layer_range(self, n_layers):
        """Decoder layers the KV budget applies to, or None for every layer."""
        if self.context_window_mode != "prune":
            return None
        if self.kv_prune_stages is not None:
            return range(self.kv_prune_stages[0], n_layers)
        start, end = self.kv_prune_layer_start, self.kv_prune_layer_end
        if start == 0 and end is None:
            return None
        end = n_layers if end is None else end
        if end > n_layers:
            raise ValueError(f"kv_prune_layer_end={end} exceeds the {n_layers} decoder layers")
        if start > n_layers:
            raise ValueError(f"kv_prune_layer_start={start} exceeds the {n_layers} decoder layers")
        return range(start, end)


    def _stage_ranges(self, n_layers):
        """PyramidDrop stages as adjacent layer ranges [s_0, s_1), ..., [s_last, n_layers)."""
        if self.context_window_mode != "prune" or self.kv_prune_stages is None:
            return None
        bounds = list(self.kv_prune_stages) + [n_layers]
        if bounds[-2] >= n_layers:
            raise ValueError(f"kv_prune_stages={self.kv_prune_stages} exceeds the {n_layers} decoder layers")
        return [range(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


    @staticmethod
    def _stage_score_layer(rng):
        """Layer whose decision row ranks a stage: the one below it, layer 0 for a stage at 0."""
        return max(rng.start - 1, 0)


    def _stage_selectors(self, n_stages):
        """Selector of each stage: kv_prune_stage_importance, 'attn' everywhere by default."""
        return self.kv_prune_stage_importance or ["attn"] * n_stages


    def _install_hidden_capture(self, layer_idx):
        """DART side DB: the hidden states entering decoder layer `layer_idx` at this chunk's
        visual positions (post-sparse-filter, after deepstack), one row per visual master slot,
        captured only while the decision forward has armed it."""
        import torch

        def hook(module, args, kwargs):
            if not self._hidden_capture_armed:
                return
            hs = args[0] if args else kwargs["hidden_states"]
            vis = self.language_model.visual_pos_masks
            if vis is None:
                return
            rows = hs[0][vis[0].to(hs.device)].detach()
            self._hidden_db = rows if self._hidden_db is None else torch.cat([self._hidden_db, rows])

        self.language_model.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)


    def _score_layer_range(self, n_layers, pruned_layers, stages=None):
        """Decoder layers whose decision rows feed the 'attn' importance score."""
        if stages is not None:
            # PyramidDrop ranks each stage by the attention read at the layer just below it
            # (a stage at layer 0 by layer 0's own row); a 'random' stage reads no row, nor does
            # a blind last stage (its budget is 0 whatever its selector says).
            scored = list(zip(stages, self._stage_selectors(len(stages))))
            if self._blind_stage:
                scored = scored[:-1]
            return tuple(sorted({self._stage_score_layer(rng) for rng, m in scored if m == "attn"}))
        if self.kv_prune_score_layer is not None:
            if self.kv_prune_score_layer >= n_layers:
                raise ValueError(f"kv_prune_score_layer={self.kv_prune_score_layer} exceeds the "
                                 f"{n_layers} decoder layers")
            return range(self.kv_prune_score_layer, self.kv_prune_score_layer + 1)
        if pruned_layers is None:
            return None
        mode = self.kv_prune_score_layers
        if mode == "pruned":
            return pruned_layers
        if mode == "all":
            return range(0, n_layers)
        if mode == "boundary":
            return range(pruned_layers.start - 1, pruned_layers.start)
        raise AssertionError(f"unhandled kv_prune_score_layers {mode!r}")


    def _record_slot_counts(self):
        """Visual/text composition of the pruned layers' cache (the whole cache when uniform)."""
        ps, st = self._kv_prune_state, self._reindex_state
        vis = ps.is_visual
        if st.layer_selective and st.alive is not None:
            vis = vis[st.alive]
        self._prune_visual_slots = int(vis.sum())
        self._prune_text_slots = int((~vis).sum())
        # Visual slots each stage's layers hold (staged arms; the last entry is the count above).
        self._prune_stage_visual = [int((ps.is_visual & st.stage_alive(s)).sum())
                                    for s in range(len(st.stages))] if st.staged and st.alive is not None else []
        # Whole-model holding: visual slots averaged over every decoder layer (unpruned layers hold
        # the master cache), and the same in frames' worth of the frames in scope.
        self._prune_visual_model = self._prune_visual_model_frames = None
        if self._prune_stage_visual and st.n_layers:
            held = (st.n_layers - len(st.pruned_layers)) * int(ps.is_visual.sum()) \
                + sum(len(rng) * n for rng, n in zip(st.stages, self._prune_stage_visual))
            counts = ps.visual_per_turn[0 if self.context_window is None
                                        else max(0, ps.n_turns - self.context_window):]
            self._prune_visual_model = held / st.n_layers
            if sum(counts):
                self._prune_visual_model_frames = self._prune_visual_model * len(counts) / sum(counts)


    def _apply_kv_prune(self):
        """Apply the frame pool, exact protected partition, and configured selector.

        Under layer-selective pruning (kv_prune_layer_start/_end) only the pruned layers
        take the budget drop: the master structures (position table, slot metadata, embed
        DB) and the unpruned layers shrink by the window's forced drop alone, and
        ReindexState.alive records which master slots the pruned layers still hold.
        PyramidDrop stages (kv_prune_stages) nest that: each stage keeps a subset of the
        stage above it under its own budget. A frame-relative budget
        (kv_prune_budget_frames) fixes the number of surviving candidates rather than the
        total cache size; the per-frame keep masks are logged when kv_prune_log_masks is on.
        """
        import time
        import torch
        from longnav.utils.kv_prune import (
            build_prune_partition, build_selected_keep, divprune_select, frame_budget,
            model_budgets, pyramid_budgets, random_select, slice_cache_layers,
            step_seed, visual_slot_embeddings,
        )

        ps = self._kv_prune_state
        st = self._reindex_state
        kv_len = self._master_len()
        self._prune_score_latency = 0.0
        self._prune_replay_count = 0
        method = self.kv_prune_importance
        layer_selective = st.layer_selective
        alive = st.alive if layer_selective else None
        frame_cut = self.context_cut == "frame"

        n_new = kv_len - (0 if ps.turn_id is None else ps.turn_id.numel())
        lm = getattr(self, "language_model", None)
        vis = getattr(lm, "visual_pos_masks", None) if lm is not None else None
        vis_row = vis[0].bool().cpu() if vis is not None else torch.zeros(n_new, dtype=torch.bool)
        raster = None
        keep_mask = getattr(lm, "vis_keep_mask", None) if lm is not None else None
        if keep_mask is not None and int(vis_row.sum()):
            raster = torch.nonzero(keep_mask.reshape(-1)).flatten().cpu()
        lead = self._turn_leads[-1] if (frame_cut and self._turn_leads) else 0
        ps.append(n_new, vis_row, lead=lead, raster=raster)

        blind = self.kv_prune_visual_blind and not self._blind_stage  # uniform blind: every layer
        first_keep = 0 if self.context_window is None else max(0, ps.n_turns - self.context_window)
        budget = None if (st.no_pruned_layers or blind) else self.kv_budget
        capacity = None
        if self.kv_prune_budget_frames is not None:
            capacity = frame_budget(ps, first_keep, self.kv_prune_budget_frames)
        tail_capacity = None  # the last stage's own budget; binds from the second frame on
        if self.kv_prune_tail_budget_frames is not None:
            tail_capacity = frame_budget(ps, first_keep, self.kv_prune_tail_budget_frames)
        elif self._blind_stage:
            tail_capacity = 0  # the blind last stage keeps no visual slot, from the first frame on
        if self.kv_prune_seed_per_step:
            ps.generator.manual_seed(step_seed(self.kv_prune_seed, self._episode_label, ps.n_turns - 1))
        partition_kwargs = dict(
            prefix_len=self._prefix_len, context_window=self.context_window,
            holdback_n=len(self.prefix_ids), recent_turns=self.kv_prune_recent_turns,
            candidate_scope=self.kv_prune_candidate_scope, frame_cut=frame_cut)
        partition = build_prune_partition(
            ps, budget=budget, capacity=capacity,
            # The blind arm removes protected visual slots too, so the protected-alive
            # invariant does not apply to it (nothing is selected).
            alive=None if blind else alive, **partition_kwargs,
        )
        candidate_idx = torch.nonzero(partition.candidates).flatten()
        score_start = time.perf_counter()
        scores = torch.zeros(kv_len, dtype=torch.float32)

        # Influence diagnostics, measured on the pre-prune cache. kv_prune_log_influence:
        # leave-one-frame-out KL over EVERY cached frame 0..t (only visual slots are masked,
        # so the prompt prefix and each turn's text survive). kv_prune_log_layer_influence:
        # KL per probe layer with the history's visual slots (and, separately, every visual
        # slot) hidden from that layer onward. Their replays and wall time are counted in
        # _prune_replay_count / _prune_score_latency like any other scorer.
        self._prune_visual_new = int(vis_row.sum())
        self._kl_influence_row = []
        self._kl_influence_logp = []
        self._kl_influence_base_logp = []
        self._kl_influence_history_all = []
        self._kl_influence_visual_all = []
        self._kl_influence_horizon = []
        self._kl_influence_keep_one = []
        self._kl_influence_keep_one_logp = []
        self._layer_influence_hist = []
        self._layer_influence_all = []
        self._keep_one_rows = []
        self._leave_one_rows = []
        probe_step = (ps.n_turns - 1) % self.kv_prune_influence_stride == 0
        if self.kv_prune_log_influence and probe_step:
            base_bias = self._replay_base_bias(partition.forced)
            if "leave_one" in self.kv_prune_influence_measures:
                base_logp, per_turn = self._kl_frame_probe(
                    torch.ones(kv_len, dtype=torch.bool), partition.forced)
                base_list = _round_list(base_logp)
                self._kl_influence_row = [float(per_turn[t][0]) if t in per_turn else 0.0
                                          for t in range(ps.n_turns)]
                self._kl_influence_logp = [_round_list(per_turn[t][1]) if t in per_turn else base_list
                                           for t in range(ps.n_turns)]
            else:
                base_logp = self._replay_action_logp(base_bias)
                base_list = _round_list(base_logp)
            self._kl_influence_base_logp = base_list
            # Joint probes on the same replayed decision: every history frame at once, every
            # visual slot at once, and the age bands (frames of age >= K; K=1 is history_all).
            cur = ps.n_turns - 1
            self._kl_influence_history_all = self._kl_joint_probe(
                ps.is_visual & (ps.turn_id < cur), base_bias, base_logp)
            layers = self._ablation_layers()
            vis_logp = self._replay_action_logp(
                base_bias, extra=self._replay_ablation_term(ps.is_visual), extra_layers=layers)
            self._kl_influence_visual_all = [self._action_kl(base_logp.to(vis_logp.device), vis_logp),
                                             _round_list(vis_logp)]
            self._kl_influence_horizon = [
                [K, *self._kl_joint_probe(ps.is_visual & (ps.turn_id <= cur - K), base_bias, base_logp)]
                for K in self.kv_prune_influence_horizons if cur - K >= 0]
            if "keep_one" in self.kv_prune_influence_measures:
                # Leave-one-IN: only frame f's visual slots stay visible (every other visual slot,
                # the current frame's included, is hidden); reference = the no-visual decision.
                vis_list = _round_list(vis_logp)
                keep_rows, keep_logps = [], []
                for f in range(ps.n_turns):
                    mine = ps.is_visual & (ps.turn_id == f)
                    if not bool(mine.any()):  # nothing survived the sparse filter: only-f == no visual
                        keep_rows.append(0.0); keep_logps.append(vis_list)
                        continue
                    only_logp = self._replay_action_logp(
                        base_bias, extra=self._replay_ablation_term(ps.is_visual & ~mine), extra_layers=layers)
                    keep_rows.append(self._action_kl(only_logp, vis_logp))
                    keep_logps.append(_round_list(only_logp))
                self._kl_influence_keep_one, self._kl_influence_keep_one_logp = keep_rows, keep_logps
        if self.kv_prune_log_layer_influence and probe_step:
            visual = partition.base_keep & ps.is_visual
            if alive is not None:
                visual &= alive
            self._layer_influence_hist = self._score_layer_influence(
                visual & ~partition.protected, partition.forced)
            self._layer_influence_all = self._score_layer_influence(visual, partition.forced)
        if self.kv_prune_log_keep_one and probe_step:
            visual = partition.base_keep & ps.is_visual
            if alive is not None:
                visual &= alive
            self._keep_one_rows, self._leave_one_rows = self._score_layer_frame_influence(
                visual, partition.forced)

        forced, first_keep = partition.forced, partition.first_keep
        stage_keeps = None
        if blind:
            # Every visual slot leaves the blind layers (the live mask already hid them
            # there during this forward); text and the frame window are untouched.
            keep = partition.base_keep & ~ps.is_visual
            ps.fold_scores(scores, 0.0)
        elif st.staged:
            # PyramidDrop: stage s keeps a subset of stage s-1's slots under budget c_s, ranked
            # by the decision row read at the layer just below the stage. Budgets follow the
            # geometric schedule from the window's original visual count down to `capacity`.
            # paper/hybrid_prune: a 'random' stage draws uniformly among the previous stage's
            # survivors instead, and with kv_prune_tail_budget_frames the last stage has its own
            # budget while the schedule ends one stage earlier.
            stages = st.stages
            selectors = self._stage_selectors(len(stages))
            tailed = self.kv_prune_tail_budget_frames is not None or self._blind_stage
            n_sched = len(stages) - (1 if tailed else 0)
            if capacity is None:
                budgets = [None] * n_sched
            else:
                total = sum(ps.visual_per_turn[max(0, first_keep):])
                if self.kv_prune_schedule_anchor == "pool":
                    # Shrink from what the first stage can draw from now (its survivors + the
                    # entering frame), not from tokens deleted long ago.
                    total = int(build_prune_partition(ps, budget=None, capacity=None, alive=st.stage_alive(0),
                                                      **partition_kwargs).candidates.sum())
                if self.kv_prune_budget_scope == "model":
                    # capacity bounds the layer-weighted mean over the whole model, tail included.
                    budgets = model_budgets(
                        total, capacity, [len(r) for r in stages[:n_sched]],
                        n_unpruned=st.n_layers - len(st.pruned_layers),
                        tail=(len(stages[-1]), tail_capacity) if tailed and tail_capacity is not None else None,
                        unpruned_total=int((partition.base_keep & ps.is_visual).sum()))
                else:
                    budgets = pyramid_budgets(total, capacity, n_sched)
            if tailed:
                budgets.append(tail_capacity if tail_capacity is None or budgets[-1] is None
                               else min(tail_capacity, budgets[-1]))
            stage_keeps, prev_keep, keep, scored = [], None, None, False
            for s_idx, (rng, c_s) in enumerate(zip(stages, budgets)):
                alive_s = st.stage_alive(s_idx)
                pool = alive_s if prev_keep is None else alive_s & prev_keep
                part_s = build_prune_partition(ps, budget=None, capacity=c_s, alive=pool,
                                               **partition_kwargs)
                cand_s = torch.nonzero(part_s.candidates).flatten()
                row = None
                if selectors[s_idx] == "attn":
                    row = st.layer_rows.get(self._stage_score_layer(rng))
                    if row is None:
                        raise RuntimeError(f"no decision row captured at layer {self._stage_score_layer(rng)} "
                                           f"for stage {s_idx}")
                    row = row.float().cpu()
                if part_s.capacity == cand_s.numel():
                    selected = cand_s
                elif row is None:
                    selected = random_select(cand_s, part_s.capacity, ps.generator)
                else:
                    selected = cand_s[torch.argsort(row[cand_s], descending=True, stable=True)[:part_s.capacity]]
                keep = build_selected_keep(part_s, selected)
                stage_keeps.append((rng, keep, alive_s))
                prev_keep = keep
                if row is not None and not scored:
                    scores, scored = row, True
            ps.fold_scores(scores, 0.0)
        else:
            if method == "attn":
                scores = st.mean_scores().cpu()
            if partition.capacity == candidate_idx.numel():
                selected = candidate_idx
            elif method == "random":
                selected = random_select(candidate_idx, partition.capacity, ps.generator)
            elif method == "divprune":
                db = visual_slot_embeddings(ps, self.past_image_embeds)
                db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
                rows = db[db_rank[candidate_idx].to(db.device)].to(getattr(self, "device", db.device))
                selected = divprune_select(rows, candidate_idx, partition.capacity)
            elif method == "dart":
                selected = self._dart_select(ps, st, candidate_idx, partition.capacity)
            elif method == "attn":
                selected = candidate_idx[torch.argsort(
                    scores[candidate_idx], descending=True, stable=True)[:partition.capacity]]
            else:
                raise AssertionError(f"unhandled pruning method {method!r}")
            ps.fold_scores(scores, self.kv_prune_ema_beta if method == "attn" else 0.0)
            keep = build_selected_keep(partition, selected)

        self._prune_score_latency = time.perf_counter() - score_start
        if frame_cut and first_keep >= ps.n_turns:
            # Single-frame window: the next chunk's lead belongs to this evicted turn.
            self._strip_next_lead = True
        # Slots already budget-dropped from the pruned layers stay in the master cache
        # (the unpruned layers hold them) without being recounted as dropped.
        budget_dropped = partition.base_keep & ~keep
        if alive is not None:
            budget_dropped &= alive
        if not bool(forced.any()) and not bool(budget_dropped.any()):
            self._record_slot_counts()
            self._log_vtp(first_keep)
            return

        # The master cache keeps a budget-dropped slot only while some unpruned layer holds it.
        shallowest_keep = stage_keeps[0][1] if stage_keeps else keep
        master_keep = shallowest_keep if st.all_layers_pruned else partition.base_keep
        master_shrinks = not bool(master_keep.all())
        with self._inference_ctx():
            if self.use_sparse and self.past_image_embeds is not None and master_shrinks:
                assert len(self.past_image_embeds) == 1, "kv pruning assumes batch size 1"
                db = self.past_image_embeds[0]
                n_vis = int(ps.is_visual.sum())
                assert db.shape[0] == n_vis, f"embed db has {db.shape[0]} rows for {n_vis} visual slots"
                self.past_image_embeds[0] = db[master_keep[ps.is_visual]]
                if self.past_image_embeds[0].shape[0] == 0:
                    # No visual slot left anywhere (visual-blind on every layer): the sparse
                    # filter expects None rather than an empty DB, and infer_step re-seeds it.
                    self.past_image_embeds = None
            if self._hidden_db is not None and master_shrinks:
                self._hidden_db = self._hidden_db[master_keep[ps.is_visual].to(self._hidden_db.device)]

            slice_cache_layers(self.past_key_values.layers, master_keep, keep, alive,
                               st.pruned_layers, stage_keeps=stage_keeps)
            if master_shrinks:
                st.slice_master(torch.nonzero(master_keep).flatten())
            st.invalidate_rotation()
            if stage_keeps is not None:
                for s_idx in range(len(stage_keeps) - 1, -1, -1):  # deepest first: nesting holds
                    st.set_stage_alive(s_idx, stage_keeps[s_idx][1][master_keep])
            elif layer_selective:
                st.set_alive(keep[master_keep])

        ps.n_budget_dropped += int(budget_dropped.sum())
        if master_shrinks:
            ps.slice(master_keep)
        self._record_slot_counts()
        if bool(forced.any()):
            self._dropped += int(forced.sum())
            self._n_evicted = first_keep
        self._log_vtp(first_keep)


    def _dart_select(self, ps, st, candidate_idx, capacity):
        """DART pivots from the rotated keys of the layer below the cut, similarity on the
        hidden states entering the cut layer (the side DB filled by _install_hidden_capture).
        With every layer pruned (layer_start=0) there is no layer below the cut: layer 0 is
        read instead, its keys and the hidden states leaving it."""
        import torch
        from longnav.utils.kv_prune import dart_select
        from longnav.utils.pre_rope import _rotate

        src = 0 if st.pruned_layers is None else max(st.pruned_layers.start - 1, 0)
        keys = self.past_key_values.layers[src].keys  # (1, n_kv, master, d), unrotated
        if keys.shape[-2] != st.master_len():
            raise RuntimeError(f"layer {src} holds {keys.shape[-2]} slots, master has {st.master_len()}")
        cos, sin = st.layer_tables(src)
        rot = _rotate(keys, cos, sin)[0].permute(1, 0, 2).reshape(keys.shape[-2], -1)
        norms = rot.float().abs().sum(dim=-1).cpu()
        db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
        if self._hidden_db is None or self._hidden_db.shape[0] != int(ps.is_visual.sum()):
            n_db = 0 if self._hidden_db is None else self._hidden_db.shape[0]
            raise RuntimeError(f"hidden DB holds {n_db} rows for {int(ps.is_visual.sum())} visual slots")
        hidden = self._hidden_db[db_rank[candidate_idx].to(self._hidden_db.device)]
        return dart_select(norms[candidate_idx], hidden, candidate_idx, capacity,
                           n_pivots=self.kv_prune_dart_pivots)


    def _log_vtp(self, first_keep):
        """Per-frame keep masks of the window after this step's prune (paper/token_prune):
        the frames the NEXT decision sees, their original visual counts, the survivors at the
        deepest pruned stage, the packed per-patch bits and the entering frame's raster."""
        if not self.kv_prune_log_masks:
            # Full-context arms switch the masks off (every frame of the episode would be
            # re-packed each step); the budget's denominator is still reported.
            if self.kv_prune_budget_frames is not None:
                self._vtp_log = {"mean/vtp_n_new": self._kv_prune_state.visual_per_turn[-1]}
            return
        import torch
        from longnav.utils.kv_prune import pack_keep_bits

        ps, st = self._kv_prune_state, self._reindex_state
        turns = list(range(max(0, first_keep), ps.n_turns))
        alive = st.alive if (st.layer_selective and st.alive is not None) \
            else torch.ones(ps.turn_id.numel(), dtype=torch.bool)
        bits, kept = pack_keep_bits(ps, alive, turns)
        self._vtp_log = {
            "vtp_frames": turns,
            "vtp_ntok": [ps.visual_per_turn[t] for t in turns],
            "vtp_kept": kept,
            "vtp_bits": bits,
            "vtp_raster": list(ps.raster_per_turn[-1]),
            "mean/vtp_n_new": ps.visual_per_turn[-1],
        }

