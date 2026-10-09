"""Context eviction and faithful window recomputation."""
import torch


class ContextCacheMixin:
    def _find_prompt_prefix_len(self, input_ids):
        '''
        Number of leading tokens to pin: the messages carrying the goal, cut at the
        <|im_start|> that opens the message holding the first image.

        Cutting at the image token itself would end the pinned span on a half-open
        '<|im_start|>user\\n<|vision_start|>' -- an unterminated turn wrapping an
        unterminated vision block -- and that is what the model reads immediately
        before the retained window. Cutting at the message boundary instead leaves the
        pinned span a whole number of complete messages.

        A template that puts the goal text in the SAME message as the first image has
        no such boundary to cut at, so there the image index stands: dropping the goal
        to tidy up the markup would defeat the point of pinning.

        Either way the span is pure text, and the sparse filter only ever drops visual
        tokens, so the index is the same in the sparsified cache.
        '''
        import torch
        seq = input_ids[0]
        hits = torch.nonzero(seq == self.processor.image_token_id, as_tuple=False)
        if not hits.numel():
            return int(input_ids.shape[1])
        img = int(hits[0])
        # prefix is '<|im_start|>assistant\n**', so prefix_ids[0] is the message opener.
        starts = torch.nonzero(seq[:img] == self.prefix_ids[0], as_tuple=False)
        if not starts.numel():
            return img
        boundary = int(starts[-1])
        # Between the opener and the image there must be nothing but the role tag;
        # anything else is goal text that has to stay pinned.
        head = self.processor.tokenizer.decode(seq[boundary + 1:img], skip_special_tokens=True)
        return boundary if "\n" in head and not head.split("\n", 1)[1].strip() else img


    def _chunk_lead(self, input_ids):
        '''
        Tokens in this turn's chunk that precede its '<|im_start|>user' opener.

        Turn chunks are cropped to start at the previous reply's action token (see the
        postfix_starts[0]-1 crop in infer_step), so for every turn but the first the lead is
        that action plus the postfix markup. Turn 0 opens on the pinned prefix and has no lead.
        The lead is pure text, so the index is the same in the sparsified cache.
        '''
        import torch
        if not self._turn_leads:
            return 0
        seq = input_ids[0]
        hits = torch.nonzero(seq == self.prefix_ids[0], as_tuple=False)
        assert hits.numel(), "turn chunk carries no '<|im_start|>' opener"
        return int(hits[0])


    def _recomputing_window(self):
        return self.context_window is not None and self.context_window_mode == "recompute"


    @staticmethod
    def _slice_rec(rec, start, end):
        '''Cut an embed record down to sequence positions [start, end).

        The deepstack embeds are indexed by visual token rather than by sequence
        position, so their bounds come from counting visual positions in the mask.
        '''
        mask = rec['visual_pos_masks']
        out = {
            'inputs_embeds': rec['inputs_embeds'][:, start:end],
            'position_ids': rec['position_ids'][..., start:end],
            'visual_pos_masks': None if mask is None else mask[:, start:end],
        }
        if rec['deepstack_visual_embeds'] is None or mask is None:
            out['deepstack_visual_embeds'] = rec['deepstack_visual_embeds']
        else:
            v0 = int(mask[0, :start].sum())
            v1 = int(mask[0, :end].sum())
            out['deepstack_visual_embeds'] = [v[v0:v1] for v in rec['deepstack_visual_embeds']]
        return out


    @staticmethod
    def _cat_recs(recs):
        '''Concatenate embed records back into one contiguous forward input.'''
        deepstack = None
        if recs[0]['deepstack_visual_embeds'] is not None:
            deepstack = [torch.cat([r['deepstack_visual_embeds'][i] for r in recs], dim=0)
                         for i in range(len(recs[0]['deepstack_visual_embeds']))]
        masks = [r['visual_pos_masks'] for r in recs]
        return {
            'inputs_embeds': torch.cat([r['inputs_embeds'] for r in recs], dim=1),
            'position_ids': torch.cat([r['position_ids'] for r in recs], dim=-1),
            'visual_pos_masks': None if any(m is None for m in masks) else torch.cat(masks, dim=1),
            'deepstack_visual_embeds': deepstack,
        }


    def _record_window_turn(self, outputs):
        '''
        Buffer this turn's post-sparsification inputs so its K/V can be recomputed later.

        These are the tensors the model actually consumed -- filtering already applied,
        absolute mRoPE positions attached -- so replaying them reproduces the turn exactly
        without re-running the vision tower or the sparse filter. Re-filtering would see a
        different embed db and keep a different patch subset, which would make the two
        context-window modes differ in more than the one thing under test.
        '''
        rec = {
            'inputs_embeds': outputs.inputs_embeds,
            'position_ids': outputs.position_ids,
            'visual_pos_masks': outputs.visual_pos_masks,
            'deepstack_visual_embeds': outputs.deepstack_visual_embeds,
        }
        if self._prefix_rec is None:
            # The pinned span is pure text and the sparse filter only ever drops visual
            # tokens, so _prefix_len indexes the sparsified sequence unchanged.
            seq_len = rec['inputs_embeds'].shape[1]
            self._prefix_rec = self._slice_rec(rec, 0, self._prefix_len)
            rec = self._slice_rec(rec, self._prefix_len, seq_len)
        self._window_recs.append(rec)


    def _drop_window_turns(self, first_keep):
        '''
        Retire turns that have fallen out of the window, holding back the boundary turn's
        assistant header exactly as the evicting path does (see the drop_end comment in
        _apply_context_window). Turn chunks start one token INSIDE the previous reply, so
        without the held-back header the window would open on a bare action token whose
        '<|im_start|>assistant\\n**' is gone -- chat markup the model never saw in training.
        The header does not accumulate: it is replaced on every eviction.
        '''
        n = first_keep - self._n_evicted
        if self.context_cut == "frame":
            # No held-back header: the oldest retained turn loses its leading action +
            # postfix instead, so the window opens on its '<|im_start|>user'. With a
            # single-frame window nothing is retained; the strip then happens on the
            # incoming chunk before it is forwarded (see infer_step), so its record is
            # already lead-free when it arrives here.
            self._boundary_rec = None
            if n < len(self._window_recs):
                rec = self._window_recs[n]
                self._window_recs[n] = self._slice_rec(rec, self._turn_leads[first_keep],
                                                       rec['inputs_embeds'].shape[1])
            del self._window_recs[:n]
            return
        boundary = self._window_recs[n - 1]
        seq_len = boundary['inputs_embeds'].shape[1]
        self._boundary_rec = self._slice_rec(boundary, seq_len - len(self.prefix_ids), seq_len)
        del self._window_recs[:n]


    def _trim_embed_db(self, first_keep):
        '''Drop the evicted turns' patches from the sparse dedup db.

        Left at full episode length it would filter re-observed geometry out as redundant
        against frames the model can no longer see, which would confound a pure
        context-length ablation.
        '''
        if not (self.use_sparse and self.past_image_embeds is not None):
            return
        evicted = self._vis_counts[self._n_evicted:first_keep]
        for idx in range(len(self.past_image_embeds)):
            self.past_image_embeds[idx] = self.past_image_embeds[idx][sum(c[idx] for c in evicted):]


    def _windowed_sparse_history(self):
        '''Return the visual-embedding history that will survive the current window step.

        Sparse filtering runs before the current turn enters the cache, while cache and DB
        eviction run afterward. Passing the canonical DB directly would therefore compare
        the current frame against turns that are due to be evicted at the end of this same
        step, permanently suppressing patches based on out-of-window content.

        This returns a sliced view without mutating the canonical DB, which must remain in
        cache/metadata lockstep until the normal post-forward eviction or prune runs.
        '''
        if not (self.use_sparse and self.past_image_embeds is not None):
            return self.past_image_embeds
        if self.context_window is None:
            return self.past_image_embeds

        if self.context_window_mode == "prune":
            ps = self._kv_prune_state
            if ps is None or ps.turn_id is None:
                return self.past_image_embeds
            assert len(self.past_image_embeds) == 1, "KV pruning assumes batch size 1"
            first_keep = ps.n_turns + 1 - self.context_window
            if first_keep <= 0:
                return self.past_image_embeds
            visual_turns = ps.turn_id[ps.is_visual]
            db = self.past_image_embeds[0]
            assert db.shape[0] == visual_turns.numel(), \
                f"embed db has {db.shape[0]} rows for {visual_turns.numel()} visual slots"
            return [db[visual_turns >= first_keep]]

        first_keep = len(self._abs_bounds) + 1 - self.context_window
        if first_keep <= self._n_evicted:
            return self.past_image_embeds
        evicted = self._vis_counts[self._n_evicted:first_keep]
        windowed = []
        for idx, db in enumerate(self.past_image_embeds):
            n_drop = sum(c[idx] for c in evicted)
            windowed.append(db[n_drop:])
        return windowed


    def _window_forward_inputs(self):
        '''
        The retained window as one contiguous forward input: the pinned prefix, the held-back
        boundary header, then the surviving turns -- the same tokens the evicting path leaves
        in the cache. The prefix is split out of turn 0's record, so it has to be prepended
        from the very first step; the boundary header only exists once something has been
        evicted.
        '''
        recs = list(self._window_recs)
        if self._n_evicted > 0 and self._boundary_rec is not None:
            recs.insert(0, self._boundary_rec)
        if self._prefix_rec is not None:
            recs.insert(0, self._prefix_rec)
        return self._cat_recs(recs)


    def _rebuild_window_cache(self):
        '''
        Rebuild the retained window's K/V from scratch against an empty cache.

        Eviction frees memory but does not remove information: a surviving token's layer>=1
        keys and values were computed from a residual stream that had attended over the
        whole episode, so the evicted frames still reach the decision through them. Replaying
        the window against a cache holding only [prefix + window] makes every retained
        representation a function of what the agent can still see, and nothing else.

        Turns keep their original absolute mRoPE positions. Rope carries phase, not content,
        and intra-window position differences are identical either way -- so this leaves the
        prefix-to-window gap exactly as the evicting path leaves it, and the two modes differ
        only in the attention scope used to compute K/V.
        '''
        # The records were captured under _inference_ctx, so stay inside it to concatenate
        # and move them; the cache this builds is handed straight back to the next step.
        with self._inference_ctx():
            merged = self._window_forward_inputs()
            merged = {k: ([t.to(self.device) for t in v] if isinstance(v, list)
                          else (v.to(self.device) if v is not None else None))
                      for k, v in merged.items()}
            # 'everything' is the non-tensor sentinel for "do not sparsify" -- the stored
            # embeds are already filtered, so re-running the filter here would both waste
            # work and change which patches survive. past_key_values=None lets the text
            # model build the cache the same way the incremental path does.
            outputs = self.language_model(**merged, attention_mask=None, past_key_values=None,
                                          use_cache=True, seq_keep_mask='everything',
                                          vis_keep_mask='everything')
        self.past_key_values = outputs.past_key_values


    def _apply_context_window(self):
        '''
        Evict whole turns older than `context_window` frames from the KV cache, pinning
        the prompt prefix so the agent keeps its goal. The boundary turn's assistant
        header survives the cut so the retained context opens on valid chat markup;
        see the drop_end comment below.

        The sparse embed db is trimmed alongside the cache: left at full episode length it
        would filter re-observed geometry out as redundant against frames the model can no
        longer see, which would confound a pure context-length ablation.

        Under context_window_mode='recompute' the survivors' K/V are rebuilt from scratch
        instead of being sliced out of the cache; see _rebuild_window_cache for why that is
        a different experiment.

        Under context_window_mode='reindex' the eviction schedule and slicing are identical
        to 'evict', but keys are cached pre-rotation (see longnav.utils.pre_rope), so the
        survivors' mRoPE positions can be renumbered here to sit flush against the prefix --
        no positional hole across the cut, StreamingLLM-style.
        '''
        import torch

        if self.context_window is None:
            return

        self._abs_bounds.append(self.past_key_values.get_seq_length() + self._dropped)
        if self.use_sparse:
            self._vis_counts.append([int(e.shape[0]) for e in self.language_model.kept_visual_embeds])

        first_keep = len(self._abs_bounds) - self.context_window
        if first_keep <= self._n_evicted:
            return

        prefix = self._prefix_len
        # Cut at the assistant header rather than at the turn boundary. Turn chunks start
        # one token INSIDE the previous reply -- the crop in infer_step begins at
        # postfix_starts[0]-1, the action token -- so a boundary-aligned cut would resume
        # the context on a bare action token whose '<|im_start|>assistant\n**' header is
        # in the evicted span, i.e. chat markup the model never saw in training. Every
        # chunk ends with exactly that header (one action token, so the chunk stops at
        # prefix_end), so holding back its length leaves the boundary reply well formed.
        # The tokens do not accumulate: the next eviction's drop_end is past them.
        if self.context_cut == "frame":
            # Cut past the boundary chunk's lead (previous action + postfix) instead, so the
            # window opens on the oldest retained turn's '<|im_start|>user': K frames and the
            # K-1 actions between them, nothing older. With a single-frame window the lead
            # belongs to a chunk that has not arrived yet, so the cache is cut to the prefix
            # and the next chunk is stripped on entry (infer_step).
            cache_len = self.past_key_values.get_seq_length()
            drop_end = self._abs_bounds[first_keep - 1] - self._dropped
            if first_keep < len(self._turn_leads):
                drop_end += self._turn_leads[first_keep]
            else:
                self._strip_next_lead = True
            assert drop_end <= cache_len, f"frame cut past the cache ({drop_end} > {cache_len})"
        else:
            drop_end = self._abs_bounds[first_keep - 1] - self._dropped - len(self.prefix_ids)
        n_drop = drop_end - prefix
        if n_drop <= 0:
            return

        # Both modes retire exactly the same tokens on exactly the same steps; they differ
        # only in whether the survivors keep the K/V they were given under full context or
        # have them recomputed against the window alone. _dropped therefore has to advance
        # in both, or the next _abs_bounds entry would misread the cache length.
        if self.context_window_mode == "recompute":
            self._drop_window_turns(first_keep)
            with self._inference_ctx():
                self._trim_embed_db(first_keep)
            self._dropped += n_drop
            self._n_evicted = first_keep
            self._rebuild_window_cache()
            return

        # Cache tensors carry the mode they were built under; keep the replacements in
        # the same one, or a no_grad cache would come back as inference tensors and
        # break the attribution replay's backward.
        with self._inference_ctx():
            for layer in self.past_key_values.layers:
                layer.keys = torch.cat([layer.keys[..., :prefix, :], layer.keys[..., drop_end:, :]], dim=-2)
                layer.values = torch.cat([layer.values[..., :prefix, :], layer.values[..., drop_end:, :]], dim=-2)

            if self.context_window_mode == "reindex":
                # The position table follows the same cut as the cache, then the surviving
                # window is renumbered to start at _prefix_len. The cut lands on the
                # held-back assistant header -- a text token, so t == h == w and one scalar
                # shifts all three mRoPE components without disturbing intra-turn 2-D patch
                # geometry (sparse-filter gaps within a turn are preserved by design).
                st = self._reindex_state
                kept_prefix = st.pos_table[..., :prefix]
                kept_window = st.pos_table[..., drop_end:]
                first = kept_window[:, 0, 0]
                assert int(first[0]) == int(first[1]) == int(first[2]), \
                    "reindex cut must land on a text token (assistant header or user opener)"
                shift = int(first[0]) - self._prefix_len
                assert shift > 0, f"eviction dropped {n_drop} tokens but no positions (shift={shift})"
                st.pos_table = torch.cat([kept_prefix, kept_window - shift], dim=-1)
                st.cos = st.sin = None  # stale length; rebuilt on the next forward's append
                # _pos_id_fast keeps assigning from self.offset, so the next turn continues
                # contiguously from the renumbered window.
                self.offset -= shift

            self._trim_embed_db(first_keep)

            # The probe's value-norm banks are indexed by absolute cache position, so
            # they follow the same cut. Ordering invariant: this runs AFTER the forward,
            # and the probe captures its row in-hook DURING the forward, so the row and
            # the bank are both in pre-eviction index space when they meet. Moving this
            # call before the forward would silently misalign them.
            if self.attn_probe is not None:
                self.attn_probe.evict(prefix, drop_end)

        # Evicted frames stay in the list so the 3D viz keeps its per-frame indexing; they
        # just contribute a blank heatmap from now on.
        for rec in self._frame_records[self._n_evicted:first_keep]:
            rec["abs_kv_idx"] = None
        for rec in self._frame_records[first_keep:]:
            rec["abs_kv_idx"] = rec["abs_kv_idx"] - n_drop

        self._dropped += n_drop
        self._n_evicted = first_keep

