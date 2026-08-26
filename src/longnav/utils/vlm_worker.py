# import os
import numpy as np
import torch
import time
import gc
import copy
import os
from typing import Optional, Any, List
from collections import defaultdict
import torch.nn as nn
from torch.optim import AdamW
from dataclasses import dataclass,field
# from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
import torch.nn.functional as F
from longnav.config_schema import VLMTrainingConfig
from longnav.utils.hamlet import (HAMLET_MODULE_NAME, attach_hamlet, make_moment_embed_hook,
                                  splice_moment_tokens, moment_positions, forward_embeds_core, mem_ratio)

CONTEXT_WINDOW_MODES = ("evict", "recompute", "reindex")

def compute_full_kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the token-level KL divergence: KL(pi || ref) = sum(pi * (log_pi - log_ref))
    
    Args:
        log_probs: [Batch, Seq, Vocab] (Normalized, i.e., LogSoftmax applied)
        ref_log_probs: [Batch, Seq, Vocab] (Normalized, i.e., LogSoftmax applied)
    
    Returns:
        kl_penalty: [Batch, Seq] (Scalar KL value per token)
    """
    # 1. Convert log_probs to probs for the weighting term
    probs = log_probs.exp()
    
    # 2. Compute KL: P * (log_P - log_Q)
    #    We sum over the last dimension (Vocab/Action Space)
    kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
    
    return kl

class VLMWorker:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct",attn_impl='sdpa',dtype='float16', prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',vocab=["stop","forward","left","right","up","down"],save_outputs=False,load_model=True,offload_cache=False,use_sparse=False,sparse_threshold=0.95,bev_canvas_size=2000,save_pixels=False,visualize_attention=False,visualize_attention_heads=False,visualize_attention_3d=False,attn3d_layers=None,context_window=None,context_window_mode="evict",attn_weighting="raw",hamlet=None):
        import transformers.modeling_flash_attention_utils as fa_utils
        def patched(position_ids, batch_size):
            return False
        fa_utils._is_packed_sequence = patched
        import torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            fix_mistral_regex=True,
        )
        self.vocab = vocab
        self.vocab_ids = self._vocab_to_ids(vocab)
        self.save_outputs = save_outputs
        self.save_pixels = save_pixels
        self.model_id = model_id
        self.attn_implementation = attn_impl
        self.dtype = dtype
        self.model=None
        self.prefix_ids = self.processor.tokenizer.encode(prefix)
        self.postfix_ids = self.processor.tokenizer.encode(postfix)
        self.offload_cache = offload_cache
        self.use_sparse = use_sparse
        self.sparse_threshold = sparse_threshold
        self.bev_canvas_size = bev_canvas_size
        self.visualize_attention = visualize_attention
        self.visualize_attention_heads = visualize_attention_heads
        self.visualize_attention_3d = visualize_attention_3d
        self.attn3d_layers = attn3d_layers if attn3d_layers is None else list(attn3d_layers)
        from longnav.utils.attn_probe import WEIGHTING_MODES
        if attn_weighting not in WEIGHTING_MODES:
            raise ValueError(f"attn_weighting must be one of {WEIGHTING_MODES}, got {attn_weighting!r}")
        self.attn_weighting = attn_weighting
        if attn_weighting == "grad" and not visualize_attention_3d:
            # The attribution is driven from the 3D viz hook; the 2D overlays have no
            # backward pass to read, so they would silently render nothing.
            raise ValueError("attn_weighting='grad' requires visualize_attention_3d=True; "
                             "the 2D attention overlays cannot produce gradients.")
        self.context_window = context_window
        if context_window_mode not in CONTEXT_WINDOW_MODES:
            raise ValueError(f"context_window_mode must be one of {CONTEXT_WINDOW_MODES}, got {context_window_mode!r}")
        self.context_window_mode = context_window_mode
        if context_window_mode == "recompute":
            if context_window is None:
                # Nothing is ever evicted at full context, so the rebuild would never fire
                # and the run would silently be an ordinary full-context eval.
                raise ValueError("context_window_mode='recompute' requires context_window to be set.")
            if self._any_attn_viz():
                # The rebuild replaces the cache wholesale, so nothing indexed by absolute KV
                # position -- the probe's rows and value-norm banks, the frame records'
                # abs_kv_idx -- can be shift-corrected the way AttentionProbe.evict does.
                raise NotImplementedError("context_window_mode='recompute' does not support attention "
                                          "visualization; the cache is rebuilt each step, so absolute "
                                          "KV indices cannot be carried across.")
        if context_window_mode == "reindex":
            if context_window is None:
                # Same rationale as recompute: nothing is ever evicted at full context, so
                # the run would silently pay the re-rotation cost for an ordinary eval.
                raise ValueError("context_window_mode='reindex' requires context_window to be set.")
            if self._any_attn_viz():
                # _record_frame_keys and _grad_attribution_pass read raw cache keys, which
                # are stored pre-rotation under reindex; the heat maps would be garbage.
                raise NotImplementedError("context_window_mode='reindex' does not support attention "
                                          "visualization; cached keys are stored pre-rotation.")
            if not use_sparse:
                # The per-slot position table is fed from TextMixin.forward, which only the
                # sparse model subclass runs through; the stock model has no hook point.
                raise NotImplementedError("context_window_mode='reindex' requires use_sparse=True.")
            if save_outputs:
                # The packed replay re-forwards the full history through the language model,
                # which would double-append the position table and desync it from the cache.
                raise ValueError("context_window_mode='reindex' does not support save_outputs=True.")
        if context_window is not None and save_outputs:
            print(f"[context_window] ⚠️ WARNING: context_window={context_window} with save_outputs=True. "
                  "The packed sequence replays the FULL uncropped history, so its logprobs will not "
                  "match the windowed rollout. Intended for eval only.")
        # HAMLET (moment tokens + memory module fused at the decision token); see
        # longnav.utils.hamlet. The module itself is created in load_model so it exists
        # on the base model before any PEFT wrapping.
        self.hamlet_cfg = dict(hamlet) if hamlet else None
        self.hamlet_enabled = bool(self.hamlet_cfg and self.hamlet_cfg.get("enabled", False))
        self.moment_ids = []
        self.mem_ids = []
        self._hamlet_hook = None
        self._moment_history = []
        self._current_mem_rows = None  # (n_mem, H) memory-token rows for the turn being forwarded
        self._last_hamlet_stats = None
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if self.hamlet_enabled:
            if not use_sparse:
                # last_hidden_state and the packed replay embeds only come out of the
                # sparse model class; the stock class returns neither.
                raise NotImplementedError("vlm.hamlet.enabled requires use_sparse=True.")
            if attn_weighting == "grad":
                # The one-token attribution replay re-derives the decision from the
                # cache alone; it has no memory fusion, so its scores would explain a
                # different policy than the one that acted.
                raise NotImplementedError("vlm.hamlet.enabled does not support attn_weighting='grad'.")
        # Attention weights are only readable with the eager attention kernel;
        # sdpa/flash return None. Auto-force eager when a heatmap viz is on.
        if self._any_attn_viz() and self.attn_implementation != "eager":
            print(f"[visualize_attention] forcing attn_impl 'eager' (was '{self.attn_implementation}') to expose attention weights.")
            self.attn_implementation = "eager"
        self.attn_probe = None  # set by load_model when any attention viz is enabled
        self.attn3d_layer_ids = []  # absolute decoder layer indices feeding the 3D viz
        self._frame_records = []  # per-frame KV bookkeeping for the growing 3D attention viz
        self._decision_inputs = None  # one-token inputs that replay the latest decision
        
        self._is_merged = None
        self._is_lora = None
        self._reindex_state = None  # set by load_model under context_window_mode='reindex'
        # Warmup the CUDA allocator
        if load_model:
            self.load_model()
        torch.cuda.empty_cache()
        self.reset()
        
    def reset(self):
        from transformers import DynamicCache,StaticCache
        import torch
        self.offset=0
        # self.past_key_values=StaticCache(config=self.model.config, offloading=self.offload_cache,max_cache_len=70000)
        self.past_key_values=None#DynamicCache(config=self.model.config, offloading=self.offload_cache)
        self.outputs = defaultdict(list)
        self.cumulative_inputs = None
        self.seq_keep_mask = None
        self.vis_keep_masks = []
        self._frame_records = []
        self._decision_inputs = None
        if self.attn_probe is not None:
            self.attn_probe.reset()
        if getattr(self, "_reindex_state", None) is not None:
            self._reindex_state.reset()
        self.past_image_embeds = None #per batch list of image embed tensors of the form N_patch by N_hidden
        self.logit_indices = []
        # HAMLET: one (n_moment, hidden) moment summary per decision so far. Kept outside
        # the KV cache on purpose -- it survives context-window eviction.
        self._moment_history = []
        self._last_hamlet_stats = None
        # Sliding context window bookkeeping (see _apply_context_window).
        self._abs_bounds = []  # cumulative cache length after each turn, as if nothing were evicted
        self._vis_counts = []  # visual patches kept per turn, for trimming the sparse embed db
        self._dropped = 0  # total tokens evicted so far
        self._n_evicted = 0  # number of turns fully evicted so far
        self._prefix_len = None  # pinned prompt tokens (goal + action space), never evicted
        # Replay buffers for context_window_mode='recompute' (see _rebuild_window_cache).
        self._window_recs = []  # one embed record per retained turn
        self._prefix_rec = None  # the pinned prefix, split off turn 0
        self._boundary_rec = None  # assistant header held back from the last evicted turn
        torch.cuda.empty_cache()

    def load_model(self):
        if not self.use_sparse:
            from transformers import AutoModelForImageTextToText
            print(f"Loading {self.model_id}...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,#"sdpa",
                device_map="cuda",
            ).eval()
            self.device = self.model.device
        else:
            from transformers import AutoConfig
            from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"Loading {self.model_id} with sparsifying patch...")
            self.model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
                self.model_id, 
                config=config,
                device_map="cuda",
                dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation = self.attn_implementation).eval()
            self.device = self.model.device
        self.model.config.use_cache = False
        self.model.to('cuda')
        self.vl_model = self.model.model
        self.language_model = self.vl_model.language_model
        if self.use_sparse:
            self.language_model.sparse_threshold = self.sparse_threshold
        if self.hamlet_enabled:
            self.moment_ids, self.mem_ids = attach_hamlet(self.model, self.hamlet_cfg, self.processor.tokenizer)
            # The incremental rollout forward looks ids up through embed_tokens; the hook
            # swaps the placeholder rows for the learnable moment embeddings and for this
            # turn's memory read-out rows (_hamlet_prepare_turn). Training replays stored
            # embeds and does both swaps explicitly (forward_embeds_core).
            self._hamlet_hook = self.model.get_input_embeddings().register_forward_hook(
                make_moment_embed_hook(self._hamlet_rows_for_hook, self.moment_ids + self.mem_ids))
            print(f"[hamlet] attached: n_moment={len(self.moment_ids)} n_mem={len(self.mem_ids)} placeholder ids "
                  f"{self.moment_ids[0]}..{self.mem_ids[-1]}, d_mem={self.hamlet_cfg.get('d_mem')}, "
                  f"memory_window={self.hamlet_cfg.get('memory_window')}")
        if self.context_window_mode == "reindex":
            from longnav.utils.pre_rope import install_pre_rope
            self._reindex_state = install_pre_rope(self.language_model)
        if self._any_attn_viz():
            self._attach_attention_probe()

    def _hamlet(self):
        """The HAMLET module, through PEFT's ModulesToSaveWrapper once wrapped (so
        disable_adapter() routes to the untrained no-op copy for the ref policy)."""
        return getattr(self.model, HAMLET_MODULE_NAME)

    def _hamlet_rows_for_hook(self):
        """Placeholder embedding table for the embed_tokens hook: the moment
        embeddings followed by the current turn's memory rows."""
        if self._current_mem_rows is None:
            raise RuntimeError("hamlet: memory rows for this turn were not prepared (_hamlet_prepare_turn)")
        return torch.cat([self._hamlet()(mode="moment_embeds"), self._current_mem_rows.to(self.device)], dim=0)

    def _hamlet_prepare_turn(self):
        """Before the forward of turn t: the memory read-out over the moment blocks
        of turns 0..t-1 becomes this turn's memory-token input rows (n_mem, H)."""
        hamlet = self._hamlet()
        t = len(self._moment_history)
        with torch.no_grad():
            if t:
                moments = torch.stack(self._moment_history)  # (t, n_moment, H)
            else:
                hidden_size = self.model.config.text_config.hidden_size
                moments = torch.zeros(0, len(self.moment_ids), hidden_size, device=self.device, dtype=self.model.dtype)
            rows = hamlet(mode="readout", moments=moments, query_blocks=torch.tensor([t], device=self.device))[0]
            self._current_mem_rows = rows.detach()
            ratio = mem_ratio(rows, hamlet(mode="mem_embeds"))
        self._last_hamlet_stats = {"mean/hamlet_mem_ratio": ratio, "max/hamlet_mem_ratio": ratio}

    def _hamlet_record_moments(self, outputs, turn_inputs):
        """After the forward of turn t: read this turn's moment hidden states off the
        chunk's last_hidden_state and append them to the episode's moment history
        (what the next turn's read-out and the training replay consume)."""
        hidden = outputs["last_hidden_state"][0]  # (S_kept, H), post final norm
        ids = turn_inputs["input_ids"][0].cpu()
        keep = self.language_model.seq_keep_mask  # cpu bool over this chunk, set by TextMixin
        if keep is not None:
            ids = ids[keep]
        if ids.shape[0] != hidden.shape[0]:
            raise RuntimeError(f"hamlet: {ids.shape[0]} kept ids vs {hidden.shape[0]} hidden rows")
        pos = moment_positions(ids, self.moment_ids)
        if pos.shape[0] != 1:
            raise RuntimeError(f"hamlet: expected one moment block per turn, found {pos.shape[0]}")
        if moment_positions(ids, self.mem_ids).shape[0] != 1:
            raise RuntimeError("hamlet: expected one memory block per turn")
        self._moment_history.append(hidden[pos[0].to(hidden.device)])
        self._current_mem_rows = None

    def _any_attn_viz(self):
        return self.visualize_attention or self.visualize_attention_heads or self.visualize_attention_3d

    def _grad_attribution_enabled(self):
        """True when the 3D heat maps measure gradients rather than attention."""
        return self.visualize_attention_3d and self.attn_weighting == "grad"

    def _inference_ctx(self):
        """no_grad instead of inference_mode when the cache must survive a backward.

        inference_mode tags everything it produces as an inference tensor, and
        autograd refuses to save those for backward. The KV cache built here is read
        again by the attribution replay, so under inference_mode that replay's graph
        would be poisoned before it started.
        """
        import torch

        return torch.no_grad() if self._grad_attribution_enabled() else torch.inference_mode()

    def _attach_attention_probe(self):
        """Probe the action-decision token's attention on every layer a viz needs.

        The 2D overlays are always last-layer; only the 3D heat video is
        layer-configurable. Per-head rows are kept solely for the 2D views.
        """
        from longnav.utils.attn_probe import AttentionProbe, resolve_layer_ids

        n_layers = len(self.language_model.layers)
        head_layers = [-1] if (self.visualize_attention or self.visualize_attention_heads) else []
        if self.visualize_attention_3d:
            self.attn3d_layer_ids = resolve_layer_ids(self.attn3d_layers, n_layers)
        else:
            self.attn3d_layer_ids = []
        text_config = self.language_model.config
        num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads
        self.attn_probe = AttentionProbe(n_layers, layers=self.attn3d_layer_ids, head_layers=head_layers,
                                         weighting=self.attn_weighting, num_kv_groups=num_kv_groups)
        self.attn_probe.attach(self.language_model.layers)
        n_probed = len(self.attn3d_layer_ids)
        if n_probed > 1:
            # History is quadratic in episode length: every step re-reports its
            # attention over all frames so far. ~3 MB per layer per 100 steps.
            print(f"[visualize_attention_3d] probing {n_probed} layers {self.attn3d_layer_ids}; "
                  f"attention history will cost roughly {3 * n_probed} MB for a 100-step episode "
                  f"and {12 * n_probed} MB for a 200-step one.")

    @property
    def _last_attn_heads(self):
        """(num_heads, kv_len) last-layer attention of the action-decision token."""
        if self.attn_probe is None:
            return None
        return self.attn_probe.head_rows.get(self.attn_probe.num_layers - 1)

    @property
    def _last_attn_row(self):
        """(kv_len,) head reduction of the above.

        Weighted rows are already in commensurable residual-stream units, so they
        sum; raw attention weights can only be averaged (each head's softmax sums to
        1 independently, so they are not comparable across heads).
        """
        heads = self._last_attn_heads
        if heads is None:
            return None
        return heads.mean(0) if self.attn_weighting == "raw" else heads.sum(0)

    def tokenize_inputs(self,messages,images):
                # Process ONLY this turn's data
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        inputs = self.processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt"
        )
        return inputs
    
    def _get_sandwich_indices(self, input_ids):
        import torch
        """
        Locates the indices of the logits that predict the sandwiched tokens.
        
        Returns:
            logit_indices (torch.Tensor): Indices relative to 'input_ids' to pass to logits_to_keep.
            target_ids (torch.Tensor): The ground truth tokens to calculate logprobs for.
        """
        # 1. Convert to NumPy for fast, robust search
        seq = input_ids[0].cpu().numpy()
        
        # Helper: NumPy sliding window search
        def search_sequence_numpy(arr, sub):
            window_size = len(sub)
            if len(arr) < window_size:
                return [-1]
            # Create strided view for O(1) comparison
            shape = (arr.shape[0] - window_size + 1, window_size)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            
            # Find all matches
            matches = np.all(windows == sub, axis=1)
            indices = np.where(matches)[0]
            
            return indices
        # 2. Find Prefix End
        prefix_np = np.array(self.prefix_ids)
        prefix_starts = search_sequence_numpy(seq, prefix_np)
        prefix_start = prefix_starts[-1]
        if prefix_start == -1:
            return None, None
        prefix_end = prefix_start + len(prefix_np)

        # 3. Find Postfix Start (Search after prefix)
        postfix_np = np.array(self.postfix_ids)
        seq_suffix = seq[prefix_end:] 
        postfix_relative_start = search_sequence_numpy(seq_suffix, postfix_np)
        assert(len(postfix_relative_start)==1) #1 prefix 1 postfix!
        postfix_relative_start=postfix_relative_start[0]

        if postfix_relative_start == -1:
            return None, None
        postfix_start = prefix_end + postfix_relative_start

        # 4. Calculate Indices
        # Target tokens are at: input_ids[prefix_end : postfix_start]
        # The hidden state at index 'i' predicts the token at 'i+1'.
        # So we need hidden states at: [prefix_end - 1 : postfix_start - 1]
        
        logit_start = prefix_end - 1
        logit_end = postfix_start - 1

        # Create the indices tensor to pass to the model
        logit_indices = torch.arange(logit_start, logit_end, device='cpu', dtype=torch.long)
        return logit_indices, prefix_starts, search_sequence_numpy(seq, postfix_np)

    def _vocab_to_ids(self,vocab):
        ids = []
        for word in vocab:
            ids +=self.processor.tokenizer.encode(word)
        if len(ids)!=len(vocab):
            raise("input vocabulary is not valid token list!")
        return ids
    
    # requires full inputs to work.
    def _accumulate_inputs(self,inputs):
        if self.cumulative_inputs is None:
            self.cumulative_inputs = dict(inputs.to('cpu'))
            if self.save_pixels:
                self.cumulative_inputs['pixel_values'] = [inputs['pixel_values'].to('cpu')]
        else:
            self.cumulative_inputs['attention_mask'] = torch.cat([self.cumulative_inputs['attention_mask'],inputs['attention_mask']],dim=-1)
            # self.cumulative_inputs['position_ids'] = torch.cat([self.cumulative_inputs['position_ids'],inputs['position_ids']],dim=-1)
            self.cumulative_inputs['input_ids'] = torch.cat([self.cumulative_inputs['input_ids'],inputs['input_ids']],dim=-1)
            self.cumulative_inputs['image_grid_thw'] = torch.cat([self.cumulative_inputs['image_grid_thw'],inputs['image_grid_thw']],dim=0) # N_image by Hidden Size (16*16*6 ?)
            if self.save_pixels:
                self.cumulative_inputs['pixel_values'].append(inputs['pixel_values'].to('cpu'))
    
    def _accumulate_custom_inputs(self,inputs,dim=0):
        if self.cumulative_inputs is None:
            self.cumulative_inputs = dict(inputs.to('cpu'))
        else:
            for k,v in inputs.items():
                if k in self.cumulative_inputs.keys():
                    self.cumulative_inputs[k] = torch.cat([self.cumulative_inputs[k],v],dim=dim)
                else:
                    self.cumulative_inputs[k] = v

    def render_cumulative_inputs(self,summarize_images = True):
        input_ids = self.cumulative_inputs['input_ids']
        if self.moment_ids:
            # HAMLET placeholders sit above the tokenizer vocabulary; render them as the
            # <|vision_pad|> special token so decode does not choke on unknown ids.
            lo, hi = self.moment_ids[0], (self.mem_ids or self.moment_ids)[-1]
            pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_pad|>")
            input_ids = torch.where((input_ids >= lo) & (input_ids <= hi), torch.full_like(input_ids, pad_id), input_ids)
        if not summarize_images:
            return self.processor.batch_decode(input_ids)
        else:
            # image_mask = self.cumulative_inputs['input_ids'] == self.processor.image_token_id
            sequences = [torch.unique_consecutive(sequence,return_counts=True) for sequence in input_ids.numpy()]
            return self.processor.batch_decode(sequences)
    
    def _calculate_pos_id(self,pos_id_kwargs=None):
        if pos_id_kwargs is None or pos_id_kwargs['mode'] == "standard":
            input_ids = self.cumulative_inputs['input_ids']
            image_grid_thw = self.cumulative_inputs['image_grid_thw']
            attention_mask = self.cumulative_inputs['attention_mask']
            # mm_token_type_ids = self.cumulative_inputs['mm_token_type_ids'] # not sure if needed but just in case
            # 1. Ask Qwen to calculate the 3D layout for this chunk
            # This returns positions starting at T=0, H=0, W=0 relative to this chunk
            position_ids, deltas = self.vl_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw, 
                video_grid_thw=None,
                attention_mask=attention_mask,
                # mm_token_type_ids=mm_token_type_ids
            )
        elif pos_id_kwargs['mode'] == 'bev':
            from longnav.utils.bev_utils import get_pos_id
            self._accumulate_custom_inputs({'patch_coords':torch.tensor(pos_id_kwargs['patch_coords']).unsqueeze(0)},dim=0) 
            patch_coords = self.cumulative_inputs['patch_coords'] # N_image by H by W by 3
            patch_coords = patch_coords-torch.amin(patch_coords[:1],dim=[1,2],keepdim=True) 
            w,t,h = patch_coords[...,0],patch_coords[...,1],patch_coords[...,2] # horrific mess here
            w = w.reshape(-1)
            t = t.reshape(-1)
            h = h.reshape(-1)

            patch_coords = torch.stack([t,t+h,t+w],dim=0).reshape(1,3,-1)
            patch_coords = self.bev_canvas_size//2*torch.ones(1,3,1)
            position_ids = get_pos_id(self.cumulative_inputs['input_ids'],patch_coords.to(self.cumulative_inputs['input_ids'].dtype),self.processor,self.bev_canvas_size)
        return position_ids
    
    def _pos_id_fast(self,turn_inputs):
        # fast version that only calculates pos ids for the current turn.
        input_ids = turn_inputs['input_ids']
        image_grid_thw = turn_inputs['image_grid_thw']
        attention_mask = turn_inputs['attention_mask']
        position_ids, deltas = self.vl_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw, 
            video_grid_thw=None,
            attention_mask=attention_mask,
            # mm_token_type_ids=turn_inputs.get('mm_token_type_ids',None)
        )
        position_ids += self.offset
        self.offset += len(turn_inputs['input_ids'][0])
        self.offset += deltas.item()
        return position_ids

    def _store_outputs(self, outputs):
        """
        Extracts cached tensors from ModelOutput and appends them to the rollout buffer.
        """
        # 1. Standard Tensors (Append to list, concatenate later)
        # These are already on CPU thanks to the TextModel code
        self.outputs['inputs_embeds'].append(outputs.inputs_embeds)
        self.outputs['position_ids'].append(outputs.position_ids)
        self.outputs['visual_pos_masks'].append(outputs.visual_pos_masks)
        
        # 2. Deepstack Inputs (List of Tensors handling)
        # outputs.deepstack_visual_embeds is a list [Layer1_Tensor, Layer2_Tensor, ...]
        # We need to store them so we can eventually concat Layer 1 across all time steps.
        if outputs.deepstack_visual_embeds is not None:
            if 'deepstack_visual_embeds' not in self.outputs:
                # Initialize list of lists: [[], [], [], ...]
                self.outputs['deepstack_visual_embeds'] = [[] for _ in outputs.deepstack_visual_embeds]
            
            # Append Layer K's tensor to the Kth list
            for layer_idx, layer_tensor in enumerate(outputs.deepstack_visual_embeds):
                self.outputs['deepstack_visual_embeds'][layer_idx].append(layer_tensor)
    def _get_sparse_logit_indices(self):
        ranks = self.seq_keep_mask.long().cumsum(dim=0)
        logits_to_keep = ranks[self.logit_indices] - 1 # logit indices of the sparsified sequence
        return logits_to_keep
    
    def _pack_embeds(self):
        '''
        pack all the embeds needed to replicate forward pass of the entire sequence.
        
        RESETS internal outputs after packing.
        '''
        assert(self.save_outputs) # must be saving outputs to use this function.
        deepstack =[torch.cat([self.outputs['deepstack_visual_embeds'][i][j] for j in range(len(self.outputs['deepstack_visual_embeds'][i]))],dim=0) for i in range(len(self.outputs['deepstack_visual_embeds']))]
        position_ids = torch.cat(self.outputs['position_ids'],dim=-1)
        visual_pos_masks = torch.cat(self.outputs['visual_pos_masks'],dim=1)
        inputs_embeds = torch.cat(self.outputs['inputs_embeds'],dim=1)
        input_ids = self.cumulative_inputs['input_ids'][:,self.seq_keep_mask]
        self.outputs = defaultdict(list) # reset outputs.


        packed = {
            "deepstack_visual_embeds": torch.stack(deepstack,dim=0).cpu(), #N_layer by N_patch by N_hidden
            "position_ids": position_ids.cpu(),
            "visual_pos_masks": visual_pos_masks.cpu(),
            "inputs_embeds": inputs_embeds.cpu(),
            "input_ids_reference": input_ids.cpu(),
            "logits_to_keep": self._get_sparse_logit_indices().cpu()
        }
        if self.hamlet_enabled:
            # HAMLET: the rollout's moment hidden states, one (n_moment, H) block per
            # decision. The replay rebuilds the memory-token rows from these (they are
            # constants of the replay; see longnav.utils.hamlet).
            packed["moment_history"] = torch.stack(self._moment_history).cpu()
        return packed

    def _pack_inputs(self):
        '''
        pack all the raw inputs needed to replicate forward pass of the entire sequence.
        "logits_to_keep" ensure only action tokens are used by the lmhead.
        '''
        input_ids = self.cumulative_inputs['input_ids']
        attention_mask = self.cumulative_inputs['attention_mask']
        image_grid_thw = self.cumulative_inputs['image_grid_thw']
        pixel_values = self.cumulative_inputs.get('pixel_values',None)
        if pixel_values is not None and isinstance(pixel_values,list):
            pixel_values = torch.cat(pixel_values,dim=0)
        position_ids = self._calculate_pos_id()
        self.cumulative_inputs = None # reset cumulative inputs.
        return {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "image_grid_thw": image_grid_thw.cpu(),
            "position_ids": position_ids.cpu(),
            "pixel_values": pixel_values.cpu() if pixel_values is not None else None,
            "seq_keep_mask": self.seq_keep_mask.cpu(),
            "vis_keep_mask": torch.cat(self.vis_keep_masks,dim=0).cpu(),
            "logits_to_keep": self._get_sparse_logit_indices().cpu()
        }

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

    def _window_forward_inputs(self):
        '''
        The retained window as one contiguous forward input: the pinned prefix, the held-back
        boundary header, then the surviving turns -- the same tokens the evicting path leaves
        in the cache. The prefix is split out of turn 0's record, so it has to be prepended
        from the very first step; the boundary header only exists once something has been
        evicted.
        '''
        recs = list(self._window_recs)
        if self._n_evicted > 0:
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
                    "reindex cut must land on a text token (assistant header)"
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

    def infer_step(self,messages,images,full_logprobs=False,temperature=1.0,check_probs=True,crop_inputs=True,pos_id_kwargs=None):
        t0 = time.time()
        self.model.gradient_checkpointing_disable()
        self.model.eval()

        if self.model is None:
            self.load_model()
            self.reset()
        if self.using_lora() and not self.is_merged():
            self.merge_adapter() # for inference speed
            pass
        # print(f"lora merge time: {time.time()-t0}",end=" ")
        
        t = time.time()
        turn_inputs = self.tokenize_inputs(messages,images)

        # print(f"tokenize time: {time.time()-t}",end=" ")
        # First we must crop the sequence so the turns properly lign up.
        logit_indices,prefix_starts,postfix_starts = self._get_sandwich_indices(turn_inputs['input_ids'])
        if crop_inputs:
            if len(prefix_starts)>1:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)]
                if 'mm_token_type_ids' in turn_inputs.keys():
                    turn_inputs["mm_token_type_ids"] = turn_inputs['mm_token_type_ids'][:,(postfix_starts[0]-1):(postfix_starts[-1]-1)] 
            else:
                turn_inputs['attention_mask'] = turn_inputs['attention_mask'][:,:(postfix_starts[-1]-1)]
                turn_inputs["input_ids"] = turn_inputs['input_ids'][:,:(postfix_starts[-1]-1)]
                if 'mm_token_type_ids' in turn_inputs.keys():
                    turn_inputs["mm_token_type_ids"] = turn_inputs['mm_token_type_ids'][:,:(postfix_starts[-1]-1)]

        # HAMLET: per turn one memory placeholder block before the frame and one block
        # of moment placeholders after it, before the assistant header so the decision
        # token stays last. The memory rows are the read-out over the previous turns.
        if self.hamlet_enabled:
            turn_inputs = splice_moment_tokens(turn_inputs, self.moment_ids, self.vision_end_id, len(self.prefix_ids),
                                               mem_ids=self.mem_ids, vision_start_id=self.vision_start_id)
            self._hamlet_prepare_turn()

        # Pin the instruction span on the first turn whenever a context window
        # needs it, or the 3D attention mass stats want to report it.
        if self._prefix_len is None and (
            self.context_window is not None or self.visualize_attention_3d
        ):
            self._prefix_len = self._find_prompt_prefix_len(turn_inputs['input_ids'])

        t = time.time()
        self._accumulate_inputs(turn_inputs)
        # print(f"accumulate time: {time.time()-t}",end=" ")
        self.logit_indices.append(self.cumulative_inputs['input_ids'].shape[1]-1) #slice index for the hidden state predicting the last token in this turn.
        turn_inputs = {k: v.to(self.device) for k, v in turn_inputs.items() if v is not None}
        t = time.time()
        if pos_id_kwargs is None or pos_id_kwargs['mode'] == "standard": # use fast pos id calculation
            turn_inputs['position_ids'] = self._pos_id_fast(turn_inputs)
            if 'position_ids' not in self.cumulative_inputs.keys():
                self.cumulative_inputs['position_ids'] = turn_inputs['position_ids'].to('cpu')
            else:
                self.cumulative_inputs['position_ids'] = torch.cat([self.cumulative_inputs['position_ids'],turn_inputs['position_ids'].to('cpu')],dim=-1)
        else:
            self.cumulative_inputs['position_ids'] = self._calculate_pos_id(pos_id_kwargs) # calculate the pos_ids for the whole sequence. hopefully not too expensive...
            turn_inputs['position_ids'] = self.cumulative_inputs['position_ids'][..., -current_len:].to(self.device)
        # print(f"pos id time: {time.time()-t}",end=" ")
         # Set up inputs for this turn
        current_len = turn_inputs['input_ids'].shape[1]
        if self.use_sparse:
            turn_inputs['past_image_embeds'] = self.past_image_embeds
            turn_inputs['save_image_db'] = True # new argument in sparse qwen to signal keeping the db as internal state
            # sparsify the input attention mask
            turn_inputs['attention_mask'] = None#turn_inputs['attention_mask'] = torch.ones((turn_inputs['input_ids'].shape[0], (self.past_key_values.get_seq_length() if self.past_key_values is not None else 0) + turn_inputs['input_ids'].shape[1]), device=self.device, dtype=turn_inputs['attention_mask'].dtype)# torch.ones(1,seql,device=self.device)
        else:
            # The cumulative mask spans the whole episode; the cache may have been cropped
            # by the context window, so take only as much as the model will actually attend to.
            past_len = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
            turn_inputs['attention_mask'] = self.cumulative_inputs['attention_mask'][:,-(past_len+current_len):].to(self.device)
        if self.save_outputs or self._recomputing_window():
            turn_inputs['save_embeds'] = True

        with self._inference_ctx():
            # Arm the attention probe only for this forward. Other forwards (the value
            # head, training) reuse the same modules and would otherwise clobber the
            # captured rows; they also carry no KV cache, which the hook checks too.
            if self.attn_probe is not None:
                self.attn_probe.enabled = True
            try:
                t = time.time()
                outputs = self.model.forward(
                    **turn_inputs,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    # logits_to_keep = logit_indices.to(self.model.device)
                    logits_to_keep=1
                )
            finally:
                if self.attn_probe is not None:
                    self.attn_probe.enabled = False
            self.past_key_values = outputs['past_key_values']
             # Compute logprobs directly (1-to-1 mapping)
            if self.hamlet_enabled:
                self._hamlet_record_moments(outputs, turn_inputs)
            relevant_logits = outputs.logits[0].float()
            if not full_logprobs:
                relevant_logits = relevant_logits[...,self.vocab_ids]
            if np.abs(temperature-1.0) > 1e-7:
                logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            else:
                logprobs = torch.log_softmax(relevant_logits, dim=-1)
            # print(f"vlm latency: {time.time()-t}",end=" ")

            if self._recomputing_window():
                self._record_window_turn(outputs)
            if self.save_outputs:
                t = time.time()
                self._store_outputs(outputs)
                # print(f"store outputs time: {time.time()-t}",end=" ")
            if self.use_sparse:
                t = time.time()
                current_keep_mask = self.language_model.seq_keep_mask
                self.vis_keep_masks.append(self.language_model.vis_keep_mask.cpu())
                if self.seq_keep_mask is None:
                    self.seq_keep_mask = current_keep_mask.cpu()
                else:
                    self.seq_keep_mask = torch.cat((self.seq_keep_mask,current_keep_mask.cpu()))
                if self.past_image_embeds is None:
                    self.past_image_embeds = self.language_model.kept_visual_embeds
                else:
                    for idx, image_embeds in enumerate(self.language_model.kept_visual_embeds):
                        self.past_image_embeds[idx] = torch.cat((self.past_image_embeds[idx],image_embeds)) #handle the batching...
                # print(f"store sparse states time: {time.time()-t}",end=" ")
        if self.visualize_attention_3d:
            self._record_frame_keys()
            self._decision_inputs = self._slice_decision_token(turn_inputs)
        if self._grad_attribution_enabled():
            self._grad_attribution_pass()
        self._apply_context_window()
        # if check_probs:
        #     try:
        #         assert(torch.argmax(logprobs,dim=-1).item() in self.vocab_ids)
        #     except:
        #         print("WARNING: prediction not in provided vocab")
        # # print("inference done!")
        # print(f" total time: {time.time()-t0}")
        return logprobs.cpu().float().numpy(),outputs

        
    def _calculate_action_logprobs(self,logits):
        import torch
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits)
        action_logprobs = torch.log_softmax(logits[...,self.vocab_ids],dim=-1)
        return action_logprobs
    
    def infer_probs(self,messages,images,**kwargs):
        logprobs,outputs = self.infer_step(messages,images,**kwargs)
        assert(len(logprobs)==1) #ensure there is a unique token position for decision making
        logprobs = logprobs[0]
        probs = np.exp(logprobs)
        probs /= np.sum(probs)
        return probs,logprobs,outputs

    def get_filter_visualization(self):
        """(vis_keep_mask_list, [t,h,w]) for the latest turn, or None."""
        if not self.use_sparse or self.cumulative_inputs is None:
            return None
        mask = getattr(self.language_model, "vis_keep_mask", None)
        if mask is None:
            return None
        grid = self.cumulative_inputs["image_grid_thw"][-1]
        return mask.cpu().numpy().astype(bool).tolist(), [int(x) for x in grid.tolist()]

    def _scatter_patch_attention(self, rows):
        """Map attention rows (..., kv_len) onto the full patch grid of the latest frame.

        Returns ((..., llm_h*llm_w) tensor with dropped patches as 0, [t,h,w]) or None.
        """
        import torch

        if self.cumulative_inputs is None:
            return None
        vpm = getattr(self.language_model, "visual_pos_masks", None)
        vis_keep = getattr(self.language_model, "vis_keep_mask", None)
        if rows is None or vpm is None or vis_keep is None:
            return None
        grid = self.cumulative_inputs["image_grid_thw"][-1]
        t, h, w = (int(x) for x in grid.tolist())
        llm_n = (h // 2) * (w // 2)

        # Current chunk occupies the last chunk_len key positions of the cached sequence.
        chunk_len = vpm.shape[1]
        kv_len = rows.shape[-1]
        past_len = kv_len - chunk_len
        local_vis_idx = torch.nonzero(vpm[0], as_tuple=False).squeeze(-1)
        if local_vis_idx.numel() == 0:
            return None
        attn_kept = rows[..., past_len + local_vis_idx]  # order matches kept-patch order

        vis_keep = vis_keep.cpu().bool()
        keep_idx = torch.nonzero(vis_keep, as_tuple=False).squeeze(-1)
        # Guard against any length mismatch between kept patches and captured attention.
        n = min(keep_idx.numel(), attn_kept.shape[-1])
        full = torch.zeros(*rows.shape[:-1], llm_n, dtype=torch.float32)
        full[..., keep_idx[:n]] = attn_kept[..., :n]
        return full, [t, h, w]

    def get_attention_visualization(self):
        """(attn_map_list, [t,h,w]) for the latest turn, or None.

        attn_map_list is a full llm_h*llm_w vector of the action-decision token's
        attention to this frame's patches (dropped patches are 0), computed from the
        last decoder layer averaged over heads.
        """
        if not self.visualize_attention:
            return None
        scattered = self._scatter_patch_attention(self._last_attn_row)
        if scattered is None:
            return None
        full, grid = scattered
        return full.numpy().tolist(), grid

    def get_attention_heads_visualization(self):
        """(per_head_attn_maps, [t,h,w]) for the latest turn, or None.

        per_head_attn_maps is a list of num_heads full llm_h*llm_w vectors, one per
        attention head of the last decoder layer (dropped patches are 0).
        """
        if not self.visualize_attention_heads:
            return None
        scattered = self._scatter_patch_attention(self._last_attn_heads)
        if scattered is None:
            return None
        full, grid = scattered
        return full.numpy().tolist(), grid

    def _record_frame_keys(self):
        """Persist the newest frame's kept-patch bookkeeping so later steps can map
        their attention rows back onto this frame. The KV cache is append-only, so
        absolute key positions recorded here stay valid for the whole episode.
        """
        import torch

        vpm = getattr(self.language_model, "visual_pos_masks", None)
        vis_keep = getattr(self.language_model, "vis_keep_mask", None)
        kv_len = None if self.attn_probe is None else self.attn_probe.kv_len
        if kv_len is None or vpm is None or vis_keep is None:
            return
        # KV positions are shared by every layer, so one record serves them all.
        past_len = kv_len - vpm.shape[1]
        local_vis_idx = torch.nonzero(vpm[0].cpu(), as_tuple=False).squeeze(-1)
        grid = self.cumulative_inputs["image_grid_thw"][-1]
        self._frame_records.append({
            "abs_kv_idx": past_len + local_vis_idx,  # kept visual keys, absolute cache positions
            "grid_idx": torch.nonzero(vis_keep.cpu().bool(), as_tuple=False).squeeze(-1),
            "grid_thw": [int(x) for x in grid.tolist()],
        })

    def _contrastive_action_score(self, logits):
        """logit[best action] - mean(logit[the others]).

        The contrast is what makes the attribution action-specific. A bare logit also
        rises for evidence that merely makes the model confident about anything, so
        differentiating it tends to light up the whole scene; subtracting the rival
        actions keeps only the evidence that picked this action over them.

        Scored on the argmax rather than the sampled action because sampling happens
        in the caller, after this runs.
        """
        action_logits = logits[self.vocab_ids].float()
        best = int(action_logits.argmax())
        rivals = torch.cat([action_logits[:best], action_logits[best + 1:]])
        return action_logits[best] - rivals.mean()

    def _slice_decision_token(self, turn_inputs):
        """The one-token inputs that reproduce the action decision against the cache.

        Everything the decision token attends to is already cached, so replaying just
        this token re-derives the decision for the price of a single-token forward.
        Kept on the worker so anything that wants to re-score the decision can --
        the gradient attribution below, or an offline audit that masks keys out of
        the cache and measures what the action logprob does.
        """
        decision = {
            "input_ids": turn_inputs["input_ids"][:, -1:],
            "position_ids": turn_inputs["position_ids"][..., -1:],
        }
        if "mm_token_type_ids" in turn_inputs:
            decision["mm_token_type_ids"] = turn_inputs["mm_token_type_ids"][:, -1:]
        return decision

    def _grad_attribution_pass(self):
        """Replace the probe's attention rows with d(action score)/d(attention).

        The decision forward runs under no_grad, so its attention carries no graph.
        Rather than pay for a grad-enabled forward of the whole chunk, this replays
        just the one token whose logits pick the action: everything it attends to is
        already in the cache, so the replay recomputes the same attention row the real
        forward produced, at the cost of a single-token forward and a backward that
        autograd prunes to the attention tensors alone.

        The token's own KV entry has to come out of the cache first. Left in, the
        query would attend to a duplicate of itself and split the softmax mass, so the
        row would not be the one that actually chose the action.

        Unreachable under context_window_mode='reindex' (viz is refused at init): the slot
        this pops and restores holds a pre-rotation key there, and the replay would also
        double-append the reindex position table.

        The entry the replay appends in its place is put back rather than kept. It is
        the same computation, but a one-token matmul reduces in a different order than
        the chunk-wide one, and in bf16 that lands a unit in the last place away from
        the original. Left in the cache that rounding compounds across steps and moves
        the trajectory, which would make the visualization alter the very decisions it
        is meant to explain. The attribution row carries the same rounding, which is
        harmless -- it explains the same computation to within bf16 precision.

        Eval freezes every parameter (PEFT base + merged adapters). A plain input_ids
        forward then builds no autograd graph -- attention weights come out with
        requires_grad=False, the probe stores nothing, and the heat videos silently
        never appear. Feeding a requires_grad input embedding reopens the graph
        through the frozen weights without unfreezing them.
        """
        import torch

        assert self.context_window_mode != "reindex", \
            "grad attribution pops raw cache keys, which are pre-rotation under reindex"
        if self.attn_probe is None or self.past_key_values is None or self._decision_inputs is None:
            return
        # Nothing else writes rows in this mode, so a step that bails below must clear
        # them rather than let the previous step's map be reported as this step's.
        self.attn_probe.rows = {}
        if torch.is_inference_mode_enabled():
            # The cache would be inference tensors and the backward would raise.
            print("[attn_weighting=grad] skipping attribution: infer_step was called "
                  "inside torch.inference_mode().")
            return
        kv_total = self.past_key_values.get_seq_length()
        if kv_total < 1:
            return

        evicted = []
        for layer in self.past_key_values.layers:
            evicted.append((layer.keys[..., -1:, :].clone(), layer.values[..., -1:, :].clone()))
            layer.keys = layer.keys[..., :-1, :]
            layer.values = layer.values[..., :-1, :]

        replay = dict(self._decision_inputs)
        input_ids = replay.pop("input_ids")
        if self.use_sparse:
            # A non-tensor keep mask is this fork's "do not sparsify" signal. Without
            # it the sparse path builds its mask from the deepstack embeds, which a
            # text-only replay does not carry.
            replay["seq_keep_mask"] = "all"
            replay["attention_mask"] = None
        else:
            replay["attention_mask"] = self.cumulative_inputs["attention_mask"][:, -kv_total:].to(self.device)

        # The sparse text model clears these at the top of every forward, and the 2D
        # attention overlays read them after infer_step returns.
        sparse_state = {k: getattr(self.language_model, k, None)
                        for k in ("visual_pos_masks", "vis_keep_mask", "seq_keep_mask")}

        self.attn_probe.enabled = True
        try:
            with torch.enable_grad():
                embeds = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                outputs = self.model.forward(
                    **replay,
                    inputs_embeds=embeds,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                self.attn_probe.backward_from(self._contrastive_action_score(outputs.logits[0, -1]))
        finally:
            self.attn_probe.enabled = False
            for k, v in sparse_state.items():
                setattr(self.language_model, k, v)
            for layer, (keys, values) in zip(self.past_key_values.layers, evicted):
                # detach first: the replayed slot carries a graph, and leaving it
                # attached would pin this step's activations for the whole episode.
                layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
                if layer.keys.shape[-2] == kv_total:
                    layer.keys[..., -1:, :] = keys
                    layer.values[..., -1:, :] = values
                else:  # the replay raised before appending; re-attach by hand
                    layer.keys = torch.cat([layer.keys, keys], dim=-2)
                    layer.values = torch.cat([layer.values, values], dim=-2)

    def get_attention_3d_visualization(self):
        """({layer_idx: per_frame_attn_maps}, per_frame_grids) for the latest turn, or None.

        The probe's per-key row for the action-decision token (attention, a weighted
        variant of it, or a gradient attribution -- see AttentionProbe.weighting), one
        entry per probed decoder layer, scattered onto the full llm_h*llm_w patch grid
        of EVERY frame observed so far in the episode (dropped patches are 0). Maps
        are raw float16 bytes (numpy arrays don't unpickle across numpy 1.x/2.x envs).
        The grids are layer-independent, so they are reported once.
        """
        import torch

        if not self.visualize_attention_3d or self.attn_probe is None or not self._frame_records:
            return None
        rows = {idx: self.attn_probe.rows[idx] for idx in self.attn3d_layer_ids if idx in self.attn_probe.rows}
        if not rows:
            return None
        per_layer = {}
        for idx, row in rows.items():
            maps = []
            for rec in self._frame_records:
                t, h, w = rec["grid_thw"]
                full = torch.zeros((h // 2) * (w // 2), dtype=torch.float32)
                if rec["abs_kv_idx"] is not None:  # None once the context window evicts the frame
                    n = min(rec["grid_idx"].numel(), rec["abs_kv_idx"].numel())
                    full[rec["grid_idx"][:n]] = row[rec["abs_kv_idx"][:n]]
                maps.append(full.numpy().astype(np.float16).tobytes())
            per_layer[idx] = maps
        grids = [rec["grid_thw"] for rec in self._frame_records]
        return per_layer, grids

    def get_attention_mass_fractions(self):
        """Visual and instruction mass fractions of the decision row, or None.

        Returns {"visual": float, "instruction": float} when both can be computed.
        "visual" is mass on still-resident image-patch keys; "instruction" is mass
        on the pinned leading prompt span (goal + action space), which is always
        pure text and stays at cache positions [0, _prefix_len).

        In raw mode this reads the probe's mean-over-heads row, which is a true
        probability distribution -- the amax row the heat maps use is not one. The
        weighted/grad rows are only peak-scaled by a scalar, so the fractions are
        unchanged by that; under 'grad' they read as "share of positive attribution"
        rather than attention probability. With a context window active, evicted
        frames drop out of the visual share, but the instruction span stays pinned.
        """
        import torch

        if not self.visualize_attention_3d or self.attn_probe is None or not self._frame_records:
            return None
        if not self.attn3d_layer_ids or self._prefix_len is None:
            return None
        layer = self.attn3d_layer_ids[-1]
        row = self.attn_probe.mass_rows.get(layer, self.attn_probe.rows.get(layer))
        if row is None:
            return None
        total = float(row.sum())
        if total <= 0:
            return None
        idxs = [rec["abs_kv_idx"] for rec in self._frame_records if rec["abs_kv_idx"] is not None]
        if not idxs:
            return None
        vis_idx = torch.cat(idxs)
        vis_idx = vis_idx[vis_idx < row.numel()]
        n_instr = min(int(self._prefix_len), row.numel())
        return {
            "visual": float(row[vis_idx].sum()) / total,
            "instruction": float(row[:n_instr].sum()) / total,
        }

    def merge_adapter(self):
        print("Merging LoRA adapters for inference...")
        self._is_merged = True
        self.model.merge_adapter()
        # o_proj is a LoRA target, so any cached W_O factors are now stale.
        if self.attn_probe is not None:
            self.attn_probe.invalidate_factors()

    def unmerge_adapter(self):
        print("Unmerging LoRA for training")
        self._is_merged = False
        self.model.unmerge_adapter()
        if self.attn_probe is not None:
            self.attn_probe.invalidate_factors()
        
    def is_merged(self):
        if self._is_merged is None:
            self._is_merged = len(self.model.get_model_status().merged_adapters) > 0
        return self._is_merged
    
    def using_lora(self):
        if self._is_lora is None:
            self._is_lora = isinstance(self.model,PeftModel)
        return self._is_lora

# Handle optional PEFT imports gracefully
try:
    from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class ValueHead(nn.Module):
    """
    A configurable MLP Value Head.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1,dtype:str='float32'):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim,dtype=dtype))
            layers.append(nn.Mish()) # why not
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        # Final projection to scalar value
        final_proj = nn.Linear(curr_dim, 1,dtype=dtype)
        # Initialize to zero for 0 value at start
        # with torch.no_grad():
        #     final_proj.weight.fill_(0.)
        #     final_proj.bias.fill_(0.)
        layers.append(final_proj)
        self.mlp = nn.Sequential(*layers)
        self.dtype = dtype

    def forward(self, x):
        return self.mlp(x)
    
class VLMWrapper(nn.Module):
    """
    Thin wrapper that enables forward pass of the language model to play nicely with DDP
    """
    def __init__(self, vlm, moment_ids=(), mem_ids=()):
        super().__init__()
        self.vlm = vlm # Can be PeftModel
        self.moment_ids = list(moment_ids)  # HAMLET placeholder ids; empty = HAMLET off
        self.mem_ids = list(mem_ids)
        self.last_stats = {}
        self._freeze_vision_tower()

    def _forward_embeds(self,embeds_inputs,compute_values=False,value_grad_scale=0.1):
        # Resolved at call time so it is the PEFT ModulesToSaveWrapper once wrapped.
        hamlet = getattr(self.vlm, HAMLET_MODULE_NAME, None) if self.moment_ids else None
        logits, values, self.last_stats = forward_embeds_core(
            embeds_inputs,
            language_model=self.vlm.model.model.language_model, #TODO: fix this mess
            lm_head=self.vlm.lm_head,
            dtype=self.vlm.dtype,
            training=self.vlm.training,
            hamlet=hamlet,
            moment_ids=self.moment_ids,
            mem_ids=self.mem_ids,
            compute_values=compute_values,
            value_head=getattr(self.vlm, "value_head", None),
            value_grad_scale=value_grad_scale,
        )
        return logits,values

    def forward(self, mode = "embeds_inputs",**inputs):
        if mode == "embeds_inputs":
            return self._forward_embeds(**inputs)
        elif mode == "standard":
            if hasattr(self.vlm, "value_head"):
                # Calculate a 0.0 scalar attached to the value head's graph
                dummy_loss = 0.0 # this hack prevents ddp freeze in sft
                for p in self.vlm.value_head.parameters():
                    if p.requires_grad:
                        dummy_loss = dummy_loss + p.sum() * 0.0
                        break
            return self.vlm(**inputs)
        elif mode == "language":
            return self.vlm.language_model(**inputs)

    def _freeze_vision_tower(self):
        """
        Locates the vision tower and ensures all parameters are frozen.
        Logs a warning if trainable parameters were found and suppressed.
        """
        # 1. unwrapping helper to get down to the base architecture
        # (Handles PeftModel, DistributedDataParallel, etc.)
        base = self.vlm

        # 2. Attempt to locate the vision module using common naming conventions
        # (Covers LLaVA, Qwen-VL, Idefics, etc.)
        vision_tower = None
        potential_names = ["vision_model", "vision_tower", "visual_model", "visual", "vit"]
        
        # Check top level
        for attr in potential_names:
            if hasattr(base, attr):
                vision_tower = getattr(base, attr)
                break
        
        # Check inside .model (Common in HF Llama-based architectures)
        if vision_tower is None and hasattr(base, "model"):
             for attr in potential_names:
                if hasattr(base.model, attr):
                    vision_tower = getattr(base.model, attr)
                    break

        # 3. Freeze and Warn
        if vision_tower is not None:
            frozen_count = 0
            example_names = []
            
            for name, param in vision_tower.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
                    if len(example_names) < 3:
                        example_names.append(name)
            
            if frozen_count > 0:
                print(f"\n[VLMWrapper] ⚠️ WARNING: Found {frozen_count} trainable parameters in Vision Tower.")
                print(f"[VLMWrapper] Examples: {example_names}")
                print("[VLMWrapper] ACTION: Forcibly FROZEN these parameters to ensure DDP compatibility in RL steps.\n")
        else:
            # Fallback info if architecture is exotic
            print("[VLMWrapper] Info: Could not auto-detect Vision Tower module to safeguard. Assuming it is correctly frozen.")

class VLMTrainingMixin:

    def setup_training(self, config: VLMTrainingConfig, rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,):
        """
        Sets up distributed training using the provided TrainConfig.
        """
        from accelerate import DistributedDataParallelKwargs
        from transformers import get_scheduler
        from accelerate import Accelerator

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #prevent value head from being killed
        # 1. Manual Environment Injection for Ray
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = "0"
        
        self.rl_algo_config=config.rl_config
        # 2. Initialize Accelerator
        print("creating accelerator")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.grad_accum_steps,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[kwargs]
        )
# or check the accelerator state
        self.gradient_checkpointing =config.gradient_checkpointing
        # 3. Gradient Checkpointing (Must run before PEFT wrapping)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            # This logic handles the edge case where input embeddings are frozen
            # causing backward() to fail with checkpointing enabled.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads() # use_reentrant=False to prevent hangs
            else:
                raise NotImplementedError("Model does not support 'enable_input_require_grads' method.")
        hidden_size = self.language_model.config.hidden_size
        
        if config.rl_config is not None:
            if config.rl_config.use_value:
                self.model.value_head = ValueHead(
                    input_dim=hidden_size,
                    hidden_dims=config.value_head_hidden_dims,
                    dropout=config.value_head_dropout,
                    dtype=getattr(torch,config.value_head_dtype)
                ).to(self.model.device)
            from verl.trainer.ppo.core_algos import get_policy_loss_fn
            self.policy_loss_fn = get_policy_loss_fn(config.rl_config.policy_loss_name)
        # 4. Apply PEFT (if config provided)
        self._setup_peft(config)
            # Print trainable parameters to verify LoRA is active
        try:
            if self.accelerator.is_local_main_process:
                self.model.print_trainable_parameters()
        except:
            print("failed to print trainable parameters...")
        # 5. Create Optimizer
        # Only optimize parameters that require gradients (i.e., the Adapters)
        print(f"accelerator device: {self.accelerator.device}")
        wrapper = VLMWrapper(self.model, moment_ids=self.moment_ids, mem_ids=self.mem_ids)
        self._wrapper = wrapper
        hamlet_key = f".{HAMLET_MODULE_NAME}."
        rest_params = [p for n, p in wrapper.named_parameters()
                       if "value_head" not in n and hamlet_key not in n and p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": rest_params,
                "lr": config.learning_rate,
                "name": "adapters"
            }
        ]

        if config.rl_config.use_value:
            head_params = [p for n, p in wrapper.named_parameters() if "value_head" in n and p.requires_grad]
            optimizer_grouped_parameters+=[
                {
                    "params": head_params,
                    "lr": config.value_head_learning_rate,
                    "name": "value_head"
                }]
        # HAMLET is a fresh module (the LoRA is not): it gets its own learning rate.
        # Only the trainable ModulesToSaveWrapper copy has requires_grad; PEFT freezes
        # the original, which disable_adapter() routes the ref policy through.
        self._hamlet_params = [p for n, p in wrapper.named_parameters() if hamlet_key in n and p.requires_grad]
        if self._hamlet_params:
            optimizer_grouped_parameters += [{
                "params": self._hamlet_params,
                "lr": float(self.hamlet_cfg.get("learning_rate", 1e-4)),
                "name": "hamlet",
            }]
            print(f"[hamlet] {sum(p.numel() for p in self._hamlet_params)/1e6:.1f}M trainable params "
                  f"at lr {float(self.hamlet_cfg.get('learning_rate', 1e-4))}")
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps, # Short warmup usually sufficient for RL
            num_training_steps=config.total_optimization_steps
        )

        # 6. Prepare with Accelerator
        # self.ddp_model becomes the sync-wrapper
        # self.model remains the direct reference (now with LoRA layers attached)

        self.ddp_model, self.optimizer,self.scheduler = self.accelerator.prepare(
            wrapper, optimizer, scheduler
        )            

        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint,True,config.load_optim,config.load_sched)
    def _setup_peft(self, config):
        if config.peft_config is not None:
            if not PEFT_AVAILABLE:
                raise ImportError("TrainConfig has peft_config, but 'peft' library is not installed.")
            from peft import LoraConfig
            from dataclasses import asdict
            # Direct application of the config object
            # CRITICAL: Add 'value_head' to modules_to_save so PEFT treats it as 
            # a full-rank trainable module (not an adapter) and saves it in the checkpoint.
            if config.peft_config.modules_to_save is None:
                config.peft_config.modules_to_save = []
            if "value_head" not in config.peft_config.modules_to_save:
                config.peft_config.modules_to_save.append("value_head")
            # Same treatment for HAMLET: full-rank trainable copy, saved in the adapter.
            # Adapters that predate the module (the stage-1 LoRA checkpoint) carry no
            # HAMLET keys; load_checkpoint fills them from the init (_fill_missing_aux_keys).
            if self.hamlet_enabled and HAMLET_MODULE_NAME not in config.peft_config.modules_to_save:
                config.peft_config.modules_to_save.append(HAMLET_MODULE_NAME)
            try:
                peft_kwargs = asdict(config.peft_config)
            except:
                from omegaconf import OmegaConf
                peft_kwargs = OmegaConf.to_container(config.peft_config,resolve=True)
            for key in ["target_modules", "modules_to_save", "modules_to_freeze"]:
                if key in peft_kwargs and peft_kwargs[key] is not None:
                    # The magic fix: list() casts ListConfig -> list
                    peft_kwargs[key] = list(peft_kwargs[key])

            real_peft_config = LoraConfig(**peft_kwargs)
            self.model = get_peft_model(self.model, real_peft_config)
        else:
            print("PEFT config not provided; training all model parameters.")
    def train_sft_step(self, batch):
        """
        Standard training step.
        """
        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            # Forward via DDP wrapper (triggers sync)
            outputs = self.ddp_model(mode="standard",**batch)
            loss = outputs.loss
            
            # Backward (handles mixed precision scaling)
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def _forward_embeds(self,rl_embeds_inputs,compute_values=False):
        # Same replay as the DDP training forward (forward_embeds_core), so old/ref
        # log-probs and the training log-probs are computed by one code path.
        hamlet = self._hamlet() if self.hamlet_enabled else None
        logits, values, _ = forward_embeds_core(
            rl_embeds_inputs,
            language_model=self.language_model,
            lm_head=self.model.lm_head,
            dtype=self.model.dtype,
            training=self.model.training,
            hamlet=hamlet,
            moment_ids=self.moment_ids,
            mem_ids=self.mem_ids,
            compute_values=compute_values,
            value_head=getattr(self.model, "value_head", None),
            value_grad_scale=None,
        )
        return logits,values
    
    def _forward_seq(self,rl_seq_inputs):
        # seq_inputs = {k:torch.tensor(v,device='cuda') for k,v in self.rl_seq_inputs.items()}
        seq_inputs = {k:v.to('cuda') for k,v in rl_seq_inputs.items()}
        output = self.model(**seq_inputs)
        return output.logits
    
    def _setup_training(self):
        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        self.reset() #clear internal state, training is (mostly) stateless
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        
    def _training_forward(self,embeds_inputs):
        # Forward via DDP wrapper (triggers sync)
        logits,vpreds = self.ddp_model(embeds_inputs = embeds_inputs,compute_values = self.rl_algo_config.use_value,value_grad_scale=self.rl_algo_config.value_grad_scale)
        return logits,vpreds 
    
    def rl_loss(self, log_probs, actions, advantages, response_mask, old_log_prob, returns, old_values, vpreds, logits, rollout_log_probs=None, ref_log_probs=None):        
        from verl.trainer.ppo.core_algos import compute_value_loss,compute_entropy_loss
        #TODO: rollout correction, rejection sampling to exclude bad tokens
        log_prob = torch.gather(log_probs, -1, actions.unsqueeze(-1).to(log_probs.device)).squeeze(-1)
        response_mask = response_mask.to(log_prob.device).bool()
        # --- CRITICAL FIX: Handle Pure DAgger Episodes ---
        if response_mask.sum() == 0:
            print("warning: empty RL mask, skipping RL loss.")
            # If PPO has no data (all tokens went to DAgger), return 0 loss safely.
            # We strictly require grad=True for DDP compatibility.
            zero_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
            return zero_loss, {'loss/pg_loss': 0.0, 'return': 0.0, 'train/vf_loss': 0.0}
        pg_loss,metrics = self.policy_loss_fn(old_log_prob=old_log_prob.to(log_prob.device),log_prob=log_prob,advantages=advantages.to(log_prob.device),response_mask=response_mask,config = self.rl_algo_config)
        metrics['loss/pg_loss'] = pg_loss.detach().item()
        metrics['return'] = torch.amax(returns).detach().item()
        if self.rl_algo_config.use_value:
            value_loss,vf_clipfrac = compute_value_loss(vpreds,returns.to(log_prob.device),old_values.to(log_prob.device),response_mask,self.rl_algo_config.cliprange_value)
            loss = pg_loss + value_loss
            metrics['critic/vf_clipfrac'] = vf_clipfrac.detach().item()
            metrics['train/vf_loss'] = value_loss.detach().item()
            valid_values = torch.masked_select(vpreds, response_mask).cpu()
            valid_returns = torch.masked_select(returns,response_mask.cpu())
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            metrics['critic/explained_variance']=(1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
        else:
            loss = pg_loss

        if self.rl_algo_config.entropy_bonus is not None:
            entropy = compute_entropy_loss(logits,response_mask)
            entropy_loss = -entropy*self.rl_algo_config.entropy_bonus
            metrics['train/entropy'] = entropy.detach().item()
            loss = loss+entropy_loss

        if ref_log_probs is not None and self.rl_algo_config.kl_coeff is not None:
            kld = compute_full_kl_penalty(log_probs,ref_log_probs.to(log_probs.device))
            metrics['train/ref_kl_divergence'] = kld.mean().item()
            loss = loss + (kld * self.rl_algo_config.kl_coeff).mean()

        if rollout_log_probs is not None:
            kld = compute_full_kl_penalty(log_probs.cpu(),rollout_log_probs.cpu())
            metrics['train/rollout_kl_divergence'] = kld.mean().item()        
        return loss,metrics
    
    def bc_loss(self, log_probs, expert_actions, dagger_mask, label_smoothing=0.1, **kwargs):
        """
        Behavior Cloning / DAgger Loss.
        """
        import torch.nn.functional as F
        
        # Flatten for CrossEntropyLoss
        # log_probs: [B, S, Vocab] -> [B*S, Vocab]
        # expert_actions: [B, S] -> [B*S]
        
        # We only want to train on the specific tokens masked for DAgger
        # (e.g. the 5% worst episodes)
        # --- ROBUSTNESS FIX: Sanitize Targets ---
        # Ensure we don't train on -1 or indices >= vocab size
        vocab_size = log_probs.size(-1)
        dagger_mask = dagger_mask.bool()
        # Create a mask of valid targets
        # This filters out -1s (Oracle failures) or garbage indices
        
        # Combine with the requested DAgger mask
        # We only train if: 1. It's selected for DAgger AND 2. The label is valid
        active_indices = dagger_mask.view(-1).to(log_probs.device)

        flat_log_probs = log_probs.view(-1, log_probs.size(-1))
        flat_targets = expert_actions.view(-1).to(log_probs.device)
        valid_target_mask = (flat_targets >= 0) & (flat_targets < vocab_size)
        active_indices = active_indices & valid_target_mask.to(active_indices.device)
        if not active_indices.any():
            zero_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
            return zero_loss, {}

        logits = kwargs.get('logits')
        if logits is not None:
            action_logits = logits[..., self.vocab_ids]
            flat_logits = action_logits.view(-1, action_logits.size(-1))
            loss = F.cross_entropy(
            flat_logits[active_indices], 
            flat_targets[active_indices],
            label_smoothing=label_smoothing
            )
        else:
            # Fallback if only log_probs available (no smoothing easily available)
            loss = F.nll_loss(
            flat_log_probs[active_indices], 
            flat_targets[active_indices]
            )
        
        metrics = {'loss/dagger_loss': loss.detach().item()}
        
        # Optional: Scale the loss if needed (usually done in config)
        return loss, metrics
    
    def generic_train_step(self,embeds_inputs,loss_fn_names,loss_kwargs_list,loss_weights=None):
        '''
        Generic train step that can be used for both RL, SFT, or any unholy combination thereof
        '''
        self._setup_training()
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            loss = torch.tensor(0.0).to(self.device)
            metrics = {}
            logits,vpreds = self._training_forward(embeds_inputs)
            log_probs = self._calculate_action_logprobs(logits) # B by S by N_action space
            if loss_weights is None:
                loss_weights = [1.0]*len(loss_fn_names)
            for loss_fn_name,weight in zip(loss_fn_names,loss_weights):
                loss_fn = getattr(self, f"{loss_fn_name}_loss")
                loss_part,metric = loss_fn(log_probs=log_probs,vpreds=vpreds,logits=logits,**loss_kwargs_list[loss_fn_name])
                loss = loss + loss_part*weight
                metrics |= metric
                
            self.accelerator.backward(loss)
            # Clip gradients and return the total norm (Global L2)
            # max_grad_norm is usually 0.5 or 1.0 in PPO papers
            grad_norm = self.accelerator.clip_grad_norm_(
                self.ddp_model.parameters(), 
                max_norm=1.0 
            )
            # Log the norm (Detect explosions if this spikes > 10.0)
            metrics['train/grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            
            self.optimizer.step()
            self.scheduler.step()
            metrics['train/lr'] = self.scheduler.get_last_lr()[0]
            self.optimizer.zero_grad()
        return metrics
            
    def train_rl_step(self,embeds_inputs,actions,old_log_prob,advantages,returns=None,old_values=None,rollout_log_probs=None,ref_log_probs=None):
        '''
        Docstring for train_rl_step
        
        :param embeds_inputs: batch of embeds for forward pass 
        :param old_log_prob: B by S 
        :param advantages: B by S advantages
        :param returns: targets for value head, B by S
        :param rollout_log_prob: Optional for rollout correction (not yet implemented)
        :param ref_logprobs: B by S by Action Space
        '''
        from verl.trainer.ppo.core_algos import compute_value_loss,compute_entropy_loss

        self.ddp_model.train()
        if self.is_merged():
            self.unmerge_adapter()
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        self.reset() #clear internal state, training is (mostly) stateless
        self.accelerator.wait_for_everyone() # ensure all workers have unmerged before training
        # Accumulate gradients (handle micro-batches)
        with self.accelerator.accumulate(self.ddp_model):
            # Forward via DDP wrapper (triggers sync)
            logits,vpreds = self.ddp_model(embeds_inputs = embeds_inputs,compute_values = self.rl_algo_config.use_value,value_grad_scale=self.rl_algo_config.value_grad_scale)
            log_probs = self._calculate_action_logprobs(logits) # B by S by N_action space
            log_prob = torch.gather(log_probs, -1, actions.unsqueeze(-1).to(log_probs.device)).squeeze(-1)
            
            '''
            old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
            log_prob (torch.Tensor):
                Log-probabilities of actions under the current policy, shape (batch_size, response_length).
            advantages (torch.Tensor):
                Advantage estimates for each action, shape (batch_size, response_length).
            response_mask (torch.Tensor):
                Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
            loss_agg_mode (str, optional):
                Aggregation mode for `agg_loss`. Defaults to "token-mean".
            config: `(verl.trainer.config.ActorConfig)`: config for the actor.
            '''
            
            response_mask = torch.ones_like(log_prob).bool() #TODO: rollout correction, rejection sampling to exclude bad tokens
            pg_loss,metrics = self.policy_loss_fn(old_log_prob=old_log_prob.to(log_prob.device),log_prob=log_prob,advantages=advantages.to(log_prob.device),response_mask=response_mask,config = self.rl_algo_config)
            metrics['loss/pg_loss'] = pg_loss.detach().item()
            metrics['return'] = torch.amax(returns).detach().item()
            if self.rl_algo_config.use_value:
                value_loss,vf_clipfrac = compute_value_loss(vpreds,returns.to(log_prob.device),old_values.to(log_prob.device),response_mask,self.rl_algo_config.cliprange_value)
                loss = pg_loss + value_loss
                metrics['critic/vf_clipfrac'] = vf_clipfrac.detach().item()
                metrics['train/vf_loss'] = value_loss.detach().item()
                valid_values = torch.masked_select(vpreds, response_mask).cpu()
                valid_returns = torch.masked_select(returns,response_mask.cpu())
                return_diff_var = torch.var(valid_returns - valid_values)
                return_var = torch.var(valid_returns)
                metrics['critic/explained_variance']=(1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
            else:
                loss = pg_loss

            if self.rl_algo_config.entropy_bonus is not None:
                entropy = compute_entropy_loss(logits,response_mask)
                entropy_loss = -entropy*self.rl_algo_config.entropy_bonus
                metrics['train/entropy'] = entropy.detach().item()
                loss = loss+entropy_loss

            if ref_log_probs is not None and self.rl_algo_config.kl_coeff is not None:
                kld = compute_full_kl_penalty(log_probs,ref_log_probs.to(log_probs.device))
                metrics['train/ref_kl_divergence'] = kld.mean().item()
                loss = loss + (kld * self.rl_algo_config.kl_coeff).mean()

            if rollout_log_probs is not None:
                kld = compute_full_kl_penalty(log_probs.cpu(),rollout_log_probs.cpu())
                metrics['train/rollout_kl_divergence'] = kld.mean().item()
            # Backward (handles mixed precision scaling)
            self.accelerator.backward(loss)
            # Clip gradients and return the total norm (Global L2)
            # max_grad_norm is usually 0.5 or 1.0 in PPO papers
            grad_norm = self.accelerator.clip_grad_norm_(
                self.ddp_model.parameters(), 
                max_norm=1.0 
            )
            # Log the norm (Detect explosions if this spikes > 10.0)
            metrics['train/grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm

            self.optimizer.step()
            self.scheduler.step()
            metrics['train/lr'] = self.scheduler.get_last_lr()[0]
            metrics['train/max_mem_GB'] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # process peak (rollout + replay)
            metrics |= self._hamlet_train_metrics()
            self.optimizer.zero_grad()
        return metrics    

    def _hamlet_train_metrics(self):
        """Gradient norm of the HAMLET parameters, the read-out projection's weight
        norm (0 at init, non-zero after the first update), the mean memory-token
        read-out ratio ||MEM - mem_embed||/||mem_embed|| of this replay and the
        replay-vs-stored moment drift. Call after optimizer.step(), before zero_grad."""
        params = getattr(self, "_hamlet_params", None)
        if not params:
            return {}
        with torch.no_grad():
            sq = sum(p.grad.float().norm() ** 2 for p in params if p.grad is not None)
            metrics = {
                'train/hamlet_grad_norm': float(sq ** 0.5) if torch.is_tensor(sq) else 0.0,
                'train/hamlet_out_proj_norm': float(self._hamlet()(mode="out_proj_norm")),
            }
        stats = getattr(self._wrapper, "last_stats", None) or {}
        for key in ("hamlet_mem_ratio", "hamlet_moment_drift"):
            if key in stats:
                metrics[f'train/{key}'] = stats[key]
        return metrics
    
    def save_adapter(self, path):
        """
        Saves ONLY the LoRA adapters. 
        Safe to call from Ray actor (handles rank check internally).
        """
        # Wait for all workers to finish their current step
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # We unwrap to get the PeftModel, then call save_pretrained
            # which knows to only save the 'adapter_model.bin'
            # unwrapped = self.accelerator.unwrap_model(self.ddp_model)
            # unwrapped.vlm.save_pretrained(path)
            self.model.save_pretrained(path)
            self._save_hamlet_reference(path)
            print(f"Adapters saved to {path}")

    def save_adapter_unsafe(self, path):
        """
        Saves ONLY the LoRA adapters. 
        Driver script is responsible for making the other VLM workers stay put, hence "unsafe"
        """
        
        self.model.save_pretrained(path)
        self._save_hamlet_reference(path)
        print(f"Adapters saved to {path}")

    def _save_hamlet_reference(self, path):
        """Persist PEFT's frozen HAMLET copy, which adapter checkpoints omit."""
        if not self.hamlet_enabled:
            return
        reference = getattr(self._hamlet(), "original_module", None)
        if reference is None:
            return
        state = {name: value.detach().cpu() for name, value in reference.state_dict().items()}
        torch.save(state, os.path.join(path, "hamlet_reference.pt"))

    def save_checkpoint_unsafe(self, path):
        """
        Ray-Optimized Saver. 
        NO BARRIERS. Call this ONLY on Rank 0 (Worker[0]).
        The Driver script MUST ensure all other workers are idle/waiting 
        via ray.get() before triggering this.
        """
        import os
        os.makedirs(path, exist_ok=True)
        # 1. Save Model (Adapters)
        # Standard DDP models are replicated, so Rank 0 has everything.
        # save_pretrained is a local I/O operation.
        self.model.save_pretrained(path)
        self._save_hamlet_reference(path)
        # 2. Save Optimizer & Scheduler
        # In Standard DDP, optimizer states are identical across ranks.
        # Saving Rank 0's copy is sufficient to restore training.
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        print(f"✅ Checkpoint saved to: {path}")

    def _fill_missing_aux_keys(self, adapter_state_dict, adapter_name="default"):
        """PEFT indexes every key of every ``modules_to_save`` wrapper when it loads
        an adapter (there is no strict=False on that path), so an adapter saved
        before a module existed -- the stage-1 LoRA checkpoint loaded with HAMLET
        on -- would raise KeyError. Fill such keys with the module's current
        (init) values instead, and say which modules were left at init."""
        from peft.utils.other import AuxiliaryTrainingWrapper
        left_at_init = set()
        for name, module in self.model.named_modules():
            if not isinstance(module, AuxiliaryTrainingWrapper):
                continue
            key_map = module.adapter_state_dict_load_map(adapter_name)
            copies = getattr(module, "modules_to_save", None)
            if not key_map or copies is None or adapter_name not in copies:
                continue
            current = copies[adapter_name].state_dict()
            for k in key_map:
                full = f"{name}.{k}"
                if full not in adapter_state_dict:
                    adapter_state_dict[full] = current[k].detach().clone()
                    left_at_init.add(name)
        if left_at_init:
            print(f" -> adapter has no weights for {sorted(left_at_init)}; left at init")
        return adapter_state_dict

    def load_checkpoint(self, path, strict_base_check=True,load_optim=True,load_sched=False):
        """
        Resumes training state fully. 
        Must be called AFTER setup_training().
        """
        import os
        from peft.utils import set_peft_model_state_dict, load_peft_weights
        from longnav.utils.factories import get_base_model
        # 1. Base Model Check
        
        if strict_base_check:
            saved_base = get_base_model(path)
            if  saved_base is None:
                print(f"⚠️ WARNING: no base model name found")
            elif self.model_id not in saved_base and saved_base not in self.model_id:
                print(f"⚠️ WARNING: Checkpoint base '{saved_base}' != Current '{self.model_id}'")
        # 2. Load Weights (Adapters + Value Head)
        # This updates self.model in-place, preserving optimizer references
        had_hamlet_state = False
        if os.path.exists(os.path.join(path, "adapter_model.bin")) or os.path.exists(os.path.join(path, "adapter_model.safetensors")):
             adapter_state_dict = load_peft_weights(path)
             had_hamlet_state = any(".hamlet." in key for key in adapter_state_dict)
             adapter_state_dict = self._fill_missing_aux_keys(adapter_state_dict)
             set_peft_model_state_dict(self.model, adapter_state_dict)
             print(" -> Adapters and (maybe) Value Head loaded.")
        else:
             print(" -> ⚠️ No adapter weights found in checkpoint.")

        # 3. Restore the frozen HAMLET reference policy. PEFT deliberately omits
        # original_module from adapter_model.safetensors, but its moment embeddings
        # affect the reference logits, so a training resume must preserve this copy.
        if self.hamlet_enabled:
            ref_path = os.path.join(path, "hamlet_reference.pt")
            reference = getattr(self._hamlet(), "original_module", None)
            if reference is not None and os.path.exists(ref_path):
                ref_state = torch.load(ref_path, map_location="cpu", weights_only=True)
                reference.load_state_dict(ref_state)
                print(" -> Frozen HAMLET reference loaded.")
            elif had_hamlet_state:
                print(" -> ⚠️ HAMLET checkpoint has no frozen reference state; using a fresh reference.")

        # 4. Load Optimizer
        opt_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(opt_path) and load_optim:
            print("loading optimizer!")
            opt_state = torch.load(opt_path, map_location=self.accelerator.device)
            self.optimizer.load_state_dict(opt_state)
            print(" -> Optimizer loaded.")
        
        # 5. Load Scheduler
        sched_path = os.path.join(path, "scheduler.pt")
        if os.path.exists(sched_path) and load_sched:
            print("loading scheduler!")
            sched_state = torch.load(sched_path, map_location=self.accelerator.device)
            self.scheduler.load_state_dict(sched_state)
            print(" -> Scheduler loaded.")

class DataGenerator:
    """Generates synthetic turn data."""
    def __init__(self, width=640, height=480, processor=None):
        self.width = width
        self.height = height
        self.processor = processor

    def create_synthetic_image(self):
        from PIL import Image
        arr = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def _prepare_turn_inputs(self, step_idx):
        """
        Creates inputs for a SINGLE turn (Image + Text).
        We do not build the full conversation history in the prompt.
        We rely on the KV cache for history.
        """
        image = self.create_synthetic_image()
        
        # Construct a standalone prompt for this step
        # We simulate the user asking for a move
        messages = [

            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Step {step_idx}: Next move?"}
                ]
            },
            # We add the Assistant start token to force the model to predict the response immediately
            {"role": "assistant", "content": "**forward**"} 
        ]
        return messages, [image]
 
if __name__ == "__main__":
    from vlm_worker import VLMWorker
    import torch
    import time
    import argparse
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("running inference test")
    # --- Constants ---
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    # MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

    
    worker = VLMWorker(model_id=MODEL_ID,attn_impl='flash_attention_2', dtype='bfloat16',offload_cache=False,use_sparse=True)
    generator = DataGenerator(IMAGE_WIDTH, IMAGE_HEIGHT, worker.processor)
    worker.reset()
    from tqdm import tqdm
    # torch.cuda.memory._record_memory_history(
    #    max_entries=3
    # )
    for i in tqdm(range(160)):
        messages,images = generator._prepare_turn_inputs(i)
        action,_,_ = worker.infer_probs(messages,images)
        # action = worker.infer_step(messages,images)
