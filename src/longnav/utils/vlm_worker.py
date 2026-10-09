import numpy as np
import torch
import time

CONTEXT_WINDOW_MODES = ("evict", "recompute", "reindex", "prune")
CONTEXT_CUTS = ("header", "frame")
KV_PRUNE_IMPORTANCE_MODES = ("attn", "random", "divprune", "dart")
KV_PRUNE_SCORE_LAYER_MODES = ("pruned", "all", "boundary")


from longnav.utils.context_cache import ContextCacheMixin
from longnav.utils.diagnostics import DiagnosticsMixin
from longnav.utils.pruning_runtime import PruningMixin


class VLMWorker(ContextCacheMixin, DiagnosticsMixin, PruningMixin):
    def __init__(
            self,
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            attn_impl='sdpa',
            dtype='float16',
            prefix='<|im_start|>assistant\n**',
            postfix='**<|im_end|>',
            vocab=["stop","forward","left","right","up","down"],
            load_model=True,
            offload_cache=False,
            use_sparse=False,
            sparse_threshold=0.95,
            bev_canvas_size=2000,
            save_pixels=False,
            visualize_attention=False,
            visualize_attention_heads=False,
            visualize_attention_3d=False,
            attn3d_layers=None,
            context_window=None,
            context_frames=None,
            context_cut="header",
            context_window_mode="evict",
            attn_weighting="raw",
            kv_budget=None,
            kv_prune_recent_turns=2,
            kv_prune_ema_beta=0.7,
            kv_prune_importance="attn",
            kv_prune_candidate_scope="all",
            kv_prune_seed=17,
            kv_prune_log_influence=False,
            kv_prune_influence_stride=1,
            kv_prune_influence_horizons=(4, 8, 16, 32),
            kv_prune_influence_measures=("leave_one",),
            influence_recompute=False,
            kv_prune_layer_start=0,
            kv_prune_layer_end=None,
            kv_prune_score_layers="pruned",
            kv_prune_score_layer=None,
            kv_prune_log_layer_influence=False,
            kv_prune_layer_influence_starts=(),
            kv_prune_visual_blind=False,
            kv_prune_log_keep_one=False,
            kv_prune_budget_frames=None,
            kv_prune_stages=None,
            kv_prune_stage_importance=None,
            kv_prune_tail_budget_frames=None,
            kv_prune_budget_scope="last_stage",
            kv_prune_schedule_anchor="original",
            kv_prune_dart_pivots=8,
            kv_prune_seed_per_step=False,
            kv_prune_log_masks=False,
            token_info_probe=False,
            horizon_probe_layers=(),
            selector_probe=False,
            selector_probe_budgets=(4, 8, 16, 32),
            selector_probe_measures=("keep_cur", "leave")):
        if kv_prune_importance not in KV_PRUNE_IMPORTANCE_MODES:
            raise ValueError(f"kv_prune_importance must be one of {KV_PRUNE_IMPORTANCE_MODES}, "
                             f"got {kv_prune_importance!r}")
        import transformers.modeling_flash_attention_utils as fa_utils
        def patched(position_ids, batch_size):
            return False
        fa_utils._is_packed_sequence = patched
        import torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vocab = vocab
        self.vocab_ids = self._vocab_to_ids(vocab)
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
        if context_frames is not None:
            # context_frames counts the frames visible at decision time, including the
            # current one; the eviction schedule below is written in past turns retained.
            if context_window is not None:
                raise ValueError("set context_frames or context_window, not both "
                                 f"(got context_frames={context_frames}, context_window={context_window}).")
            if context_frames < 1:
                raise ValueError(f"context_frames must be >= 1 (1 = current frame only), got {context_frames}.")
            context_window = context_frames - 1
        if context_window is not None and context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {context_window}.")
        self.context_window = context_window
        if context_window_mode not in CONTEXT_WINDOW_MODES:
            raise ValueError(f"context_window_mode must be one of {CONTEXT_WINDOW_MODES}, got {context_window_mode!r}")
        self.context_window_mode = context_window_mode
        if context_cut not in CONTEXT_CUTS:
            raise ValueError(f"context_cut must be one of {CONTEXT_CUTS}, got {context_cut!r}")
        self.context_cut = context_cut
        if context_window_mode == "recompute":
            if context_window is None:
                # Nothing is ever evicted at full context, so the rebuild would never fire
                # and the run would silently be an ordinary full-context eval.
                raise ValueError("context_window_mode='recompute' requires context_window to be set.")
            if not use_sparse:
                # _record_window_turn replays the embeds the sparse subclass exposes on its
                # output (inputs_embeds, position_ids, deepstack); the stock model has none.
                raise NotImplementedError("context_window_mode='recompute' requires use_sparse=True.")
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
        self.kv_budget = kv_budget
        self.kv_prune_recent_turns = kv_prune_recent_turns
        self.kv_prune_ema_beta = kv_prune_ema_beta
        self.kv_prune_importance = kv_prune_importance
        self.kv_prune_candidate_scope = kv_prune_candidate_scope
        self.kv_prune_seed = int(kv_prune_seed)
        # paper/token_prune: frame-relative budget, PyramidDrop stages, DART pivots, per-step
        # seeding and the per-frame keep-mask log (see _apply_kv_prune / vtp_log).
        self.kv_prune_budget_frames = None if kv_prune_budget_frames is None else float(kv_prune_budget_frames)
        self.kv_prune_stages = None if kv_prune_stages is None else [int(l) for l in kv_prune_stages]
        # paper/hybrid_prune: per-stage selector and the last stage's own frame-relative budget.
        self.kv_prune_stage_importance = None if kv_prune_stage_importance is None \
            else [str(m) for m in kv_prune_stage_importance]
        self.kv_prune_tail_budget_frames = None if kv_prune_tail_budget_frames is None \
            else float(kv_prune_tail_budget_frames)
        self.kv_prune_budget_scope = str(kv_prune_budget_scope)
        self.kv_prune_schedule_anchor = str(kv_prune_schedule_anchor)
        self.kv_prune_dart_pivots = int(kv_prune_dart_pivots)
        self.kv_prune_seed_per_step = bool(kv_prune_seed_per_step)
        self.kv_prune_log_masks = bool(kv_prune_log_masks)
        self.kv_prune_log_influence = bool(kv_prune_log_influence)
        self.kv_prune_influence_stride = int(kv_prune_influence_stride)
        self.kv_prune_influence_horizons = tuple(sorted({int(k) for k in (kv_prune_influence_horizons or ())}))
        self._kl_influence_row = []  # KL(P_t || P_t^{-f}) per cached frame f, when logging influence
        self._kl_influence_logp = []  # per cached frame f: action log-probs with f's visual slots hidden
        self._kl_influence_base_logp = []  # action log-probs of the replayed full-context decision
        self._kl_influence_history_all = []  # [kl, logp]: every history frame's visual slots hidden
        self._kl_influence_visual_all = []  # [kl, logp]: every visual slot hidden (text-only decision)
        self._kl_influence_horizon = []  # [[K, kl, logp], ...]: frames of age >= K hidden
        self._kl_influence_keep_one = []  # keep-one-in: KL(P only f's visual slots || P no visual) per frame f
        self._kl_influence_keep_one_logp = []  # per cached frame f: action log-probs with only f's visual slots
        self.kv_prune_influence_measures = tuple(str(m) for m in (kv_prune_influence_measures or ()))
        if not self.kv_prune_influence_measures or any(
                m not in ("leave_one", "keep_one") for m in self.kv_prune_influence_measures):
            raise ValueError("kv_prune_influence_measures must be a non-empty subset of "
                             f"{{'leave_one', 'keep_one'}}, got {self.kv_prune_influence_measures}")
        self._prune_visual_new = 0  # visual slots this turn added (0 => its KL row entry is vacuous)
        # Layer-selective pruning: the budget applies to decoder layers [start, end) only.
        self.kv_prune_layer_start = int(kv_prune_layer_start)
        self.kv_prune_layer_end = None if kv_prune_layer_end is None else int(kv_prune_layer_end)
        self.kv_prune_score_layers = kv_prune_score_layers
        self.kv_prune_score_layer = None if kv_prune_score_layer is None else int(kv_prune_score_layer)
        self.kv_prune_log_layer_influence = bool(kv_prune_log_layer_influence)
        self.kv_prune_layer_influence_starts = tuple(
            int(i) for i in (kv_prune_layer_influence_starts or ()))
        self.kv_prune_visual_blind = bool(kv_prune_visual_blind)
        # paper/hybrid_prune 'uniform' arm: visual-blind together with kv_prune_stages makes the LAST
        # stage blind -- its layers hide every visual column (history and the current frame) during
        # the forward and hold no visual slot afterwards (an implicit tail budget of 0).
        self._blind_stage = self.kv_prune_visual_blind and self.kv_prune_stages is not None
        self.kv_prune_log_keep_one = bool(kv_prune_log_keep_one)
        self._keep_one_rows = []  # per probe layer: keep-one-in KL per cached frame
        self._leave_one_rows = []  # per probe layer: leave-one-out KL per cached frame
        self._layer_influence_hist = []  # KL per probe layer, history visual slots hidden
        self._layer_influence_all = []  # KL per probe layer, every visual slot hidden
        self._kv_prune_state = None
        if self.kv_prune_log_influence and context_window_mode != "prune":
            raise ValueError("kv_prune_log_influence requires context_window_mode='prune' "
                             "(the per-slot turn/visual bookkeeping lives in KVPruneState)")
        if any(k < 2 for k in self.kv_prune_influence_horizons):
            # K frames survive including the current one; K=1 is the history_all probe.
            raise ValueError("kv_prune_influence_horizons must all be >= 2")
        # Leak-free leave-one-frame-out probes: every probe re-prefills prefix + all turns with the
        # removed frame's visual tokens deleted (see _recompute_influence_probe).
        self.influence_recompute = bool(influence_recompute)
        self._influence_recs = []  # device copies of the stored turn records (prefix first)
        self._influence_probe_count = 0
        self._influence_probe_latency = 0.0
        if self.influence_recompute:
            if not use_sparse:
                raise NotImplementedError("influence_recompute replays the embeds the sparse subclass "
                                          "exposes on its output; it requires use_sparse=True.")
            if context_window is not None or context_frames is not None or context_window_mode != "evict":
                raise ValueError("influence_recompute probes the full episode context; use "
                                 "context_window=None with context_window_mode='evict'.")
            if self._any_attn_viz():
                raise NotImplementedError("influence_recompute does not combine with attention visualization.")
            if self.kv_prune_influence_stride < 1:
                raise ValueError("kv_prune_influence_stride must be at least 1")
        # paper/layer_token_info: keep what the per-visual-token layer probes need without
        # running any per-step probe -- the decision-token inputs (prune mode, leaky replay arm)
        # and the stored turn records (evict mode, leak-free re-prefill arm). The probes
        # themselves live in longnav.utils.token_info and are called by the driver.
        self.token_info_probe = bool(token_info_probe)
        if self.token_info_probe:
            if not use_sparse:
                raise NotImplementedError("token_info_probe requires use_sparse=True (visual slot "
                                          "bookkeeping and stored post-filter embeds).")
            if context_window is not None or context_frames is not None:
                raise ValueError("token_info_probe probes the full episode context; use "
                                 "context_window=None / context_frames=None.")
            if context_window_mode not in ("prune", "evict"):
                raise ValueError("token_info_probe runs on context_window_mode='prune' (leaky "
                                 "decision-replay arm) or 'evict' (leak-free re-prefill arm), "
                                 f"got {context_window_mode!r}")
            if self._any_attn_viz():
                raise NotImplementedError("token_info_probe does not combine with attention visualization.")
            if attn_impl != "sdpa":
                raise NotImplementedError("token_info_probe's per-layer column hiding builds bool sdpa "
                                          "masks; use attn_impl='sdpa'.")
        # paper/info_horizon: after every decision, the current frame's frame-level keep-one-in /
        # leave-one-out KL with its visual tokens hidden from each listed decoder layer on --
        # leak-free re-prefill in evict mode, leaky decision replay in prune mode (see
        # longnav.utils.token_info.horizon_probe). Rides on the token_info_probe substrate.
        self.horizon_probe_layers = tuple(sorted({int(l) for l in (horizon_probe_layers or ())}))
        if self.horizon_probe_layers:
            if not self.token_info_probe:
                raise ValueError("horizon_probe_layers requires token_info_probe=True")
            if any(l < 0 for l in self.horizon_probe_layers):
                raise ValueError("horizon_probe_layers must be non-negative")
        self._horizon_row = {}
        self._horizon_probe_count = 0
        self._horizon_probe_latency = 0.0
        # paper/selector_kl: after every decision, the flat-variant selectors' layer-0 criteria
        # for each visual token of the current frame next to its leaky keep-one-in / leave-one-out
        # KL (longnav.utils.selector_probe). Rides on the prune substrate of token_info_probe with
        # nothing ever pruned; the probe arms the attention-row and DART hidden captures itself.
        self.selector_probe = bool(selector_probe)
        self.selector_probe_budgets = tuple(sorted({int(k) for k in (selector_probe_budgets or ())}))
        self.selector_probe_measures = tuple(str(m) for m in (selector_probe_measures or ()))
        self._selector_row = {}
        self._selector_probe_count = 0
        self._selector_probe_latency = 0.0
        if self.selector_probe:
            from longnav.utils.token_info import ALL_MEASURES
            if not self.token_info_probe:
                raise ValueError("selector_probe requires token_info_probe=True")
            if context_window_mode != "prune":
                raise ValueError("selector_probe reads the live prune-substrate cache; use "
                                 "context_window_mode='prune'")
            if kv_prune_candidate_scope != "visual" or int(kv_prune_recent_turns) != 0:
                raise ValueError("selector_probe ranks the flat variant's pool: every visual slot, the "
                                 "current frame included (kv_prune_candidate_scope='visual', "
                                 "kv_prune_recent_turns=0)")
            if self.kv_prune_layer_start != 0 or self.kv_prune_layer_end is not None \
                    or kv_prune_stages is not None or kv_prune_visual_blind:
                raise ValueError("selector_probe scores at decoder layer 0 with every layer in play: "
                                 "no layer-selective pruning, stages or visual blinding")
            if kv_prune_budget_frames is not None:
                raise ValueError("selector_probe never prunes (the pool must stay whole); drop "
                                 "kv_prune_budget_frames and keep kv_budget unreachable")
            if not self.selector_probe_budgets or any(k < 1 for k in self.selector_probe_budgets):
                raise ValueError(f"selector_probe_budgets must be positive frames' worth, got "
                                 f"{selector_probe_budgets}")
            bad = [m for m in self.selector_probe_measures if m not in ALL_MEASURES]
            if bad or not self.selector_probe_measures:
                raise ValueError(f"selector_probe_measures must be a non-empty subset of {ALL_MEASURES}, "
                                 f"got {self.selector_probe_measures}")
            if self.kv_prune_score_layer not in (None, 0):
                raise ValueError("selector_probe reads the layer-0 decision row (the flat variant's "
                                 f"FastV); kv_prune_score_layer={self.kv_prune_score_layer} conflicts")
            self.kv_prune_score_layer = 0  # FastV at layer 0: fold that one row only
        if (self.kv_prune_log_layer_influence or self.kv_prune_log_keep_one) \
                and context_window_mode != "prune":
            raise ValueError("kv_prune_log_layer_influence / kv_prune_log_keep_one require "
                             "context_window_mode='prune' (the per-layer replay path lives on the "
                             "pre-rotation cache)")
        if context_window_mode == "prune":
            if kv_budget is not None and self.kv_prune_budget_frames is not None:
                raise ValueError("set either kv_budget (total cache slots) or kv_prune_budget_frames "
                                 "(candidate slots worth N history frames), not both")
            if self.kv_prune_budget_frames is not None:
                if self.kv_prune_budget_frames <= 0:
                    raise ValueError("kv_prune_budget_frames must be positive")
                # Without a window the budget is relative to every history frame of the episode
                # (paper/token_prune flat variant): k x the running mean tokens per frame.
            if kv_budget is None and self.kv_prune_budget_frames is None and not self.kv_prune_visual_blind:
                # Nothing is ever budget-pruned: the identity arm of paper/token_prune, which
                # must reproduce the plain window (or, without one, the full context) bit-for-bit.
                print("[prune] no kv_budget / kv_prune_budget_frames: identity arm, only the window prunes.")
            if self.kv_prune_stages is not None:
                stages = self.kv_prune_stages
                if stages != sorted(stages) or len(set(stages)) != len(stages) or stages[0] < 0:
                    raise ValueError(f"kv_prune_stages must be strictly increasing layer indices >= 0, got {stages}")
                if self.kv_prune_budget_scope not in ("last_stage", "model"):
                    raise ValueError(f"kv_prune_budget_scope must be 'last_stage' or 'model', "
                                     f"got {self.kv_prune_budget_scope!r}")
                if self.kv_prune_schedule_anchor not in ("original", "pool"):
                    raise ValueError(f"kv_prune_schedule_anchor must be 'original' or 'pool', "
                                     f"got {self.kv_prune_schedule_anchor!r}")
                if self.kv_prune_budget_scope == "model" and self.kv_prune_budget_frames is None:
                    raise ValueError("kv_prune_budget_scope='model' bounds the whole-model holding by "
                                     "kv_prune_budget_frames; set it")
                if self.kv_prune_layer_start != 0 or self.kv_prune_layer_end is not None:
                    raise ValueError("kv_prune_stages fixes the pruned layers itself; leave "
                                     "kv_prune_layer_start / kv_prune_layer_end at their defaults")
                if kv_prune_importance != "attn":
                    raise ValueError("kv_prune_stages (PyramidDrop) requires kv_prune_importance='attn', "
                                     "with attention-ranked or random stages")
                if self._blind_stage:
                    if len(stages) < 2:
                        raise ValueError("kv_prune_visual_blind with kv_prune_stages blinds the LAST stage; "
                                         "it needs at least two stages")
                    if self.kv_prune_tail_budget_frames is not None:
                        raise ValueError("kv_prune_visual_blind with kv_prune_stages gives the last stage an "
                                         "implicit budget of 0; leave kv_prune_tail_budget_frames unset")
                sel = self.kv_prune_stage_importance
                if sel is not None and (len(sel) != len(stages) or any(m not in ("attn", "random") for m in sel)):
                    raise ValueError(f"kv_prune_stage_importance needs one of 'attn'/'random' per stage "
                                     f"({len(stages)} stages), got {sel}")
                tail = self.kv_prune_tail_budget_frames
                if tail is not None:
                    if len(stages) < 2 or self.kv_prune_budget_frames is None:
                        raise ValueError("kv_prune_tail_budget_frames sets the last stage's budget below the "
                                         "schedule's: it needs at least two stages and kv_prune_budget_frames")
                    if not 0 <= tail <= self.kv_prune_budget_frames:
                        raise ValueError(f"kv_prune_tail_budget_frames must lie in [0, kv_prune_budget_frames], "
                                         f"got {tail}")
            elif self.kv_prune_stage_importance is not None or self.kv_prune_tail_budget_frames is not None \
                    or self.kv_prune_budget_scope != "last_stage" or self.kv_prune_schedule_anchor != "original":
                raise ValueError("kv_prune_stage_importance / kv_prune_tail_budget_frames / kv_prune_budget_scope / "
                                 "kv_prune_schedule_anchor describe the stages of kv_prune_stages; set it")
            if kv_prune_importance == "dart":
                # layer_start=0 (every layer pruned) reads layer 0: its keys and its output.
                if self.kv_prune_stages is not None:
                    raise ValueError("kv_prune_importance='dart' reads keys and hidden states at "
                                     "kv_prune_layer_start-1; it does not combine with kv_prune_stages")
                if self.kv_prune_dart_pivots < 1:
                    raise ValueError("kv_prune_dart_pivots must be at least 1")
            if kv_prune_candidate_scope not in ("all", "visual"):
                raise ValueError("kv_prune_candidate_scope must be 'all' or 'visual', "
                                 f"got {kv_prune_candidate_scope!r}")
            if self.kv_prune_influence_stride < 1:
                raise ValueError("kv_prune_influence_stride must be at least 1")
            layer_selective = self.kv_prune_layer_start > 0 or self.kv_prune_layer_end is not None
            if self.kv_prune_layer_start < 0:
                raise ValueError("kv_prune_layer_start must be nonnegative")
            if self.kv_prune_layer_end is not None and self.kv_prune_layer_end < self.kv_prune_layer_start:
                raise ValueError("kv_prune_layer_end must not precede kv_prune_layer_start")
            if kv_prune_score_layers not in KV_PRUNE_SCORE_LAYER_MODES:
                raise ValueError(f"kv_prune_score_layers must be one of {KV_PRUNE_SCORE_LAYER_MODES}, "
                                 f"got {kv_prune_score_layers!r}")
            if kv_prune_score_layers == "boundary" and self.kv_prune_layer_start < 1:
                raise ValueError("kv_prune_score_layers='boundary' reads layer_start-1, so "
                                 "kv_prune_layer_start must be at least 1")
            if self.kv_prune_score_layer is not None:
                if self.kv_prune_score_layer < 0:
                    raise ValueError("kv_prune_score_layer must be nonnegative")
                if (kv_prune_importance != "attn" and not self.selector_probe) or self.kv_prune_stages is not None:
                    raise ValueError("kv_prune_score_layer picks the one decoder layer whose decision row "
                                     "ranks the slots; it requires kv_prune_importance='attn' and no "
                                     "kv_prune_stages")
            if (self.kv_prune_log_layer_influence or self.kv_prune_log_keep_one) \
                    and not self.kv_prune_layer_influence_starts:
                raise ValueError("the per-layer diagnostics need at least one probe layer in "
                                 "kv_prune_layer_influence_starts")
            if attn_impl == "flash_attention_2" and (
                    self.kv_prune_log_influence
                    or self.kv_prune_log_layer_influence or self.kv_prune_log_keep_one):
                raise NotImplementedError(
                    "kl scoring and the influence diagnostics replay the decision with a 4-D "
                    "additive attention bias, which the flash_attention_2 integration misreads "
                    "as a 2-D padding mask; use attn_impl='sdpa' or 'eager'.")
            if self._any_attn_viz():
                # Same as reindex: cached keys are stored pre-rotation, and arbitrary-slot
                # drops cannot be shift-corrected the way AttentionProbe.evict does.
                raise NotImplementedError("context_window_mode='prune' does not support attention "
                                          "visualization; cached keys are stored pre-rotation and "
                                          "slots are dropped non-contiguously.")
            if not use_sparse:
                # Needs the pre-rope hook in TextMixin.forward AND the per-chunk
                # visual_pos_masks / embed DB that only the sparse subclass maintains.
                raise NotImplementedError("context_window_mode='prune' requires use_sparse=True.")
            if offload_cache:
                # The prune step reassigns .layers[i].keys/.values directly, which an
                # offloaded cache would silently undo.
                raise ValueError("context_window_mode='prune' does not support offload_cache=True.")
            from longnav.utils.kv_prune import KVPruneState
            self._kv_prune_state = KVPruneState(seed=self.kv_prune_seed)
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
        self.cumulative_inputs = None
        self.seq_keep_mask = None
        self.vis_keep_masks = []
        self._frame_records = []
        self._decision_inputs = None
        self._kl_influence_row = []
        self._kl_influence_logp = []
        self._kl_influence_base_logp = []
        self._kl_influence_history_all = []
        self._kl_influence_visual_all = []
        self._kl_influence_horizon = []
        self._kl_influence_keep_one = []
        self._kl_influence_keep_one_logp = []
        self._influence_recs = []
        self._influence_probe_count = 0
        self._influence_probe_latency = 0.0
        self._prune_visual_new = 0
        self._layer_influence_hist = []
        self._layer_influence_all = []
        self._keep_one_rows = []
        self._leave_one_rows = []
        self._prune_score_latency = 0.0
        self._prune_replay_count = 0
        self._prune_visual_slots = 0
        self._prune_text_slots = 0
        self._prune_stage_visual = []
        self._prune_visual_model = self._prune_visual_model_frames = None
        self._vtp_log = {}
        self._horizon_row = {}
        self._horizon_probe_count = 0
        self._horizon_probe_latency = 0.0
        self._selector_row = {}
        self._selector_probe_count = 0
        self._selector_probe_latency = 0.0
        self._episode_label = getattr(self, "_episode_label", "")
        self._hidden_db = None  # DART: (n_visual_master, hidden) states entering the cut layer
        self._hidden_capture_armed = False
        if self.attn_probe is not None:
            self.attn_probe.reset()
        if getattr(self, "_reindex_state", None) is not None:
            self._reindex_state.reset()
        if getattr(self, "_kv_prune_state", None) is not None:
            self._kv_prune_state.reset()
        self.past_image_embeds = None #per batch list of image embed tensors of the form N_patch by N_hidden
        self.logit_indices = []
        # Sliding context window bookkeeping (see _apply_context_window).
        self._abs_bounds = []  # cumulative cache length after each turn, as if nothing were evicted
        self._vis_counts = []  # visual patches kept per turn, for trimming the sparse embed db
        self._dropped = 0  # total tokens evicted so far
        self._n_evicted = 0  # number of turns fully evicted so far
        self._prefix_len = None  # pinned prompt tokens (goal + action space), never evicted
        # context_cut='frame' bookkeeping: per turn, how many tokens precede that turn's
        # '<|im_start|>user' opener in its chunk (the previous action + postfix; 0 for turn 0),
        # and whether the next incoming chunk must lose that lead before it is forwarded
        # (only when the window is a single frame, so the lead is never cached at all).
        self._turn_leads = []
        self._strip_next_lead = False
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
        if self.context_window_mode in ("reindex", "prune"):
            from longnav.utils.pre_rope import install_pre_rope
            n_layers = len(self.language_model.layers)
            pruned_layers = self._pruned_layer_range(n_layers)
            stages = self._stage_ranges(n_layers)
            blind_layers = None
            if self.context_window_mode == "prune" and self._blind_stage:
                blind_layers = stages[-1]  # the last stage's layers see no visual token
            elif self.context_window_mode == "prune" and self.kv_prune_visual_blind:
                end = n_layers if self.kv_prune_layer_end is None else self.kv_prune_layer_end
                blind_layers = range(self.kv_prune_layer_start, end)
            score_layers = self._score_layer_range(n_layers, pruned_layers, stages)
            self._reindex_state = install_pre_rope(
                self.language_model, pruned_layers=pruned_layers, score_layers=score_layers,
                blind_layers=blind_layers, stages=stages)
            if self.context_window_mode == "prune" and (self.kv_prune_importance == "dart"
                                                        or self.selector_probe):
                # The hidden states leaving the scored layer: layer_start-1, or layer 0 when
                # every layer is pruned (the selector probe reads DART's layer-0 criterion).
                self._install_hidden_capture(max(self.kv_prune_layer_start, 1))
        if self._any_attn_viz():
            self._attach_attention_probe()

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

        needs_grad_cache = self._grad_attribution_enabled()
        return torch.no_grad() if needs_grad_cache else torch.inference_mode()

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
        if not summarize_images:
            return self.processor.batch_decode(self.cumulative_inputs['input_ids'])
        else:
            # image_mask = self.cumulative_inputs['input_ids'] == self.processor.image_token_id
            sequences = [torch.unique_consecutive(sequence,return_counts=True) for sequence in self.cumulative_inputs['input_ids'].numpy()]
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

    


    def _master_len(self):
        """Slots in the master cache (every slot held by at least one layer)."""
        st = getattr(self, "_reindex_state", None)
        if st is not None and st.pos_table is not None:
            return st.master_len()
        return self.past_key_values.get_seq_length() if self.past_key_values is not None else 0

    def _kv_lengths(self):
        """Per-layer resident cache lengths (they differ under layer-selective pruning)."""
        if self.past_key_values is None:
            return []
        return [int(layer.keys.shape[-2]) for layer in self.past_key_values.layers]

    def kv_stats(self):
        """(layer-mean, layer-max, master) cache lengths, or None before the first forward.

        The layer mean is the memory-equivalent length: the cache's byte size is
        proportional to the sum over layers, so it equals the uniform kv_len when every
        layer holds the same slots and lies between the pruned and master lengths otherwise.
        """
        lengths = self._kv_lengths()
        if not lengths:
            return None
        return float(np.mean(lengths)), int(max(lengths)), int(self._master_len())


    def begin_episode(self, label):
        """Name the episode (per-step seeding and the mask log key on it)."""
        self._episode_label = "" if label is None else str(label)

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

        if (self.context_window is not None or self.context_window_mode == "prune"
                or self.influence_recompute or self.token_info_probe) and self._prefix_len is None:
            self._prefix_len = self._find_prompt_prefix_len(turn_inputs['input_ids'])
        if self.context_window is not None and self.context_cut == "frame":
            lead = self._chunk_lead(turn_inputs['input_ids'])
            if self._strip_next_lead:
                # The previous step evicted everything but the prefix; this chunk's leading
                # action + postfix belong to that evicted turn, so they never enter the cache.
                for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
                    if key in turn_inputs and turn_inputs[key] is not None:
                        turn_inputs[key] = turn_inputs[key][:, lead:]
                self._strip_next_lead = False
                lead = 0
            self._turn_leads.append(lead)

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
            turn_inputs['past_image_embeds'] = self._windowed_sparse_history()
            turn_inputs['save_image_db'] = True # new argument in sparse qwen to signal keeping the db as internal state
            # sparsify the input attention mask
            turn_inputs['attention_mask'] = None#turn_inputs['attention_mask'] = torch.ones((turn_inputs['input_ids'].shape[0], (self.past_key_values.get_seq_length() if self.past_key_values is not None else 0) + turn_inputs['input_ids'].shape[1]), device=self.device, dtype=turn_inputs['attention_mask'].dtype)# torch.ones(1,seql,device=self.device)
        else:
            # The cumulative mask spans the whole episode; the cache may have been cropped
            # by the context window, so take only as much as the model will actually attend to.
            past_len = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
            turn_inputs['attention_mask'] = self.cumulative_inputs['attention_mask'][:,-(past_len+current_len):].to(self.device)
        if (self._recomputing_window() or self.influence_recompute
                or self.token_info_probe):
            turn_inputs['save_embeds'] = True

        with self._inference_ctx():
            # Arm the attention probe only for this forward. Other forwards (the value
            # head, training) reuse the same modules and would otherwise clobber the
            # captured rows; they also carry no KV cache, which the hook checks too.
            if self.attn_probe is not None:
                self.attn_probe.enabled = True
            # Arm the decision-row attention capture for this forward only; the score row
            # feeds the prune step's importance EMA (see _apply_kv_prune).
            if (self._kv_prune_state is not None and (not self.kv_prune_visual_blind or self._blind_stage)
                    and (self.kv_prune_importance == "attn" or self.selector_probe)):
                self._reindex_state.arm_capture()
            self._hidden_capture_armed = self.kv_prune_importance == "dart" or self.selector_probe
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
                if self._kv_prune_state is not None:
                    self._reindex_state.capture_scores = False
                self._hidden_capture_armed = False
            self.past_key_values = outputs['past_key_values']
             # Compute logprobs directly (1-to-1 mapping)
            relevant_logits = outputs.logits[0].float()
            if not full_logprobs:
                relevant_logits = relevant_logits[...,self.vocab_ids]
            if np.abs(temperature-1.0) > 1e-7:
                logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)
            else:
                logprobs = torch.log_softmax(relevant_logits, dim=-1)
            # print(f"vlm latency: {time.time()-t}",end=" ")

            if self._recomputing_window() or self.influence_recompute or self.token_info_probe:
                self._record_window_turn(outputs)
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
        if (self.visualize_attention_3d or
                self.kv_prune_log_influence or self.kv_prune_log_layer_influence
                or self.kv_prune_log_keep_one or self.token_info_probe):
            self._decision_inputs = self._slice_decision_token(turn_inputs)
        if self._grad_attribution_enabled():
            self._grad_attribution_pass()
        if self.influence_recompute:
            self._recompute_influence_probe()
        if self.context_window_mode == "prune":
            self._apply_kv_prune()
        else:
            self._apply_context_window()
        if self.horizon_probe_layers:
            # After the (no-op) prune / window pass: the prune substrate registers the new turn's
            # slots there, and at full context nothing is evicted, so both arms see the whole cache.
            self._horizon_probe()
        if self.selector_probe:
            # Same placement as the horizon probe: the new turn's slots are registered in the
            # prune state and, at an unreachable budget, nothing has left the cache.
            self._selector_probe()
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

    def merge_adapter(self):
        print("Merging LoRA adapters for inference...")
        self._is_merged = True
        self.model.merge_adapter()
        # o_proj is a LoRA target, so any cached W_O factors are now stale.
        if self.attn_probe is not None:
            self.attn_probe.invalidate_factors()

    def unmerge_adapter(self):
        print("Unmerging LoRA adapter")
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
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

    


 
