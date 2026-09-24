from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# --- 1. Resource & Environment Config ---
@dataclass
class ResourceConfig:
    ray_address: str = "local"
    object_spilling_directory: str = "./ray_object_spilling"
    osm_gb: int = 128  # Object Store Memory in GB
    vlm_resource_tag: str = "env_a"
    sim_resource_tag: str = "env_b"
    master_addr: str = 'localhost'
    master_port: Optional[int] = None #port for accelerate/ddp
    num_vlms: int = 1
    num_sims: int = 1
    vlm_conda_env: Optional[str] = "longnav_vlm"
    habitat_conda_env: Optional[str] = "vln"
    vlm_gpu_fraction: float = 0.7
    sim_gpu_fraction: float = 0.14
    vlm_cpus: int = 4
    sim_cpus: int = 4

# --- 3. Model & Worker Configs ---
@dataclass
class VLMConfig:
    model_id: Optional[str] = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
    attn_impl: str = "sdpa"
    dtype: str = "bfloat16"
    prefix: str = '<|im_start|>assistant\n**'
    postfix: str = '**<|im_end|>\n'
    vocab: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])
    offload_cache: bool = False
    use_sparse: bool = True
    sparse_threshold: float = 0.95  # cosine sim cutoff for visual token filtering (keep if sim < threshold)
    save_outputs: bool = False # only need this for RL
    # Sliding context window. Two ways to size it; set at most one:
    #   context_frames=K  -- K observation frames are visible when the action is read (the
    #                        current frame plus the K-1 before it). K=1 is the current frame
    #                        alone. This is the paper-facing knob.
    #   context_window=N  -- legacy: N past turns survive each step's eviction, so N+1 frames
    #                        are visible at decision time (context_frames = N + 1).
    # None on both = full episode context.
    context_frames: Optional[int] = None
    context_window: Optional[int] = None
    # Where the cut lands inside the boundary turn. Turn chunks begin one token inside the
    # previous reply (the action token), so a cut at the raw chunk boundary would open the
    # window on a bare action with no assistant header. "header" (legacy) holds back the
    # previous turn's '<|im_start|>assistant\n**' so the window opens on `header + action`
    # of the oldest evicted turn. "frame" instead drops that action and its postfix too, so
    # the window opens on the oldest retained '<|im_start|>user' turn: exactly K frames and
    # the K-1 actions between them, nothing from before. mode='prune' supports it at slot
    # granularity only.
    context_cut: str = "header"
    # How the retained window is produced. "evict" slices the older turns' K/V out of the
    # cache: cheap, but the surviving keys and values were computed while attending over the
    # whole episode, so evicted frames still reach the decision through them. "recompute"
    # rebuilds the window's K/V against a cache holding only the pinned prefix plus the
    # window, making the decision a strict function of what the agent can still see. The
    # difference between the two is the information leaked by eviction. "reindex" evicts on
    # the same schedule as "evict" but caches keys pre-rotation and applies RoPE at
    # attention time, renumbering survivors to contiguous mRoPE positions
    # (StreamingLLM-style; no positional hole across the cut). Requires use_sparse.
    # "prune" enforces a hard KV token budget (kv_budget) on top of the reindex substrate:
    # each step the lowest-importance unprotected slots are dropped, importance being an
    # EMA of the decision token's attention to each slot. context_window is optional here
    # and acts as the selection pool (older turns force-dropped on the evict schedule).
    context_window_mode: str = "evict"
    # --- context_window_mode='prune' knobs (see longnav.utils.kv_prune) ---
    kv_budget: Optional[int] = None  # hard cap on cache slots; required for mode='prune'
    kv_prune_recent_turns: int = 2  # last N turns are never budget-pruned
    kv_prune_ema_beta: float = 0.7  # importance EMA decay; 0 = last decision only (TOVA)
    kv_prune_merge: bool = False  # merge dropped visual slots into nearest kept slot (CaM-style)
    # 'attn' (decision-row attention EMA), 'random' (control), 'stratified' (equal per-turn),
    # 'diversity' (FPS on sparse embeds), 'kl' (leave-one-frame-out KL, frame-proportional),
    # 'keep_one' (keep-one-frame-in KL vs the no-history prior -- the CVPR'26 standalone
    # information measure, frame-proportional), 'fisher' / 'fisher_diversity' (gradient
    # Fisher), 'grid' (random's per-frame quotas, filled by farthest-point picks on the mRoPE
    # (row, col) token grid).
    kv_prune_importance: str = "attn"
    kv_prune_granularity: str = "slot"  # 'slot' (TOVA-style) or 'turn' (whole-frame keyframe selection)
    kv_prune_candidate_scope: str = "all"  # 'visual' protects every in-window text slot
    kv_prune_seed: int = 17  # reset per episode for scheduling-independent sampling
    kv_prune_fisher_pool_factor: float = 2.0  # Fisher candidates passed to hybrid FPS
    # --- 'voxel_dedup' / 'voxel_strat': need sim.voxel_kwargs (per-patch world voxels from
    # depth + pose), sim.output_schema.obs.patch_coords=true and rollout.pos_id_mode=standard.
    kv_prune_voxel_cap: int = 1  # voxel_dedup: max visual slots kept per world cell (newest first)
    kv_prune_voxel_scale: int = 2  # integer coarsening of the sim's voxel ids (0.15 m * scale per cell)
    kv_prune_voxel_2d: bool = True  # bucket on (x, z) only -- habitat y is up
    # --- leave-one-frame-out influence (paper/frame_influence): every
    # scored step, KL(P_t || P_t^{-f}) for EVERY cached frame f (one decision replay per frame,
    # only f's visual slots hidden), logged raw in sequence.json as sup/kl_influence together with
    # the hidden-frame action log-probs (sup/kl_influence_logp), the replayed base log-probs
    # (sup/kl_influence_base_logp) and the joint probes: every history frame hidden
    # (sup/kl_influence_history_all), every visual slot hidden (sup/kl_influence_visual_all) and
    # the age bands (sup/kl_influence_horizon: frames of age >= K hidden, one [K, kl, logp] per K
    # in kv_prune_influence_horizons; K counts surviving frames incl. the current one, so K=1 is
    # history_all and every K must be >= 2). Needs prune mode; run it with an unreachable
    # kv_budget and context_window=null so the full cache is probed (+experiment=paper_influence).
    kv_prune_log_influence: bool = False
    kv_prune_influence_stride: int = 1  # score every k-th step; skipped steps log []
    kv_prune_influence_horizons: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    # Which per-frame measures the influence probes log (subset of leave_one, keep_one; applies to
    # kv_prune_log_influence and influence_recompute alike). leave_one: the rows above.
    # keep_one (leave-one-IN): for EVERY cached frame f, KL(P with only f's visual slots visible ||
    # P with no visual slot visible) -- the text (prefix, goal, action history) stays, every other
    # frame's visual slots incl. the current frame's are hidden, and the reference is the
    # visual_all probe; expectation under the only-f distribution, as in the other keep-one
    # diagnostics (kv_prune_log_keep_one, horizon_probe_layers). Logged as sup/kl_influence_keep_one
    # (t+1 KLs) and sup/kl_influence_keep_one_logp (the only-f action log-probs). One replay per
    # frame per measure; the joint probes are logged whatever the measures.
    kv_prune_influence_measures: List[str] = field(default_factory=lambda: ["leave_one"])
    # Leak-free variant of the same probes (+experiment=paper_influence_recompute): instead of
    # hiding a frame's key columns from the replayed decision, every probe re-prefills prefix +
    # all turns from their stored post-filter embeddings with that frame's visual tokens deleted
    # (survivors keep their absolute mRoPE positions), so the decision is a strict function of
    # what remains. Same log keys, same stride/horizons. Full context only (context_window=None,
    # context_window_mode='evict'), requires use_sparse; one full prefill per probe, so the cost
    # per episode grows with T^3.
    influence_recompute: bool = False
    # paper/layer_token_info: retain the decision-token inputs (prune mode) / the stored turn
    # records (evict mode) so longnav.utils.token_info can run per-visual-token, per-layer
    # keep-one-in / leave-one-out probes on the current frame after a step. No per-step probe
    # runs. Full context only, requires use_sparse and attn_impl='sdpa'.
    token_info_probe: bool = False
    # paper/info_horizon: after every decision, the current frame's frame-level keep-one-in /
    # leave-one-out KL with its visual tokens hidden from each listed decoder layer on (a value
    # equal to the layer count is the "none" point, 0 by construction). Leak-free re-prefill in
    # evict mode, leaky decision replay in prune mode; 1 + 3 * len(layers) forwards per step.
    # Requires token_info_probe. Rows land raw in sequence.json as sup/horizon_*.
    horizon_probe_layers: List[int] = field(default_factory=list)
    # --- layer-selective pruning ("When Token Pruning is Worse than Random", CVPR'26, arXiv
    # 2512.07580): the budget applies only to decoder layers [layer_start, layer_end); the other
    # layers keep the window-only cache, so memory savings scale with the fraction of layers
    # pruned (sup/mean_kv_len reports the layer-mean length, kv_len_layer_max / kv_len_master
    # the unpruned length). Refuses kv_prune_granularity='turn' and kv_prune_merge.
    kv_prune_layer_start: int = 0  # first decoder layer whose cache is budget-pruned (0 = every layer)
    kv_prune_layer_end: Optional[int] = None  # exclusive end of the pruned range; None = through the last layer
    # 'attn' selector under layer-selective pruning: whose decision rows are averaged. 'pruned' =
    # the pruned range, 'all' = every layer, 'boundary' = layer_start-1 only (FastV-style).
    kv_prune_score_layers: str = "pruned"
    # 'attn' selector: rank by the decision row of this ONE decoder layer, whatever is pruned
    # (overrides kv_prune_score_layers). kv_prune_score_layer=0 with layer_start=0 is FastV scored
    # and applied at layer 0 (paper/token_prune flat variant).
    kv_prune_score_layer: Optional[int] = None
    # Visual-blind arm (hypothesis test for the information horizon): on decoder layers
    # [layer_start, layer_end) no query may attend to ANY visual slot -- cached history and the
    # current frame alike -- and visual slots are dropped from those layers' caches after each
    # step. Text and the frame window are untouched; kv_budget / importance are ignored.
    kv_prune_visual_blind: bool = False
    # --- layer-influence diagnostic (tools/layer_influence.py): per step, KL(P_t || P_t with the
    # visual slots hidden from layers >= i) for each probe layer i, once for history-only visual
    # slots (sup/layer_influence_hist) and once including the recent turns (sup/layer_influence_all).
    # One decision replay per probe and row. Needs prune mode; run with an unreachable kv_budget.
    kv_prune_log_layer_influence: bool = False
    kv_prune_layer_influence_starts: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28])
    # --- keep-one-in vs leave-one-out diagnostic (tools/keep_one_influence.py): per probe layer i
    # and EVERY cached frame f, KL(P only f || P no visual) and KL(P full || P without f), hidings
    # on layers >= i (sup/keep_one_influence, sup/leave_one_influence). 1 + 2 * n_frames replays per
    # probe layer per scored step -- use a window and kv_prune_influence_stride.
    kv_prune_log_keep_one: bool = False
    # --- paper/token_prune: visual-token pruning under a recency window (permanent deletion).
    # kv_prune_budget_frames=k keeps exactly k x (mean original post-filter visual count of the
    # window's frames) history visual slots each step -- "what k frames cost this model" -- in
    # place of the total-cache kv_budget; nothing is pruned while the window holds <= k frames.
    # Use with context_frames=32, context_cut='frame', kv_prune_candidate_scope='visual' and
    # kv_prune_recent_turns=0 (the just-decided frame joins the candidates). Methods:
    # random / divprune (DivPrune, input-level) / attn with layer_start=2 + score_layers='boundary'
    # (FastV) / dart with layer_start=2 (DART) / attn with kv_prune_stages=[7,14,21] (PyramidDrop:
    # nested budgets V0*lambda^s, each stage ranked at the layer below it).
    # Flat variant (+experiment=paper_prune_flat): no window at all -- the budget is then k x the
    # running mean over every history frame of the episode -- and every method at layer 0:
    # attn with kv_prune_score_layer=0, dart with layer_start=0.
    kv_prune_budget_frames: Optional[float] = None
    kv_prune_stages: Optional[List[int]] = None  # first layer of each PyramidDrop stage (ascending; 0 = every layer pruned)
    # --- paper/hybrid_prune: per-stage rule on top of kv_prune_stages. kv_prune_stage_importance
    # names each stage's selector ('attn' = ranked at the layer below the stage, 'random' = uniform
    # draw among the previous stage's survivors; None = every stage 'attn'). With
    # kv_prune_tail_budget_frames=j the LAST stage keeps j frames' worth on its own (from the
    # second frame on, whatever k is) and the geometric schedule spans the stages before it, ending
    # at kv_prune_budget_frames. Hybrid arm: stages=[7,14,20], [attn,attn,random], tail 1.
    kv_prune_stage_importance: Optional[List[str]] = None
    # What kv_prune_budget_frames bounds under kv_prune_stages: 'last_stage' (PyramidDrop's own
    # reading, Exp 4: the deepest scheduled stage holds k frames' worth) or 'model' (the history
    # visual slots AVERAGED OVER ALL DECODER LAYERS are k frames' worth -- lambda is solved each step,
    # model_budgets; the tail stage counts). 'model' normally needs a stage at layer 0
    # (kv_prune_stages=[0,...], ranked by the layer-0 row), or the unpruned layers alone exceed it.
    kv_prune_budget_scope: str = "last_stage"
    # What the geometric schedule shrinks FROM. 'original' (Exp 4): the original token count of every
    # frame in scope -- bounded under a window, but without one it grows all episode and the ratio
    # collapses (deep stages starve). 'pool': the slots the first stage can draw from NOW (its
    # survivors + the entering frame) -- "keep a fraction lambda of what the previous stage holds",
    # stationary and independent of episode length. paper/hybrid_prune uses 'pool' for both protocols.
    kv_prune_schedule_anchor: str = "original"
    kv_prune_tail_budget_frames: Optional[float] = None
    kv_prune_dart_pivots: int = 8  # DART: pivots by key L1 norm at layer_start-1 (layer 0 when layer_start=0)
    kv_prune_seed_per_step: bool = False  # 'random': reseed from (seed, episode label, step) each step
    kv_prune_log_masks: bool = False  # log sup/vtp_{frames,ntok,kept,bits,raster} per step (sequence.json)
    # What the attention visualizations measure. "raw" = attention weight alpha (max over
    # heads). "value_norm" = alpha*||v||, "wo_norm" = alpha*||W_O v|| (both summed over
    # heads): alpha only routes, so a key with high alpha and a small value vector
    # contributes nothing. Weighted modes also make heads commensurable.
    # "grad" = alpha*d(action score)/d(alpha), i.e. what actually drove the decision
    # rather than what was merely loud. It costs one extra single-token forward and
    # backward per step, requires visualize_attention_3d, and runs the decision forward
    # under no_grad instead of inference_mode so the cache can be differentiated.
    attn_weighting: str = "raw"

@dataclass 
class PolicyLossConfig:
    clip_cov_ratio: Optional[float] = 0.0002
    clip_cov_ub: Optional[float] = 5.0
    clip_cov_lb: Optional[float] = 1.0
    
@dataclass 
class RLAlgoConfig:
    # generic on policy params
    use_value: bool = False
    value_grad_scale: float = 0.1
    advantage_estimator: str = "reinforce_plus_plus"
    policy_loss_name: str = "vanilla"
    n_rollout: int = 12 # note: must be divisible by num vlms times gradient accumulation
    n_adv: int = 256 # number of trajectories for advantage estimation, must > n_rollout
    n_epoch: int = 2 # number of policy gradient epochs

    # PPO Hyperparameters
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    clip_ratio_c: float = 3.0
    loss_agg_mode: Optional[str] = "token-mean" #"seq-mean-token-sum" seq-mean-token-mean
    
    # clip cov parameters
    policy_loss:PolicyLossConfig = field(default_factory=PolicyLossConfig)
    # GAE Hyperparameters
    gamma: float = 0.99
    lam: float = 0.95

    time_kernel_sigma: float = 50.0
    time_alignment: str = "start" #or end
    time_loto: bool = False # leave one trajectory out
    
    distance_kernel_sigma: float = 0.5
    distance_clip_max: Optional[float] = 17.0
    distance_clip_percentile: Optional[float] = 0.95
    distance_pad_mode: Optional[str] = "replicate"
    distance_pad_val: Optional[float] = None
    # Value & Entropy
    cliprange_value: float = 0.2
    entropy_bonus: float = 0.0

    # Ref KL Control
    use_ref: bool = True
    kl_coeff: float = 0.001
    kl_target: float = 0.1

    # # Compatibility for verl's agg_loss
    # @property
    # def global_batch_info(self):
    #     # For single-worker testing, batch size is 1
    #     return {}# "dp_size": 1, "global_batch_size": 1
    global_batch_info: Optional[Dict[str,Any]] = field(default_factory=lambda:{})
    # Helper to support config.get("key", default) used in loss functions
    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class SFTConfig:
    pass

# --- training configs ---
@dataclass
class HydraLoraConfig:
    """
    A Hydra-compatible mirror of peft.LoraConfig.
    Removes Union types (like str | List[str]) that crash OmegaConf.
    """
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Enforce List[str] to satisfy Hydra. 
    # If you need regex (str), you can change this to Any, but List is safer.
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # modules_to_save is also a list, defaulting to None is fine for Hydra
    modules_to_save: Optional[List[str]] = None
    use_rslora: bool = False #rank stabilized lora. should use?

@dataclass
class VLMTrainingConfig:
    # checkpoints
    checkpoint:Optional[str] = None
    load_optim:bool = False # l
    load_sched:bool = False

    # Optimization
    learning_rate: float = 5e-6
    grad_accum_steps: int = 1
    mixed_precision: Optional[str] = "no" #['no', 'fp8', 'fp16', 'bf16']
    gradient_checkpointing: bool = True
    total_optimization_steps: int = 100000 # used for linear LR schedule
    warmup_steps: int = 64
    save_step: Optional[int] = 10

    # Value Head Configuration
    value_head_learning_rate: float = 5e-4  # Often higher than Adapter LR
    value_head_dropout: float = 0.0
    value_head_dtype: str = "float32"  
    # List of hidden layer sizes. Empty list [] implies a single linear layer (Linear Probe).
    value_head_hidden_dims: List[int] = field(default_factory=lambda:[1024,512])

    # PEFT: Pass the actual configuration object here (e.g., LoraConfig)
    # Typed as Any to avoid crashing if peft isn't installed on the driver
    peft_config: Optional[Any] = field(default_factory=HydraLoraConfig) 

    rl_config:Optional[RLAlgoConfig] = field(default_factory=RLAlgoConfig) # RL Algorithm 
    sft_config:Optional[SFTConfig] = None

# --- habitat sim configs ---
@dataclass
class HabitatConfig:
    config_path: str = "habitat_configs/objectnav_hm3d_rgbd_semantic.yaml"
    dataset_path: Optional[str] = None
    workspace: Optional[str] = "."
    scenes_dir: Optional[str] = None
    split: str = "val"
    fp_guard: bool = False
    fn_guard: bool = False
    # Per-patch world voxels from depth + pose (longnav.utils.voxel_utils.patch_voxels kwargs:
    # patch_size, resolution, fov_degrees). Empty = off. A dict default (not None) so Hydra can
    # add keys on the command line: `+sim.voxel_kwargs.patch_size=32 ...` (see
    # tools/run_kv_prune_ablation.sh VOXEL=true).
    voxel_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "obs": {"rgb": True, "instr_or_goal": True, "patch_coords": False},
        "info": {"episode_label": True, "spl": True, "soft_spl":True, "success": True,"distance_to_goal":True},
        "done": True,
        "reward": True,
        "stuck": True,
        "fp_stop": True
    })
    auto_flush: bool = False # automatically flush logs upon reset
    ep_seed: Optional[bool] = None # if set, episode iterators are deterministic with same set seed all habitat workers
    explr_bonus: Optional[float] = 0.13
    collision_penalty: Optional[float] = 0.05
    fpstop_penalty: Optional[float] = 0.3
    add_top_down_map:bool = False
    visualize_3d:bool = False # render accumulated 3D patch-filtering video (video_3d.mp4)
    visualize_attn3d:bool = False # render accumulated 3D attention-heat video (video_attn3d.mp4)
    # How attention heatmaps are mapped to colour. "peak" = x/max, where one dominant
    # patch sets the scale and crushes the rest to the ghost floor. "robust" clips both
    # tails to percentiles of the nonzero entries so the hottest patches saturate
    # instead of dictating the scale. See attn3d.attention_range.
    attn_norm_mode: str = "peak"
# --- Rollouts (both for Eval and RL) ---
@dataclass
class RolloutConfig:
    max_steps: int = 350
    temperature: float = 1.0
    deterministic: bool = False  # True = argmax; False = sample from action probs
    action_space_str: str = "[stop, forward, left, right, up, down]"
    system_prompt: str = "${read_text:src/longnav/conf/prompts/objectnav_prompt.txt}"
    action_space: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])
    # Templates are lists of dicts (JSON-like)
    convo_start_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "user", "content": [{"type": "text", "text": "${rollout.system_prompt}"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])
    
    convo_turn_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])
    stop_prob_threshold: Optional[float] = None
    visualize_token_filtering: bool = False  # dim filtered visual patches in rollout videos
    visualize_attention: bool = False  # overlay action-attention heatmap on RGB in rollout videos
    visualize_attention_heads: bool = False  # append a 4-wide grid of per-head attention heatmaps to rollout videos
    visualize_attention_3d: bool = False  # capture growing max-over-heads attention history for the 3D heat video
    # Decoder layers feeding the 3D heat video, one video each. Negative indices
    # count from the end; null probes every layer.
    attn3d_layers: Optional[List[int]] = field(default_factory=lambda: [-1])
    # Position ids when the sim attaches per-patch world coords (sim.voxel_kwargs set):
    # 'auto' = 'bev' whenever coords arrive (legacy behaviour); 'standard' = stock mRoPE,
    # the coords are only forwarded to the KV-prune voxel selectors; 'bev' = force bev.
    pos_id_mode: str = "auto"


# --- Experiment housekeeping ---
@dataclass
class RunConfig:
    run_name: str = "debug_run"
    wandb_project: Optional[str] = None
    shard_size: int = 6
    subset_label: str = "sample400_a"
    episode_json: str = ""
    output_dir: str = "./dump/results"
    jobtype: str = "eval"

# --- ROOT CONFIGs ---
@dataclass
class InferenceConfig:
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sim: HabitatConfig = field(default_factory=HabitatConfig)
    task: RunConfig = field(default_factory=RunConfig)

@dataclass
class RLConfig(InferenceConfig):
    training: VLMTrainingConfig = field(default_factory=VLMTrainingConfig)
    hab_config_list: Optional[List] = None
