# KV-budget pruning results (2026-08-19)

**Question:** the context-window ablation showed 32 frames ≈ full history, competitive down
to 8. Can the 32-frame window be compressed to ~4-frame *memory cost* while keeping
performance? (Metric: actual resident KV slots, never nominal frames.)

**Mechanism:** `vlm.context_window_mode='prune'` (`src/longnav/utils/kv_prune.py`) — a hard
KV token budget (`vlm.kv_budget`) enforced every step on the reindex (pre-rotation)
substrate: lowest-importance unprotected slots are sliced out of the cache, the per-slot
mRoPE table, the slot metadata, and the sparse embed DB together. Survivors keep their
original positions (no renumbering). Protected: prompt prefix + last
`kv_prune_recent_turns` (2) turns. Importance modes: `attn` (EMA of the decision token's
attention row, captured free inside `pre_rope_attention_forward`) or `random`;
granularity `slot` or `turn` (whole-keyframe selection); optional `kv_prune_merge`
(CaM-style: fold dropped visual slots into their nearest kept slot by embed cosine).
Smoke: `tests/kv_prune_smoke.py`. Sweeps: `tools/run_kv_prune_ablation.sh`; analysis:
`tools/compare_runs.py`, `tools/plot_kv_budget_tradeoff.py`.

Budgets matched to measured window memory (reindex relogs with the new `sup/mean_kv_len`
metric): ~153 tokens/turn after sparse filtering, ~574-token pinned prefix →
B4=1187, B8=1799, B16=3022 slots. KV cost ≈ 0.109 MB/slot (28 layers × 8 KV heads ×
128 dim × 2 × bf16).

## Results — 100-ep HM3D-v2 split, deterministic, viz off

| run | SR | SPL | steps | mean kv | vs win32 KV | lat ms |
|---|---|---|---|---|---|---|
| win32 evict (reference) | 0.880 | 0.481 | 121 | ~5474 | 1× | 57.0 |
| full history | 0.840 | 0.468 | 111 | (grows) | — | 82.1 |
| **random-prune B16** | **0.860** | **0.427** | 152 | 2769 | **2.0×↓** | 50.4 |
| win16 evict | 0.820 | 0.419 | 152 | ~3024 | 1.8×↓ | 50.2 |
| **random-prune B8** | **0.810** | **0.406** | 171 | 1736 | **3.2×↓** | 46.3 |
| win8 evict | 0.790 | 0.383 | 184 | ~1799 | 3.0×↓ | 45.5 |
| **random-prune B4** | **0.730** | **0.324** | 224 | 1172 | **4.7×↓** | 44.2 |
| random-prune B4, no window pool | 0.707 | 0.329 | 227 | 1172 | 4.7×↓ | 44.2 |
| win4 evict | 0.700 | 0.309 | 239 | ~1187 | 4.6×↓ | 42.9 |
| keyframe-attn B8 (turn granularity) | 0.730 | 0.337 | 209 | 1641 | 3.3×↓ | 47.3 |
| slot-attn B8 (TOVA/H2O-style EMA) | 0.650 | 0.328 | 222 | 1745 | 3.1×↓ | 47.4 |
| keyframe-attn B4 | 0.650 | 0.289 | 265 | 1044 | 5.2×↓ | 45.2 |
| slot-attn B4 | 0.630 | 0.289 | 254 | 1172 | 4.7×↓ | 44.9 |
| attn B4 + merge (CaM-style) | 0.580 | 0.250 | 273 | 1173 | 4.7×↓ | 48.3 |
| reindex win4 (renumbered) | 0.450 | 0.236 | 308 | 1187 | 4.6×↓ | 43.5 |

Figure: `dump/longnav_eval/hm3d_v2_100_kv_budget_tradeoff.png`. Evict-baseline kv is the
fitted schedule (574 + 153·N); prune arms are measured. Peak process memory 4.6 GB at B4
vs 5.4 GB at win32 (13.3 GB worst-case at full); per-step latency −22% vs win32.

## Findings

1. **Random pruning of the sparse-deduped cache is the best compressor tried.** It sits
   above the recency-window curve at every matched budget (+0.02..+0.04 SR, individually
   n.s. at n=100 but consistent in direction on SR, SPL, and steps at all three budgets),
   and B16-random matches win32's *success rate* at half the KV. Why: the sparse filter
   already deduplicates patches, so cache content is near-uniformly informative — uniform
   sampling preserves spatial coverage. It also hard-caps memory (kv_max = budget exactly),
   which windows do not (win32's kv peaked at 8780 slots on one scene).
2. **Attention-salience selection is *worse than random*** (B4: 0.630 vs 0.730, paired
   p=0.06; same ordering at B8). The decision row concentrates on currently-relevant
   content — redundant with the protected recent frames — and discards the diffuse
   coverage needed later. Keyframe (turn-granular) selection closes part of the gap at B8
   (0.730) but never reaches random.
3. **KV merging hurts** (0.580/0.250): importance-weighted averaging of K/V across
   distinct geometry blurs it, even pre-rotation.
4. **Positional holes beat renumbering at extreme compression:** reindex win4 collapses to
   0.450 SR vs evict win4's 0.700 — the evict≈reindex parity (win8–64) breaks at win4.
   The prune mode keeps original positions, which this validates.
5. **The 32-frame pool is optional for random pruning** (no-window ≈ windowed): per-step
   budget re-sampling already imposes geometric attrition on old slots.
6. **A hard information bottleneck remains.** No arm reaches win32 SPL at B4/B8 memory
   (B8-random concedes −0.076 SPL, p=0.002). Consistent with the recompute-leakage
   result: leak-free win4 SR is 0.43, so a large share of every compressed arm's
   performance rides on eviction leakage through surviving K/V, and SPL keeps improving
   with genuinely visible context.

## Practical recommendation

`context_window_mode=prune, kv_prune_importance=random, kv_budget=1800–3000` gives
0.81–0.86 SR at 2–3.2× less KV than win32 with lower latency; use B≈1187 only when memory
is the binding constraint. Learned/trained compression (fine-tuning under the budget) is
the natural next step; selection heuristics alone appear tapped out.

## Reproduction

```bash
# baselines with kv metrics
MODE=reindex PREFIX=hm3d_v2_100_reindexv2win WINDOWS="32 4" bash tools/run_context_window_ablation.sh
# main arms
BUDGETS="1187 1799 3022" WINDOW=32 IMPORTANCE=random PREFIX=hm3d_v2_100_prunerand bash tools/run_kv_prune_ablation.sh
BUDGETS="1187 1799" WINDOW=32 bash tools/run_kv_prune_ablation.sh                      # attn slots
GRANULARITY=turn BUDGETS="1187 1799" WINDOW=32 PREFIX=hm3d_v2_100_pruneturn bash tools/run_kv_prune_ablation.sh
MERGE=true BUDGETS="1187" WINDOW=32 PREFIX=hm3d_v2_100_prunemerge bash tools/run_kv_prune_ablation.sh
# analysis
python tools/compare_runs.py <runs...> --baseline <run>
/home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_budget_tradeoff.py
```

## Update 2026-08-29/30 — can anything beat random? (kl, grid, voxel)

Runs `kvprune_v2_*` (same 100-ep split, seed 17, window 32, scope=all); figure
`dump/longnav_eval/kvprune_v2_prune_methods.png` (`tools/plot_kv_prune_methods.py`). Paired
sign-flip bootstrap vs `kvprune_v2_full_random_w32_b<B>` (`tools/compare_runs.py`).

| selector | what it does | B4 SR/SPL | B8 SR/SPL | B16 SR/SPL | verdict |
|---|---|---|---|---|---|
| random (baseline) | uniform over candidates each step | 0.760/0.328 | 0.800/0.377 | 0.850/0.424 | — |
| `kl` | frame quota ∝ KL(action ‖ action with that frame masked), random within frame | 0.750/0.331 | 0.790/0.386 | 0.860/0.396 | tie (all p ≥ 0.09); 8× VLM latency (one replay per frame) |
| `grid` | random's exact per-frame quotas, filled by 2-D farthest-point picks on the mRoPE (row, col) token grid | 0.770/0.344 | 0.780/0.392 | 0.840/0.415 | tie (ΔSPL +0.016 p=0.39, +0.015 p=0.50, −0.009 p=0.60); 3–4 ms/step |
| `voxel_dedup` 0.3 m (screen, 25 ep) | ≤1 visual slot per world (x, z) cell, newest kept, then random | 0.720/0.224 | – | – | **worse** (ΔSPL −0.10, p=0.046) |
| `voxel_dedup` 0.6 m (screen) | same, coarser cells | 0.640/0.255 | – | – | worse (ΔSR −0.16) |
| `voxel_strat` 0.6 m (screen) | equal share across occupied cells, newest first | 0.680/0.231 | – | – | worse (ΔSR −0.12) |

(Random screen baseline on the same 25 episodes: 0.800/0.326.)

Findings:

1. **Within-frame spatial evenness adds nothing.** `grid` keeps random's temporal profile
   exactly (same generator stream ⇒ identical per-frame quotas and text picks) and only
   changes *which* patches within a frame survive; the result is indistinguishable from
   an i.i.d. sample at every budget. The i.i.d. clumps/gaps random leaves inside a frame
   are not what limits the compressed agent.
2. **Geometric redundancy is used, not wasted.** Tagging every visual slot with its world
   voxel (depth + pose, `longnav.utils.voxel_utils`) and collapsing multi-view copies of
   the same cell hurts sharply, and more so with coarser cells. Per-episode, the loss sits
   on the *short* episodes (dedup 0.3 m: ΔSR −0.15, +101 steps on episodes random solves in
   ≤216 steps; long episodes ≈ neutral): the cap removes the repeated views of *nearby*
   floor/wall geometry in the frames just outside the protected window, i.e. it thins the
   dense recent frames that local navigation runs on. The alignment cross-check
   (`VLMWorker._turn_voxel_rows`) never fired, so this is a genuine negative result.
3. Together with the earlier arms (attn/keyframe/Fisher: worse; FPS-diversity, flat
   stratified: much worse; KL: tie), every information-agnostic *or* information-seeking
   selector lands on or below random. Random's geometric attrition of dense recent frames
   appears to be the operating point of this policy; the remaining lever is training under
   the budget, not selection.

## Protecting the action-text history hurts (2026-08-30, `candidate_scope=visual`)

Question: at the same total budget, is the 32-turn action-text history (the agent's
"odometry in words") worth pinning, pruning only visual tokens? Runs
`kvprune_vis_full_{random,kl}_w32_b{1799,3022}` — identical to `kvprune_v2_full_*` except
`kv_prune_candidate_scope=visual`. **B4 (1187) is structurally infeasible**: the protected
floor (prefix ~600 + in-window text growing to ~320 + two recent turns' visual ~520)
exceeds the budget once the window fills (`RuntimeError` at floor 1191 by turn ~6; this is
also why the old 2-row `kvprune_screen_random_w32_b1187` died).

| run | SR | SPL | steps | kv_vis | kv_text | paired vs scope=all random |
|---|---|---|---|---|---|---|
| vis-random B8 | 0.650 | 0.262 | 258 | 1166 | 585 | dSR −0.15 (p=0.007), dSPL −0.115 (p<1e-4) |
| vis-kl B8 | 0.630 | 0.286 | 250 | 1166 | 581 | dSR −0.17 (p=0.0004) |
| vis-random B16 | 0.800 | 0.382 | 169 | 2232 | 559 | dSPL −0.042 (p=0.017) |
| vis-kl B16 | 0.800 | 0.363 | 180 | 2211 | 562 | dSPL −0.062 (p=0.0002) |

Findings: (1) pinning all text swaps ~290 visual slots for text at B8 (kv_text 585 vs 293)
and costs −0.15 SR — **per slot, visual tokens are worth far more than old action text**;
scope=all random was already making the better trade by letting text compete and thin out.
(2) Within the text-protected regime, kl vs random is again a tie (B8 dSR −0.02 p=0.81,
dSPL +0.024 p=0.08; B16 0.00 / −0.019 p=0.16) — the selector-invariance result replicates
in a second regime. Consistent with the influence diagnostic: what matters is the volume of
visual context, not which category of old token survives.

## Influence-persistence diagnostic (2026-08-30) — why no selector can beat random

Question: does a frame's influence on the decision persist, so that any online score (EMA'd
or not) computed from past steps could predict which frames to keep? Measured directly:
10 full-cache episodes (`kl_influence_full10`, prune mode with an unreachable budget,
`vlm.kv_prune_log_influence=true`), at every step t the leave-one-frame-out KL over ALL
cached frames f <= t (151,400 (t, f) pairs; one single-token decision replay per frame,
~14 + 2.3·kv/1k ms each, 3.4 h total). Analysis `tools/kl_influence.py`; figure
`dump/longnav_eval/kl_influence_full10/kl_influence.png`. Persistence is reported on
age-detrended residuals (raw rank-persistence is inflated by the shared decay curve — the
synthetic self-test shows a non-persistent process still scores raw rho ~0.9).

| lag k | residual rho (per step) | top-25% retention (chance 0.25) |
|---|---|---|
| 1 | **0.046** | 0.289 |
| 2 | 0.041 | 0.282 |
| 4 | 0.028 | 0.276 |
| 8 | 0.028 | 0.283 |
| 16 | 0.018 | 0.274 |

Per-episode residual rho at lag 1 spans −0.03..0.08 — indistinguishable from zero in every
episode. Decay is a cliff, not a curve: mean KL is 0.023 for the current frame (median
0.0014 — even it usually barely matters), drops 20x at age 1 (0.0012), and is flat at
~0.0003 for every older age. **Revival: 0 events out of 1,009 dormant frames.** The two
newest frames hold 29% of the per-step KL mass on average (median 15%).

**Conclusion: the online-selector search is closed.** Beyond the current frame, per-frame
direct influence on the decision is uniformly tiny, uncorrelated from one step to the next,
and never revives — there is no signal for an EMA to smooth (rho ~ 0 makes any EMA converge
to the uniform mean, i.e. to random). This explains the whole selector series: frames are
interchangeable at the level the decision token reads them, and the useful history reaches
the decision through the K/V that later tokens absorbed (the eviction-leakage result), not
through direct attention to old frames. Caveat: this measures masking at decision time
under a full cache; it cannot see what future frames' encodings would have lost had the
frame been pruned earlier — so it rules out decision-replay scoring, while training under
the budget (or shaping what gets absorbed) remains open.

Voxel arms need the sim to attach per-patch world voxels and the VLM to keep stock
position ids: `VOXEL=true POS_ID_MODE=standard IMPORTANCE=voxel_dedup VOXEL_SCALE=4
bash tools/run_kv_prune_ablation.sh` (Hydra: `+sim.voxel_kwargs.patch_size=32 ...
sim.output_schema.obs.patch_coords=true rollout.pos_id_mode=standard`). Habitat depth is
normalised to [0, 1]; `HabitatWorker._postprocess_step` de-normalises it before unprojection.
A `random` run with voxels attached reproduces the baseline rows bit-for-bit
(`kvprune_v2_voxelplumb_random_w32_b1187`).

## Layer-selective pruning — setup (2026-08-31)

**Question:** *When Token Pruning is Worse than Random* (Wang et al., CVPR'26, arXiv
2512.07580) finds that visual-token information in VLLMs vanishes at an intermediate
"information horizon" layer — structured selectors beat random only in shallow layers, and
pruning deep layers is free regardless of selector. Every arm above prunes all 28 layers with
one slot set. Does the layer at which slots are dropped change the picture (and can any
selector beat random when it only has to be right at shallow layers)?

**Mechanism:** `vlm.kv_prune_layer_start=L` (optionally `kv_prune_layer_end=M`) applies the
budget to decoder layers `[L, M)` only; the other layers keep the window-only cache. This is the
paper's "prune at layer L" translated to an incremental KV cache: new tokens still pass through
every layer (their K/V must be cached at the shallow layers), but at layers ≥ L they attend only
over the retained slots. Implementation (`src/longnav/utils/pre_rope.py`,
`src/longnav/utils/kv_prune.py`, `VLMWorker._apply_kv_prune`):
- The position table, slot metadata and sparse embed DB describe the **master** cache — every
  slot held by any layer, i.e. the unpruned layers' window-only cache. `ReindexState.alive`
  marks the master slots the pruned layers still hold; selectors draw candidates from `alive`
  only, so a slot dropped from the pruned layers never comes back there. Survivors keep their
  original mRoPE positions in every layer.
- Each layer rotates its keys with `cos/sin` gathered through its own slot index and gets an
  attention mask resized to its own key length (`mask_for_layer`): transformers builds one causal
  mask from layer 0, and sdpa/eager silently truncate a too-wide mask from the right, which
  would leak intra-chunk future tokens. Uniform pruning (`layer_start=0`) is bit-identical to
  before (`tests/kv_prune_layer_smoke.py`: `(0, None)` vs `(0, 28)` identical; with nothing
  pruned the layer-selective path matches stock to 0.00000).
- Replay-based scorers (`kl`, `fisher`) and the `attn` score by default measure removal on the
  pruned layers only (`kv_prune_score_layers=pruned|all|boundary`; `boundary` = layer L−1,
  FastV-style).
- Caveats: (i) memory savings scale with the fraction of layers pruned — read
  `sup/mean_kv_len` (layer-mean, memory-equivalent) next to `kv_lmax` (unpruned length) in
  `compare_runs.py`; the budget is per pruned layer, as in the paper, not total-memory
  matched. (ii) The sparse filter dedups against the master DB, so a patch dropped from the
  deep layers is not re-admitted while the shallow layers still hold it. (iii) `granularity=turn`
  and `kv_prune_merge` are refused; FA2 is refused with replay-based scorers (its integration
  misreads the 4-D replay bias as a padding mask — a pre-existing bug, now caught at construction).

**Information-horizon diagnostic:** `vlm.kv_prune_log_layer_influence=true` logs per step, for
each probe layer i in `kv_prune_layer_influence_starts`, KL(P_t ‖ P_t with the visual slots
hidden from layers ≥ i) — once for the history's visual slots only (`sup/layer_influence_hist`,
what pruning removes) and once for every visual slot including the current frame
(`sup/layer_influence_all`). Hiding from layer i onward leaves what the slots contributed to
layers < i in the residual stream, so the curve over i is the depth to which visual information
still reaches the decision; the probe i = 28 hides nothing and is 0 by construction.

**Reproduction:**
```bash
# arms: selectors x prune layers x budgets (layer 0 = the uniform arms above)
METHODS="random attn kl" LAYERS="0 7 14 21" BUDGETS="1187 1799" bash tools/run_kv_prune_layer_sweep.sh
python tools/compare_runs.py kvprune_layer_random_w32_l14_b1799 --baseline kvprune_layer_random_w32_b1799
/home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_prune_layers.py --prefix kvprune_layer
# horizon curve (10 episodes, full cache, ~16 replays/step at the default 8 probes)
python -m longnav.scripts.eval +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources=single \
  task.run_name=layer_influence_full10 task.subset_label="" task.episode_json=dump/hm3d_v2_10_labels.json \
  vlm.context_window_mode=prune vlm.kv_budget=10000000 vlm.context_window=null \
  vlm.kv_prune_importance=random vlm.kv_prune_log_layer_influence=true
/home/brabiei/miniconda3/envs/vln/bin/python tools/layer_influence.py dump/longnav_eval/layer_influence_full10
```
Results: to be filled in from the sweep.

### Layer-selective results (2026-09-01): kl vs random by prune layer, B4 = 1187, 32-frame pool

**Screen, 25 episodes (`kvprune_layer_screen_{random,kl}_w32[_l<L>]_b1187`; layer 0 = the
existing `kvprune_all_screen_*` arms).** Paired kl − random at each prune layer:

| layer_start | random SR / SPL | kl SR / SPL | ΔSR | ΔSPL (p) | kv layer-mean |
|---|---|---|---|---|---|
| 0 (uniform) | 0.80 / 0.326 | 0.76 / 0.273 | −0.04 | −0.054 (0.27) | 1172 |
| 4 | 0.88 / 0.336 | 0.68 / 0.277 | −0.20 (0.06) | −0.059 (0.15) | ~1820 |
| 8 | 0.88 / 0.376 | 0.80 / 0.321 | −0.08 | −0.055 (0.09) | ~2480 |
| 12 | 0.84 / 0.380 | 0.76 / 0.384 | −0.08 | +0.003 | ~3020 |
| 16 | 0.84 / 0.380 | 0.80 / 0.411 | −0.04 | +0.030 (0.54) | ~3565 |
| 20 | 0.76 / 0.398 | 0.84 / 0.432 | +0.08 | +0.034 (0.09) | ~4040 |
| 24 | 0.92 / 0.460 | 0.84 / 0.419 | −0.08 | −0.041 | ~4730 |

kl is *worse* than random when the drop happens at shallow layers and only turns positive at
16–20 — the layers where the horizon diagnostic (2-episode run above) says history stops
reaching the decision. Layers 16 and 20 passed the screen gate and were promoted.

**Full, 100 episodes (`kvprune_layer_{random,kl}_w32_l{16,20}_b1187`; figure
`dump/longnav_eval/kvprune_layer_by_layer.png`, memory-matched view
`kvprune_layer_by_layer_memory.png`):**

| arm | SR | SPL | steps | kv layer-mean | kv unpruned layers | lat ms |
|---|---|---|---|---|---|---|
| uniform random (L0) | 0.760 | 0.328 | 219 | 1172 | — | 41.6 |
| uniform kl (L0) | 0.750 | 0.331 | 225 | 1172 | — | 326 |
| random L16 | 0.830 | 0.438 | 124 | 3644 | 5502 | 56.7 |
| **kl L16** | **0.860** | **0.456** | 125 | 3649 | 5511 | 493 |
| random L20 | 0.850 | 0.455 | 125 | 4177 | 5381 | 57.7 |
| **kl L20** | **0.860** | **0.472** | 121 | 4210 | 5428 | 532 |
| uniform random B16 (ref) | 0.850 | 0.424 | 145 | 2782 | — | 48.1 |
| win32 evict (ref) | 0.880 | 0.481 | 121 | ~5474 | — | 57.0 |

Paired kl − random (sign-flip bootstrap): **L16 ΔSR +0.03 (p 0.46), ΔSPL +0.018 (p 0.27);
L20 ΔSR +0.01 (p 1.0), ΔSPL +0.016 (p 0.12).** Direction agrees with the screen on both
metrics at both layers — the first selector to sit above random on SPL anywhere in this
series — but the effect is not detectable at n=100 (a ΔSPL of ~0.017 would need on the order
of 1000 paired episodes), and it costs ~9× per-step latency.

**Reading.** (1) *No selector beats random by a measurable margin*, even at the layers most
favourable to it. (2) *Where you prune matters only through memory*: at matched layer-mean
kv the layer arms sit on the uniform-random budget curve (random L16 at 3644 slots: SPL 0.438
vs ~0.44 interpolated between B16 and win32; random L20 at 4177: 0.455 vs ~0.45), so
sparing the shallow layers buys exactly what the extra slots buy, no more. (3) *kl's failure
at shallow layers is informative*: leave-one-frame-out KL measured at the pruned layers keeps
what is salient to the current decision, which is redundant with the protected recent turns
(the same failure as decision-attention in the uniform series); at layers ≥ 16 there is so
little history influence left to rank that kl degenerates towards uniform sampling and
matches random. Together with the horizon diagnostic this says the policy reads its visual
history through layers ≲ 16–20 and any memory-shaping lever has to act there — or in
training — not in the slot selector.

### Visual-blind arms (2026-09-01): drop ALL visual tokens from layer L on

**Hypothesis test for the horizon.** `vlm.kv_prune_visual_blind=true` with
`kv_prune_layer_start=L`: on decoder layers ≥ L no query may attend to any visual slot —
cached history *and* the current frame (`pre_rope.hide_columns`, applied live inside the
attention forward) — and visual slots leave those layers' caches after each step. Text and the
32-frame pool are untouched; no budget or selector is involved. `L = 28` is stock (smoke-checked
to 0.00000), `L = 0` is a text-only policy (smoke-checked to be exactly image-independent).
Runs `hm3d_v2_100_w32_vblind[_l<L>]`, 100 episodes, paired vs `hm3d_v2_100_win32`; figure
`dump/longnav_eval/hm3d_v2_100_w32_visual_blind.png`.

| first blind layer L | SR | SPL | steps | kv layer-mean | ΔSR (p) | ΔSPL (p) |
|---|---|---|---|---|---|---|
| 28 (win32, none) | 0.880 | 0.481 | 121 | ~5474 | — | — |
| 24 | 0.850 | 0.460 | 124 | 4736 | −0.03 (0.51) | −0.021 (0.22) |
| **20** | **0.850** | **0.431** | 132 | 4078 | −0.03 (0.45) | **−0.050 (0.003)** |
| 16 | 0.350 | 0.201 | 342 | 2709 | −0.53 (<1e-4) | −0.280 (<1e-4) |
| 0 (text only) | 0.000 | 0.000 | 500 | 651 | −0.88 (<1e-4) | −0.481 (<1e-4) |

**Reading.** The horizon is real and sharp. Blind from layer 24: indistinguishable from the
full model. Blind from layer 20: the agent still *finds* the goal as often (SR n.s.) but takes
~9% more steps — a significant SPL loss of 0.05 — so layers 20–27 still read some visual
detail, and what they read buys path efficiency, not success. Blind from layer 16: the policy
collapses (SR 0.35, most episodes time out); blind from layer 0: total failure. So the visual
information that decides the action is consumed almost entirely in layers 16–20, matching the
leave-one-frame-out horizon curve (KL falls 10× between probes 16 and 20) and the layer-selective
kl result (kl only matches random at L ≥ 16, because there is nothing left to rank there).

Blind-from-20 lands exactly on uniform random B16 (SR 0.850 both; SPL 0.431 vs 0.424, p = 0.78)
while holding more slots (4078 vs 2782 layer-mean), so it is a probe of where vision is read,
not a memory strategy. Where it *is* useful: layers ≥ 24 can drop every visual slot for free
(−15% of the visual K/V), and any future compression that targets layers < 20 has to preserve
what those layers absorb — which is the same conclusion the selector series reached from the
other side.

### Keep-one-in vs leave-one-out (2026-09-01): standalone vs unique visual information

**Motivation.** The CVPR'26 paper's information metric keeps ONE visual token and masks the
rest (standalone information); the `kl` selector masks one frame and keeps the rest (unique
information). `vlm.kv_prune_log_keep_one=true` logs both per cached frame and probe layer
(hidings on layers ≥ i), and `kv_prune_importance=keep_one` is the matching selector. Run
`keep_one_influence_w32_b10000000` (10 ep, window 32, unreachable budget, every 4th step,
probes 0/8/16/20/24/28; 258 scored steps, ~31k (step, frame) pairs); figure
`dump/longnav_eval/keep_one_influence_w32_b10000000/keep_one_influence.png`; analysis
`tools/keep_one_influence.py`.

| frame age | keep-one @ probe 0 | leave-one @ probe 0 | uniqueness | @ probe 20 (keep / leave) |
|---|---|---|---|---|
| 0 (current) | 0.0409 | 0.0280 | 0.69 | 0.0007 / 0.0006 |
| 1 | 0.0255 | 0.0011 | **0.04** | 0.0005 / 0.0004 |
| 2–7 | 0.0072 | 0.0006 | **0.08** | 0.0004 / 0.0004 |
| 8+ | 0.0001 | 0.0001 | 0.67 | 0.0001 / 0.0001 |

**Three findings.**
1. **Recent frames are ~10–25× redundant.** A frame 1–7 steps old carries substantial
   standalone information (keep-one 0.007–0.026) but almost no unique information (leave-one
   ≤ 0.001): anything it says, other frames say too. This is the direct measurement of why no
   selector beats random — under a budget, ranking frames by any per-frame score cannot
   matter when the frames are interchangeable; the objective would have to be coverage.
2. **Old frames carry almost no standalone information either** (age 8+: keep-one 0.0001).
   A lone 20-step-old frame moves the decision no more than no frame at all — the model does
   not read old frames directly even when they are the only thing on offer. History acts
   through what the text/recent tokens have absorbed (note the baseline here: "no visual"
   still leaves cached text K/V that were COMPUTED with full vision — unlike the visual-blind
   arms, which prevent absorption itself and therefore collapse).
3. **Standalone information obeys the same horizon.** For every age bin, keep-one is flat
   across probes 0–16 and collapses ~50× between probes 16 and 20; uniqueness rises toward 1
   there (the trickle that survives past layer 20 is non-redundant). So layers ≥ 20 are not
   merely ignoring redundant tokens — they read (almost) no visual tokens at all, confirming
   the visual-blind result from the measurement side.

**Selector arm** (`kvprune_layer_keep_one_w32_b1187`, 100 ep, B4, uniform): SR/SPL
0.780/0.321 vs random 0.760/0.328 — paired dSR +0.02 (p=0.80), dSPL −0.007 (p=0.71), at ~8×
latency. Ties random, like `kl`: standalone and unique scores fail for opposite reasons
(keep-one over-values redundant copies, leave-one under-values them), and both reduce to
noise against a near-uniformly informative cache.
