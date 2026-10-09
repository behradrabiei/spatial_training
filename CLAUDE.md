# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Two conda environments — read this first

Nothing in this repo runs in `base`. There are two envs, and the split is load-bearing:

| Env | Holds | Used by |
|-----|-------|---------|
| `longnav_vlm` | torch 2.8.0+cu128, transformers, ray 2.53.0, verl (editable), longnav (editable), numpy 2.x | **the driver** — always activate this to launch `train_rl`/`eval`/`serve`; also the VLM/RL Ray actors |
| `vln` | habitat-sim + habitat-lab 0.3.3, ray 2.53.0, matplotlib, longnav (editable, `--no-deps`), numpy 1.26 | Habitat sim Ray actors; anything importing `habitat` or plotting |

You activate `longnav_vlm` and run the script. Ray then places actors into the right env itself via `runtime_env={"conda": ...}` (see `factories.py`), driven by `resources.vlm_conda_env` / `resources.habitat_conda_env`. Never `conda activate vln` to run a training or eval script.

Consequences worth remembering:
- `import habitat` fails in `longnav_vlm` **by design** — the driver never touches habitat directly, only through Ray actor handles.
- `matplotlib` is only in `vln`, so `tools/plot_*.py` needs `/home/brabiei/miniconda3/envs/vln/bin/python`.
- Setting `resources.vlm_conda_env=None` makes VLM actors inherit the driver env — that is how the smoke tests run without a conda round-trip.

`SETUP_NOTES.md` records how both envs were built on this machine (RTX 5090 / `sm_120`; flash-attn is deliberately skipped, `attn_impl=sdpa` is the working default).

## Commands

```bash
conda activate longnav_vlm

# Smoke tests — no habitat needed, both use DummyEnvActor
python3 tests/eval_smoke.py    # rollout collection end-to-end
python3 tests/rl_smoke.py      # one real LoRA training step on GPU
python3 tests/grad_attn_smoke.py  # gradient-attribution attention path
python3 tests/attn_evict_smoke.py # attention viz under a sliding context window
python3 tests/ctx_recompute_smoke.py # context_window_mode='recompute' (first check is CPU-only)
python3 tests/ctx_frames_smoke.py    # context_frames / context_cut='frame' (first two checks CPU-only)
python3 tests/vtp_flat_smoke.py      # flat token-prune arms: no window, every method at layer 0 (first check CPU-only)
python3 tests/vtp_hybrid_smoke.py    # hybrid protocol (attn stages + random tail) and staged pruning without a window (first three checks CPU-only)

# Dummy end-to-end (no habitat deps)
python3 -m longnav.scripts.train_dummy

# Eval
python -m longnav.scripts.eval +checkpoint=longnav +dataset=hm3d_v2_val \
  +experiment=eval +resources=single task.run_name=my_eval_run

# RL training
python3 -m longnav.scripts.train_rl +checkpoint=sft +dataset=hm3d_train \
  +experiment=train +resources=octo +training=hapo task.run_name=my_training_run

# Sim-to-real FastAPI server (see tools/client.py for the API)
# NB: README says `longnav.serve`; the module actually lives at longnav.scripts.serve
python3 -m longnav.scripts.serve
```

There is no pytest suite, no linter, and no build step — `tests/*.py` are standalone scripts. Hydra tab completion: `eval "$(python -m longnav.scripts.train_rl -sc install=bash)"` (works with `python`, not `python3`).

`NOTES_TO_SELF.md` holds the exact invocations for the full and smoke HM3D-v2 eval runs.

## Config system

Hydra + structured configs. Schemas are dataclasses in `src/longnav/config_schema.py` (`InferenceConfig` → `RLConfig` is the root); presets live in `src/longnav/config/<group>/*.yaml` and compose with `+group=name`:

`checkpoint` (which adapter to load) · `dataset` (habitat yaml + scene/episode paths) · `experiment` (eval vs train: output dir, wandb project, viz toggles) · `resources` (`single`/`quad`/`octo`, plus `l2`/`l3`/`l4` registered in code) · `training` (`hapo`/`rpp` RL hyperparameters).

Every preset yaml must start with `# @package _global_`. `register_configs()` also installs a `${read_text:<path>}` OmegaConf resolver, used to pull the system prompt out of `src/longnav/conf/prompts/`.

Config landmines:
- `resources.num_vlms` must divide `training.rl_config.n_rollout` (default 16) or **training hangs**. It must also be ≤ GPU count. `num_sims` should be `num_vlms + 1`.
- `resources.osm_gb` is the Ray object store size — tune to shared memory or you get spilling into `ray_object_spilling/`.
- `task.run_name` is the wandb run key, and **default behavior is to resume**: already-logged `episode_label`s are read back from wandb history and skipped (`WandbFactory.create`). Reusing a run name silently means "continue", not "redo".
- `sim.workspace` is the directory containing `data/`, per habitat-lab's DATASETS.md.

## Architecture

### Ray actor topology

`ExpBootstrapper` (`utils/factories.py`) owns cluster setup and spawns three actor kinds, each pinned by a custom Ray resource tag (`env_a` for VLMs, `env_b` for sims) so placement is explicit rather than left to the scheduler. All actors are created with `max_restarts=0` — a crashed worker must not silently come back with a fresh KV cache.

The driver holds no model and no simulator. It only shuffles `ObjectRef`s.

### The rollout event loop

`collect_rollouts()` in `utils/rollout_core.py` is the heart of the system: a single-threaded event loop over `ray.wait`, juggling four future pools (`pending_resets`, `active_episodes`, `pending_postproc`, `pending_logs`) against a deque of idle VLMs and ready sims. An episode is dispatched only when a VLM *and* a reset sim are both free; when it finishes, the VLM goes to post-processing and the sim independently goes to log-flush → reshard → reset, so neither blocks the other.

Sharding is pull-based: sims request a new shard from `shard_iterator` only once `is_exhausted()`. `get_shard_iterator` sources episodes from `task.subset_label` (a key into `constants.episode_labels_table`), or `task.episode_json` (a JSON list of labels), or yields the trivial shard when `shard_size <= 0` (habitat loads the whole dataset itself).

There is a 6-minute stall detector that warns and re-arms rather than aborting — long attention-video renders legitimately hold a sim actor for minutes.

### Env interface

Any actor matching `env/env_base.py:DummyEnvActor` can be dropped into `collect_rollouts`: `reset()`, `step(action, supplementary_logs)`, `assign_shard(episodes)`, `flush_logs_to_disk()`, `is_exhausted()`.

The one non-obvious contract: **RGB is returned separately from the state dict** — `(rgb, state_dict)`, or `(rgb, patch_coords, state_dict)` in BEV mode. This keeps the heavy array out of the pickled dict for Ray zero-copy. `HabitatEnvActor` (`env/habitat.py`) implements this by popping `rgb` out of `obs` and injecting `is_exhausted` into the state.

`sim.output_schema` filters what each step returns; `HabitatWorker` also carries ObjectNav-specific reward shaping (`explr_bonus`, `collision_penalty`, `fpstop_penalty`) and optional FP/FN guards.

### VLMWorker and the incremental KV cache

`utils/vlm_worker.py` (the largest and subtlest file). The agent does **not** re-encode its history each step. It keeps one growing `DynamicCache` across the whole episode and feeds only the new turn, which is what makes 350-step episodes tractable. Everything else follows from that:

- **Action decoding is not generation.** The action is read from logits at a fixed "sandwich" position located by searching for `prefix`/`postfix` token patterns (`_get_sandwich_indices`) — hence the `**action**` markup in the conversation templates. `vocab` entries must each be exactly one token.
- **Sparse visual tokens** (`vlm.use_sparse`, default on): `utils/modeling.py` subclasses Qwen3-VL to drop visual patches whose cosine similarity to already-cached patches exceeds `sparse_threshold`, i.e. re-observed geometry is not re-stored.
- **Context window** (`vlm.context_window`, default `None` = full episode): `_apply_context_window` evicts whole turns from the cache while pinning the prompt prefix so the goal survives, and trims the sparse embed DB alongside the cache — otherwise old frames the model can no longer see would still suppress new patches as "redundant" and confound the ablation. Mind the counting: `context_window=N` keeps N *past* turns, so N+1 frames are visible when the action is read. The paper-facing knob is **`vlm.context_frames=K`** (K frames visible incl. the current one, K=1 = current frame only; internally `context_window=K-1`). **`vlm.context_cut`** picks where the cut lands: `header` (legacy default) holds back the previous assistant header so the window opens on `header + action` of the oldest evicted turn; `frame` drops that action too so the window opens on the oldest retained `<|im_start|>user` — exactly K frames and the K−1 actions between them (at K=1 the incoming chunk's lead is stripped on entry, `_strip_next_lead`). `frame` works with `prune` at slot granularity (`build_prune_partition(frame_cut=True)` retires the boundary turn's lead via `KVPruneState.is_lead`; `tests/vtp_prune_smoke.py` pins prune≡evict parity). `tests/ctx_frames_smoke.py` pins parity, the legacy counts, and the K=1 evict≡recompute control.
- **Context window mode** (`vlm.context_window_mode`, default `evict`): eviction frees memory but does not remove information — a surviving token's layer≥1 K/V were computed from a residual stream that had attended over the whole episode, so evicted frames still reach the decision through them. `recompute` instead rebuilds the window's K/V against a cache holding only prefix + window, making the decision a strict function of what the agent can still see; the gap between the two modes is the leak. Both retire the same tokens on the same steps (`tests/ctx_recompute_smoke.py` pins this down) and retained turns keep their original absolute mRoPE positions, so the arms differ in one variable only. `recompute` replays stored post-ViT embeds rather than re-running the vision tower or the sparse filter, costs one window-sized text prefill per step, and refuses to run with attention viz. `reindex` (StreamingLLM-style, `utils/pre_rope.py`) evicts on the same schedule as `evict` but caches keys pre-rotation and applies RoPE at attention time from a per-slot position table, renumbering survivors to contiguous mRoPE positions — no positional hole across the cut; requires `use_sparse`, refuses attention viz and `save_outputs` (`tests/reindex_smoke.py` pins schedule parity, position contiguity, and bit-identical fidelity to stock when nothing is evicted).
- **KV budget pruning** (`context_window_mode=prune`, `utils/kv_prune.py`): a hard slot budget (`vlm.kv_budget`) enforced every step on the pre-rotation substrate; survivors keep their original positions. Selectors in `vlm.kv_prune_importance`; results and protocol in `KV_PRUNE_RESULTS.md`. **Layer-selective pruning** (`vlm.kv_prune_layer_start` / `kv_prune_layer_end`) confines the budget to decoder layers `[start, end)` while the other layers keep the window-only cache — the incremental-cache analogue of "prune at layer L" from *When Token Pruning is Worse than Random* (CVPR'26). The position table, slot metadata and embed DB then describe the **master** cache (what the unpruned layers hold); `ReindexState.alive` marks what the pruned layers still hold, and `pre_rope_attention_forward` gathers each layer's rotation phases and mask columns (`mask_for_layer`) — transformers builds one causal mask from layer 0 and sdpa/eager silently truncate it, so every layer's mask must be resized to its own key length. `kv_prune_score_layers` picks which layers feed the `attn` score (`pruned`/`all`/`boundary`); `kv_prune_log_layer_influence` logs the per-layer information-horizon KL, `kv_prune_log_keep_one` the per-frame keep-one-in vs leave-one-out KL by layer (`tools/keep_one_influence.py`; `kv_prune_importance=keep_one` is the matching selector); `kv_prune_visual_blind` is the horizon hypothesis test — on layers `[start, end)` no query may attend to any visual slot (cached or current frame; `hide_columns`) and those slots leave the blind layers' caches, no budget involved. Refuses `granularity='turn'`, `kv_prune_merge`, and FA2 with replay-based scorers. `tests/kv_prune_layer_smoke.py`; sweep `tools/run_kv_prune_layer_sweep.sh`; plots `tools/plot_kv_prune_layers.py`, `tools/layer_influence.py`.
- **Visual-token pruning arms** (`paper/token_prune`, preset `+experiment=paper_prune`): the prune substrate with `context_frames=32 context_cut=frame`, `kv_prune_candidate_scope=visual`, `kv_prune_recent_turns=0` (the just-decided frame is history) and a **frame-relative budget** `vlm.kv_prune_budget_frames=k` (keep k × the window's mean original post-filter visual count; `frame_budget`, replaces `kv_budget`; nothing pruned while the window holds ≤ k frames). Methods: `random` (per-step reseed `kv_prune_seed_per_step`), `divprune` (`divprune_select`, cosine max-min on the embed DB), FastV = `attn` + `layer_start=2` + `score_layers=boundary`, `dart` (`dart_select`: key-L1-norm pivots at layer 1, cosine on the layer-2 input hidden states kept in `VLMWorker._hidden_db` by a pre-hook), PyramidDrop = `attn` + `vlm.kv_prune_stages=[7,14,21]` (**staged** `ReindexState.stages`: nested `alive` masks, per-stage rows in `layer_rows`, budgets `pyramid_budgets`). `kv_prune_log_masks` writes the per-frame keep masks (`sup/vtp_{frames,ntok,kept,bits,raster}`) into `sequence.json`. Without a budget the arm is the identity and is **bit-identical** to `r1_evict_k32` (`tests/vtp_prune_smoke.py`). Uni-NaVid side: `baselines/uninavid_hm3d/prune.py` (`--prune-method/--prune-k`). **Flat variant** (`+experiment=paper_prune_flat`, `VARIANT=flat` in the run scripts, `--variant flat` in `aggregate.py`/`plot.py`, data in `paper_results/token_prune_flat/`): no window at all (`kv_prune_budget_frames` is then k × the running mean over every history frame) and every method at layer 0 — `vlm.kv_prune_score_layer=0` makes `attn` rank by the layer-0 decision row alone (works with nothing layer-selective; `ReindexState.configure` keeps an explicit `score_layers` when `pruned_layers is None`), `dart` with `layer_start=0` reads layer-0 keys + layer-0 output (`_dart_select`, hidden capture at `max(layer_start, 1)`), masks off (`_log_vtp` would re-pack the whole episode each step). Uni-NaVid: `--prune-layer 0` = layer-0 scoring pre-pass + row removal (`TokenPruner._prepass`), `--context-frames` ≤ 64. Own wandb project (`longnav_paper_flat`) because arm names repeat. `tests/vtp_flat_smoke.py`.
- **Hybrid protocol vs PyramidDrop** (`paper/hybrid_prune`, preset `+experiment=paper_hybrid`, data in `paper_results/hybrid_prune/`, wandb project `longnav_paper_hybrid`): staged pruning with a per-stage rule. `vlm.kv_prune_stage_importance` names each stage's selector (`attn` = ranked at the layer below the stage, `random` = uniform draw nested in the previous stage's survivors; a random stage captures no decision row — `_score_layer_range` skips it) and `vlm.kv_prune_tail_budget_frames=j` gives the LAST stage its own frame-relative budget (binds from the second frame on, whatever k is) while `pyramid_budgets` spans the stages before it and ends at `kv_prune_budget_frames`. Hybrid arm = K=32 frame cut + `kv_prune_stages=[7,14,20]` + `[attn,attn,random]` + tail 1; the comparison arm is PyramidDrop `[7,14,21]` with **no window** (`context_frames=null context_cut=header`; the master cache then never shrinks). Top-level `kv_prune_importance` stays `attn`; with neither knob set the staged path is bit-identical to Exp 4's. `sup/mean_kv_stage_visual_<s>` logs the visual slots each stage's layers hold. Uni-NaVid side: `--prune-method hybrid` (`STAGES['hybrid']`, attention stages 8/16, random stage from layer 24 keeping `TAIL_FRAMES`×4 tokens, drawn when layer 24's prefill starts). `tests/vtp_hybrid_smoke.py`, `baselines/tests/test_prune.py`. **Uniform protocol (Exp 7b, third arm `<model>_uniform_mp<k>`):** `kv_prune_visual_blind=true` **with** `kv_prune_stages` makes the LAST stage blind (`VLMWorker._blind_stage`: `blind_layers` = that stage, implicit tail budget 0, its selector entry ignored, `kv_prune_tail_budget_frames` must stay unset; a tail of exactly 0 without the blind is the history-only variant). Arm = `stages=[0,18]`, `[random,random]`, blind, model scope, pool anchor: layers 0–17 hold one uniform draw of `28k/18` frames' worth, layers 18–27 no visual token at all (current frame included). Uni-NaVid `--prune-method uniform` (`STAGES['uniform']=((0,0),(None,20))`, `BLIND_STAGE`; needs `--prune-budget-scope model`): input-level seeded draw with row removal at stage 0, a budget-0 random stage at layer 20 whose mask also covers the current frame's group and 64 fine tokens (`_blind_cols`). `aggregate.py`/`plot.py --protocols` narrow the protocol set (the MP3D replicate has no uniform arms). **MP3D replicate:** `DATASET=mp3d` on the run scripts, `--dataset mp3d` on `aggregate.py`/`plot.py` (`mp3d_100_seed17.json`, data in `paper_results/hybrid_prune_mp3d/`, wandb project `longnav_paper_hybrid_mp3d`).
- **Per-token layer information** (`paper/layer_token_info`, `utils/token_info.py`, `vlm.token_info_probe`): after a step, keep-one-in / leave-one-out KL for every visual token of the current frame at every decoder layer (hiding on layers ≥ i). Leaky arm = per-slot decision replay on the prune substrate (`probe_mask_arm`); leak-free arm = hooked full re-prefill in evict mode where a forward pre-hook on each `self_attn` swaps in a `hide_columns` mask (`probe_recompute_arm`, `_ColumnHider`). Every leak-free forward, base included, goes through the masked sdpa kernel path: an unmasked or different-length bf16 prefill differs by ~2e-2 in probability, the same floor as rebuilt-vs-live. Both arms read the decision through an **fp32 projection of the final hidden state onto the four action rows** (`_ActionHead`): the bf16 lm_head quantises KL to ~1e-3-nat levels and a single token's removal sits below that quantum. `probe_*_arm(measures=...)` selects the measures; `NOHIST_MEASURES` (`keep_all` + `leave_nohist`) describe the current frame with every history frame's visual tokens hidden — the multi-scene / multi-timestep study under `paper_results/layer_token_info/nohist/` (`probe_episode.py --study`, `run_nohist.sh`, panel layers fixed at 1/12/18). `tests/token_info_smoke.py`.
- **Selector-criterion vs KL probe** (`paper/selector_kl`, preset `+experiment=paper_selector_probe`, `vlm.selector_probe`, `utils/selector_probe.py`, loader `analysis/selector.py`): the flat variant's pool with nothing pruned and the sparse filter OFF (full grids, as `paper_token_info`); after every decision, for each visual token of the current frame, FastV's layer-0 decision-row attention, DivPrune's nearest-neighbour cosine distance (`divprune_nn`) and greedy position, DART's pivot dissimilarity (`dart_dis`), and every method's keep rank at `selector_probe_budgets` frames' worth (one-shot on the whole cache, `frame_budget`) next to the leaky `keep_cur` KL of `token_info.probe_mask_arm` at layer 0 (`selector_probe_measures`). Forces `kv_prune_score_layer=0` and arms both the attention-row and the DART hidden captures itself (independent of `kv_prune_importance`). Rows in `sequence.json` as `sup/selector_*`; the showcase RGB needs `showcase_frames.py` (video frames carry burned-in text). `tests/selector_probe_smoke.py`. Uni-NaVid side: `--selector-probe` (`baselines/uninavid_hm3d/selector_probe.py`, `SelectorProbe` on the `LayerProbe` capture; criteria from one layer-0 pass, KL by batched column masks through an fp32 action-row head; sidecar `selector/<label>.jsonl`; `baselines/tests/test_selector_probe.py`).
- **Attention visualization** forces `attn_impl='eager'` automatically (sdpa/flash return no attention weights). `vlm.attn_weighting` selects what the heatmaps mean: `raw` α, `value_norm`/`wo_norm` (α weighted by the token's actual contribution to the residual stream), or `grad` (α · ∂score/∂α, requiring `visualize_attention_3d` and an extra forward+backward under `no_grad` instead of `inference_mode`).

Class chain: `VLMWorker` → `RolloutWorker` (+`EpisodeRolloutMixin`) → `RLWorker` (+`VLMTrainingMixin`) → `RLActor`. Training is LoRA/PEFT via `utils/trainers.py`, with losses and advantage estimators pulled from the vendored `verl` submodule.

### Output layout

```
dump/<experiment output_dir>/<run_name>/
  config.yaml                    # resolved config, saved at bootstrap
  rollout/
    results_<pid>                # JSONL, one line per episode, one file per sim worker
    <scene_id>/<episode_label>/  # video.mp4, thumbnail.jpg, sequence.json, summary.json
```

Aggregating a run means globbing `rollout/results_*` and reading JSONL — `longnav.analysis.results_io.load_r1_run` does this (and `load_uninavid_run` for the baseline's `episodes/*.json`).

## Paper experiments

`paper/<experiment>/` holds each paper experiment's code: a README with the protocol, run scripts, `aggregate.py` (→ `results/*.csv`, `table.tex`) and `plot.py` (→ `figures/*.pdf|png`, run in `vln`). Its data lives in **`paper_results/<experiment>/`** (tracked, not `dump/`): `runs/<arm>/` raw output with configs and logs (bulky per-episode dirs and traces gitignored via `paper_results/.gitignore`), `results/`, `figures/`. Shared loaders/stats live in `src/longnav/analysis/`. Hydra presets for paper protocols are `+experiment=paper_*` (e.g. `paper_ctx`), which pin the episode manifest (`src/longnav/conf/episode_jsons/hm3d_v2_100.json`), viz off, deterministic decoding and the sparse regime; an arm is then one or two overrides. The Uni-NaVid baseline harness is `baselines/` (tracked; its runtime, weights and results are ignored via `baselines/.gitignore`) — see `paper/ctx_ablation/README.md`.

## Submodules

`verl` (RL algorithms — advantage estimators, policy losses) and `ovon` are git submodules; `git submodule update --init --recursive`. `verl` is installed with `--no-dependencies` because its `numpy<2.0.0` pin conflicts with longnav's. habitat-lab is a gitignored clone at `.habitat-lab/`.

## Attention viz: the invariant to preserve

Anything indexed by absolute KV position — the probe's captured rows, its value-norm banks, the frame records' `abs_kv_idx` — must be cropped or shifted together when `_apply_context_window` evicts. They are read *after* `infer_step` returns, so a tensor left in pre-eviction index space silently points every surviving frame at the wrong key. `AttentionProbe.evict` handles the probe side; `_apply_context_window` handles the frame records. `tests/attn_evict_smoke.py` pins this down. See `LOOK_INTO.md` for the remaining caveats on how the heat videos should be read.
