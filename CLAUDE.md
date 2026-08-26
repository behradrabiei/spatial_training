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
python3 tests/hamlet_smoke.py     # HAMLET moment tokens + memory: rollout/replay/train/checkpoint invariants

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

On NCSA Delta everything runs from `cluster/delta/` (`env.sh` paths under `/work/nvme/bgon/brabiei/longnav_runtime`, offline HF, staged models via `stage_models.py`); `prepare_hm3d_train.sh` extracts the HM3D train scenes once, `run_rl_train.sh` is the RL driver invocation, `train_hamlet.sbatch` the 8 h 4×A100-40GB job (with a smoke-test gate), `test_hamlet.sbatch` a 1-GPU pre-flight (`ONLY_MINI=1 MINI_STEPS=350 ...` knobs), `test_ddp.sbatch` a 2-GPU multi-actor (DDP/NCCL) pre-flight, and `eval_ckpt.sbatch`/`run_eval.sh` a 1-GPU deterministic eval of one adapter on `hm3d_v2_val36.json` by default (`EVAL_CHECKPOINT`, `EVAL_HAMLET=on|off`, or an explicit `EVAL_EPISODES`/legacy `EVAL_SUBSET`) — the way to compare checkpoints, since training-time success/SPL come from different random episodes each cycle. `eval_sweep.sh <specs> <episodes.json>` queues a list of such evals (≤2 in flight, the per-user QOS limit) and prints one summary line each. Valid labels for Delta's v2 val are `scene_<index 0-27>`; the `constants.episode_labels_table` subsets do not match that file. The gate's dummy episodes are fixed-length (`FixedLengthDummyEnv`) so it cannot fail on sampled actions — a 1-step episode would make verl's `masked_var` raise. Two Delta-only facts: `env.sh` exports `RAY_CONDA_HOME` because compute nodes have no `conda` on PATH (Ray's conda `runtime_env` shells out to it), and `run_rl_train.sh` uses `objectnav_hm3d_v2.yaml` (no semantic sensor) because the `vln` env there has numpy 2, under which habitat-lab's semantic-sensor gym Box overflows. `training.max_wallclock_hours` makes `train_rl` stop cleanly and write `checkpoints/checkpoint_final`; each cycle also appends to `<run>/progress.json`. `START_CHECKPOINT` performs a full state resume, preserves global cycle numbering, and new HAMLET checkpoints include `hamlet_reference.pt` so the frozen PPO reference survives restarts.

## Config system

Hydra + structured configs. Schemas are dataclasses in `src/longnav/config_schema.py` (`InferenceConfig` → `RLConfig` is the root); presets live in `src/longnav/config/<group>/*.yaml` and compose with `+group=name`:

`checkpoint` (which adapter to load) · `dataset` (habitat yaml + scene/episode paths) · `experiment` (eval vs train: output dir, wandb project, viz toggles) · `resources` (`single`/`quad`/`octo`, plus `l2`/`l3`/`l4` registered in code) · `training` (`hapo`/`rpp` RL hyperparameters).

Every preset yaml must start with `# @package _global_`. `register_configs()` also installs a `${read_text:<path>}` OmegaConf resolver, used to pull the system prompt out of `src/longnav/conf/prompts/`.

Config landmines:
- `resources.num_vlms` must divide `training.rl_config.n_rollout` (default 16) or **training hangs**. It must also be ≤ GPU count. `num_sims` should be `num_vlms + 1`.
- `resources.osm_gb` is the Ray object store size — tune to shared memory or you get spilling into `ray_object_spilling/`. Related trap: every postproc result is one plasma object (trajectory + packed embeds), and `ray.get` returns zero-copy views into it — holding any array from it (e.g. the trajectory for the `n_adv` baseline history) pins the whole object; `train_rl.py` deep-copies the history for that reason.
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
- **Context window** (`vlm.context_window`, default `None` = full episode): `_apply_context_window` evicts whole turns from the cache while pinning the prompt prefix so the goal survives. It cuts at the assistant header rather than the turn boundary, and trims the sparse embed DB alongside the cache — otherwise old frames the model can no longer see would still suppress new patches as "redundant" and confound the ablation.
- **Context window mode** (`vlm.context_window_mode`, default `evict`): eviction frees memory but does not remove information — a surviving token's layer≥1 K/V were computed from a residual stream that had attended over the whole episode, so evicted frames still reach the decision through them. `recompute` instead rebuilds the window's K/V against a cache holding only prefix + window, making the decision a strict function of what the agent can still see; the gap between the two modes is the leak. Both retire the same tokens on the same steps (`tests/ctx_recompute_smoke.py` pins this down) and retained turns keep their original absolute mRoPE positions, so the arms differ in one variable only. `recompute` replays stored post-ViT embeds rather than re-running the vision tower or the sparse filter, costs one window-sized text prefill per step, and refuses to run with attention viz. `reindex` (StreamingLLM-style, `utils/pre_rope.py`) evicts on the same schedule as `evict` but caches keys pre-rotation and applies RoPE at attention time from a per-slot position table, renumbering survivors to contiguous mRoPE positions — no positional hole across the cut; requires `use_sparse`, refuses attention viz and `save_outputs` (`tests/reindex_smoke.py` pins schedule parity, position contiguity, and bit-identical fidelity to stock when nothing is evicted).
- **Attention visualization** forces `attn_impl='eager'` automatically (sdpa/flash return no attention weights). `vlm.attn_weighting` selects what the heatmaps mean: `raw` α, `value_norm`/`wo_norm` (α weighted by the token's actual contribution to the residual stream), or `grad` (α · ∂score/∂α, requiring `visualize_attention_3d` and an extra forward+backward under `no_grad` instead of `inference_mode`).

- **Training replay mask** (`forward_embeds_core` in `utils/hamlet.py`, shared by the DDP training forward and the `old`/`ref` log-prob passes): the packed episode is replayed with `use_cache=False` and an explicit all-ones `attention_mask`. The mask is load-bearing — with neither a mask nor a cache (what gradient checkpointing forces in train mode) transformers ≥ 4.53 reads the non-monotonic mRoPE image positions as packed-sequence boundaries, the causal mask goes block-diagonal, and the training log-probs silently diverge from the rollout (`actor/ppo_kl` ≈ 1 at the very first step). `tests/hamlet_smoke.py` stage [2b] pins the train/eval replay equality.

Class chain: `VLMWorker` → `RolloutWorker` (+`EpisodeRolloutMixin`) → `RLWorker` (+`VLMTrainingMixin`) → `RLActor`.

### HAMLET (`vlm.hamlet`, `+hamlet=on`)

`utils/hamlet.py` ports HAMLET (arXiv 2510.00695): `n_moment` learnable **moment tokens** are spliced into every turn right after `<|vision_end|>` as placeholder ids above the tokenizer vocabulary (spare rows of the embedding matrix; tokenizer untouched), and a 2-layer block-causal **memory transformer** over the episode's moment hidden states produces a read-out that is added to the decision token's final hidden state *before* `lm_head` through a zero-init projection. Fusion is post-LM on purpose: the training replay is one causal forward over stored embeds, so feeding memory back into the input would need two passes and break the rollout/`old_logprobs` equivalence. Rollout (`_hamlet_readout`, read-out for the current block) and replay (`forward_embeds_core`, read-outs for all blocks) run the same per-block computation. The module lives on the base model as `model.hamlet` (attached in `load_model`, before PEFT) and rides in the adapter via `modules_to_save`; `disable_adapter()` therefore routes the ref-policy pass through its untrained zero copy, which is why `out_proj` must stay zero-init. Requires `use_sparse=True`; refuses `attn_weighting='grad'`. `tests/hamlet_smoke.py` pins the invariants (one moment block per decision, delta==0 at init, replay==rollout, ref==old at init, gradients reach the module, checkpoint round-trip, eviction keeps the moment history). The memory history lives outside the KV cache, so it survives `context_window` eviction — that combination is the intended eval ablation. Training is LoRA/PEFT via `utils/trainers.py`, with losses and advantage estimators pulled from the vendored `verl` submodule.

### Output layout

```
dump/<experiment output_dir>/<run_name>/
  config.yaml                    # resolved config, saved at bootstrap
  rollout/
    results_<pid>                # JSONL, one line per episode, one file per sim worker
    <scene_id>/<episode_label>/  # video.mp4, thumbnail.jpg, sequence.json, summary.json
```

Aggregating a run means globbing `rollout/results_*` and reading JSONL — see `tools/plot_context_window_ablation.py`.

## Submodules

`verl` (RL algorithms — advantage estimators, policy losses) and `ovon` are git submodules; `git submodule update --init --recursive`. `verl` is installed with `--no-dependencies` because its `numpy<2.0.0` pin conflicts with longnav's. habitat-lab is a gitignored clone at `.habitat-lab/`.

## Attention viz: the invariant to preserve

Anything indexed by absolute KV position — the probe's captured rows, its value-norm banks, the frame records' `abs_kv_idx` — must be cropped or shifted together when `_apply_context_window` evicts. They are read *after* `infer_step` returns, so a tensor left in pre-eviction index space silently points every surviving frame at the wrong key. `AttentionProbe.evict` handles the probe side; `_apply_context_window` handles the frame records. `tests/attn_evict_smoke.py` pins this down. See `LOOK_INTO.md` for the remaining caveats on how the heat videos should be read.
