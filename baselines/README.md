# Uni-NaVid inference and navigation benchmarks

This directory contains standalone, inference-only Uni-NaVid evaluation, teleoperation,
and multi-goal interfaces. Single-goal evaluation defaults to the 100 HM3D-v2 labels in
`dump/hm3d_v2_100_labels.json` (relative to the repository root). That manifest is a protected
local input; see the [restoration guide](../paper_results/REPRODUCIBILITY.md) for a fresh clone.
The harness does not import LongNav or install into the existing `vln` or `longnav_vlm` environments.
Paper launchers also support MP3D; the [study index](../paper_results/README.md) links their protocols.

## Layout and isolation

Paths in this list are relative to `baselines/`.

- `.envs/uninavid-hm3d`: relocated local Conda prefix.
- `.cache`: Conda, pip, Hugging Face, Torch, XDG, and temporary caches.
- `Uni-NaVid`: pinned editable upstream clone.
- `habitat-lab`: pinned editable Habitat-Lab clone.
- `uninavid_hm3d`: ObjectNav adapters, pruning, probes, and teleoperation.
- `results`: resumable per-episode outputs and aggregate summaries.

The large/generated paths are ignored by `baselines/.gitignore`. HM3D scenes
are referenced by symlink and are only read by the evaluator.

## Commands

Run these commands from the repository root. The wrapper selects the isolated runtime.

```bash
bash baselines/setup.sh
bash baselines/run.sh download
bash baselines/run.sh validate
bash baselines/run.sh model-check
bash baselines/run.sh smoke
bash baselines/run.sh full
```

Interactively drive one single-object HM3D-v2 episode and then press Space to
hand control permanently to Uni-NaVid with:

```bash
bash baselines/run.sh teleop --episode-index 0
```

Use `--seed N` (default `30`) for a reproducible teleop/model run. The seed is
applied to Habitat and independently to each Uni-NaVid generation call, so
stochastic sampling cannot be shifted by unrelated CUDA random-number use. To
reproduce the actions, keep the same episode, seed, inference options, hardware
and software environment, and enter the same manual key sequence before Space.

The controls are `W/A/D/X` for forward/left/right/stop, Space for model
handoff, and `Q` to abort. During teleoperation, Uni-NaVid runs a fresh preview
on every observation and retains the visual history of the manually driven
trajectory. Its normal buffered-action behavior resumes after handoff.

The live view and video place the annotated RGB observation beside Habitat's
top-down map with goal instances and the agent trajectory. They and the JSON
trace are written under `baselines/results/uninavid_teleop/<index>_<episode-label>/` as `current.png`,
`episode.mp4`, and `trace.json`. Attention heat maps are intentionally disabled:
Uni-NaVid's mixed 2x2, 8x8, and similarity-merged visual tokens do not retain the
spatial provenance required by LongNav's 3D attention renderer.

The multi-goal commands target the original 236-episode OneMap sequential multi-object
benchmark. **The default assets are missing in this checkout:**
`src/longnav/conf/episode_jsons/onemap_multi.json` and
`habitat_configs/objectnav_hm3d_multi.yaml`. Restore those original assets before running
these commands, or supply their actual locations using `--labels` and `--config`
(absolute paths). The interface tests do not replace this benchmark integration check.

```bash
bash baselines/run.sh multi-validate
bash baselines/run.sh multi-smoke
bash baselines/run.sh multi-full
bash baselines/run.sh multi-all-goals-smoke
bash baselines/run.sh multi-all-goals-full
```

The full multi-object run is resumable and writes to
`baselines/results/onemap_multi_uninavid_sequential`. Goals are disclosed one at a time.
When the model successfully stops at an intermediate goal, the evaluator keeps
the visual history, discards any buffered future action, and appends the new task
to the text context. Model actions are never added to that context. Stop actions
are scored as emitted: no false-positive or false-negative oracle stop guard and
no post-switch stop veto are used.

The `multi-all-goals-*` modes instead put the complete ordered sequence in the
initial task (for example, `Find chair, then plant, then bed, in that order.`),
while retaining the same visual memory, task updates, and no-oracle stopping.
Their full results are written to
`baselines/results/onemap_multi_uninavid_all_goals`.


The full command uses `--resume`; completed episode JSON files are skipped.
Pass evaluator overrides after the command, for example:

```bash
bash baselines/run.sh smoke --temperature 0.1 --actions-per-inference 1
```

Context-window and extreme ablations (`--context-frames K`, `--no-visual-tokens`,
`--current-frame-only-64`) are driven from the paper scripts: `paper/ctx_ablation/run_uninavid.sh`
sweeps K on the 100-episode manifest with the official decoding, and the flags can be passed to
`baselines/run.sh eval` directly. `--current-frame-only-64` removes the current frame's
duplicate four-token history representation while retaining its 64 detailed tokens; the
leave-one-frame-out influence probes (`--frame-influence`, `paper/frame_influence/run_uninavid.sh`)
are documented in `paper/frame_influence/README.md`. Every evaluation can be resumed by
rerunning its command with `--resume`.

## Modifying inference

Uni-NaVid is installed editable, so changes under `baselines/Uni-NaVid/uninavid`
take effect immediately. The observation-history cache and online token merging
are implemented in `uninavid/model/uninavid_arch.py`; decoding and action
buffering are controlled by [uninavid_hm3d/eval.py](uninavid_hm3d/eval.py). No training runtime is required.

Retained pruning methods are `identity`, `random`, `fastv`, `pyramiddrop`, `divprune`,
`dart`, `hybrid`, `uniform`, and the whole-frame `kl_keep_one`, `entropy_keep_one` and
`recency` (`--prune-method`; the whole-frame ones need `--prune-layer 0`). Budget scope is `last_stage` or `model`;
`uniform` requires `model`. Use the [windowed/flat pruning protocol](../paper/token_prune/README.md)
or [hybrid/uniform/PyramidDrop protocol](../paper/hybrid_prune/README.md) for compatible
windows, budgets, and stages. [prune.py](uninavid_hm3d/prune.py) owns these mechanisms;
the [frame-influence](../paper/frame_influence/README.md),
[horizon](../paper/info_horizon/README.md), and [selector-KL](../paper/selector_kl/README.md)
protocols document the retained diagnostic probes.

Each run records the source revisions, dependency versions, label checksum,
effective inference settings, raw model outputs, chosen actions, timing, and
Habitat metrics in its result directory.

Per-step trace (`traces/<label>.jsonl`, one line per env step). On inference
steps (`inference: true`) the record also carries the cost of that LLM call, added
for the cost benchmark of `paper/hybrid_prune` (2026-09-23): `inference_seconds`
(wall time of `model.generate`), `prompt_tokens` (the tokens the LLM prefilled —
instruction plus the visual history kept — read by a forward pre-hook on decoder
layer 0 that keeps the *last* multi-token forward of the call, so a pruning
pre-pass over the unpruned prompt is not what is counted), `cuda_peak_bytes`
(`torch.cuda.max_memory_allocated()` over the call, after a reset) and
`cuda_allocated_bytes` (resident after the call: weights + the per-frame feature
cache). Older runs can lack these fields; missing telemetry is not a measured zero.

## LightNav-0 baseline

[LightNav-0](https://github.com/lightorigins/LightNav-0) (Qwen3-VL-4B with RVQ action tokens,
Apache-2.0) is evaluated with its own, unmodified Habitat pipeline on the same 100 HM3D-v2
labels. The upstream clone is pinned at `3015508b` and nothing in it is edited; the glue under
`lightnav_hm3d/` only prepares the episode subset and converts the outputs. Nothing is
installed into `vln`, `longnav_vlm`, or the Uni-NaVid prefix.

Layout (relative to `baselines/`, all ignored by Git except `lightnav_hm3d/`, the scripts
and `tests/test_lightnav*.py`):

- `.envs/lightnav-habitat`: Python 3.9 + habitat-sim 0.3.1 + habitat-lab 0.3.20231024, the
  env-server stack upstream verified (`habitat_server/environment.yml`).
- `.envs/lightnav`: Python 3.11 + torch 2.10 / transformers 5.8.0 / vLLM 0.19.1 for the GPU client.
- `LightNav-0`: pinned editable upstream clone; the checkpoint
  (`LightOriginsHQ/LightNav-0`, 9.7 GB, ships `eval_config.json` + `action_tokenizer/`) is
  downloaded to `LightNav-0/checkpoints/LightNav-0`.
- `.cache/datasets/objectnav_hm3d_v2_100`: filtered copy of the HM3D-v2 val dataset
  (stub + one `content/` shard per scene) holding exactly the 100 labelled episodes.
  Upstream's server has no episode filter, so it is pointed at this copy with
  `--data-path`. The labels name the k-th episode of each scene shard (habitat-lab renumbers
  episode ids on load); the generator rewrites each kept episode's id to that index, which
  the loader then reports as `raw_episode_id`, so every result maps back to its label.
- `results/lightnav_hm3d_v2_100[_native]`: `raw/` holds LightNav's own `results.jsonl`,
  `summary.json` and the server/client logs; `episodes/<label>.json`, `summary.json` and
  `run.json` are the converted Uni-NaVid-layout records read by `longnav.analysis.results_io`.

```bash
bash baselines/setup_lightnav.sh
bash baselines/run_lightnav.sh download
bash baselines/run_lightnav.sh validate      # boots the env server on the subset and probes it
bash baselines/run_lightnav.sh smoke         # 1 episode, comparable protocol (+ overlay video)
bash baselines/run_lightnav.sh full          # 100 episodes, success radius 1.0 m
bash baselines/run_lightnav.sh full-native   # 100 episodes, LightNav's published protocol
python3 -m unittest baselines.tests.test_lightnav
```

Each run starts upstream's ZMQ env server (`lightnav_habitat.serve --task objectnav`, 480x270
RGB at 120 deg hfov, 0.88 m camera, `velocity_control` with 0.25 m / 30 deg per step, 500-step
budget, episodes in dataset order) and upstream's client (`lightnav.cli.eval_habitat`, greedy
decoding, the checkpoint's own history window and RVQ decoder, an explicit STOP on the last
budget step, after which Habitat scores the final pose; the paper aggregators re-score these
`forced_stop` episodes as failures via `longnav.analysis.results_io.load_lightnav_run`, since
the LongNav-R1 and Uni-NaVid harnesses cannot succeed at the cap). The two variants differ only
in server flags: `full` passes
`--success-distance 1.0` on the shipped navmesh, matching the Uni-NaVid and LongNav
protocol in this repository; `full-native` keeps LightNav's default 0.1 m radius and
`--navmesh-cell-height 0.05`, the setting its paper numbers use (HM3D-v2 goal view points sit
0.05-0.15 m below the shipped navmesh, which floors `distance_to_goal`). The camera and
action space are the model's and are not changed for either variant, so LightNav sees a
different view of the same episodes than the discrete-action 640x480 / 79 deg agent used by
Uni-NaVid. `results.jsonl` is append-only, so a run directory is never reused; `BACKEND=hf`
switches the client off vLLM, `GPU_MEM_UTIL` caps the vLLM fraction (the simulator renders
on the same GPU). `run.json` records both command lines, the upstream and checkpoint
revisions, package versions, the label checksum and the server-log counts the launcher checks
(navmesh re-bakes, server errors).

Two launcher knobs serve the paper's context-window ablation (`paper/ctx_ablation/`):
`RUN_DIR=<abs dir>` writes the run somewhere other than `results/$RUN_NAME`, and
`CONTEXT_FRAMES=K` runs the client through `lightnav_hm3d/eval_ctx.py` instead of
`lightnav.cli.eval_habitat`. The wrapper cuts the visual history to the newest `K` observed
frames (current frame included) and hands every other argument to upstream. LightNav-0 keeps
the whole episode and samples it by age with the checkpoint's SlowFast tiers
(`--num_history_frames` does not shorten that buffer for SlowFast checkpoints), so the wrapper
rebinds `lightnav.inference.samples.slowfast_video_segments` in-process: with more than `K`
frames observed it builds the segment layout upstream uses at step `K-1` of a fresh episode,
shifts the frame ids by the number of dropped frames and indexes the full buffer. The frame
timestamps are age-relative and the ViT cache is keyed by absolute frame id, so the model
input equals that of an episode that started `K` steps ago (the clone is still not edited;
`baselines/tests/test_lightnav_ctx.py` checks the equivalence). `run.json` records
`context_frames` and `client_entry`.

A third knob serves the information-horizon experiment (`paper/info_horizon/`):
`BLIND_LAYER=L` runs the client through `lightnav_hm3d/eval_blind.py`, which hides every
`<|video_pad|>` token from decoder layer `L` on (history and current frame; vision markers and
timestamps stay visible). LightNav decodes inside vLLM, whose FlashAttention kernel takes no
mask and whose compiled CUDA-graph decode runs no Python hook, so the wrapper forces eager vLLM
(`VLN_VLLM_ENFORCE_EAGER=1`) and replaces `FlashAttentionImpl.forward` on every decoder layer
with an exact masked attention (SDPA over the K/V gathered from the paged cache); layers `>= L`
additionally drop the visual columns. `L = 36` (the model's depth) hides nothing and is the
matched reference for the blind arms, since the kernel path differs from the unmodified run.
Before each `LLM.generate` the wrapper records the prompt's visual positions and afterwards
checks the masked kernel ran on every layer of every forward. `BLIND_LAYER` and
`CONTEXT_FRAMES` are not combined. `run.json` records `blind_layer` and `enforce_eager`;
`baselines/tests/test_lightnav_blind.py` checks the kernel and the hook offline.

A fourth knob serves the flat token-prune study (`paper/token_prune/`, "Flat variant"):
`PRUNE_METHOD=M PRUNE_K=k` (M in `random|fastv|divprune|dart`) runs the client through
`lightnav_hm3d/eval_prune.py`, which keeps `k x 14` of the prompt's history visual tokens under
rule M and hides the rest from every query on every layer, on the same eager masked-attention
kernel as `BLIND_LAYER` (the information-horizon reference `lightnav_ref` is therefore its
unpruned reference). Deletions are permanent through a per-episode ledger keyed by tubelet
identity `(frame0, frame1, pool, cell)`; the current 2-frame tubelet is exempt. FastV and DART
score the prompt in a pre-pass (`generate` with one token, ledger hidden) that captures the
last prompt token's layer-0 attention row, the layer-0 keys and the layer-1 input, then the real
generate runs with the dropped columns hidden; random and DivPrune (LLM input embeddings) need no
pre-pass. One record per inference lands in `raw/prune/episode_NNN.jsonl`, which the converter
moves to `prune/<label>.jsonl`. Not combined with the other two wrappers. `run.json` records
`prune_method`, `prune_k` and the wrapper's `meta.json`; `baselines/tests/test_lightnav_prune.py`
checks the layout, the ledger, the selection rules and the hooks offline.

`lightnav_hm3d/serve_navverse.py` serves the same model and wrappers over HTTP for a simulator
that is not Habitat: it speaks the `/act` contract of `longnav.scripts.serve` (one JPEG per step,
the goal on the first step, a discrete `action_id` back), so NavVerse's LongNav client drives
LightNav-0 on its Isaac Sim benchmark (`paper/ctx_ablation/run_navverse_lightnav.sh`). It
discretizes LightNav's 1 s velocity step into STOP / FORWARD / TURN (Habitat's stop guard, then the
larger normalized component), feeds frames at their native aspect ratio (`--aspect-mode keep`;
640x480 -> 384x288, 30 tokens per pooled history pair, so `--prune-tokens-per-frame 15`), and can
install the pruning wrapper (`--prune-method ... --prune-k ...`) before the engine is built.
`baselines/tests/test_lightnav_serve_navverse.py` checks the rule, the contract and the step-log
merge offline.
