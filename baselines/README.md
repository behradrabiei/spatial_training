# Uni-NaVid HM3D-v2 baseline

This directory contains a standalone, inference-only Uni-NaVid evaluation on
the 100 labels in `../dump/hm3d_v2_100_labels.json`. It does not import LongNav
or install anything into the existing `vln` or `longnav_vlm` environments.

## Layout and isolation

- `.envs/uninavid-hm3d`: relocated local Conda prefix.
- `.cache`: Conda, pip, Hugging Face, Torch, XDG, and temporary caches.
- `Uni-NaVid`: pinned editable upstream clone.
- `habitat-lab`: pinned editable Habitat-Lab clone.
- `uninavid_hm3d`: HM3D ObjectNav adapter.
- `results`: resumable per-episode outputs and aggregate summaries.

The large/generated paths are ignored by `baselines/.gitignore`. HM3D scenes
are referenced by symlink and are only read by the evaluator.

## Commands

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
trace are written under `results/uninavid_teleop/<index>_<episode-label>/` as `current.png`,
`episode.mp4`, and `trace.json`. Attention heat maps are intentionally disabled:
Uni-NaVid's mixed 2x2, 8x8, and similarity-merged visual tokens do not retain the
spatial provenance required by LongNav's 3D attention renderer.

Run Uni-NaVid on the 236-episode OneMap sequential multi-object benchmark with:

```bash
bash baselines/run.sh multi-validate
bash baselines/run.sh multi-smoke
bash baselines/run.sh multi-full
bash baselines/run.sh multi-all-goals-smoke
bash baselines/run.sh multi-all-goals-full
```

The full multi-object run is resumable and writes to
`results/onemap_multi_uninavid_sequential`. Goals are disclosed one at a time.
When the model successfully stops at an intermediate goal, the evaluator keeps
the visual history, discards any buffered future action, and appends the new task
to the text context. Model actions are never added to that context. Stop actions
are scored as emitted: no false-positive or false-negative oracle stop guard and
no post-switch stop veto are used.

The `multi-all-goals-*` modes instead put the complete ordered sequence in the
initial task (for example, `Find chair, then plant, then bed, in that order.`),
while retaining the same visual memory, task updates, and no-oracle stopping.
Their full results are written to
`results/onemap_multi_uninavid_all_goals`.


The full command uses `--resume`; completed episode JSON files are skipped.
Pass evaluator overrides after the command, for example:

```bash
bash baselines/run.sh smoke --temperature 0.1 --actions-per-inference 1
```

Context-window and extreme ablations (`--context-frames K`, `--no-visual-tokens`,
`--current-frame-only-64`) are driven from the paper scripts: `paper/ctx_ablation/run_uninavid.sh`
sweeps K on the 100-episode manifest with the official decoding, and the flags can be passed to
`bash baselines/run.sh eval ...` directly. `--current-frame-only-64` removes the current frame's
duplicate four-token history representation while retaining its 64 detailed tokens; the
leave-one-frame-out influence probes (`--frame-influence`, `paper/frame_influence/run_uninavid.sh`)
are documented in `paper/frame_influence/README.md`. Every evaluation can be resumed by
rerunning its command with `--resume`.

## Modifying inference

Uni-NaVid is installed editable, so changes under `baselines/Uni-NaVid/uninavid`
take effect immediately. The observation-history cache and online token merging
are implemented in `uninavid/model/uninavid_arch.py`; decoding and action
buffering are controlled by `uninavid_hm3d/eval.py`. No training packages are
installed or required.

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
cache). Runs from before that date have `null` in the three new fields.
