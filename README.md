# LongNav

Inference and paper reproduction for LongNav-R1 and Uni-NaVid navigation models.
Pretrained SFT and RL-derived checkpoints remain supported; this repository does not train models.

## Installation

Keep model inference and Habitat in separate environments. The tested local setup is recorded in
[SETUP_NOTES.md](SETUP_NOTES.md); cleanup did not upgrade that stack.

In `longnav_vlm` (Python 3.10):

```bash
pip install -e .
```

SDPA is the default attention implementation. FlashAttention is optional. PEFT is required for
pretrained adapter loading. No `verl`, TRL, optimizer, or training runtime is required.

Install Habitat Lab / Habitat Sim in `vln`, then register the inference package without changing
Habitat's pinned dependencies:

```bash
pip install 'ray[default]==2.53.0'
pip install --no-deps -e .
```

Use the same Ray version in both environments. Dataset paths are configured by
`src/longnav/config/dataset/`; `sim.workspace` is the root from which Habitat resolves relative paths.
Uni-NaVid has its own runtime; see [baselines/README.md](baselines/README.md).

## Evaluation

Run from the repository root with `longnav_vlm` active:

```bash
python -m longnav.scripts.eval +checkpoint=longnav +dataset=hm3d_v2_val \
  +experiment=paper_ctx +resources=single task.run_name=my_eval \
  task.wandb_project=null resources.osm_gb=10
```

`checkpoint=base`, `sft`, and `longnav` select the maintained pretrained models. Adapter settings live
under `adapter.path` and `adapter.peft_config`. Historical `training.checkpoint` and PEFT fields are
translated at bootstrap; conflicting settings raise an error. Historical optimizer/loss fields are
ignored only when reading a saved configuration. Removed pruning policies and active removed knobs
raise errors.

Resource presets set the number of model/simulator actors and Ray object-store size. Adjust them to
available GPUs, CPUs, and shared memory. Each simulator flushes episode results before reuse.
`task.wandb_project=null` disables remote logging; results are still written locally.

## Serving and other interfaces

```bash
python -m longnav.scripts.serve
```

The FastAPI application loads the model at startup. POST an image to `/act` with `instr_or_goal` to
start or reset an episode; subsequent requests continue its cached context. A single server worker
owns one conversation. See [tools/client.py](tools/client.py) for the client interface.

Uni-NaVid evaluation, teleoperation, action buffering, and multi-goal interfaces are documented in
[baselines/README.md](baselines/README.md). Its default single-goal manifest is
`dump/hm3d_v2_100_labels.json`, a protected local input. The original multi-goal manifest and Habitat
configuration are missing in this checkout; see [CLEANUP.md](CLEANUP.md) before an integration run.

## Retained methods and paper results

Context eviction, recomputation, and reindexing are retained, as are random, FastV, DivPrune, DART,
PyramidDrop, hybrid, uniform, visual-blind controls, and paper diagnostics. Both whole-model and
last-stage budget calculations remain available. KL/keep-one diagnostics and gradient attention
attribution remain; they are independent of model training.

[paper_results/README.md](paper_results/README.md) links the study protocols.
[paper_results/REPRODUCIBILITY.md](paper_results/REPRODUCIBILITY.md) describes the checksummed input
manifest, local backups, isolated rebuilds, and restoration. Preserve raw traces and probe arrays,
not just summaries. Unknown artifacts remain in place; confirmed obsolete artifacts are in the
ignored `.cleanup_archive/before-codebase-cleanup/` directory.

## Code and validation

- `utils/checkpoints.py`: shared pretrained model and adapter loading.
- `utils/vlm_worker.py`: model inference and attention attribution.
- `utils/context_cache.py`: context windows, cache eviction, and recomputation.
- `utils/kv_prune.py`, `utils/pruning_runtime.py`: selectors and staged pruning.
- `utils/diagnostics.py`: replay-based paper diagnostics.
- `utils/rollout_core.py`: episode collection and Ray scheduling.
- `env/`: Habitat metrics, stopping, rendering, and simulator interfaces.

```bash
python tests/inference_unit.py
python tests/run_inference_checks.py --output /tmp/longnav-checks
python tests/eval_smoke.py
python tools/paper_inventory.py verify
python tools/reproduce_paper.py --name my-rebuild
```

The full smoke suite needs cached model weights and a GPU. One-off cleanup comparison scripts
and generated validation artifacts have been removed; their source remains in Git history.
See [CLEANUP.md](CLEANUP.md) for measured validation results and remaining integration limitations.
