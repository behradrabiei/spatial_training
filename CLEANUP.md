# Inference cleanup and reproducibility

Baseline: `bf4e65ff` (`before codebase cleanup`). Implementation branch:
`codex/inference-cleanup`. Changes are ordinary commits; Git history was not rewritten.

## Scope

Evaluation, serving, and paper probes now share `InferenceWorker` and the checkpoint loader.
Pretrained base/SFT/LongNav-R1 checkpoints, PEFT loading, adapter merge timing, and attention hooks
are preserved. SFT/RL/DAgger entrypoints, training configs, losses, optimizers, value heads,
training-only packing/data builders, and the `verl` dependency were removed. The existing model,
Habitat, and Uni-NaVid environments and cached weights were not changed.

The VLM worker is split into inference, context/cache, pruning, and diagnostic components. Retained
selectors are random, FastV, DivPrune, DART, PyramidDrop, hybrid, and uniform. Both budget definitions
remain because the windowed paper studies use `last_stage`. Removed methods include stratified,
legacy diversity/grid/voxel selection, KL/keep-one selection, Fisher, cache merging, and whole-turn
budget pruning. KL/keep-one probes, gradient attention attribution, BEV positions, sparse filtering,
Habitat metrics, stopping, visualization, and Uni-NaVid interfaces remain.

The server initializes at application startup and uses the correct prompt location. Evaluation keeps
one collection queue until its finite split ends, avoiding loss of prefetched episodes between
batches; pending log flushes finish before shutdown. Saved checkpoint configs migrate through a
small compatibility layer, with explicit errors for conflicts or active removed options.

## Data protection and archival

`paper_results/protected_inputs.json` inventories 54,321 scientific files (7.09 GiB), study commands,
run episode membership/counts, configurations, shared references, outputs, and SHA-256 checksums.
All protected inputs remain at their original paths and have independent checksummed local copies.
This includes the keep-one horizon data in `dump/` and Uni-NaVid's default label manifest.

13,318 confirmed obsolete files (5.29 GiB) are under
`.cleanup_archive/before-codebase-cleanup/obsolete/`, preserving their original relative paths:

- Superseded hybrid `*_b<k>` runs, aborted `*_m<k>` runs, and last-stage report outputs.
- Removed-selector sweeps, training outputs/code, exploratory notebooks, and unused launchers.
- Inactive paper launch/driver logs and locks; scientific configuration and provenance remain.

Earlier cost benchmarks are protected. Unclassified `dump/`, `outputs/`, and `wandb/` artifacts,
datasets, environments, model caches, baseline clones, and the OVON runtime were left in place.
The archive is local and ignored, so it does not reduce local disk usage. It removes obsolete files
from active directories and future Git snapshots.

## Validation

The compact validation summary is in `cleanup_validation.json`. After validation, the one-off
parity/comparison scripts, baseline source copy, generated paper mirrors, recorded test outputs,
and test logs were removed. Those scripts remain recoverable from commit `52b21afd`. Scientific
backups, obsolete-artifact archives, and their restoration manifests remain intact.

- Real cached LongNav-R1 adapter, 12 configurations × 2 repeats × 8 fixed image inputs: exact action,
  cache-length, retained-slot identity, and probability matches against `bf4e65ff` (maximum difference
  0.0; comparison floor 1e-7, informed by repeated baseline runs). Includes window boundaries,
  recomputation/reindexing, every retained selector, staged protocols, and visual blindness.
- Real HM3D/Ray episode `4ok3usBNeis_0`: 48 steps, success 1.0, SPL 0.7537797097049764; matching actions,
  probabilities, rewards, and Habitat metrics. A separate two-simulator check verifies finite shard
  scheduling, episode reset, and log flushing for three episodes.
- 18 retained check scripts passed: configuration/API/scheduling, context/cache behavior, staged
  pruning, attention and gradient attribution, KL replay, token information, horizon, and selector
  diagnostics. The selector probes change live decisions by 0.0 in their regression check.
- All 116 Uni-NaVid unit tests passed, covering its selectors, buffering, teleoperation, and multi-goal
  logic. A real baseline/current GPU comparison of `4ok3usBNeis_0` also matches every buffered
  action, generated decision, and Habitat metric: 55 steps, 28 inference calls, success 1.0,
  SPL 0.6389751896037087. The baseline runtime and vendored model implementation are unchanged.
- All 32 reproduction commands passed. Rebuilt numerical outputs and figures are compared in
  temporary mirrors, including episode
  membership: 50 CSVs, 10 TeX tables, 5 Parquet datasets, 8 JSON files, 2 compressed records,
  and 90 PNG/PDF figure pairs all match. The comparison also verifies that these outputs were
  regenerated, not merely copied. PNG comparisons are pixel-exact; PDF comparisons ignore only export timestamps.
  Representative protocol and token-information figures were visually inspected.
- All 48 retained hybrid/uniform/PyramidDrop arms across HM3D and MP3D have 100/100 episodes.
  All protected originals and backups pass checksum verification after archival.
- Maintained configuration composition and 20 shell launchers pass their checks. No imports of
  removed owned modules remain; inference tests reject `verl`/TRL imports and removed options.

The default multi-goal files `src/longnav/conf/episode_jsons/onemap_multi.json` and
`habitat_configs/objectnav_hm3d_multi.yaml` were already missing at baseline. Real multi-goal
integration requires those original assets; the unit tests do not replace that benchmark.

## Restoration

Keep the entire ignored archive when relocating this project; a Git clone alone is insufficient for
large paper inputs. See `paper_results/REPRODUCIBILITY.md` for scientific-input restoration.

Archive batches are inventoried in `selectors-archive.json`, `remaining-training-archive.json`,
`obsolete-results-archive.json`, and `inactive-paper-logs-archive.json`. `complete-archive.json` also
covers the earlier training/`verl` removals. Each records original paths, sizes, and checksums.

```bash
# Example: restore superseded runs and exploratory artifacts from their local archive.
python tools/archive_cleanup.py restore --name obsolete-results
```

Restoration verifies archive checksums first, refuses to overwrite differing current files, and
keeps the backup. To restore individual items, copy them from `obsolete/<original-relative-path>`.
Use the baseline commit for original source implementations. If restoring the former `verl`
submodule checkout, also restore its Git registration from that commit; its archived `.git` pointer
is relative to its original location. No external dataset or weight download is part of restoration.
