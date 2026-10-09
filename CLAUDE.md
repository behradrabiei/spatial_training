# Working in LongNav

This is an inference and paper-reproduction repository. Read README.md for installation and
CLEANUP.md for cleanup evidence. Do not reintroduce training dependencies.

## Runtime boundaries

`longnav_vlm` runs model inference, Ray drivers, serving, and numerical aggregation. `vln` runs
Habitat workers and plotting. Uni-NaVid uses its isolated runtime under baselines/.envs; do not
upgrade or edit vendored model implementations as part of unrelated work. Both environments must
have matching Ray versions. Never change cached checkpoint weights.

Evaluation, serving, and in-process probes use InferenceWorker and utils/checkpoints.py. Legacy
checkpoint configuration is normalized once at bootstrap. VLMWorker combines context/cache,
pruning, and diagnostic mixins. Keep cache updates, hook order, tensor operations, and random draws
stable during refactoring: they affect navigation decisions.

## Scientific inputs

paper_results/protected_inputs.json maps retained studies and files to SHA-256 checksums at
bf4e65ff. Scientific inputs stay at their canonical paths, including
`dump/longnav_eval/keep_one_influence_w32_b10000000` and `dump/hm3d_v2_100_labels.json`.
Large traces, arrays, and showcase images can be ignored by Git while still being required.
Consult the manifest before deleting or moving anything. Rebuild outputs with
`tools/reproduce_paper.py` in an isolated mirror, never over the original results.

Confirmed obsolete files have local copies under `.cleanup_archive/before-codebase-cleanup/obsolete`.
Archive manifests record original paths and checksums. Do not sweep-delete dump/, outputs/,
wandb/, datasets, environments, or baseline runtime clones. Unclassified artifacts remain in place.

## Inference invariants

Retain random, FastV (attn), DivPrune, DART, PyramidDrop, hybrid, uniform, visual blindness,
context eviction/recomputation/reindexing, and both last_stage/model budgets. KL/keep-one replay
is a paper diagnostic, not a selection policy. Gradient attribution intentionally uses backward;
it is not training. World-coordinate utilities still support BEV inference and visualization.

Preserve Habitat measures, stop guards, sparse filtering, episode ordering, log flushing, Uni-NaVid
action buffering, teleoperation handoff, and multi-goal transitions. Missing multi-goal benchmark
assets must be supplied from the original source, not replaced with a new benchmark.

## Checks

`tests/inference_unit.py` covers configuration migration, rejected old options, server lifecycle,
and scheduling. `tests/run_inference_checks.py` runs retained GPU/cache/probe tests serially.
`tests/inference_parity.py` supports a baseline src tree through PYTHONPATH and compares action
probabilities, cache lengths, and retained-slot identities. `tests/habitat_parity.py` exercises a
canonical episode with Ray and Habitat. Uni-NaVid tests live under baselines/tests.

Use `tools/paper_inventory.py verify` and `verify --backup` to audit inputs. Compare rebuilt tables,
arrays, and figures using `tools/compare_paper_outputs.py`. Tests and rebuilds must write to temporary
or ignored validation locations, not the protected paper directories.
