# Protected paper inputs

`protected_inputs.json` records the baseline commit, retained studies, run episode labels,
reproduction commands, file sizes, and SHA-256 checksums. It includes ignored scientific
inputs, shared reference runs, episode manifests, and the keep-one diagnostic in `dump/`.
It is a frozen baseline inventory, not a list of files safe to delete.

Before the cleanup, all listed files were copied to
`.cleanup_archive/before-codebase-cleanup/protected-inputs/`, with their original relative
paths. The backup is local and ignored by Git. Keep that directory when moving this project;
a Git clone alone does not contain the large traces, probe arrays, or all showcase images.
Restore a missing input by copying `protected-inputs/<path>` to `<path>` in the checkout.

Verify originals and backup:

```bash
python tools/paper_inventory.py verify
python tools/paper_inventory.py verify --backup
```

Rebuild in an isolated mirror without overwriting published results:

```bash
python tools/reproduce_paper.py --name reproduction-check
```

This uses the existing `longnav_vlm` and `vln` environments. The named validation directory
must not already exist. Command logs and exit statuses are saved beside the rebuilt outputs
under `.cleanup_archive/before-codebase-cleanup/validation/`. Missing inputs or a failed
command must be resolved before treating those outputs as reproducible.

Model checkpoints, Habitat datasets, external runtime clones, and Conda environments are
not part of this backup. Their existing locations are unchanged by the cleanup.

The manifest also lists older showcase and alternate cost-plot commands; the reproduction runner
executes them after their aggregations. Compare two mirrors with the model environment:

```bash
python tools/compare_paper_outputs.py \
  .cleanup_archive/before-codebase-cleanup/validation/baseline \
  .cleanup_archive/before-codebase-cleanup/validation/cleaned \
  --require-rebuilt --report /tmp/paper-comparison.json
```

`--require-rebuilt` rejects outputs that were only copied into the mirrors. Numerical tables and
arrays must agree exactly. PNGs are compared pixel-for-pixel, and PDFs are compared with only
creation/modification timestamps removed. The older cost benchmark's historical published plot
style is preserved in the original files; the baseline and cleaned code both rebuild using the
plotting behavior at `bf4e65ff`.

Local archive restoration for confirmed obsolete artifacts is documented in [../CLEANUP.md](../CLEANUP.md).
