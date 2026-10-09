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
executes them after their aggregations. The completed cleanup comparison matched numerical tables
and arrays exactly, PNGs pixel-for-pixel, and PDFs with only export timestamps removed. Its
one-off comparison script is recoverable from commit `52b21afd`; the temporary mirrors and logs
were removed after recording the validation summary in `cleanup_validation.json`.

Future rebuilds create disposable output under `validation/<name>/`. After reviewing the results,
remove that named mirror when no longer needed. Keep `protected-inputs/`, `obsolete/`, and the
archive manifests: those hold the scientific backups and restoration data.

The older cost benchmark's historical published plot style is preserved in the original files;
the maintained code rebuilds using the plotting behavior at `bf4e65ff`.

Local archive restoration for confirmed obsolete artifacts is documented in [../CLEANUP.md](../CLEANUP.md).
