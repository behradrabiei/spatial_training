# Archived paper studies

Studies that are not used in the paper, moved here from `paper_results/` to keep that directory
to what the draft needs. Only the run data is kept; tables and figures are rebuilt on demand next
to the runs (`<study>/results/`, `<study>/figures/`, git-ignored). The run files are still listed,
with their original checksums, in `paper_results/protected_inputs.json`.

| Directory | Study | Rebuild |
|---|---|---|
| `frame_influence/` | [Leave-one-frame-out influence](../paper/frame_influence/README.md) | `aggregate.py` / `plot.py --measure leave_one` |
| `frame_influence_keepone/` | [Keep-one-in influence](../paper/frame_influence/README.md) | `--measure keep_one`; `--measure keep_one_full` writes `frame_influence_keepone_full/` from the same runs |

```bash
for m in leave_one keep_one keep_one_full; do
  conda run -n longnav_vlm python paper/frame_influence/aggregate.py --measure $m
  conda run -n vln python paper/frame_influence/plot.py --measure $m
done
```

`paper_results/frame_influence_recent/` (the variant the paper uses) reads these runs for its
reference checks, so they must stay at these paths.

## Deleted: windowed token pruning (`token_prune/`)

The [windowed visual-token pruning study](../paper/token_prune/README.md) (32-frame window; the
paper uses `paper_results/token_prune_flat/`) was deleted from the working tree on 2026-11-21, runs
included. No other study reads it. Its 12,648 checksummed run files (configs, episode records,
pruning records and the LongNav-R1 `sequence.json` traces) remain in the local backup and in
`paper_results/protected_inputs.json`; videos, images and logs are gone. To bring it back:

```bash
cp -r .cleanup_archive/before-codebase-cleanup/protected-inputs/paper_results_archive/token_prune paper_results_archive/
conda run -n longnav_vlm python paper/token_prune/aggregate.py --strict   # ~2 min, also reads ctx_ablation/results/per_episode.csv
conda run -n vln python paper/token_prune/plot.py
```

The tables rebuilt this way were checked byte-identical to the deleted ones before the deletion.
