# Archived paper studies

Studies that are not used in the paper, moved here from `paper_results/` to keep that directory
to what the draft needs. Only the run data is kept; tables and figures are rebuilt on demand next
to the runs (`<study>/results/`, `<study>/figures/`, git-ignored). The run files are still listed,
with their original checksums, in `paper_results/protected_inputs.json` (except `ctx_ablation_navverse/`, which
was never part of that manifest or of the backup).

| Directory | Study | Rebuild |
|---|---|---|
| `frame_influence/` | [Leave-one-frame-out influence](../paper/frame_influence/README.md) | `aggregate.py` / `plot.py --measure leave_one` |
| `frame_influence_keepone/` | [Keep-one-in influence](../paper/frame_influence/README.md) | `--measure keep_one`; `--measure keep_one_full` writes `frame_influence_keepone_full/` from the same runs |
| `protocol_mp3d/` | [MP3D replicate of the pruning-protocol study](../paper/protocol/README.md) (was `paper_results/hybrid_prune_mp3d/`), archived 2026-11-22 | `paper/protocol/aggregate.py --dataset mp3d --strict` / `plot.py --dataset mp3d` |
| `info_horizon_mp3d/` | [MP3D replicate of the information-horizon study](../paper/info_horizon/README.md), archived 2026-11-23 | `paper/info_horizon/aggregate.py --dataset mp3d` / `plot.py --dataset mp3d` |
| `ctx_ablation_navverse/` | [NavVerse indoor replicate of the context-window ablation](../paper/ctx_ablation/README.md) (LongNav-R1 and LightNav-0), archived 2026-11-23 | `paper/ctx_ablation/aggregate.py --dataset navverse` / `plot.py --dataset navverse` |

```bash
for m in leave_one keep_one keep_one_full; do
  conda run -n longnav_vlm python paper/frame_influence/aggregate.py --measure $m
  conda run -n vln python paper/frame_influence/plot.py --measure $m
done
```

`paper_results/frame_influence_recent/` (the variant the paper uses) reads the frame-influence runs for its
reference checks, so they must stay at these paths.

```bash
conda run -n longnav_vlm python paper/protocol/aggregate.py --dataset mp3d --strict
conda run -n vln python paper/protocol/plot.py --dataset mp3d   # add --diagnostics --models LongNav-R1 Uni-NaVid --protocols pyramiddrop uniform hybrid for the old figure set
```

`protocol_mp3d/` was moved with its `results/` and `figures/` as they stood (local, git-ignored here; the
tables rebuild byte-identically, the figures are the pre-2026-11-22 set with Uni-NaVid and the hybrid). Its
unpruned reference line reads `info_horizon_mp3d/results/horizon.csv` (archived here on 2026-11-23; rebuild it
first if that `results/` directory is missing).

```bash
conda run -n longnav_vlm python paper/info_horizon/aggregate.py --dataset mp3d
conda run -n vln python paper/info_horizon/plot.py --dataset mp3d
conda run -n longnav_vlm python paper/ctx_ablation/aggregate.py --dataset navverse   # no --strict: the leak-free nv_recompute_* arms were never run
conda run -n vln python paper/ctx_ablation/plot.py --dataset navverse                # the overlay also reads paper_results/ctx_ablation/results/summary.csv
```

`info_horizon_mp3d/` and `ctx_ablation_navverse/` were moved on 2026-11-23 with their `results/` and `figures/` as
they stood (local, git-ignored here). Both sets of tables were rebuilt from the archived runs and compared
byte-identical. In `protected_inputs.json` and the backup the MP3D horizon runs are at
`paper_results_archive/info_horizon_mp3d/runs/`; its 12 frozen tables and figures keep their original
`paper_results/info_horizon_mp3d/` paths there and exist only in the backup, like those of the other archived
studies. `ctx_ablation_navverse/` has no backup copy: its `runs/` here (and the tracked files in git) are the
only ones. The launchers of both replicates write to these archive paths.

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
