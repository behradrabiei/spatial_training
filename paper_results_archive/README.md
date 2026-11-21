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
