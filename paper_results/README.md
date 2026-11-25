# Paper results and study index

Retained experiment data and outputs, produced by the code in `paper/`. The
[protected-input manifest](protected_inputs.json) connects studies, run arms, episode membership,
reproduction commands, and SHA-256 checksums to the baseline commit `bf4e65ff`.

Compact configurations, episode records, pruning records, and curated tables/figures can be
tracked deliberately. Some required traces, probe sidecars, and showcase images are ignored and
remain local. **A Git clone alone is insufficient to reproduce the paper.** Follow the
[reproduction and restoration guide](REPRODUCIBILITY.md) to restore missing inputs and rebuild
outputs in an isolated directory. Ignore rules do not determine whether a scientific input is disposable.

Typical layout (exact files vary by study):

```text
paper_results/<study>/
  runs/<arm>/      config.yaml + rollout/results_*                 (LongNav-R1)
                  manifest.json + summary.json + episodes/*.json  (Uni-NaVid)
                  per-step traces, pruning records, probe arrays as required
  results/        aggregate CSV/Parquet/JSON tables and table.tex
  figures/        PDF and PNG figures
```

New launches also produce ignored logs and locks. Historical launch/driver logs and confirmed
obsolete artifacts were archived locally; their restoration manifests are documented in
[the cleanup record](../CLEANUP.md).

| Retained data directory | Study and protocol |
|---|---|
| `ctx_ablation/` | [Context-window ablation](../paper/ctx_ablation/README.md), both models; full-context runs are shared references. |
| `../paper_results_archive/frame_influence/` | [Leave-one-frame-out influence](../paper/frame_influence/README.md), including LongNav-R1's leak-free recomputation arm. Archived 2026-11-21 (not used in the paper): runs only, `results/` and `figures/` rebuild from them (see the linked README). |
| `../paper_results_archive/frame_influence_keepone/` | [Keep-one-in influence](../paper/frame_influence/README.md), including the documented masking/splicing comparison. Archived 2026-11-21, runs only, as above. |
| `token_prune_flat/` | [Flat pruning](../paper/token_prune/README.md): random, FastV, DivPrune, DART; layer-0 selection without the 32-frame window. |
| `layer_token_info/` and its `nohist/` study | [Layer-token information](../paper/layer_token_info/README.md): leaky/leak-free token KL, the no-visual-history study, and the historical windowed horizon figure. |
| `info_horizon/` | [Information horizon](../paper/info_horizon/README.md): full-context probes and visual-blind SR/SPL on HM3D. |
| `../paper_results_archive/info_horizon_mp3d/` | The MP3D replicate of `info_horizon/` (18 arms). Archived 2026-11-23 (not used in the paper); `results/` and `figures/` rebuild from `runs/`. The archived `protocol_mp3d` figure reads its `results/horizon.csv` for the unpruned reference. |
| `../paper_results_archive/ctx_ablation_navverse/` | The [NavVerse indoor replicate](../paper/ctx_ablation/README.md) of the context-window ablation (LongNav-R1 and LightNav-0, 12 arms). Archived 2026-11-23 (not used in the paper); not part of `protected_inputs.json`; `results/` and `figures/` rebuild from `runs/`. |
| `protocol/` | [Pruning protocol (uniform) vs PyramidDrop, and the retired hybrid](../paper/protocol/README.md) on HM3D: whole-model budgets, 24 complete 100-episode arms for LongNav-R1 and Uni-NaVid plus 8 LightNav-0 arms; documented cost benchmarks also retained. Named `hybrid_prune/` until 2026-11-22. |
| `../paper_results_archive/protocol_mp3d/` | The MP3D replicate of `protocol/` (24 complete 100-episode arms). Archived 2026-11-22 (not used in the paper); `results/` and `figures/` rebuild from `runs/`. |
| `selector_kl/` | [Selector criteria vs no-history keep-one-in KL](../paper/selector_kl/README.md), both models; LongNav-R1 showcase heatmaps. |

The windowed horizon figure additionally reads
`dump/longnav_eval/keep_one_influence_w32_b10000000`. Uni-NaVid's default evaluation and
teleoperation manifest is `dump/hm3d_v2_100_labels.json`. Both are protected even though they
live outside `paper_results/`.

The [windowed visual-token pruning study](../paper/token_prune/README.md) (`token_prune/`: random,
FastV, PyramidDrop, DivPrune, DART under the 32-frame window) was deleted on 2026-11-21; the paper
uses the flat variant. Its checksummed inputs survive only in the backup, see
[the archive index](../paper_results_archive/README.md) for how to restore them.

Superseded hybrid `*_b<k>` and aborted `*_m<k>` runs, together with the old
`results_last_stage/` and `figures_last_stage/`, are archived. Current hybrid reporting uses
only whole-model budgets. This does not remove the last-stage budget implementation needed
by the windowed token-pruning study. Earlier documented cost benchmarks remain available.
