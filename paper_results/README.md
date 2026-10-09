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
| `frame_influence/` | [Leave-one-frame-out influence](../paper/frame_influence/README.md), including LongNav-R1's leak-free recomputation arm. |
| `frame_influence_keepone/` | [Keep-one-in influence](../paper/frame_influence/README.md), including the documented masking/splicing comparison. |
| `token_prune/` | [Windowed visual-token pruning](../paper/token_prune/README.md): random, FastV, PyramidDrop, DivPrune, DART; last-stage budgets retained. |
| `token_prune_flat/` | [Flat pruning](../paper/token_prune/README.md): random, FastV, DivPrune, DART; layer-0 selection without the 32-frame window. |
| `layer_token_info/` and its `nohist/` study | [Layer-token information](../paper/layer_token_info/README.md): leaky/leak-free token KL, the no-visual-history study, and the historical windowed horizon figure. |
| `info_horizon/`, `info_horizon_mp3d/` | [Information horizon](../paper/info_horizon/README.md): full-context probes and visual-blind SR/SPL on HM3D and MP3D. |
| `hybrid_prune/`, `hybrid_prune_mp3d/` | [Hybrid, uniform, and PyramidDrop](../paper/hybrid_prune/README.md): whole-model budgets, 24 complete 100-episode arms per dataset; documented cost benchmarks also retained. |
| `selector_kl/` | [Selector criteria vs no-history keep-one-in KL](../paper/selector_kl/README.md), both models; LongNav-R1 showcase heatmaps. |

The windowed horizon figure additionally reads
`dump/longnav_eval/keep_one_influence_w32_b10000000`. Uni-NaVid's default evaluation and
teleoperation manifest is `dump/hm3d_v2_100_labels.json`. Both are protected even though they
live outside `paper_results/`.

Superseded hybrid `*_b<k>` and aborted `*_m<k>` runs, together with the old
`results_last_stage/` and `figures_last_stage/`, are archived. Current hybrid reporting uses
only whole-model budgets. This does not remove the last-stage budget implementation needed
by the windowed token-pruning study. Earlier documented cost benchmarks remain available.
