# paper_results

Data for the paper experiments, one directory per experiment, produced by the code in
`paper/<experiment>/`. Curated `results/` and `figures/`, compact episode records, manifests, and configurations
can be tracked deliberately. Large traces, probe arrays, renders, ordinary logs, and locks
remain ignored. Ignored scientific inputs are still required: see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and `protected_inputs.json` for their backup and checksums.

```
paper_results/<experiment>/
  runs/<arm>/      raw run output: config.yaml + launch.log + rollout/results_* (LongNav)
                   or manifest.json + summary.json + episodes/*.json (Uni-NaVid)
  results/         per_episode.csv, summary.csv, table.tex   (aggregate.py)
  figures/         *.pdf + *.png                            (plot.py)
```

- `ctx_ablation/` — context-window ablation; protocol in `paper/ctx_ablation/README.md`.
- `frame_influence/` — leave-one-frame-out influence by frame age (both models); protocol in `paper/frame_influence/README.md`.
- `token_prune/` — visual-token pruning (Random / FastV / PyramidDrop / DivPrune / DART) under a 32-frame window, both models; protocol in `paper/token_prune/README.md`.
- `hybrid_prune/` — the pruning protocols at a whole-model budget of k frames' worth, both models: `hybrid` (K=32 window, attention stages, random tail from layer 20) and `uniform` (K=32 window, one uniform random draw at layer 0, visual-blind from layer 18 / 20) vs no-window PyramidDrop, plus the k=16 cost benchmark; `hybrid_prune_mp3d/` the MP3D replicate (hybrid, uniform, and PyramidDrop); protocol in `paper/hybrid_prune/README.md`.
- `layer_token_info/` — per-visual-token, per-decoder-layer information of the current observation at one decision (keep-one-in / leave-one-out KL, leaky vs leak-free), the Fig.-7 analogue; protocol in `paper/layer_token_info/README.md`.
- `info_horizon/` — information horizon by decoder layer (frame-level keep-one-in / leave-one-out KL of the current frame and visual-blind SR/SPL), full context, leak-free, both models; protocol in `paper/info_horizon/README.md`.
- `selector_kl/` — flat-variant selector criteria (FastV layer-0 attention, DivPrune, DART, random) vs the per-token no-history keep-one-in KL of the current frame at layer 0, both models (LongNav-R1 with heatmaps, Uni-NaVid Spearman only), every decision of ten short episodes; protocol in `paper/selector_kl/README.md`.
- `info_horizon_mp3d/` — the same on MP3D val (100 seed-17 episodes for SR/SPL, 10 for the KL probes); `DATASET=mp3d` in `paper/info_horizon/`.
