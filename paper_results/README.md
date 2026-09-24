# paper_results

Data for the paper experiments, one directory per experiment, produced by the code in
`paper/<experiment>/`. Everything here is tracked except the bulky per-step artifacts listed in
`.gitignore` (per-episode rollout dirs, traces, videos), which stay on disk only.

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
- `layer_token_info/` — per-visual-token, per-decoder-layer information of the current observation at one decision (keep-one-in / leave-one-out KL, leaky vs leak-free), the Fig.-7 analogue; protocol in `paper/layer_token_info/README.md`.
- `info_horizon/` — information horizon by decoder layer (frame-level keep-one-in / leave-one-out KL of the current frame and visual-blind SR/SPL), full context, leak-free, both models; protocol in `paper/info_horizon/README.md`.
- `info_horizon_mp3d/` — the same on MP3D val (100 seed-17 episodes for SR/SPL, 10 for the KL probes); `DATASET=mp3d` in `paper/info_horizon/`.
