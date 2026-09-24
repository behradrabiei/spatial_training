# Context-window ablation

How much visual history does an ObjectNav VLM agent need, and how much of a KV-cache agent's
apparent small-window performance is information leaking through the cache rather than
memory? SR / SPL / steps vs the number of frames in context, for LongNav-R1 (two variants)
and Uni-NaVid, on the 100-episode HM3D-v2 val split.

## Definitions

**Context size `K`** — the number of observation frames the policy can see when it emits an
action, *including the current frame*. `K=1` is the current frame alone; `full` is the whole
episode. The split, the goal prompt and the action space are identical across arms.

*LongNav-R1.* `vlm.context_frames=K` with `vlm.context_cut=frame` (`+experiment=paper_ctx`).
At decision time the KV cache holds the pinned prompt prefix (system prompt + goal) plus
exactly the last `K` user turns (`<|im_start|>user … <image> … <|im_end|>`), each followed by
the assistant reply that was produced after it. The window opens on the oldest retained
`<|im_start|>user`: the action that preceded it is gone too, so `K` frames come with the
`K−1` actions taken between them and nothing older. Older turns are retired one whole turn per
step. (`vlm.context_window=N`, the legacy knob, keeps `N` *past* turns and cuts at the assistant
header, i.e. `N+1` frames plus one dangling action token; every run before this experiment used
that convention.)

The sparse visual-token filter stays **on** (`use_sparse=true, sparse_threshold=0.95`) — it is
the regime the checkpoint was trained in. Each frame's patches are deduplicated only against
retained frames that will still be in the window after this step, and the dedup database is
trimmed with the cache, so a frame the model can no longer see never suppresses new patches.
At `K=1` nothing is deduplicated.

*Uni-NaVid.* `--context-frames K`. Uni-NaVid re-encodes its single-turn prompt on every
inference (instruction + visual tokens, no action history); its cross-step memory is a
per-frame feature cache whose rows are computed independently per frame. The harness keeps the
last `K` rows and drops the compressed long-term cache under any finite window, so `K` means
the same thing as for LongNav-R1. There is no cache to leak through, hence a single arm.

## Arms

| arm | model | history mechanism | leak-free? |
|---|---|---|---|
| `r1_full` | LongNav-R1 | growing KV cache, whole episode | n/a (nothing evicted; shared baseline of both R1 arms) |
| `r1_evict_k{32,16,8,4,1}` | LongNav-R1 | KV eviction (`context_window_mode=evict`) | **no** — a surviving token's layer≥1 K/V were computed while attending over frames that are now gone |
| `r1_recompute_k{32,16,8,4,1}` | LongNav-R1 | window K/V rebuilt each step against prefix+window only (`context_window_mode=recompute`) | yes — the decision is a strict function of the visible frames |
| `uninavid_full` | Uni-NaVid | native feature cache + long-term compression | yes |
| `uninavid_k{32,16,8,4,1}` | Uni-NaVid | last `K` per-frame features | yes |

`evict − recompute` at matched `K` is the leakage. Both arms retire the same tokens on the
same steps and keep the same absolute mRoPE positions (`tests/ctx_frames_smoke.py`,
`tests/ctx_recompute_smoke.py`). At `K=1` the two are the same computation (the cache is
prefix-only after every step and the prefix K/V never saw a frame), so `r1_recompute_k1` is a
built-in control that must reproduce `r1_evict_k1`.

## Protocol

- Split: `src/longnav/conf/episode_jsons/hm3d_v2_100.json` (100 episodes, HM3D-v2 val),
  identical for every arm; an arm is used only if it holds all 100.
- LongNav-R1: checkpoint `+checkpoint=longnav` (RPP + kernel-embedding RL, ckpt 231 on the
  sparse SFT backbone), `sdpa`, bf16, argmax actions, 500-step cap, `ep_seed=17`, no guards,
  no visualization. Everything is in `src/longnav/config/experiment/paper_ctx.yaml`; each arm
  adds only `vlm.context_frames=K vlm.context_window_mode={evict|recompute}`. The resolved
  config of every run is saved to `paper_results/ctx_ablation/runs/<arm>/config.yaml`.
- Uni-NaVid: the official evaluator settings as shipped in `baselines/` — seeded sampling
  (seed 30, temperature 0.2), 2 actions per LLM call, 500-step cap, same success radius. Note
  the decoding asymmetry with LongNav-R1 (sampled vs argmax); it is the published protocol of
  each model. Each run's `manifest.json` records the flags, seeds, package versions and the
  label-file hash. Seeding fixes the sampler's random stream, not fp16 kernel arithmetic, so
  sampled Uni-NaVid runs are not bit-reproducible: two runs of `uninavid_full` (2026-08-22 vs
  2026-09-04, identical config) gave SR 0.76 vs 0.72, SPL 0.413 vs 0.417, paired p=0.22 —
  a run-to-run floor of roughly ±0.04 SR. Treat differences between Uni-NaVid points below
  that as noise. LongNav-R1 is argmax and reproduces exactly.
- Statistics: episode means with 95% percentile-bootstrap CIs (2000 resamples, seed 17);
  differences between arms are paired on `episode_label` and tested with a two-sided sign-flip
  permutation test (10 000 resamples). Read the curve on SPL and steps as well as SR: with a
  500-step budget a memory-starved agent can still stumble onto the goal.

## Running

```bash
# Uni-NaVid (~10–15 min per arm, isolated runtime in baselines/.envs)
bash paper/ctx_ablation/run_uninavid.sh

# LongNav-R1 (evict 25–70 min per arm, recompute ~2 h per arm; r1_full first)
conda activate longnav_vlm
bash paper/ctx_ablation/run_r1.sh                # or ARMS="r1_full r1_evict_k8" bash ...

# Aggregate (longnav_vlm) and plot (vln has matplotlib)
python paper/ctx_ablation/aggregate.py
/home/brabiei/miniconda3/envs/vln/bin/python paper/ctx_ablation/plot.py
```

Do not run the two model families concurrently on one GPU (Uni-NaVid-7B alone takes ~15 GB).
Both run scripts skip arms that are already complete and resume partial ones.

## Outputs — all under `paper_results/ctx_ablation/` (tracked)

- `runs/<arm>/` — raw run output. LongNav-R1: `config.yaml` (resolved Hydra config),
  `launch.log`, `rollout/results_*` (per-episode JSONL); the per-episode `rollout/<scene>/…`
  dirs (`sequence.json`, thumbnails) stay on disk but are gitignored. Uni-NaVid: `manifest.json`
  (flags, seeds, versions, label hash), `summary.json`, `episodes/*.json`, `launch.log`;
  `traces/` gitignored. `driver.log` is the chained launch log.
- `results/per_episode.csv`, `results/summary.csv` (means, CIs, p-values vs each model's
  `full`, leakage gap per `K`), `results/table.tex` (booktabs, stars vs `full`).
- `figures/ctx_ablation.{pdf,png}` — SR | SPL | steps vs `K`;
  `figures/ctx_leakage.{pdf,png}` — evict − recompute gap vs `K`.

The pre-paper sweep scripts (`tools/run_context_window_ablation.sh`, `tools/plot_context_window_ablation.py`,
`baselines/run_context_window_ablation.sh`; legacy `context_window`, header cut, no CIs) were removed
once this experiment replaced them; the runs they produced remain under `dump/`.
