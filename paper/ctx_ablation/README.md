# Context-window ablation

How much visual history does an ObjectNav VLM agent need, and how much of a KV-cache agent's
apparent small-window performance is information leaking through the cache rather than
memory? SR / SPL / steps vs the number of frames in context, for LongNav-R1 (two variants),
Uni-NaVid and LightNav-0, on the 100-episode HM3D-v2 val split.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`; plots use `vln`;
Uni-NaVid and LightNav-0 launchers use their isolated baseline runtimes. To rebuild the retained tables and
figures without overwriting published outputs, follow the
[reproduction guide](../../paper_results/REPRODUCIBILITY.md). A Git clone alone is
insufficient: restore required local inputs from the checksummed backup first. Ordinary
launch logs and locks are ignored; protected traces, arrays, and images remain scientific inputs.

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
step. (`vlm.context_window=N`, the legacy size knob, keeps `N` *past* turns. With the historical
`vlm.context_cut=header` setting this gives `N+1` frames plus one dangling action token.
Size and cut are separate settings; this experiment uses the frame cut.)

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

*LightNav-0.* `CONTEXT_FRAMES=K` on `baselines/run_lightnav.sh` (added 2026-10-15). LightNav-0
caches every RGB observation of the episode and, at each inference, samples the history by
age with the checkpoint's SlowFast tiers (the two newest frames at full resolution, ages 2–33
dense at half resolution, sparser pairs beyond, and the absolute first two frames as an
anchor); the single-turn prompt holds one `<video>` per tier, no action history and no step
counter, and each frame pair carries an age-relative `<t seconds>` prefix. The harness wrapper
`baselines/lightnav_hm3d/eval_ctx.py` restricts the sampler to the newest `K` observed frames:
with more than `K` frames observed it uses the segment layout upstream computes at step `K−1`
of a fresh episode, shifted onto the absolute ids of the last `K` frames (timestamps are
age-relative, the ViT-tubelet cache is keyed by absolute id), so the model input is that of
an episode that started `K` steps ago. For every `K` in the sweep (≤ 34) the window lies
inside the dense tiers, i.e. the model simply sees the newest `K` frames. The clone,
checkpoint, tiers, decoding and camera are untouched; `baselines/tests/test_lightnav_ctx.py`
checks the equivalence against the unmodified sampler. Like Uni-NaVid, the prompt is rebuilt
from the visible frames every step, so there is nothing to leak through and a single arm.

## Arms

| arm | model | history mechanism | leak-free? |
|---|---|---|---|
| `r1_full` | LongNav-R1 | growing KV cache, whole episode | n/a (nothing evicted; shared baseline of both R1 arms) |
| `r1_evict_k{32,16,8,4,2,1}` | LongNav-R1 | KV eviction (`context_window_mode=evict`) | **no** — a surviving token's layer≥1 K/V were computed while attending over frames that are now gone |
| `r1_recompute_k{32,16,8,4,2,1}` | LongNav-R1 | window K/V rebuilt each step against prefix+window only (`context_window_mode=recompute`) | yes — the decision is a strict function of the visible frames |
| `uninavid_full` | Uni-NaVid | native feature cache + long-term compression | yes |
| `uninavid_k{32,16,8,4,1}` | Uni-NaVid | last `K` per-frame features | yes |
| `lightnav_full` | LightNav-0 | whole-episode frame cache, SlowFast age sampling | yes |
| `lightnav_k{32,16,8,4,1}` | LightNav-0 | SlowFast sampling over the last `K` frames | yes |
| `lightnav_keep_k{32,16,8,4,2,1}` | LightNav-0 | the window's image tokens (newest `K` frames; at `K = 2` the full-resolution current pair and no history, at `K = 1` the current frame doubled) with every older segment's timestamp and vision markers left in the prompt — the flat token-prune study's `lightnav_recency_b{K−2}` runs, `lightnav_random_b0` and `lightnav_recency_b0_cur1`, read from `paper_results/token_prune_flat/runs/` | yes |

`keep − window` at matched `K` (`gap_*` in `summary.csv`) is what LightNav-0 loses to the *removal of the
prompt skeleton* rather than to the removal of the frames: see "LightNav-0: the window cliff is the
skeleton" below. `evict − recompute` at matched `K` is the leakage. Both arms retire the same tokens on the
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
- LightNav-0: upstream's own pipeline as shipped in `baselines/` (`run_lightnav.sh full`):
  480×270 RGB at 120° hfov, `velocity_control` actions, greedy decoding, explicit STOP on the
  last budget step, 500-step cap, success radius 1.0 m on the shipped navmesh (the comparable
  protocol, not LightNav's native 0.1 m + re-bake). `lightnav_full` is the comparable
  100-episode run of 2026-10-14 (`baselines/results/lightnav_hm3d_v2_100`), imported into
  `runs/lightnav_full` rather than rerun; the windowed arms use the same launcher with
  `CONTEXT_FRAMES=K`. Decoding is greedy, so runs are reproducible up to bf16 kernel order.
  Each run's `run.json` records both command lines, `context_frames`, the clone and checkpoint
  revisions and the label-file hash. **Step-cap rule (2026-10-16):** upstream's forced STOP lets
  Habitat score the final pose, so an agent parked inside the 1.0 m radius when the budget runs
  out counts as a success there, whereas LongNav-R1 and Uni-NaVid episodes are truncated at the
  cap without a stop call and cannot succeed. `aggregate.py` therefore loads LightNav-0 arms with
  `results_io.load_lightnav_run`, which scores every `forced_stop` episode as a failure (SR = SPL
  = 0; the harness's own values stay in the run's `episodes/*.json` and in the loader's
  `success_raw` / `spl_raw`). This changes 28 of the 600 LightNav-0 episodes: `lightnav_full` .81
  → .80, `k32` .84 → .82, `k8` .75 → .68, `k4` .49 → .34 (15 episodes that reached the goal
  area but never stopped), `k1` .03 → .00; `k16` unchanged.
- Uni-NaVid: the official evaluator settings as shipped in `baselines/` — global RNG seed 30
  (temperature 0.2; no per-call `--sampling-seed`), 2 actions per LLM call, 500-step cap, same success radius. Note
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

## LightNav-0: the window cliff is the skeleton, not the frames (2026-10-25/27)

LightNav-0's prompt is one block per frame pair, `<t seconds> <|vision_start|> image tokens <|vision_end|>`,
oldest first, with age-relative timestamps and no step counter. The window wrapper deletes the old blocks
outright, so at `K = 4` nothing in the prompt is older than 0.6 s and the input is that of step 3 of a fresh
episode, forever. The flat token-prune study found that random pruning down to the same 28 history tokens
keeps SR .82 (`paper/token_prune/README.md`, "Layout check"), so the control `lightnav_keep_k{K}` was run: the
prune wrapper (`baselines/lightnav_hm3d/eval_prune.py --prune-method recency`, eager vLLM, masked attention
columns) hides the image tokens of every frame older than the newest `K` and leaves the empty old blocks with
their timestamps. At every step ≥ 34 the visible frames are exactly the window's (verified on upstream's
layout: the window's history at `K` is the newest `K − 2` frames of the full layout, all at the model's
28-token-per-pair pooling; before step 34 upstream pads the history on even steps, so the newest pair is
`(t−2, t−2)` there). Keep-mask audit of the four arms: 0 violations. The green "LightNav-0" line of
`figures/ctx_ablation.*` is this control (the window arms with the old blocks removed are no longer drawn, see
`results/summary.csv`); it shares the full-context point with them (its matched
eager-kernel reference is the prune study's `lightnav_identity`, .84 / .458 / 139 steps, vs the compiled
`lightnav_full` .80 / .450 / 160; paired ΔSR +.04, p = .39). The `K = 1` control (`lightnav_recency_b0_cur1`,
`PRUNE_CURRENT_FRAMES=1`) hides every history token and replaces the current pair (t − 1, t) with the doubled
current frame (t, t) in the sampler's layout, upstream's own step-0 layout and exactly what the `K = 1` window
shows; its keep-mask audit is also 0 violations (45,123 step records). The `K = 2` point is the prune study's
`lightnav_random_b0` (`--prune-k 0`: the full-resolution current pair and no history token, skeleton kept); there
is no `K = 2` window arm, so that row has no keep − window column.

| K | window SR / SPL / steps | forced stops (parked ≤ 1 m) | old blocks kept SR / SPL / steps | forced stops | keep − window ΔSR (p) / ΔSPL (p) / Δsteps (p) |
|---|---|---|---|---|---|
| 32 | .82 / .467 / 177 | 16 (2) | .81 / .445 / 155 | 10 | −.01 (1.0) / −.022 (.34) / −22 (.13) |
| 16 | .80 / .444 / 189 | 19 (0) | .79 / .439 / 174 | 14 | −.01 (1.0) / −.005 (.87) / −14 (.30) |
| 8 | .68 / .365 / 236 | 27 (7) | .84 / .475 / 164 | 11 | **+.16 (.0002)** / **+.110 (.0002)** / **−72 (<.001)** |
| 4 | .34 / .188 / 376 | 65 (15) | .79 / .431 / 203 | 18 | **+.45 (<.001)** / **+.243 (<.001)** / **−173 (<.001)** |
| 2 | — | — | .59 / .332 / 286 | 38 (8) | — |
| 1 | .00 / .000 / 500 | 100 (3) | .02 / .014 / 452 | 88 (2) | +.02 (.51) / +.014 (.51) / −48 (.001) |

With the same frames visible, restoring the old blocks' timestamps removes the whole cliff down to `K = 4`: the
control is flat at .79–.84 from `K = 32` to `K = 4` and never differs from `lightnav_full` (all p ≥ .44), while
the window loses .12 SR at `K = 8` and .46 at `K = 4`. At `K = 1` the skeleton does not help (.02 vs .00): with
a single frame the model has no motion cue and stops on its own in only 12 episodes, 2 of them at the goal; the
`K = 2` point (the full-resolution current pair and no history, skeleton kept) scores .59, so the `K = 1`
collapse is the missing second current frame, a different failure from the `K = 4` / `K = 8` one, and the
dashed line's `K = 2 → 4` rise (.59 → .79) is what visual history is worth once the skeleton and the current
pair are both in place. The gap is entirely the stop decision: under the window the agent
succeeds .93–.99 of the time whenever it does stop, and its deficit is episodes forced to stop at the cap (27
and 65 of 100, 7 and 15 of them parked within 1 m of the goal); the control's forced stops stay at 10–18 and
none of them are parked at the goal. Read the LightNav-0 window curve as "what happens when the model loses its
sense of elapsed time", not as its memory requirement; its memory requirement, measured with the skeleton
intact, is below four frames (the no-history arm `lightnav_random_b0` of the prune study still scores .59).
The control's residual cost against the unpruned model is steps, not SR (+64 at `K = 4`, +35 at `K = 16`,
+15 at `K = 32`), and it is the only LightNav-0 arm that improves nothing at large `K`, so the 28-token
history pairs the model sees under a 32-frame window carry about as much as the whole history does.
Runs: `lightnav_recency_b2` 2026-10-25 (118 min), `b6` / `b14` / `b30` 2026-10-27 06:40 → 10:13 (71 / 76 / 66 min),
`lightnav_recency_b0_cur1` 2026-10-27 11:10 → 15:00 (230 min, nearly every episode at the 500-step cap).

## Running

```bash
# Uni-NaVid (~10–15 min per arm, isolated runtime in baselines/.envs)
bash paper/ctx_ablation/run_uninavid.sh

# LongNav-R1 (evict 25–70 min per arm, recompute ~2 h per arm; r1_full first)
conda activate longnav_vlm
bash paper/ctx_ablation/run_r1.sh
# A subset: ARMS="r1_full r1_evict_k8" bash paper/ctx_ablation/run_r1.sh

# LightNav-0 (~50 min per arm, isolated runtime in baselines/.envs; imports lightnav_full)
bash paper/ctx_ablation/run_lightnav.sh

# Aggregate (longnav_vlm) and plot (vln has matplotlib)
python paper/ctx_ablation/aggregate.py
conda run -n vln python paper/ctx_ablation/plot.py
```

Do not run the model families concurrently on one GPU (Uni-NaVid-7B alone takes ~15 GB;
LightNav-0 runs its simulator and vLLM on the same device). All run scripts skip arms that are
already complete; the R1 and Uni-NaVid scripts resume partial ones, while LightNav cannot
resume (append-only `results.jsonl`) and asks for a partial arm to be moved away.

## Outputs — all under `paper_results/ctx_ablation/`

- `runs/<arm>/` — raw run output. LongNav-R1: `config.yaml` (resolved Hydra config),
  `rollout/results_*` (per-episode JSONL); the per-episode `rollout/<scene>/…`
  dirs (`sequence.json`, thumbnails) stay on disk but are gitignored. Uni-NaVid: `manifest.json`
  (flags, seeds, versions, label hash), `summary.json`, `episodes/*.json`;
  `traces/` gitignored. LightNav-0: `run.json` (commands, `context_frames`, revisions, label
  hash), `raw/results.jsonl` + `raw/summary.json` (upstream's own output), `episodes/*.json` and
  `summary.json` (converted to the Uni-NaVid layout); `raw/logs/` gitignored. New launches generate ignored `launch.log` / `driver.log` files; historical logs were archived.
- `results/per_episode.csv`, `results/summary.csv` (means, CIs, p-values vs each model's
  `full`, leakage gap per `K`), `results/table.tex` (booktabs, stars vs `full`).
- `figures/ctx_ablation.{pdf,png}` — SR | SPL vs `K` for LongNav-R1 (KV eviction; leak-free recompute
  dashed) and LightNav-0 (old blocks kept), drawn at single-column width (3.5 × 1.5 in, 8 pt text, metric name on the y axis) for a
  two-column paper. Uni-NaVid, the LightNav-0 window arms (old blocks removed), steps, and the
  evict − recompute gap (`leak_*` columns) are in `summary.csv` / `table.tex` only.
  `figures/ctx_ablation_sr.{pdf,png}` and `figures/ctx_ablation_spl.{pdf,png}` are the same two panels as
  standalone figures (3.5 × 1.9 in); `plot.py` writes the pair and both singles for every SR | SPL figure.
- Kept locally, not on GitHub (code and documentation only for new outputs): the standalone `_sr` / `_spl`
  figures (`plot.py` rewrites them). The two LongNav-R1 `K = 2` arms added 2026-11-20 (`r1_evict_k2`,
  `r1_recompute_k2`; about 1.5 h on one GPU) are tracked like the other arms, so `aggregate.py --strict`
  works on a fresh clone.

## NavVerse indoor replicate (LongNav-R1, and LightNav-0 old blocks kept)

The same K arms of our LongNav-R1 stack on NavVerse's indoor ObjectNav benchmark
(`NavVerse-Benchmark/`, Isaac Sim, GRScenes commercial scenes), added 2026-10-20, and the
LightNav-0 "old blocks kept" arms behind the same client (added 2026-10-28, see the subsection
below). Outputs live under `paper_results/ctx_ablation_navverse/` with the HM3D layout (`runs/`,
`results/`, `figures/`).

- **Model and serving.** NavVerse's LongNav client (`baselines/longnav_r1/longnav_agent.py`)
  posts one JPEG frame per step and the goal word on the first step to an HTTP `/act` endpoint
  and keeps no history itself, so the context window is applied by *our* server exactly as in
  the Habitat arms: `python -m longnav.scripts.serve --experiment paper_ctx
  [--set vlm.context_frames=K --set vlm.context_window_mode=evict]` (same adapter, argmax actions,
  sparse filter, frame cut; `server_config.json` records the effective config, `server_steps.jsonl`
  the per-call latency / cache length / peak memory). NavVerse's vendored LongNav-R1 clone is not used.
- **Evaluator.** NavVerse runs in its Docker image (`navverse-release:local`) with host
  networking, `--preset benchmark_longnav --longnav_server_url http://127.0.0.1:8000/act
  --longnav_max_steps 500 --sim_episode_timeout_s 1800` (Spot, POV camera 640x480, ~82° HFOV),
  discrete actions 0.25 m / 30° executed physically by a waypoint follower. One simulator+server
  pair per arm: two pairs fit the GPU for the K arms (~13 GB each) but not the 62 GB host (Isaac
  Sim takes 12-16 GB of RAM per process on some scenes, NavVerse's task-queue actor grows to
  ~13 GB over an arm, each agent worker holds 4-5 GB of frames), so Ray's memory monitor kills
  the pairs in a loop; `K_WORKERS=2` keeps that option (each worker posts to port 8000+i). The
  `--sim_episode_timeout_s` flag is a one-file patch on the NavVerse clone's branch
  `ctx-ablation-timeout` (bind-mounted into the image): NavVerse's stock 300 s simulated-time
  limit ended episodes after ~230 steps because its 2 s per-action limit makes a step cost
  ~1.3 s of simulated time, so the 500-step cap never bound.
- **Episodes.** `src/longnav/conf/episode_jsons/navverse_indoor_100_seed17.json`: a seed-17
  sample (`tools/sample_episode_manifest.py`) of 100 of the 160 `benchmark500_indoor` labels
  (`navverse_indoor_160.json`), i.e. 5 held-out commercial scenes, all ObjectNav.
- **Scoring.** NavVerse's own measures: success = STOP called within **1.6 m** of the goal
  (Habitat arms: 1.0 m), SPL over the 3D path length, path-based distance to goal. The LongNav
  client returns STOP itself at the 500-step cap and NavVerse scores that pose (LightNav-0's
  forced stop), so `results_io.load_navverse_run` re-scores those episodes as failures
  (`success_raw` / `spl_raw` keep the harness values), making the cap mean what it means for
  the other arms. An episode can also end by NavVerse's 300 s simulated-time limit
  (`termination_reason=time_out`, a failure); `summary.csv` counts forced stops and time-outs.
- **Failure modes.** Every episode is put in exactly one of five outcome classes
  (`results_io.navverse_failure_mode`, in this priority order): `success`; `fell` (the
  simulator's `bad_orientation` termination, the robot tipped over); `cap_never_near_goal`
  (forced STOP at the 500-step cap, never within the 1.6 m success radius, NavVerse's
  `oracle_success` = 0); `cap_near_goal_no_stop` (forced STOP at the cap after having been within
  the radius, i.e. a missed stop); `wrong_stop` (the model called STOP before the cap outside the
  radius). `per_episode.csv` carries `oracle_success` and `failure_mode`, `summary.csv` the
  per-arm counts, `results/failure_modes.csv` one count column per class, and
  `figures/failure_modes.{pdf,png}` draws them as one stacked bar of 100 episodes per K
  (success at the bottom), so the bars share a total. A `stuck` class was considered and
  dropped: by the last 100 steps nearly every cap episode sits inside a 0.5 m box, so it would
  only rename the two cap classes.
- **Known deviation, to address (found in the 2026-10-21 review).** `longnav.scripts.serve`
  reads `objectnav_prompt.txt` with `read_text()` (no strip) while the Hydra path's `read_text`
  resolver strips it, so every NavVerse arm of the 2026-10-20/22 sweep used a system prompt
  ending in one extra newline token. The deviation is identical across the six arms (the
  within-NavVerse K comparison is valid) but the sweep is not an exact replicate of the Habitat
  protocol: on 10 Habitat seed-17 episodes under the K=8 protocol, a stripped-prompt control
  reproduces the reference rollouts 10/10 bit-identically while the newline prompt diverges on
  4/10 (one failure becomes a success; SR .70 vs .80), i.e. a seed-like perturbation, not a bias.
  The same un-stripped read is in NavVerse's vendored adapter (`baselines/longnav_r1/
  serve_longnav.py`), which additionally samples actions instead of taking the argmax, so
  LongNav-R1 numbers produced through NavVerse's stock path carry both differences. TODO after
  the sweep: `.strip()` the prompt in `serve.py` (the sweep was left untouched so all arms share
  one prompt) and decide whether to rerun the six arms (~40 h) for an exact replicate.
- **Cost.** Isaac Sim executes every action physically at ~1.3x real time, so an episode costs
  ~1 s per step of wall time regardless of the model (the VLM call is 40–80 ms): 5–10 h per
  100-episode arm, versus 25–70 min on Habitat. Arms resume (`--resume_existing_results`).

```bash
# all evict arms (nv_full, nv_evict_k{32,16,8,4,1}); needs the GPU alone (~9 GB sim + 5–13 GB server)
bash paper/ctx_ablation/run_navverse.sh
ARMS="nv_evict_k8" bash paper/ctx_ablation/run_navverse.sh                # a subset
bash paper/ctx_ablation/run_navverse_lightnav.sh                           # nvl_full, nvl_keep_k{32,16,8,4,1}
conda run -n longnav_vlm python paper/ctx_ablation/aggregate.py --dataset navverse
conda run -n vln python paper/ctx_ablation/plot.py --dataset navverse        # + hm3d-vs-navverse overlay
```

### LightNav-0 on NavVerse: old blocks kept (2026-10-28)

The HM3D "old blocks kept" line (labelled "LightNav-0" in the figures; the window's frames with every older
segment's timestamp and vision markers left in the prompt) replicated on the same 100 NavVerse
episodes: arms `nvl_full`, `nvl_keep_k{32,16,8,4}` and `nvl_keep_k1` (`paper/ctx_ablation/run_navverse_lightnav.sh`;
no window arms, no K = 2 point; K = 1 = no history and the current frame doubled, `--prune-k 0
--prune-current-frames 1`, the HM3D `lightnav_keep_k1` = `lightnav_recency_b0_cur1` layout, added 2026-10-30). `aggregate.py --dataset navverse` loads them as model
`nvl` (same loader and step-cap re-scoring as the LongNav-R1 arms, a fifth table column group);
`plot.py --dataset navverse` draws them in green in `ctx_ablation`, adds both models' HM3D vs
NavVerse pairs to the overlay, and writes `figures/failure_modes_lightnav.{pdf,png}`.

- **Serving.** NavVerse has no LightNav agent. `baselines/lightnav_hm3d/serve_navverse.py`
  (LightNav's isolated runtime, in-process vLLM) speaks the `/act` contract of our LongNav-R1
  server, so NavVerse's LongNav client drives LightNav-0 unchanged: one 640x480 JPEG per step,
  the goal word on the first step (formatted as LightNav's HM3D server does, `Find the <goal>.`),
  the client's own STOP at the 500-step cap (re-scored as a failure, as above). The prompt is the
  model's own full-episode SlowFast history rebuilt every step (`kv_len` in `server_steps.jsonl`
  is therefore the step's prompt length, not a cache).
- **Actions.** LightNav-0 executes the first waypoint of its predicted trajectory as a 1 s
  `velocity_control` step on Habitat (forward in [0, 0.25] m, yaw in [-30, 30] deg, both at
  once). The client only takes STOP / FORWARD (0.25 m) / a left turn / a right turn, so the
  server discretizes: STOP when Habitat's stop guard would fire (|lin| < 0.025 m/s and
  |ang| < 1 deg/s -- the explicit stop code, a decode failure and an all-zero trajectory all
  produce that), else whichever of the normalized forward (lin / 0.25 m) and turn (|ang| / the
  turn size) components is larger (ties to FORWARD, left for a positive = counter-clockwise yaw).
  A combined move-and-turn step thus becomes one or the other; the step log records the waypoint
  (`fwd_m`, `yaw_deg`) and the rule (`reason`) behind every action. **Turn size: 15 deg**
  (`--longnav_action_map STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL`, the server's
  `--turn-deg 15`), the closest discrete turn to LightNav's own 18 deg rotate-in-place step, where
  the LongNav-R1 arms turn 30 deg (R1's native discrete turn). On the smoke episode
  (grCommercial_scene15_0) LightNav requested a pure 18 deg rotation on >90% of steps; with 30 deg
  turns it over-rotated and oscillated left/right (88 of 133 turn runs one step long, 7% FORWARD,
  never near the goal), with 15 deg turns it swept in one direction and moved forward three times
  as often (20% FORWARD, reached the goal area) -- so each model runs at its own turn granularity.
- **Frames.** The LongNav-R1 arms saw the 4:3 NavVerse frames at their native aspect (its
  processor never stretches), so LightNav-0 runs with upstream's `aspect_mode=keep`: 640x480 ->
  384x288 (the closest 4:3 size at the checkpoint's pixel budget) -> a 9x12 token grid, 108 image
  tokens per frame against 112 for the trained 448x256, and, with the model's ceil pooling, **30
  tokens per pooled 2-frame history pair** (28 on HM3D) and 9 per long-tier pair. The server
  refuses a different model frame size (`--expect-video-size 288 384`) because the budget below
  assumes that grid. The camera itself differs from LightNav's training camera (82 deg HFOV
  here, 120 deg there), and NavVerse's ~1.3 s of simulated time per step does not match the
  model's 4 Hz timestamps (HM3D's 1 s steps do not either); both are the same across the arms.
- **Keep arms.** `--prune-method recency --prune-k K-2 --prune-tokens-per-frame 15`
  (`eval_prune.py`'s rule under the eager masked kernel on every layer, a deletion ledger per
  episode): the current pair plus the K-2 newest history frames keep their image tokens, every
  older block's image tokens are hidden while its timestamp and vision markers stay -- the HM3D
  `lightnav_keep_k{K}` = `lightnav_recency_b{K-2}` protocol with the budget expressed in 30-token
  pairs (K = 32: 450 tokens = 15 pairs; 16: 7; 8: 3; 4: 1). `nvl_full` runs the unmodified compiled
  kernel (as HM3D's `lightnav_full`; the kernel swap was noise-level there, +.04 SR). Pruning
  records go to `runs/<arm>/prune/<launch>/episode_NNN.jsonl` (gitignored; a sub-directory per
  server launch so a resumed arm never appends to the first launch's files); the step log carries
  `n_kept` / `budget` / `n_pool` per step.
- **Cost.** Sim-bound like the LongNav-R1 arms (~1 s per step; LightNav's eager step is
  100-400 ms): 5-10 h per arm, one arm at a time (one vLLM instance plus one Isaac Sim worker
  fill the GPU). The sweep ran 2026-10-28 13:50 to 2026-10-30 05:13 (6.3 / 6.7 / 7.5 / 9.8 /
  9.1 h for full / 32 / 16 / 8 / 4; the server held 12.0-12.5 GB, 113 ms per step on the
  compiled kernel, 250-280 ms eager; longest prompt 1443 tokens of the 2560 limit; no decode
  failures).
- **Findings (100 episodes, SR / SPL / mean steps / forced stops).** `nvl_full` .12 / .056 /
  206 / 21; keep K = 32 .15 / .037 / 187 / 20 (dSR +.03, p = .60); K = 16 .12 / .031 / 220 / 25
  (0, p = 1); K = 8 .06 / .021 / 305 / 45 (dSR -.06, p = .11; dSPL -.035, p = .02; +99 steps,
  p = .0003); K = 4 .04 / .009 / 307 / 43 (dSR -.08, p = .06; dSPL -.047, p = .004); K = 1
  (no history, current frame doubled; run 2026-10-30, 9.3 h) .01 / .003 / 384 / 66 (dSR -.11,
  p = .001; dSPL -.053, p = .001; 5 cap episodes within the radius re-scored, 16 oracle
  successes, FORWARD on 3% of steps). LightNav-0
  lands where LongNav-R1 does on this benchmark (full .12 vs .10, K = 4 .04 vs .04, K = 1 .01
  vs .00) and its curve has the same shape: flat from full context down to K = 16, then a drop
  at K <= 8 and a collapse at K = 1 (as on HM3D, where K = 1 scores .02). That
  is the opposite of HM3D, where the same "old blocks kept" protocol is flat to K = 4
  (.79-.84): on NavVerse keeping the prompt skeleton does not rescue the short windows. The
  loss at K <= 8 is wandering, not stopping: forced stops at the cap without ever reaching the
  goal radius rise from 20 to 34 / 37 episodes (53 at K = 1) while wrong stops fall from 46 to 37 / 32 (10 at K = 1)
  (`figures/failure_modes_lightnav.png`), and NavVerse's oracle success drops from 21-24 to
  20 / 12. At every K the dominant failure is a STOP at the wrong place (32-46 of 100; for
  LongNav-R1 it is running out of steps), and the model spends 82-92% of its steps turning
  (FORWARD on 8-18% of steps, highest at K = 16), consistent with the smoke episode. The
  robot tipped over in 12-23 episodes per arm, as for LongNav-R1. Caveats as above: 82 deg
  camera, 4:3 frames, discretized 15 deg turns, and vLLM run-to-run drift, all shared by the
  five arms.

The pre-paper sweep scripts (`tools/run_context_window_ablation.sh`, `tools/plot_context_window_ablation.py`,
`baselines/run_context_window_ablation.sh`; legacy `context_window`, header cut, no CIs) were removed
once this experiment replaced them; the runs they produced remain under `dump/`.
