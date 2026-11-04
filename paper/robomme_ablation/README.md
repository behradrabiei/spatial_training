# RoboMME memory ablations on a manipulation VLA

The three navigation studies (context window, information horizon by layer, token pruning)
asked of a manipulation policy with an explicit visual memory: the MME-VLA suite's
`perceptual-framesamp-*` checkpoints (pi0.5 backbone + a 512-token perceptual memory of 32
sampled frames), in their two integration variants, on the RoboMME **Counting** suite (BinFill,
PickXtimes, SwingXtimes, StopCube; 4 × 50 test episodes, no demonstration video, so execution
history is the only memory). One run per arm, model seed 7, as in the navigation studies.

| model | memory integration | nav analogue | status (2026-11-03) |
|---|---|---|---|
| `context` = `perceptual-framesamp-context` | the 512 memory tokens are prepended to the VLM prefix and attended by every query in all 18 layers | the closest analogue of the navigation models' KV cache; **run first** (user decision 2026-11-03) | ctx window **complete** (11 arms); layer horizon **complete** (8 arms, cliff between L=14 and 16); pruning **paused** (user decision 2026-11-03, after the modul families) |
| `modul` = `perceptual-framesamp-modul` | each of the 18 action-expert layers cross-attends its action tokens to the memory and FiLM-modulates the pre-FFN norm; the VLM prefix never sees memory | memory as a modulator (the RoboMME paper's best variant, 44.5 % average) | ctx window **complete** (14 nav arms + 13 side arms: uniform, horizon_w, fill), horizon **complete** (2026-11-04 03:10); sweep finished 2026-11-04 07:58, `blind_l17` added 09:25, nothing running; pruning not queued |

## Layout

```
paper/robomme_ablation/            code
  arms.py                          arm table: name -> family, x, ablation JSON (RUN_ORDER)
  run_all.sh                       sequential driver (MODEL=..., ARMS="..." subset, skips finished arms)
  aggregate.py                     raw runs -> per-model CSVs + combined summary
  plot.py                          per-model + combined figures (vln env)
paper_results/robomme_ablation/    results (untracked; raw runs stay on the vault)
  results/summary.csv              all models, one row per arm, leading `model` column
  figures/                         combined figures, one colour per model
  modul/   {results, figures, runs}   per-model: summary.csv + per_episode.csv, figures, compact run copies
  context/ {results, figures, runs}
/mnt/vault/robomme/runs/ablation/          raw modul runs,   <arm>/ckpt79999/seed7/{episodes.jsonl,log.json,progress.json,policy_calls.jsonl,videos/}
/mnt/vault/robomme/runs/ablation_context/  raw context runs (same layout); driver.log in each
/mnt/vault/robomme/runs/ablation_checks{,_context}/   identity-check action dumps
```

Figures (`*_partial` while a family is running, final name once complete): `ctx_ablation` (SR |
steps vs K; recency solid, "old frames' timestamps kept" dashed; the modul uniform / fill side
arms are in summary.csv only, not drawn) and
`information_perf` (SR | steps vs first blinded layer) are **final for both models** (2026-11-04);
`ctx_horizon` (time-span window, modul side arms only) is final for modul; `budget_curves` (SR |
steps vs frames' worth kept; one row per model in the combined figure) exists only as `_partial`
(pruning not run; the figure shows the recency reference line alone).

Setup of the simulator and the policy server: `robomme_benchmark/SETUP_LOCAL.md`. The ablation
code is a patch of the pinned policy repo (`robomme_benchmark/patches/policy_repo_ablation.diff`,
branch `ablation` of the submodule): `src/mme_vla_suite/ablation.py` (selection logic, unit-tested
in `tests/test_ablation.py`), hooks in `policies/policy.py` (memory assembly per call) and
`models/integration/history_gemma.py` + `history_pi0.py` (per-layer blinding), `--ablation` on
`scripts/serve_policy.py`, and an `episodes.jsonl` written by `examples/robomme/eval.py`.

## How the models remember (facts the arms rely on)

Shared by both models:

- Memory = 32 frames × 16 tokens (4×4-pooled SigLIP features, 2048-d) + a 3-D positional embedding
  keyed by the absolute step index, right-padded to 512 slots with a mask. The frames are evenly
  spaced over the whole episode so far and always include step 0 and the current step. Training
  used the same sampler, so all 32 slots are filled from step 31 on.
- The current front and wrist images (256 tokens each) sit in the VLM prefix regardless of memory.
- The policy is called once per 16 env steps (a 16-action chunk); the buffer holds every env
  step. "Decision-time frames" are the buffer steps at which the policy was called.
- All arms are masking or selection over the fixed 512-slot memory: no recompilation, and no
  leakage question (the memory is rebuilt from the full buffer at every call; there is no
  cross-call cache).

Integration-specific:

- `context`: the memory tokens are the first 512 positions of the prefix (then images, then text);
  all prefix and action queries attend to them in all 18 layers. Blinding from layer L masks the
  memory and current-image key columns for every query at layers ≥ L.
- `modul`: a per-layer `MemoryAttention` from the action tokens to the memory conditions the
  pre-FFN norm (scale/shift). Blinding from layer L masks the current-image key columns and zeroes
  the modulation (falls back to its learned bias).

## Arms (`arms.py`)

| Family | Arms | Navigation analogue |
|---|---|---|
| ctx | `full` (native 32-frame sampler, shared reference); `recency_k{32,16,8,4}` = newest K decision-time frames incl. the current one, in the first K slots, the rest masked; `k1` = current frame only; `recency_k{32,16,8,4}_keep`, `k1_keep` = the "old blocks kept" control: the native 32 frames with their own position embeddings (timestamps), the image + state content of every frame older than the window's oldest step replaced by that step's content | Exp 1 window K (K counts the current frame) and its `lightnav_keep_k{K}` control (window frames visible, older blocks' timestamps kept) |
| ctx side arms (modul only, not in the nav protocol) | `uniform_k{16,8,4}` = K evenly spaced frames (the model's own sampler at budget K); `horizon_w{512,…,16}` = the native 32-frame sampler restricted to the newest W env steps (window in time, frame count unchanged); `uniform_k{16,8,4,1}_fill` = the uniform arms with the K frames duplicated cyclically to the 32 slots (count control) | none; added to separate frame count from time span after the modul recency arms collapsed |
| horizon | `blind_l{2,4,…,16,17}`: from layer L on, no query reads a visual token (see above); text, state/time and action tokens untouched; none = `full` | Exp 6 blind-from-L (all visual tokens incl. the current frame hidden from layers ≥ L, text never) |
| prune | `{random,divprune,dart}_b{16,8,4,2,1}`: 16·k history memory tokens kept by a sticky per-episode ledger (candidates = survivors ∪ the previous decision frame's 16 tokens; dropped tokens never return); the current frame's 16 tokens are protected and not counted; b0 = `k1`; recency = the ctx recency arms at K−1 | Exp 4b flat pruning (no window, k frames' worth, current frame protected, permanent deletion, applied at layer 0 for all layers) |

Selectors: `random` draws uniformly (seeded by (7, episode)); `divprune` is greedy max–min cosine
distance on the memory-encoder output (post-projection, as the navigation DivPrune used
post-projector embeddings); `dart` takes 8 pivots by largest L2 norm and keeps the tokens least
cosine-similar to each pivot. FastV (attention salience of the action queries) is not run: the
decision tokens are noisy flow-matching action tokens, so an attention score needs the layer-0
memory cross-attention of a noise sample; left as a stretch arm.

Dropped on purpose (no analogue for a continuous-action flow-matching policy): the KL keep-one /
leave-one probes and the KL / entropy keep-one selectors; SPL (no shortest path: we report SR and
episode steps); the evict-vs-recompute leakage arms.

## Run, aggregate, plot

```bash
MODEL=perceptual-framesamp-context ARMS="full recency_k32 ..." bash paper/robomme_ablation/run_all.sh   # tmux robomme_abl_ctx
MODEL=perceptual-framesamp-modul bash paper/robomme_ablation/run_all.sh                                # tmux robomme_abl; skips finished arms
python paper/robomme_ablation/aggregate.py                      # all models -> <model>/results + results/summary.csv
/home/brabiei/miniconda3/envs/vln/bin/python paper/robomme_ablation/plot.py [--suffix _partial]   # per-model + combined figures
```

`--model modul|context` restricts either script to one model. Metrics: SR and steps per episode
with 95 % bootstrap CIs (2000 resamples, seed 17), deltas vs `full` paired on (task, episode) with
a sign-flip test (10 000 resamples), per-task SR, timeout rate. `plot.py` leaves out arms that
are still running (their task mix is biased: tasks run in sequence and differ a lot in SR).

## Checks

- `tests/test_ablation.py` (policy venv, `JAX_PLATFORMS=cpu`): window indices (recency, uniform,
  horizon, fill, keep), ledger budget/permanence/seeding, slot layout, blind vectors — 15 passed.
- Identity: a 40-call synthetic episode (`scripts/ablation_check_client.py`, seeded random images
  and states, 16 new frames per call so the sampler runs past 32 frames) against live servers of
  the unpatched commit (ac846a9, git worktree `/mnt/vault/robomme/policy_repo_ref`) and the
  patched branch, same checkpoint and seed; action chunks compared bit for bit.
- Smokes: `uniform_k32` (the native sampler through the ablation path) equals `full` up to the
  noise floor; `blind_l0` changes the actions by O(1); every family's logged selections
  (`policy_calls.jsonl`: frames, n_mem_tokens, visible_from) are as specified.

| comparison (max abs diff; mean) | modul | context |
|---|---|---|
| unpatched vs unpatched, same process, two episodes | 0 (bit-identical) | 0 (bit-identical) |
| unpatched vs unpatched, two processes | 6.0e-3; 2.7e-4 (XLA autotuning differs per process) | 0 (bit-identical) |
| patched, no ablation vs unpatched | 6.2e-3; 3.4e-4 (noise floor) | 1.6e-2; 2.7e-4 |
| patched `blind_from_layer=18` vs unpatched | **0, identical md5** (second process) | 1.9e-2; 2.6e-4 |
| patched `uniform_k32` vs unpatched | 5.6e-3 (noise floor) | 1.6e-2; 2.7e-4 (= patched, no ablation) |
| `blind_from_layer=0` vs unpatched | 2.0 | 2.0; 1.1e-1 |
| gentlest real ablation (`keep4`) | — | 6e-3 mean |

The modul patch reproduces an unpatched process exactly; the 6e-3 spread is that model's own
cross-process kernel-autotuning noise. The context model is bit-reproducible across processes,
and its patched graph (extra per-layer mask selects threaded through the layer scan) compiles to
slightly different kernels: mean 2.7e-4, the same profile as the modul model's cross-process
noise and 20× below the gentlest ablation. All arms of a model share the patched graph, so
arm-to-arm comparisons are unaffected. Raw dumps: `/mnt/vault/robomme/runs/ablation_checks{,_context}/`.

## Results

Numbers are SR (ΔSR vs `full`, paired; p from the sign-flip test), then steps per episode and the
timeout rate; per-task SR in `<model>/results/summary.csv`.

### Context window

Figures `figures/ctx_ablation.png` (frame-count axis) and `figures/ctx_horizon.png` (time-span
axis, modul only). **context complete 2026-11-03 17:07; modul complete 2026-11-04 07:58 (nav
arms 01:33, side arms horizon_w / fill 07:58).**

| arm | context SR (ΔSR) | context steps / timeouts | modul SR (ΔSR) | modul steps / timeouts |
|---|---|---|---|---|
| full | .50 | 402 / 0 | .70 | 431 / .01 |
| recency_k32 | .34 (−.17, p<.001) | 449 / 0 | .23 (−.47) | 437 / .01 |
| recency_k16 | .30 (−.21, p<.001) | 449 / 0 | .23 (−.47) | 509 / .05 |
| recency_k8 | .37 (−.13, p=.003) | 454 / 0 | .23 (−.47) | 555 / .09 |
| recency_k4 | .37 (−.13, p=.002) | 473 / .01 | .25 (−.45) | 500 / .10 |
| k1 | .05 (−.46) | 876 / .52 | .21 (−.49) | 499 / .10 |
| recency_k32_keep | .51 (+.01) | 404 / 0 | .67 (−.03, p=.31) | 423 / 0 |
| recency_k16_keep | .53 (+.03, p=.29) | 412 / .01 | .68 (−.02, p=.50) | 416 / 0 |
| recency_k8_keep | .49 (−.02, p=.72) | 399 / 0 | .64 (−.06, p=.035) | 418 / 0 |
| recency_k4_keep | .47 (−.04, p=.33) | 415 / .01 | .54 (−.16, p<.001) | 404 / .01 |
| k1_keep | .24 (−.27) | 556 / .13 | .26 (−.44) | 527 / .07 |

Per task (`full`): context BinFill .42 / PickXtimes .58 / SwingXtimes .82 / StopCube .18; modul
.40 / .92 / .96 / .50. The context checkpoint is the weaker policy (SwingXtimes, StopCube), so
only shapes are comparable across the two models, not levels.

Context model:

- No history (`k1`) is a floor at SR .05 with half the episodes timing out: this model uses its
  memory tokens.
- The decision-frame recency window loses .13–.21 SR at every K with no ordering in K, already at
  K=32, which removes no frame in episodes shorter than 512 steps. The loss is the changed memory
  layout (stride 16, one frame per policy call, ≤ K slots filled), not forgetting. SwingXtimes
  carries most of it (.82 → .42–.50), steps +47–71.
- The "old blocks kept" control recovers `full` at every K ≥ 4 (.47–.53, all p ≥ .29): the native
  32 slots with their own timestamps, with the content older than the window replaced by the
  cutoff step's content, cost nothing down to a 4-call (48-step) window of real content. Same
  shape as LightNav-0 on HM3D (keep line flat at K ≥ 4 against the window's cliff).
- `k1_keep` (32 slots all carrying the current frame's content, native timestamps) sits at .24,
  far above `k1` (.05) and far below `full`: the slot skeleton alone is worth ~.2 SR, and the
  remaining ~.25 needs real past content, which 4 decision frames already supply.

Modul model (side arms `uniform_k16/8/4` = .18 / .22 / .27, timeouts .13–.20, steps +75–200):

- Every recency arm sits at the no-history level (`k1` .21), including K=32, so the collapse is
  the frame grid, not forgetting. `uniform_k16` (the model's own sampler at budget 16, full
  coverage) collapses too, so coverage is not the issue either: the model reads the number of
  filled memory slots (always 32 past step 31 in training) as elapsed time, and any memory with
  fewer frames looks like "episode just started".
- The `_keep` arms confirm it: with the 32 slots and timestamps in place, hiding content older
  than the newest 32 / 16 calls costs nothing (.67 / .68 vs .70), 8 calls (112 steps) costs .06
  (StopCube .50 → .36), 4 calls (48 steps) costs .16 (SwingXtimes .96 → .54, StopCube → .38).
  So, unlike the context model (flat to K=4), this model does read content older than ~50–100
  steps on the two tasks that need the longest look back; but the solid line's −.47 at every K
  is almost entirely the slot structure. `k1_keep` (32 slots of the current frame) .26 ≈ `k1` .21:
  the skeleton alone is worth little here (vs ~.2 on the context model).
- Per-task at `recency_k4_keep`: BinFill .44 / PickXtimes .78 / SwingXtimes .54 / StopCube .38.

Modul time-span window (`horizon_wW`: the native 32-frame sampler run inside the newest W env
steps, own timestamps; the memory of an episode that started W steps ago) against its control
(`_keep` at the matching span) and the decision-frame recency arm; figure `ctx_horizon.png`:

| span W (env steps) | window `horizon_wW` | control `recency_k{W/16}_keep` | `recency_k{W/16}` | per task, window (B/P/S/C) |
|---|---|---|---|---|
| full | .70 | .70 | .70 | .40 / .92 / .96 / .50 |
| 512 | .65 (−.05, p=.048) | .67 | .23 | .32 / .84 / .94 / .50 |
| 256 | .48 (−.22) | .68 | .23 | .38 / .64 / .52 / .36 |
| 128 | .36 (−.34) | .64 | .23 | .38 / .42 / .40 / .22 |
| 64 | .27 (−.43) | .54 | .25 | .24 / .36 / .32 / .16 |
| 32 | .15 (−.55; timeouts .29, steps 737) | – | – | .12 / .20 / .20 / .08 |
| 16 | .20 (−.50) | .26 (`k1_keep`) | .21 (`k1`) | .16 / .32 / .26 / .04 |

- The time-span window is the navigation window shape: SR falls monotonically with W (.65 → .48
  → .36 → .27 → .15/.20), steps and timeouts rise below 64, and W=512 is `full` for most
  episodes (mean length 431). Unlike the frame-count recency line it is graded, because the slot
  count stays at 32 and only the visible span moves.
- Most of that fall is still the skeleton, not forgetting. At every span the window has *more*
  real recent content than its control (32 frames inside W vs W/16 real frames plus duplicated
  cutoff content in the old slots), yet the control is far above it: at W=256 .68 vs .48, at
  128 .64 vs .36, at 64 .54 vs .27. What the window loses and the control keeps is the native
  grid: a memory whose first slot is step 0 and whose stride is `episode length / 32`. The model
  has never seen a memory starting at step `t − W`, and it reads that skeleton (the time stamps
  of the 32 slots) as elapsed time; moving the skeleton forward costs .2–.3 SR on its own. The
  genuine content cost at matched span is the control's own loss: ≤ .06 down to 112 steps, .16
  at 48 steps, as above.
- Decomposition at W=64 (−.43 total): .27 is the moved skeleton (window vs control), .16 is the
  forgotten content (control vs `full`). At W=128 (−.34): .28 skeleton, .06 content. At W=256
  (−.22): .20 skeleton, .02 content. So on this model a time window of any length below the
  episode is dominated by the layout change, and the forgetting signal only becomes visible
  through the control.
- `horizon_w32` is below `horizon_w16` (.15 vs .20, timeouts .29 vs .09): at W=32 the sampler
  places 32 frames on consecutive steps (stride 1) with a 32-step-old first slot, the densest
  OOD memory; at W=16 the window holds fewer distinct steps than slots, so it degenerates to a
  near-duplicate current-frame memory like `k1_keep`.

Modul count controls (`uniform_kK_fill`: the K evenly spaced frames of `uniform_kK`, each
duplicated cyclically to fill the 32 slots with its own timestamp): k16 .49 / k8 .29 / k4 .18 /
k1 .20 vs `uniform_k16/8/4` .18 / .22 / .27 and `k1` .21.

- Filling the slots rescues a 16-frame memory (.18 → .49, SwingXtimes .20 → .70) but not an
  8-frame one (.29) or below, and `k1_fill` = `k1`. So the number of filled slots is part of what
  the model reads, but not all of it: `recency_k16_keep` (32 distinct native timestamps, 16 real
  frames) is at .68 against `uniform_k16_fill`'s .49 with the same 16 real frames on 16 distinct
  timestamps. Distinct native timestamps in all 32 slots matter beyond the slot count.

#### Is the recency arm the navigation window? (discussion 2026-11-03; modul horizon_w run 2026-11-04, context arms still pending)

Mostly, with one difference that drives the flat solid line. What matches: K counts decision
frames incl. the current one; a removed frame is unattendable at every layer; the memory is rebuilt
per call, so this is nav's leak-free recompute mode; each kept frame keeps its own timestamp;
`_keep` is the LightNav "old blocks kept" control. What is off: in nav, window K is `full` with
the oldest frames deleted and nothing else changed. Here `full` is 32 frames evenly spaced over
all env steps (step 176: 0, 5, 11, …) while `recency_kK` keeps the newest K policy-call steps
(K=4: 128, 144, 160, 176), i.e. mostly frames that were not in `full`, on a stride-16 grid the
native sampler never produces after step 31, in K slots instead of 32. So the solid line changes
three things at once (forgets old content, re-grids the memory, shrinks the slot count) and only
the first is the nav operation; hence `recency_k32`, which forgets nothing in a 400-step episode,
loses as much as `recency_k4`, and K is not comparable to nav's K (nav K=32 deletes most of a
200-step episode; here it deletes nothing).

The clean analogue is `horizon_wW`: the model's own rule ("32 frames spanning the episode so
far") applied inside the newest W env steps, so the memory looks like the native memory of an
episode that started W steps ago (32 slots, own timestamps), and only the visible time span
changes. The `_keep` arms are already its control (cutoff = oldest call step: k4_keep ↔ w64,
k8_keep ↔ w128, k16_keep ↔ w256, k32_keep ↔ w512; `ctx_horizon` figure pairs them). Proposed:
add `horizon_w{512,…,16}` to the context sweep (6 arms, ~25 min each) and report the window
family on a time-span axis (horizon_w = window, keep = control, recency = cost of the
decision-frame grid). Done for modul (table above, `ctx_horizon.png`): the time-span window is
graded in W as in nav, and the control shows that most of its loss is the moved skeleton. Not yet
run for context (the sweep ended 2026-11-04 07:58; `MODEL=perceptual-framesamp-context
ARMS="horizon_w512 horizon_w256 horizon_w128 horizon_w64 horizon_w32 horizon_w16" bash
paper/robomme_ablation/run_all.sh` after removing the `STOP` file in the context runs dir). Remaining model-side difference: history frames are 16 pooled tokens vs 256
for the current frame in the prefix (nav history frames keep their token count); it bears on the
pruning family, not the window.

### Information horizon by layer

Figure `figures/information_perf.png` (context: `context/figures/information_perf.png`).
**Both models complete (context 2026-11-03 23:28, modul 2026-11-04 03:10; `blind_l17` added 2026-11-04 09:25).**

| first blinded layer L (0-based; layers L..17 blind) | context SR | context steps / timeouts | modul SR | modul steps / timeouts |
|---|---|---|---|---|
| none (`full`) | .50 | 402 / 0 | .70 | 431 / .01 |
| 2, 4, 6, 8, 10, 12 | .00 each | 1050 / .75 each | .00 each | 1050 / .75 each |
| 14 | .01 (2 successes) | 1017 / .68 | .005 (1 success) | 914 / .49 |
| 16 | .51 (+.01, p=1.0) | 420 / .01 | .61 (−.09, p=.006) | 425 / 0 |
| 17 (last layer only) | .49 (−.01, p=.80) | 413 / .01 | .69 (−.01, p=1.0) | 419 / .01 |

Per task at L=16: context BinFill .46 / PickXtimes .58 / SwingXtimes .82 / StopCube .16 (= `full`
on every task); modul BinFill .46 / PickXtimes .90 / SwingXtimes .92 / StopCube **.14** (`full`:
.40 / .92 / .96 / .50) — the modul loss at L=16 is StopCube alone (−.36), the other three tasks
are within noise.

- The cliff is sharp and late: blinding the last 4 layers (L=14) still collapses the policy to
  the no-success floor (most episodes at the step cap), blinding the last 2 (L=16) costs nothing
  on any task. So the action stream has absorbed everything it needs from the visual tokens
  (memory + current images) by the end of layer 15 of 18, and not before layer 14.
- Relative depth 14/18 = 78 %, later than the navigation models (LongNav-R1 recovers between
  16 and 20 of 28 = 57–71 %; LightNav-0 between 20 and 24 of 36 = 56–67 %). The flow-matching
  suffix is one reason to expect this: the action tokens start every denoising pass as noise, so
  all visual information must be pulled in within the unblinded layers of each pass, whereas a
  nav decoder's decision token sits on a residual stream that was built over many prior tokens.
- The collapse below the cliff is total (no successes, 75 % timeouts) rather than graded, as in
  nav below its cliff; the one graded point is L=14 (context 2 successes, timeouts .68; modul
  1 success, timeouts .49 — modul fails faster rather than better).
- **Modul is not fully recovered at L=16**: SR .61 vs .70 (p=.006), entirely StopCube (.50 → .14;
  BinFill/PickXtimes/SwingXtimes unchanged). StopCube is the one Counting task whose success
  hinges on reacting to a visual cue (stop when the cube count is reached) rather than on
  repeating a motion, and on modul the last two layers' modulation still carries that cue. On the
  context model StopCube is at .16–.18 with or without the last two layers, i.e. the context
  model never solves it well enough for the blind cut to show. So the "intact at L=16" verdict
  holds for the three motion-repetition tasks on both models and for the cue task only on context,
  where there is little to lose. `blind_l17` (last layer only) restores modul to `full` (.69,
  StopCube .40 vs .50, p=1.0 overall) and leaves context at `full` (.49): the StopCube cue is
  absorbed in layer 16, and the last layer reads nothing from vision on either model.
- A finer sweep (L=15, possibly 13) would locate the cliff to one layer; not run. With L=17 the
  profile is: blind from ≤14 collapses, from 16 costs only modul's StopCube, from 17 costs nothing.

### Token pruning

Figure `figures/budget_curves.png`. **Paused for both models** (user decision 2026-11-03 23:05; context
family relaunch command in Run; modul not queued). The recency line of the figure is already the ctx
recency arms at K−1.
