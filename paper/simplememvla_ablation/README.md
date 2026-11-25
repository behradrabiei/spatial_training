# SimpleMemVLA context-window and layer-depth ablations (RoboMME)

How many history frames does SimpleMemVLA need on RoboMME's memory tasks? The navigation
context-window experiment (`paper/ctx_ablation/`) on a manipulation VLA: success rate and steps
vs the number of history frames the model is shown, on the Permanence suite (VideoUnmask,
ButtonUnmask, VideoUnmaskSwap, ButtonUnmaskSwap; 4 × 50 test episodes).

Model and environment setup: `robomme_benchmark/SETUP_LOCAL.md` (SimpleMemVLA section). The
released RoboMME checkpoint is evaluated as released (seed 0, `sdpa`, non-pipelined, 16 executed
actions per decision, 1,300-step cap, group size 4). The MME-VLA ablation in
`paper/robomme_ablation/` is a different model family and is not continued.

## How the model sees history

At every decision (every 16 simulator steps) the prompt is rebuilt from scratch and always has the
same shape: 60 history slots, each a two-frame patch of the front camera with a fixed timestamp
(`<0.2 seconds>` … `<59.2 seconds>`, 64 tokens per slot), then the current wrist image, then the
instruction (4,541 tokens). The 120 slot frames are the current frame and the 119 before it at one
frame per 10 simulator steps (2 fps). Slots that reach back before the episode start hold the
episode's first frame (`max(0, cur + offset)` in `robomme_sim/policy.py`). Nothing is carried
between decisions, so this video is the model's only memory. For the video tasks the reset
delivers the demonstration video (67–169 frames) into the same buffer.

The released RoboMME `config.json` has no `variable_history` key, so upstream's evaluator uses this
fixed, first-frame-filled layout; the repository's current training recipe describes a video that
grows early in the episode instead. The released checkpoint is used as is.

## Definition

**Context size K** = the number of history frames visible at decision time, including the current
one (the navigation definition). One frame is 0.5 s = 10 simulator steps.

Arm `k{K}` (`CONTEXT_FRAMES=K`) applies the model's own history rule to a history that starts
(K−1)·10 steps ago (`windowed_history_positions` in `robomme_sim/policy.py`). The model input is
exactly what the unmodified policy produces for an episode that began K frames ago: the newest K
frames as they are, every older slot filled with the oldest visible frame. Prompt shape,
timestamps, wrist image, instruction, proprioception and action generation are unchanged. `full`
leaves the option unset and runs upstream's code path.

| K | reach | | K | reach |
|---|---|---|---|---|
| full | 59.5 s (1,190 steps) | | 8 | 3.5 s (70 steps) |
| 32 | 15.5 s (310 steps) | | 4 | 1.5 s (30 steps) |
| 16 | 7.5 s (150 steps) | | 1 | current frame only (not run) |

K = 1 is implemented but was skipped on Permanence (2026-11-16): K = 4 is already at .155.

### Parallels with the navigation experiment

| Navigation (`paper/ctx_ablation/`) | Here |
|---|---|
| K = frames visible incl. the current one | same; a frame is one 2 fps history sample |
| Window = the input of an episode that started K steps ago (LightNav-0 window, Uni-NaVid, leak-free LongNav-R1) | same statement, checked: window = native rule on a truncated history |
| Leak-free arms for models that rebuild their prompt | prompt and backbone are recomputed every decision and each two-frame patch is encoded on its own, so there is one arm per K |
| LightNav-0 "old blocks kept" control (its window deleted the prompt skeleton) | no separate arm needed: the prompt skeleton is constant by construction |
| Sampling grid untouched | native stride-10 grid anchored at the current frame (the MME-VLA `recency` arm re-gridded) |
| K = 1 = current frame only | every slot holds the current frame, the model's native step-0 input |
| Large-K negative control | K = 32 reaches 310 steps, about a whole Permanence episode |

Limit: the model's only cue for elapsed time is the content of its history (how many leading slots
are static copies), so a K-window also makes the episode look K frames old. Unlike LightNav-0
there is no layout that removes old content while keeping that cue.

## Running

```bash
bash paper/simplememvla_ablation/run_ctx.sh                    # full, k32, k16, k8, k4, k1
ARMS="full k32 k16 k8 k4" bash paper/simplememvla_ablation/run_ctx.sh   # what was run on Permanence
ARMS="k8 k4" bash paper/simplememvla_ablation/run_ctx.sh       # a subset
SUITE=counting bash paper/simplememvla_ablation/run_ctx.sh     # another suite
robomme_benchmark/.venv-simplememvla/bin/python paper/simplememvla_ablation/aggregate.py
/home/brabiei/miniconda3/envs/vln/bin/python paper/simplememvla_ablation/plot.py   # --suffix _partial
```

Raw runs: `/mnt/vault/robomme/runs/simplememvla_ablation/<suite>/<arm>/` (`episodes.jsonl`,
`results.json`, `run_config.json`, `eval.log`, `videos/`). Finished arms are skipped, partial arms
resume, `touch <suite>/STOP` stops the driver before the next arm. Outputs:
`paper_results/simplememvla_ablation/<suite>/{results,figures,runs}`.

Statistics as in the navigation studies: 95 % percentile-bootstrap CIs (2,000 resamples, seed 17);
differences vs `full` paired on (task, episode), two-sided sign-flip test (10,000 resamples).

## Checks

- Window rule, CPU only: K = 120 reproduces the native frame selection at every step; for each K
  the window equals the native rule applied to a history that starts K frames ago, and equals the
  native selection clamped at the cutoff; at most K distinct frames, none older than (K−1)·10
  steps; the distinct frames occupy the newest K slots, aligned with the two-frame patches.
- Every `episodes.jsonl` record carries `max_history_age_steps` and `max_history_frames` (the
  oldest and the most distinct history frames shown in that episode); `summary.csv` reports the
  maximum per arm.
- K = 8 smoke (8 episodes): every record at 70 steps / 8 frames.

## Results

Permanence suite, 200 episodes per arm (50 per task). SR with ΔSR vs `full` (paired; sign-flip p),
mean steps, and per-task SR. Sweep complete 2026-11-16 (14:40–18:40, 4 h); K = 1 not run. Figures:
`paper_results/simplememvla_ablation/permanence/figures/ctx_ablation.png` (SR | steps vs K) and
`ctx_by_task.png` (the same two panels with one line per task), both in the paper's single-column format (3.5 in wide, 8 pt text, means only;
`paper/figstyle.py`, the format of `paper_results/ctx_ablation/figures/ctx_ablation`).

| arm | SR (ΔSR; p) | steps | fail / timeout | VideoUnmask | ButtonUnmask | VideoUnmaskSwap | ButtonUnmaskSwap |
|---|---|---|---|---|---|---|---|
| `full` | .960 [.930, .985] | 257 | .03 / .01 | 1.00 | .98 | .90 | .96 |
| `k32` | .880 [.835, .925] (−.080; p = .0002) | 267 | .11 / .01 | 1.00 | .96 | .80 | .76 |
| `k16` | .580 [.510, .650] (−.380; p = .0001) | 316 | .38 / .04 | .84 | .88 | .52 | .08 |
| `k8` | .320 [.260, .385] (−.640; p = .0001) | 478 | .43 / .25 | .76 | .36 | .16 | .00 |
| `k4` | .155 [.105, .205] (−.805; p = .0001) | 611 | .49 / .35 | .28 | .20 | .14 | .00 |

**`full` (2026-11-16, 33 min).** The unmodified model scores 96.0 %, the paper's Permanence average
(their per-task figure: 100 / 100 / 90 / 94; here 100 / 98 / 90 / 96). Episodes are short: 139
steps on VideoUnmask, 286 on ButtonUnmask, 193 on VideoUnmaskSwap, 410 on ButtonUnmaskSwap, so the
model never has more than 35 distinct history frames on VideoUnmask, and only the two timeouts
(1,301 steps) ever fill all 120 slots.

**`k32` (2026-11-16, 35 min).** −.08 SR, all of it on the two Swap tasks (VideoUnmaskSwap .90 → .80,
ButtonUnmaskSwap .96 → .76); VideoUnmask and ButtonUnmask are unchanged. K = 32 was expected to be a
negative control, and it is one exactly where it should be: taking the episodes `full` solves, **none
of the 118 that fit inside the window's 310-step reach is lost, and all 17 losses are among the 74
longer ones** (VideoUnmaskSwap 6 of 19, ButtonUnmaskSwap 10 of 39, ButtonUnmask 1 of 11, VideoUnmask
0 of 5; the lost episodes are 408–1,120 steps long including the reset video). The Swap episodes are
the long ones (median 264 and 320 steps with the video against 162 and 224), so their reveal is the
first thing to fall out of a 15.5 s window. Losses are wrong picks, not timeouts (fail .03 → .11),
and steps barely move.

**`k16` (2026-11-16, 45 min).** −.38 SR. Every task is now affected and the order follows episode
length: ButtonUnmaskSwap collapses (.96 → .08), VideoUnmaskSwap halves (.90 → .52), ButtonUnmask and
VideoUnmask lose .10 and .16. The reach rule holds again: of the episodes `full` solves, **none of
the 18 that fit inside the 150-step reach is lost** (all VideoUnmask) and the 76 losses are all among
the 174 longer ones (VideoUnmask 8 of 32, ButtonUnmask 5 of 49, VideoUnmaskSwap 19 of 45,
ButtonUnmaskSwap 44 of 48). Still mostly wrong picks (fail .38) with a few timeouts appearing on the
Swap tasks (.04); ButtonUnmaskSwap episodes lengthen (410 → 590 steps).

**`k8` (2026-11-16, 55 min).** −.64 SR. No episode fits inside the 70-step reach any more (the
shortest `full` episode is 146 steps with its video), so the reach rule has nothing left to protect:
128 of the 192 episodes `full` solves are lost, and no episode is gained. The tasks separate by how
they fail. ButtonUnmaskSwap is at zero and **all 50 episodes run to the 1,300-step cap** (timeout
.25 overall, the whole of it this task; this is also why the arm takes 55 min). ButtonUnmask (.98 →
.36) and VideoUnmaskSwap (.90 → .16) fail by wrong picks at unchanged episode length (268 and 200
steps against 286 and 193). VideoUnmask keeps .76 although its reveal video leaves the window about
70 steps into the episode; the episodes are short (144 steps), so a likely reading is that the pick
is committed while the video is still visible and the rest follows from the scene (not checked).

**`k4` (2026-11-16, 71 min).** −.805 SR; 161 of the 192 episodes `full` solves are lost, none gained.
VideoUnmask, which held at K = 8, now breaks (.76 → .28) with a 30-step reach, and its episodes get
shorter (117 steps against 139): wrong picks made early. ButtonUnmask (.36 → .20) starts running to
the cap (20 timeouts, 20 wrong picks; 818 steps against 286), ButtonUnmaskSwap again times out on
all 50 episodes, and VideoUnmaskSwap stays where it was at K = 8 (.16 → .14, all wrong picks).

### Summary

- SR falls monotonically with the window: .96 → .88 → .58 → .32 → .155 for full / 32 / 16 / 8 / 4.
  There is no plateau: even K = 32 (15.5 s) costs .08.
- What decides whether an episode survives is whether it still fits inside the window. Among the
  episodes `full` solves, none that fits is lost at K = 32 (0 of 118) or K = 16 (0 of 18); every
  loss is an episode longer than the window's reach. From K = 8 down nothing fits.
- Task order follows episode length: ButtonUnmaskSwap (longest) is at .08 by K = 16, VideoUnmask
  (shortest) keeps .76 at K = 8.
- Failure mode shifts from wrong picks (K ≥ 16) to running out the 1,300-step cap on the button
  tasks (K ≤ 8: ButtonUnmaskSwap 50 of 50, ButtonUnmask 20 of 50 at K = 4); the video tasks keep
  failing by wrong picks.
- Against navigation (`paper/ctx_ablation/`, HM3D): Uni-NaVid is flat down to K = 8 and LightNav-0
  down to K = 16; here the task needs one specific past event (the reveal), so the curve is a
  reach cutoff with no flat part.

---

# Layer-depth ablation (information horizon)

From which decoder layer on can the visual tokens be dropped at no cost? The navigation
blind-rollout arm (`paper/info_horizon/`, "Blind SR/SPL") on SimpleMemVLA: the live rollout with
all visual tokens hidden from decoder layer L on, SR and steps vs L, same suite, episodes, seed
and settings as above. The reference (nothing hidden, L = 32) is the `full` arm above; with the
option unset no hook is installed.

## How visual information reaches the action

Per decision the 32-layer Qwen3.5 text decoder prefills the prompt (3,840 history-video tokens +
64 wrist-image tokens + text), then decodes a short sub-task string. The DiT action head reads only
the last-layer states of those generated sub-task tokens, their token embeddings and the
proprioceptive state. So vision can reach the action only through the decoder layers, by text
tokens reading visual tokens.

The decoder alternates two token mixers: three `linear_attention` layers (gated delta rule: a
recurrent state plus a 4-tap causal convolution) then one `full_attention` layer, eight times
(attention at layers 3, 7, …, 31, zero-based).

## Definition

**Hiding from layer L** (`BLIND_LAYER=L`, arm `blind_l<L>`, `robomme_sim/layer_blind.py`): on
decoder layers ≥ L no token can read a visual token (history video and current wrist image; text,
timestamps and vision markers stay). Layers < L run unchanged; positions never move.

- attention layers ≥ L: the visual tokens' key columns are closed for every query, on the prefill
  and on every decode step (the navigation definition);
- recurrent layers ≥ L: the layer runs on the sequence with the visual positions removed, so they
  do not write to the recurrent state, do not decay it and do not enter the convolution of later
  tokens. This is the recurrent counterpart of a closed key column: the other tokens are computed
  as if the visual tokens were absent.

| Navigation (`paper/info_horizon/`) | Here |
|---|---|
| all visual tokens (history + current frame) hidden from L on, live rollout | same: history video + wrist image |
| key columns closed for every query, layers ≥ L | same on the 8 attention layers; skip on the 24 recurrent layers |
| separators / timestamps stay visible | same |
| reference = unblinded run | `full` (L = 32) |
| grid around the cliff, L = 0, 4 dropped as known zero | L ∈ {8, 12, 16, 20, 24, 28}, block boundaries; 0 and 4 not run |

## Running

```bash
bash paper/simplememvla_ablation/run_depth.sh                     # blind_l28 … blind_l8, deep to shallow
ARMS="blind_l4 blind_l0" bash paper/simplememvla_ablation/run_depth.sh
```

Aggregation and plots as above (`information_perf.png`: SR | steps vs L; `depth_by_task.png`: the same per task), in
the same single-column format as the context-window figures. The last tick, L = 32, is the
unblinded `full` arm.

## Checks (2026-11-17, real decision, 2 VideoUnmask episodes, same process)

- No leak: with the hook at L ∈ {0, 8, 16, 18, 24, 28, 30, 31}, replacing the visual rows that enter
  layer L by noise changes neither the sub-task nor the action (max |ΔA| = 0, exactly); without the
  hook the same noise changes the action. Holds for an unpadded and a left-padded batch.
- Pass-through: hooks on all 32 layers with no token flagged as visual give the baseline action
  exactly on a padded batch and within 0.008 on an unpadded one (the hook passes an explicit causal
  mask where upstream lets the kernel build it; the baseline's own padded-vs-unpadded difference is
  also 0.008).
- Removing the hooks restores the baseline exactly; a blinded row's action is the same in a padded
  and an unpadded batch within 0.008.
- Smoke (`BLIND_LAYER=16`, 4 episodes) end to end; every record carries `blind_layer`.

## Results

Complete: six arms, 200 episodes each, 2026-11-17 00:26–08:20 (7.9 h). Figures
`paper_results/simplememvla_ablation/permanence/figures/information_perf.png` and `depth_by_task.png`.

| arm | SR (ΔSR; p) | steps | fail / timeout | VideoUnmask | ButtonUnmask | VideoUnmaskSwap | ButtonUnmaskSwap |
|---|---|---|---|---|---|---|---|
| `full` (none) | .960 [.930, .985] | 257 | .03 / .01 | 1.00 | .98 | .90 | .96 |
| `blind_l28` | .955 [.925, .980] (−.005; p = 1.0) | 250 | .04 / .00 | 1.00 | 1.00 | .90 | .92 |
| `blind_l24` | .890 [.845, .935] (−.070; p = .0001) | 326 | .04 / .07 | .96 | .96 | .82 | .82 |
| `blind_l20` | .555 [.485, .630] (−.405; p = .0001) | 719 | .08 / .37 | .82 | .48 | .62 | .30 |
| `blind_l16` | .300 [.235, .360] (−.660; p = .0001) | 973 | .04 / .67 | .72 | .12 | .32 | .04 |
| `blind_l12` | .190 [.135, .250] (−.770; p = .0001) | 1089 | .07 / .75 | .52 | .16 | .08 | .00 |
| `blind_l8` | .090 [.055, .130] (−.870; p = .0001) | 1158 | .07 / .85 | .14 | .12 | .10 | .00 |

**`blind_l28` (2026-11-17, 33 min).** Hiding the visual tokens in the last four layers (28–31) costs
nothing: .955 against .960, 2 episodes lost and 1 gained, every task within two episodes of `full`,
steps unchanged (250 against 257).

**`blind_l24` (2026-11-17, 41 min).** First loss: −.07 SR, 14 episodes lost and none gained, spread
over all four tasks. The lost episodes are mostly timeouts, not wrong picks (fail .03 → .04, timeout
.01 → .07; 13 episodes at the 1,300-step cap against 2), and episodes get longer (326 steps against
257). So layers 24–27 are still reading the image, and what suffers first is carrying the motion
through, not choosing the cup.

**`blind_l20` (2026-11-17, 83 min).** The drop: −.405 SR, 81 episodes lost and none gained. 70 of the
81 are timeouts (73 episodes at the cap in total against 2 for `full`), 11 are wrong picks (fail .03 →
.08, 10 of the 16 on VideoUnmaskSwap). The button tasks lose most (ButtonUnmask .98 → .48, 25
timeouts; ButtonUnmaskSwap .96 → .30, 30 timeouts); the video tasks keep more (VideoUnmask .82,
VideoUnmaskSwap .62). Episodes that still succeed take longer, on the button tasks about twice as long
(ButtonUnmask 598 steps against 266 at `blind_l28`, ButtonUnmaskSwap 560 against 414). Mean length is
719 steps against 257. So the horizon lies between layers 20 and 28: most of the loss falls between 20
and 24, a small part between 24 and 28.

**`blind_l16` (2026-11-17, 102 min).** Further down, but not to zero: .300, 133 episodes lost and 1
gained, 128 of the 133 timeouts (timeout rate .67; wrong picks stay at .04). The tasks separate
clearly. VideoUnmask still succeeds in 36 of 50 episodes with the upper half of the decoder blind, and
at the usual length (163 steps on successes). The button tasks are almost gone (ButtonUnmask .12,
ButtonUnmaskSwap .04, 90 of 100 episodes at the cap), VideoUnmaskSwap is at .32. So the curve is a
ramp over layers 16–28, not a single cliff, and its position depends on the task: the tasks that begin
with a button press need the image in deeper layers than the plain pick.

**`blind_l12` (2026-11-17, 107 min).** .190, 154 episodes lost and none gained, 144 of them timeouts
(timeout rate .75). The Swap tasks are now gone (VideoUnmaskSwap .08 with 12 wrong picks,
ButtonUnmaskSwap 0 of 50, every episode at the cap); ButtonUnmask stays at its `blind_l16` level (.16
against .12, the same within noise). VideoUnmask is still at .52, but its successes are no longer
quick: 422 steps on successes against 163 at `blind_l16`. So layers 12–15 matter for every task
except ButtonUnmask, which was already near its floor.

**`blind_l8` (2026-11-17, 108 min).** .090, 174 episodes lost and none gained, 162 of them timeouts
(timeout rate .85). VideoUnmask falls from .52 to .14, so every task is now at .14 or below. The 18
remaining successes are quick again (293 steps on average), so they are episodes the model solves
with what layers 0–7 read, not slow recoveries. L = 4 and L = 0 were not run.

## Summary

- **The last four layers do not need the image.** Hiding the visual tokens from layer 28 on costs
  nothing (.955 against .960).
- **Below 28 the loss is a ramp, not a cliff.** SR falls at every 4-layer step: .890 at 24, .555 at
  20, .300 at 16, .190 at 12, .090 at 8. The steepest part is 20–24 (−.335). The curve has not reached
  zero at the shallowest layer tested.
- **The ramp's position depends on the task.** VideoUnmask keeps .72 with layers 16–31 blind and .52
  with 12–31 blind; the button tasks fall between 24 and 16 (ButtonUnmask .96 → .12,
  ButtonUnmaskSwap .82 → .04); VideoUnmaskSwap lies between.
- **The failure mode is timeouts.** From layer 24 down, 79–96 % of the lost episodes ran to the
  1,300-step cap; the wrong-pick rate stays between .04 and .08 in every arm. The policy stops
  completing the task, it does not pick a different cup. No arm gains more than one episode over
  `full`.
- **Against navigation (`paper/info_horizon/`).** LongNav-R1 has a cliff between layers 16 and 20 of
  28 and LightNav-0 a ramp over 20–24 of 36: both lose everything inside about four layers, at 55–70 %
  of the depth, and are at or near the reference above. Here the free zone starts later (88 % of the
  depth), the first loss is at 75 %, and the decline is spread over at least 20 of the 32 layers. One
  difference in the method should be kept in mind: 24 of the 32 layers are recurrent and are blinded
  by skipping the visual positions, not by closing attention columns.

# Velocity-field influence of the context window

The navigation paper measures how much old frames move a decision with a KL between the
full-context action distribution and the one with only the K most recent frames
(`recent_k`, `paper/frame_influence/`). SimpleMemVLA's action head is a flow-matching DiT, so the
counterpart here is the squared L2 distance between velocity fields:

    Infl(K) = E_{a_t, t} || v_full(a_t, t) − v_K(a_t, t) ||²

## Definition

Measured offline, at every decision of an ordinary full-context rollout (the `full` arm; the
actions that are executed come from the unmodified model):

* **`v_full`** is the DiT conditioned on the rollout's own decision: the generated sub-task
  tokens, their last-layer hidden states, proprioception.
* **`v_K`** is the DiT conditioned on the same decision run again with the K-frame window of the
  context-window ablation (`context_frames = K`, same definition of K as above). The window run
  decodes its own sub-task, so the measure is end to end: a changed sub-task and changed hidden
  states both count.
* **`(a_t, t)`** are the states the full-context sampler visits: 8 noise draws per decision,
  integrated with `v_full` by the model's own rule (10 Euler steps, t = 1.0 … 0.1), every step
  kept, 80 points per decision. Both fields are evaluated at exactly these points.
* The squared norm is over the whole action chunk (30 steps × 8 dimensions) in the model's
  normalised action space; `infl_exec` restricts it to the 16 executed steps and `rel` divides by
  the same mean of `||v_full||²`.

Arms: K ∈ {1, 2, 4, 8, 16, 32, 64} and `replay`, the full input once more through the probe path
(numerical floor). Model weights, prompt shape, wrist image, instruction and proprioception are
the same in both evaluations; only the history frames differ.

A window that covers the whole history changes nothing: the input is identical and the influence
is exactly 0 (`same_input`). The tables therefore give three decision sets: `cut` (the window
removes at least one frame, n_history > K), `all` (every decision, the zeros included: what a K
window does to the episode as it ran) and `fixed` (more than 32 history frames, the same decisions
for every K ≤ 32). Means are episode-balanced ± SEM across episodes.

## Running

```bash
bash paper/simplememvla_ablation/run_influence.sh          # tmux; Permanence, test episodes 0–11 per task
robomme_benchmark/.venv-simplememvla/bin/python paper/simplememvla_ablation/aggregate_influence.py
/home/brabiei/miniconda3/envs/vln/bin/python paper/simplememvla_ablation/plot_influence.py
```

`KS=4,8,16,32`, `NOISE=8`, `EPISODES_PER_TASK`, `SUITE`, `TASKS` as environment variables.
Twelve consecutive episodes per task because difficulty cycles with the episode index mod 4.
Code: `robomme_sim/influence_probe.py` (`VelocityInfluenceProbe`), `--influence_ks` in
`robomme_sim/eval_success.py`, a guarded `condition_tap` in `robomme_sim/batched_policy.py`
(policy submodule, branch `local`); `INFLUENCE_KS` in `robomme_benchmark/run_simplememvla_eval.sh`.
Raw run: `/mnt/vault/robomme/runs/simplememvla_ablation/<suite>/influence_recent/`
(`influence.jsonl`, one line per decision). Tables and the figure go to
`paper_results/simplememvla_influence/<suite>/` and are kept locally (git-ignored).

The probe leaves the rollout as it is: probe passes return from `generate_batch` before the action
head samples, the probe's noise comes from its own generator, and the rollout buffer's caches are
only read. `measure` raises if the global random state changed.

## Checks

* `tests/simplememvla_influence_unit.py` (CPU, fake action head): the probe states follow the
  sampler's Euler rule, identical conditions give exactly 0, known gaps are recovered, the global
  random stream is untouched, probe passes never sample.
* GPU smoke (2026-11-23, VideoUnmask episodes 0–3, one group of 4): `replay` influence exactly 0 on
  all 37 decisions; influence exactly 0 wherever the window covers the history; success and step
  counts identical to the `full` arm's records for the same episodes (94 / 100 / 102 / 264 steps).

## Results (2026-11-23, Permanence, test episodes 0–11 of each task, 48 episodes, 795 decisions)

Run time 54 min (probe 45 min). The rollout reproduces the `full` arm: same outcome on all 48
episodes (SR .917), identical step counts on the 12 VideoUnmask episodes (same noise stream; the
later tasks start at another offset of the stream because 12 instead of 50 episodes precede them).
`replay` influence is exactly 0 on all 795 decisions. Mean `||v_full||²` is 433.

Episode-balanced means; `cut` = decisions where the window removes a frame. "Sub-task changed" =
the window run decoded a different sub-task. The last column is the context-window sweep's
success rate on the same 48 episodes (`full` .917).

| K | cut decisions (episodes) | Infl, cut (± SEM) | Infl / ‖v‖² | sub-task changed | Infl, all decisions | normalised to K = 1, median (mean) | SR with window K |
|---|---|---|---|---|---|---|---|
| 1 | 771 (48) | 225 ± 17 | .57 | 37 % | 217 | 100 % | — |
| 2 | 771 (48) | 184 ± 13 | .47 | 34 % | 178 | 83 % (84) | — |
| 4 | 747 (48) | 136 ± 11 | .35 | 29 % | 127 | 61 % (67) | .17 |
| 8 | 663 (48) | 85 ± 13 | .21 | 20 % | 71 | 25 % (40) | .35 |
| 16 | 465 (44) | 52 ± 9 | .13 | 7 % | 35 | 2.9 % (32) | .50 |
| 32 | 157 (23) | 24 ± 8 | .055 | 0.7 % | 4.7 | 0 % (1.7) | .83 |
| 64 | 2 (1) | 0.004 | 10⁻⁵ | 0 % | 0 | 0 % | — |

Figure `figures/influence_recent_k`: per K, the box of the 48 episodes' influence as a percentage
of the same episode's K = 1 influence (all decisions), median drawn through the boxes.

* **The influence falls with K but slowly**: half of the no-history influence is still there at
  K = 4 and a quarter at K = 8 for the median episode. It reaches zero only where the window covers
  the episode (K = 32 for the median episode; 23 of 48 episodes are longer than that).
* **Mean and median part at K ≥ 8** (K = 16: median 2.9 %, mean 32 %): the episodes split into
  those whose history fits the window and those where it does not. By task, the normalised median
  at K = 16 is 0 % (VideoUnmask, ButtonUnmask) against 26 % / 22 % (VideoUnmaskSwap,
  ButtonUnmaskSwap) — the order of the success-rate sweep.
* **The offline measure predicts the closed-loop outcome.** Among the 44 episodes `full` solves,
  the episode's mean Infl(K) separates the episodes the K-window run loses from those it keeps:
  AUC .97 (K = 32, 5 lost), 1.00 (16, 20 lost), 1.00 (8, 27 lost), .87 (4, 36 lost). It is not only
  episode length: restricted to episodes where the window removes a frame at some decision, the
  mean over those decisions gives AUC .96 / 1.00 / 1.00 / .85, above the number of such decisions
  (.76 / .96 / .95 / .83). No episode is lost without a cut decision.
* **Two channels.** Decisions with a changed sub-task carry 83 % of the summed influence at K = 1
  (50 % of decisions), 51 % at K = 16 (9 %) and 9 % at K = 32 (0.4 %); their influence is about
  `‖v_full‖²` (440 at K = 1, 270 at K = 16). With the sub-task unchanged the influence is small for
  most decisions (median 21 at K = 1, 0.4 at K = 4, 0.003 at K = 16) with a tail (90th percentile
  280 / 99 / 17): the hidden states alone can move the velocity field as far as a sub-task change.
* **Where in the flow**: the gap grows towards the data end, from 54 at t = 1.0 to 697 at t = 0.2
  (K = 1; the same shape at every K), so the average is dominated by the last steps of the sampler.
* **Where in the chunk**: the 16 executed steps hold 31 % of the influence at K = 1 (21 % at
  K = 16) against 54 % of `‖v_full‖²`; the two fields disagree more on the later part of the chunk.

Reading notes: the values are in normalised action units and scale with the chunk size, so they
are not comparable with the navigation KL or across models; compare shapes, the normalised curve
and the sub-task change rate. The K-window input is the context-window ablation's (older slots
hold the oldest visible frame), so the measure includes the effect of that padding.
