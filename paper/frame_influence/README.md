# Leave-one-frame-out influence

How much does each historical frame, individually, move the action distribution of an ObjectNav
VLM agent, and how does that influence depend on the frame's age? For LongNav-R1 and Uni-NaVid,
on 10 HM3D-v2 val episodes, at every decision step `t` and for every history frame `i < t`:

    Infl_t(i) = KL( π(· | c_t)  ‖  π(· | c_t \ v_i) )

`π(· | c_t)` is the full-context action distribution, `c_t \ v_i` the same context with frame
`i`'s visual tokens removed and everything else unchanged. The per-(model, episode, step, frame)
table is the retention-independent quantity a later experiment joins against any pruning
method's retention log to compute captured influence.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`; plots use `vln`;
Uni-NaVid launchers use the isolated baseline runtime. To rebuild the retained tables and
figures without overwriting published outputs, follow the
[reproduction guide](../../paper_results/REPRODUCIBILITY.md). A Git clone alone is
insufficient: restore required local inputs from the checksummed backup first. Ordinary
launch logs and locks are ignored; protected traces, arrays, and images remain scientific inputs.

> **Archived 2026-11-21 (paper clean-up).** The paper uses the recent-window variant only
> (`paper_results/frame_influence_recent/`). The leave-one-out study and the leave-one-in variant moved
> out of `paper_results/` to `paper_results_archive/frame_influence/` and
> `paper_results_archive/frame_influence_keepone/`, where only their `runs/` are kept; the full-reference
> variant has no runs of its own and rebuilds from the keep-one runs. The `results/` and `figures/` of all
> three were deleted and rebuild next to the archived runs in under a minute (checked before deletion:
> every table and figure came back identical); rebuilt outputs are git-ignored. The numbers and file
> references in the sections below describe those rebuilt outputs. The recent-window aggregation still
> reads the archived runs for its reference checks.
>
> ```bash
> for m in leave_one keep_one keep_one_full; do
>   conda run -n longnav_vlm python paper/frame_influence/aggregate.py --measure $m
>   conda run -n vln python paper/frame_influence/plot.py --measure $m
> done
> ```

## Definitions

**Evaluation set.** `src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json`: 10 labels,
`random.Random(17).sample(<the 100-episode split>, 10)` (`tools/sample_episode_manifest.py`),
the same 10 episodes the earlier influence-persistence diagnostic used
(`dump/kl_influence_10_seed17_labels.json`). Both models run exactly these labels.

**Evaluated step `t`.** Every decision. LongNav-R1 decides at every env step; Uni-NaVid decides
every second env step (`actions_per_inference=2`, the official protocol), so its rows exist only
on inference steps. `step` is always the env step index and frame `i` is the observation made at
env step `i`, so `age = t − i` counts env steps for both models and `i = t−1` is the most recent
history frame. Rows are one per `(model, episode, step, frame)`.

**Frame `i`'s tokens.** Its *visual* tokens only. LongNav-R1: the frame's visual K/V slots that
survived the sparse filter (`use_sparse=true`, `sparse_threshold=0.95`, the checkpoint's native
regime); frames whose patches were all deduplicated have no tokens and are excluded from
statistics (`n_slots = 0`). Uni-NaVid: the frame's 4-token history group, or its contribution to
a merged long-term slot. Text (system prompt, goal, LongNav-R1's action history) always stays.
This is the only definition comparable across the two models — Uni-NaVid has no action text in
its context.

**Influence.** KL in nats over the four actions `{stop, forward, left, right}` at temperature 1,
full distribution first: the ablated distribution's deviation is weighted by what the agent
actually does. LongNav-R1 reads the action from the logits at the decision position restricted to
the action vocabulary; Uni-NaVid's distribution is the renormalised softmax over the four
single-token action words at the position after `ASSISTANT:` (their probability mass under the
full vocabulary, `p_mass`, is logged and is ≈1). Raw logits are stored so other temperatures can
be recomputed.

**Counterfactual semantics.** The removal is truncation-style, consistent with the context-window
ablation: nothing that stays is recomputed.

* *LongNav-R1* keeps one growing KV cache and reads the action from a single decision token, so
  removing frame `i` = hiding its visual key columns from that token's attention (an additive
  `−inf` mask on the replayed decision token, `VLMWorker._kl_frame_probe`). For a single query
  this is identical to physically deleting those K/V; every other slot keeps its original K/V and
  its original mRoPE position (a positional hole remains, exactly as under `evict`). What it does
  **not** remove is what frame `i` already contributed to later tokens' K/V while they were
  encoded — the same information that leaks through eviction in the context-window ablation. The
  measured quantity is therefore the *direct* influence: what a KV-cache pruner acting at decision
  time can still remove.
* *LongNav-R1, leak-free arm* (`influence_recompute`): the same removal, but the context is rebuilt
  from scratch — prefix plus every turn's stored post-filter embeddings, minus the removed frame's
  visual tokens, forwarded against an empty cache with the survivors' original absolute mRoPE
  positions (the recompute machinery of the context-window ablation). Nothing survives from a
  computation that saw the frame, so this is the strict counterfactual, and evict-style minus
  leak-free at matched K is the decision-level analogue of Experiment 1's leakage gap. With
  nothing removed the rebuilt base reproduces the live decision to bf16 precision (max |Δp| ≈ 1e-2
  in the smoke test), a tighter floor than the single-token replay.
* *Uni-NaVid* rebuilds its single-turn prompt every inference from a per-frame feature cache
  whose rows are computed independently per frame, so removing frame `i` = deleting its group from
  the assembled prompt embeddings and re-running the LLM (`baselines/uninavid_hm3d/influence.py`);
  the prompt is re-encoded contiguously (positions shift), the natural "frame never existed" for
  this architecture. Frames older than 64 steps live in 1-token long-term slots merged by cosine
  similarity; a merged slot minus one member is rebuilt as the fp32 mean of its remaining members
  (the fp16 closed form `(w·slot − x)/(w−1)` amplifies rounding ~`w`-fold), and a `merged_control`
  probe (slot rebuilt from all its members) measures that reconstruction's noise floor.

**Trajectory.** Full context, official decoding: LongNav-R1 argmax (`deterministic=true`,
`ep_seed=17`), Uni-NaVid sampled at temperature 0.2 with per-call seeding (`--sampling-seed 30`,
seed `30 + inference_index` per `generate` call) so the sampled trajectory is reproducible. Probes
are side computations: LongNav-R1 replays the decision token and restores the cache
(`tests/kl_influence_smoke.py` pins that the decision does not move); Uni-NaVid scores ablations
*after* the on-policy sample has been drawn, consuming no RNG. The LongNav-R1 run reproduces the
context-window ablation's `r1_full` trajectories step for step (checked by `aggregate.py`).

**Joint probes per step** (one extra replay each):

| probe | removed | why |
|---|---|---|
| `history_all` | every frame `i < t` | joint history influence, vs the sum of single-frame influences (redundancy) |
| `horizon_K`, K ∈ {4, 8, 16, 32} | every frame of age ≥ K (the K most recent frames incl. the current survive) | per-step information horizon, the decision-level twin of the context-window ablation's `K`; K=1 is `history_all` |
| `visual_all` | every visual token (current frame included) | text-only prior; Uni-NaVid: the `--no-visual-tokens` prompt |
| `age0` | the current frame | anchor for the age axis |
| `merged_control` (Uni-NaVid) | nothing — a merged slot rebuilt from all its members | fp16 reconstruction noise floor |

Ablated log-probs are stored for every probe, so argmax flips can be counted.

## Arms

| arm | model | run |
|---|---|---|
| `r1_influence` | LongNav-R1 (`+checkpoint=longnav`, sparse SFT + RPP RL, ckpt 231) | `+experiment=paper_influence`: full context on the prune substrate with an unreachable budget (`kv_budget=1e9`, nothing is ever pruned), `kv_prune_log_influence=true` on every step (evict-style probes: key columns hidden from the replayed decision) |
| `r1_influence_recompute` | LongNav-R1, same checkpoint and trajectories | `+experiment=paper_influence_recompute`: `vlm.influence_recompute=true`; every probe re-prefills prefix + all turns from their stored post-filter embeddings with the removed frame's visual tokens deleted (leak-free: nothing survives from a cache that saw the frame). One full prefill per probe, so cost grows with T³; run on the 9 episodes shorter than 150 steps (`hm3d_v2_9_seed17_short.json`) |
| `uninavid_influence` | Uni-NaVid-7B (`baselines/`) | official evaluator + `--frame-influence --sampling-seed 30` |
| `lightnav_influence_keepone` | LightNav-0 (`baselines/`, keep-one only) | `baselines/run_lightnav.sh` with `INFLUENCE=1` (`lightnav_hm3d/eval_influence.py`): normal navigation on the eager masked kernel of `lightnav_ref`, then the keep-one probes after every decision (see the LightNav-0 paragraph below) |

## Leave-one-in variant (`paper_results_archive/frame_influence_keepone/`)

The same experiment with the complementary per-frame measure, for every frame `f = 0..t`
(current frame included):

    KeepOne_t(f) = KL( π(· | c_t^{only f})  ‖  π(· | c_t^{no visual}) )

`c_t^{only f}` keeps frame `f`'s visual tokens and removes every other frame's (the current
frame's included); `c_t^{no visual}` is the text-only context of the `visual_all` probe (prefix,
goal, action history / specials stay). The KL is taken with the expectation under the only-`f`
distribution, as in the repo's other keep-one diagnostics (`vlm.kv_prune_log_keep_one`,
`vlm.horizon_probe_layers`, Uni-NaVid's `--layer-probe`), so age 0 is exactly Experiment 6's
full-context keep-one number. Counterfactual semantics as above: LongNav-R1 hides every other
visual key column from the replayed decision (evict-style; the leak-free arm is implemented —
`vlm.kv_prune_influence_measures` on the recompute path — but was not run); Uni-NaVid keeps the
full prompt and closes every visual key column except frame `f`'s group for every query on every
layer (`--influence-measures keep_one_mask`, the column masking of the layer-blind arms; a long-term
slot is first rebuilt as `f`'s own mean token; separators and specials stay open), so the lone
group keeps its place in the sequence; the reference is the same prompt with every visual column
closed (`visual_all_mask`) and age 0 is the current frame's columns alone (`only_current_mask`,
identical to Experiment 6's layer-0 keep-one probe on the same seeded episodes).

*Why masking, not splicing, for Uni-NaVid.* The first attempt (`--influence-measures keep_one`,
kept as `runs/uninavid_influence_keepone_splice/`) spliced the other groups, the separators and
the fine tokens out, leaving the canonical one-group prompt. Uni-NaVid re-encodes contiguously, so
that prompt is byte-identical at every later decision: a frame's value was constant over time
(median SD 1e-4 across decisions) and the age curve was flat at ≈0.25 nats by construction — the
model cannot tell how old a lone frame is. LongNav-R1's replay keeps the frame's original mRoPE
position and the growing action text, so its curve decays with age either way.

Arms: `r1_influence_keepone` (`+experiment=paper_influence_keepone`, i.e. `paper_influence` with
`vlm.kv_prune_influence_measures=[keep_one]`; rows `sup/kl_influence_keep_one`,
`sup/kl_influence_keep_one_logp`) and `uninavid_influence_keepone` (`--influence-measures
keep_one_mask`; records `keep_one[]`, `keep_one_age0`, `visual_all_mask`, `keep_one_mode`). The
leave-one-out rows are not re-logged (same
deterministic / seeded trajectories, so the data above stands); the joint probes are logged as
before and keep their full-context meaning. Column semantics in `results/influence.csv.gz` for this
measure: `infl` = keep-one KL, `logp_full` = the reference (no-visual) log-probs, `logp_removed` =
the only-frame log-probs, `flip` = the frame alone moves the argmax off the no-visual decision;
`steps.csv`'s `sum_loo` is the sum of keep-one over the history frames.

*LightNav-0 (arm `lightnav_influence_keepone`, added 2026-10-26; keep-one only).* The third
model re-prefills its whole prompt inside vLLM every step, so `baselines/lightnav_hm3d/eval_influence.py`
navigates on the eager masked-SDPA kernel of the information-horizon reference `lightnav_ref`
(nothing hidden for the real decision; the trajectory is the model's own) and, after each decision,
re-runs the same prompt with `<|video_pad|>` columns closed for every query on every layer
(timestamps, vision markers and text stay; positions unchanged): once with every visual column
closed (`visual_all_mask`, the reference), once with only the current tubelet open (`keep_one_age0`),
once per history **tubelet** of age `t - f1 <= 33` (`MAX_AGE`; the current tubelet plus the whole
fast tier, 16 two-frame tubelets at pool 2 — the mid/long tiers are not probed, which halves the
cost and covers the ages the age curve is read on), plus one re-run with nothing hidden (the
kernel's noise floor, `kl_live_vs_replay`). The frame unit is LightNav's 2-frame tubelet:
`frame` = its newer frame `f1`, `age = t - f1`, `tier = p<pool>`, `weight` = 2, `n_slots` = its
visual tokens. **Action distribution.** LightNav emits `<apos><opos><act_l0><act_l1><act_l2>`:
two pointing tokens, then three residual-VQ codes whose coarse level `act_l0` (256 codes, ~0.9 m
resolution; the two residual levels refine by 7 and 4 cm and change the executed command class
for 4 % of code pairs) carries the motion. Every probe **free-runs its own two pointing tokens**
under its mask (3-token greedy generate; nothing from the full-context decision enters a probe, as
for the other two models, whose action word is the first generated token) and the logits of the
third token are restricted to the 256 `<act_l0_*>` ids and log-softmaxed at T = 1 (`p_mass` = the
full-vocabulary mass those ids hold). Two KLs are logged per probe: `kl` over the 256 codes (model
label **LightNav-0**) and `kl_bin` over the same probabilities folded onto the six executed-action
classes of the velocity controller — stop, forward, forward-left/right, rotate-left/right, from
decoding each coarse code alone through LightNav's first-waypoint → `velocity_control` mapping
(dt 1 s, 0–0.25 m/s, ±30 deg/s; 8 / 80 / 39 / 42 / 43 / 44 codes per class) — the alphabet closest
to the other models' four words (label **LightNav-0 (6-bin)**, dashed; a lower bound of the
256-way KL by data processing). The 256 log-probs of every probe are stored (float16) in
`influence/<label>.jsonl` with the pointing tokens' top-1 probabilities (`p_apos`, `p_opos`: how
confident the free-run prefix is), `meta.json` holds the code→class table. vLLM's prefix cache is
reset before every generate and each prefill is asserted to cover the whole prompt, so K/V
computed under one mask are never reused under another; the decision's logits are checked to
reproduce the emitted tokens. Column semantics as for Uni-NaVid: `logp_full` = reference,
`logp_removed` = only-tubelet (the 6-bin log-probs for the bin label, empty for the 256-way label
whose `argmax_*` are code ids), `chosen_action` / `action_at_frame` = the executed class at `t` /
at the frame's step; the joint leave-out probes (`history_all`, `visual_all`, `horizon_K`) do not
exist for this arm. Unit tests: `baselines/tests/test_lightnav_influence.py` (client env).

```bash
MEASURE=keep_one nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &   # Uni-NaVid (KEEP_ONE_MODE=mask|splice), R1, then LightNav-0
bash paper/frame_influence/run_lightnav.sh                                         # LightNav-0 arm alone (MAX_AGE=33)
python paper/frame_influence/aggregate.py --measure keep_one
conda run -n vln python paper/frame_influence/plot.py --measure keep_one
```

**Findings (runs of 2026-09-23, R1's 500-step episode added the same evening; LightNav-0 run
2026-10-26; `paper_results_archive/frame_influence_keepone/results/summary.json`).**
All three models on all 10 episodes (R1 trajectories identical to `r1_full`; Uni-NaVid the same
seeded trajectories as the leave-one-out arm; LightNav-0 its own eager-kernel trajectories, SR 0.9
after forced-stop rescoring, 1 696 decisions, 24 416 tubelet probes in 90 min — they are not the
trajectories of any earlier LightNav run: the closed loop diverges after the first few dozen
steps as between every pair of LightNav runs so far, two episodes crawl to 485/500 steps here).
Nats; age 0 = the current frame (LightNav: current tubelet) alone; single-frame rows are
episode-balanced means. LightNav ages are even in steady state (the current tubelet holds frames
`t-1` and `t`, so the newest history tubelet has age 2); odd ages exist only in the first 32 steps
of an episode (10–150 rows each) and are left out of the row.

|  | R1, keep-one | Uni-NaVid, keep-one (masked) | LightNav-0, keep-one (256 codes) | LightNav-0, keep-one (6-bin) | R1, leave-one-out | Uni-NaVid, leave-one-out |
|---|---|---|---|---|---|---|
| age 0 | 0.054 (pooled 0.038; the 500-step episode holds 45 % of the decisions) | 0.422 | 9.24 (median 9.9) | 3.33 | 0.023 | 0.355 |
| age 1 / 2 / 4 / 8 / 16 / 32 (balanced mean) | 0.031 / 0.021 / 0.0061 / 0.0013 / 0.0006 / 0.0004 | 0.161 / 0.162 / 0.170 / 0.153 / 0.137 / 0.100 | — / 6.12 / 5.10 / 4.98 / 5.08 / 3.88 | — / 2.85 / 2.21 / 2.33 / 2.77 / 2.35 | 0.0012 / — / 0.0003 / 0.0003 / — / — | 0.219 / — / 0.0045 / 0.0011 / — / — |
| merged 64+ slots (Uni-NaVid) | — | 0.026 | — (ages > 33 not probed) | — | — | 2e-4 |
| floor | 3.8e-4 (replay) | 1e-5 (merged control) | 0 (replay; the eager masked kernel is bit-deterministic) | 0 | 3.0e-4 | 1.1e-5 |
| age-adjusted persistence ρ, shortest lag | 0.12 | 0.97 | 0.67 (lag 2 = the same tubelet; lag 1 0.33 on the 160 steps where a tubelet keeps its `f1`) | 0.67 | 0.03 | 0.10 |

* Both age-0 values reproduce Experiment 6's full-context keep-one probes (R1 leaky 0.060;
  Uni-NaVid layer 0: median difference 0 over the 650 decisions, worst 1.6e-2 from fp16 batch
  composition), which is the external check that the masked implementation is right.
* **LongNav-R1**: a single frame on its own is worth 0.031 nats at age 1 against the no-visual
  prior — 25× its leave-one-out value (0.0012) — and decays two orders of magnitude to the replay
  floor by age ~12. Old frames carry no decision-relevant information *on their own*, in either
  direction of the counterfactual.
* **Uni-NaVid**: a single frame in its original position is worth 0.16 nats at any age from 1 to
  8, and still 0.10 at age 32; only the merged 64+ slots drop to 0.03. The decay across the window
  is a factor ~1.6 where leave-one-out falls by two orders. A lone old frame is nearly as
  informative as a lone recent one for Uni-NaVid: its history groups are content-addressed rather
  than age-addressed, which is also why its persistence ρ is 0.97 (a frame's keep-one value barely
  changes from one decision to the next; median SD 0.02 across decisions).
* The splice form is kept for reference (`runs/uninavid_influence_keepone_splice/`): flat at 0.25
  nats by construction, not comparable.
* **LightNav-0**: the curve is flat like Uni-NaVid's, not decaying like R1's — a lone tubelet of
  age 32 is worth 3.9 nats against the no-visual prior where the current tubelet is worth 9.2
  (ratio 0.42; Uni-NaVid 0.24, R1 0.007) — but the nats are two orders larger than the other
  models' because the 256-code distributions are near one-hot: a probe either leaves the decision
  at the prior (20 % of tubelet probes below 0.1 nats) or moves it with confidence to another code
  (47 % above 7 nats ≈ −log of that code's prior probability); the 6-bin fold halves the level
  (41 % below 0.1, 19 % above 7) and keeps the shape. So for LightNav the keep-one KL reads as
  "how often does this tubelet alone override the prior, times how surprised the prior is": any
  single fast-tier tubelet does so about as often as any other, and the current one almost always
  (age-0 KL below 0.1 nats at 0.2 % of decisions). The flip rate is 0.98 on codes (the only-tubelet
  argmax code differs from the no-visual argmax) and 0.83 on bins; the only-tubelet 6-bin argmax
  equals the executed class at 16 % of pairs — a single tubelet almost never reproduces the
  full-context decision. Persistence is 0.67 at lag 2 (a tubelet's value barely changes between
  the two decisions it survives unchanged), between Uni-NaVid's 0.97 and R1's 0.12. The rank
  heatmap blanks most LightNav cells: its noise rule (2× the row median) is above every cell when
  the row median itself is 5 nats, so only decisions with a bimodal row show colour. Caveats: the
  free-run pointing prefix is not confident (`p_apos` top-1 0.36, `p_opos` 0.92), so each probe's
  action distribution is the one along the probe's own greedy pointing path, not a marginal over
  pointings; only the current tubelet and the 16 fast-tier tubelets (14.4 older tubelets per
  decision unprobed) are measured, so no statement about ages beyond 33 or the mid/long tiers.
* **LightNav-0 scored against the full decision instead of the prior** (`results/lightnav_agreement.csv`,
  `figures/lightnav_agreement.*`; computed from the stored 256-way log-probs, no extra run): the
  age structure the saturated keep-one KL hides. Episode-balanced probability that a lone
  tubelet's argmax is the full-context decision's argmax over the six executed-action classes,
  ages 0 / 2 / 4 / 8 / 16 / 32: 0.61 / 0.33 / 0.26 / 0.30 / 0.27 / 0.14 (256 codes: 0.20 at age 0,
  ≤ 0.04 at any history age); always answering the most common class scores 0.62 and the
  no-visual prior 0.06, but that constant scores 1 / 0 on decisions that were forward / a turn or
  stop, whereas the current tubelet alone scores 0.73 / 0.43 (ages 2 / 8 / 32: 0.34 / 0.35 / 0.12
  on forward decisions, 0.33 / 0.20 / 0.20 on turns and stops; Cohen's κ 0.36 at age 0, 0.14 at
  age 2, 0.00 from age 8); the figure is one box per age over the episodes' overall agreement rates (`results/lightnav_agreement_episodes.csv`). KL(full ‖ only tubelet) rises monotonically 3.2 → 7.2 → 8.0 → 8.6 → 9.2 →
  9.9 nats over the same ages. So a tubelet older than a few steps, on its own, yields a confident
  action for a viewpoint the robot has left — below the majority-class rate — while the current
  tubelet alone reproduces the decision 61 % of the time. Old frames are not each carrying the
  current decision; what they carry is redundant across frames (random token pruning to a quarter
  of the tokens costs nothing, Exp 4b) and lost only when the window drops them all (the K = 4 cut,
  Exp 1).
* **LightNav-0 action spread: single- or multi-modal?** (`paper/frame_influence/action_spread.py`,
  `conda run -n vln`, writes to `outputs/validation/lightnav_action_spread/`; computed from the
  same stored 256-way log-probs, no run, 2026-10-28.) The 256-way entropy is metric-free, so the
  stored distribution is pushed through the action decoder exactly (probability-weighted moments
  of the 256 coarse codes decoded alone, no sampling) and a Gaussian is fitted to the chunk's
  endpoint pose (x, y, unwrapped heading; ridge 1e-4). Per step: `H_cat`, the Gaussian entropy
  `H_gauss`, and `d_exec`, the Mahalanobis distance of the greedy code's endpoint from the
  pushforward mean. Result: effectively unimodal in trajectory space. `d_exec` median 0.72, p90
  1.12, above 1.5 on 1.1 % of steps (19, mostly an episode's first steps where rotate-in-place and
  go-forward compete); endpoint spread median 13 cm forward / 10 cm lateral / 11° heading.
  Spearman ρ(`H_cat`, `H_gauss`) 0.74 over steps, rotate steps above the trend and forward steps
  below; `H_gauss` falls over the episode in 9/10 episodes (sign test p = 0.02) against 8/10
  (p = 0.11) for `H_cat`. Figures `spread_scatter`, `spread_over_time`, `spread_showcase`.

## Full-reference variant (`paper_results_archive/frame_influence_keepone_full/`)

This results directory is kept locally and is git-ignored (2026-11-20: only code and documentation
go to GitHub). It is computed offline from the keep-one runs, so `aggregate.py --measure keep_one_full` rebuilds it.

The complementary single-frame measure, with the decision the agent actually took as the
reference and the lone frame as the ablated distribution (added 2026-11-07 at the user's request
for LongNav-R1 leaky and LightNav-0; Uni-NaVid added the same day):

    Full_t(f) = KL( π(· | c_t)  ‖  π(· | c_t^{only f}) )

Low = frame `f` on its own reproduces the full-context decision; high = it does not. The
expectation is under the full distribution, so a lone frame is charged for missing probability
where the agent actually acts (the sufficiency direction; the reverse direction would charge
confident wrong actions instead). Its ceiling for a frame that carries nothing is the `visual_all`
joint probe, KL(full ‖ no visual): a frame above the ceiling is actively misleading, not merely
uninformative. For LightNav-0 this is the `kl_full_only` quantity of the agreement analysis above,
promoted to a measure so it gets the standard age-curve / box / persistence outputs (no rank
heatmap for this measure, user decision).

**No run of its own.** Both keep-one arms stored the two distributions at every (episode, step,
frame): LongNav-R1 logs the replayed full decision (`sup/kl_influence_base_logp`, same bf16
single-token replay path as the probes, so the replay floor cancels as in the other measures) next
to every only-frame distribution (`sup/kl_influence_keep_one_logp`); LightNav-0 stores the 256-way
log-probs of the live decision (`full.logp_f16`) and of every probe; Uni-NaVid's masked keep-one
records hold `logp_full` and every probe's `logp` (its ceiling is the column-masked no-visual
prompt, `visual_all_mask`, rescored the same way). `--measure keep_one_full`
reads the keep-one runs (`RUNS_ROOT` in `aggregate.py`) and writes to the `_keepone_full` root.
Column semantics: `infl` = KL(full ‖ only f), `logp_full` = the full decision, `logp_removed` = only-f, `flip` = the lone frame's argmax differs from the full
one, `steps.age0` = KL(full ‖ only the current frame), `steps.visual_all` = the ceiling (filled for
LightNav too; `summary.visual_all_balanced` is its episode-balanced mean). Unit tests:
`tests/influence_loaders_unit.py` (LongNav-R1, synthetic `sequence.json`) and
`test_keep_one_full_matches_agreement` in `baselines/tests/test_lightnav_influence.py`.

```bash
python paper/frame_influence/aggregate.py --measure keep_one_full          # longnav_vlm; reads the keep_one runs
conda run -n vln python paper/frame_influence/plot.py --measure keep_one_full
```

**Findings (aggregated 2026-11-07 from the 2026-09-23 / 2026-10-26 keep-one runs; nats,
episode-balanced means; `results/summary.json`, `results/age_curve.csv`).** LightNav ages are even
in steady state, as above. The keep-one aggregation was re-run before this change and is
byte-identical; the LightNav values equal `lightnav_agreement.csv`'s `kl_full_only(_bin)_balanced`
at every age (max difference 0). Reading direction: this is a distance to the full decision, so
the most recent frame has the SMALLEST value; a frame's influence in the usual sense is how far
below the ceiling it sits.

|  | LongNav-R1 (leaky) | Uni-NaVid (masked) | LightNav-0 (256 codes) |
|---|---|---|---|
| age 0 (current frame alone) | 0.012 (pooled 0.009) | 0.21 (pooled 0.19) | 3.22 |
| age 1 / 2 / 4 / 8 / 16 / 32 | 0.034 / 0.032 / 0.042 / 0.043 / 0.047 / 0.044 | 0.43 / 0.46 / 0.49 / 0.46 / 0.72 / 0.46 | — / 7.20 / 7.97 / 8.63 / 9.22 / 9.87 |
| pooled age bins 1 / 2–3 / 4–7 / 8–15 / 16–31 / 32–63 / 64+ | — | 0.31 / 0.34 / 0.36 / 0.39 / 0.42 / 0.44 / 0.47 | — |
| ceiling KL(full ‖ no visual) | 0.048 ± 0.005 (pooled 0.034) | 0.53 ± 0.05 (pooled 0.57) | 9.94 ± 0.16 |
| flip rate (lone frame's argmax ≠ full) | 0.016 | 0.45 | 0.98 |

* **LongNav-R1**: the current frame alone sits a quarter of the way from the full decision to the
  no-visual ceiling (0.012 of 0.048); a frame of age 1–2 alone is at 0.03, two thirds of the way;
  from age 4 on a lone frame is at the ceiling (0.042–0.047 vs 0.048), i.e. indistinguishable from
  showing the model nothing. No age sits above the ceiling, so old frames are inert rather than
  misleading on their own, in agreement with the keep-one decay to the floor by age ~12. The
  argmax flips at 1.6 % of pairs: on the 4-action alphabet the lone frame usually leaves the argmax
  alone and the KL is carried by probability mass, not by the decision.
* **Uni-NaVid**: the current frame alone recovers 60 % of the gap to the full decision (0.21 of
  0.53); a lone history frame of any age sits at 0.43–0.49 (balanced), i.e. 10–20 % of the way
  back from the ceiling, with only a gentle drift towards it (pooled bins 0.31 → 0.47 from age 1
  to the merged 64+ slots). The balanced bump at ages 14–22 (0.72 at 16, up to 1.1 at 20) is three
  short episodes (`a8BtkwhxdRV_2`, `k1cupFYWXJ6_0`, `q5QZSEeHe5g_0`, 20–25 steps) whose *first*
  frames alone sit far above the ceiling (1–3.7 nats: an episode's opening views on their own give
  a confident action that contradicts the later decision) and which leave the balanced mean past
  age 22; the other seven episodes are flat at 0.3–0.6 across those ages. So
  a lone old frame reproduces the decision about as well as a lone recent one, the full-reference
  counterpart of its flat keep-one curve: Uni-NaVid's history groups are content-addressed, and
  what a single group carries is small and age-independent. The argmax flips on 45 % of pairs.
* **LightNav-0**: the curve rises monotonically from 3.2 nats (current tubelet) through 7.2 (age 2)
  to 9.9 at age 32, where it meets the ceiling (9.9): a lone fast-tier tubelet is progressively
  less able to reproduce the decision the further back it lies, and by age ~16–32 is as far from it
  as no vision at all — the age structure the saturated keep-one KL hid. The nats stay two orders
  above LongNav-R1's because the 256-code decisions are near one-hot: the lone tubelet's argmax
  differs from the full one on 98 % of pairs, so the KL is essentially −log of the probability the
  lone tubelet gives the decision's code. Only the 256-code reading is kept for this measure (the
  6-bin fold, still produced for keep-one, was dropped here at the user's request; the loader still
  supports ``space="bins"``).
* Read together with the keep-one variant: on LongNav-R1 both directions agree that an old frame
  alone carries nothing; on Uni-NaVid both are flat in age (a lone frame moves the model a fixed
  0.16 nats off the prior and lands a fixed ~0.45 nats from the decision whatever its age); on
  LightNav-0 an old tubelet alone moves the model confidently off the prior (keep-one flat at 4–6
  nats) but to a stale decision (this measure at the ceiling), which is the "confident action for
  a viewpoint the robot has left" reading of the agreement analysis.
  Caveats as for the keep-one LightNav arm (free-run pointing prefix, ages ≤ 33 only).

## Recent-window variant (`paper_results/frame_influence_recent/`)

This results directory is kept locally and is git-ignored (2026-11-20: only code and documentation
go to GitHub). Rebuilding it on a fresh clone means re-running the arms below (about 5 h on one GPU).

The cumulative counterpart of the full-reference measure: the reference is again the decision
the agent took, and the comparison is the decision made with only the `K` most recent frames'
visual tokens (added 2026-11-18 at the user's request, LongNav-R1 leaky and Uni-NaVid; LightNav-0,
both with its old blocks kept and without, 2026-11-19):

    Recent_t(K) = KL( π(· | c_t)  ‖  π(· | c_t^{last K frames}) ),   K = 0, 1, ..., 32

`K` counts frames including the current one, as `context_frames` does in the context-window
ablation: `K = 0` is no frame at all (the `visual_all` probe), `K = 1` the current frame only
(`history_all`), `K = 2` the current frame and the previous one, and so on. Text always stays.
These are the joint age-band probes of the leave-one-out runs (`horizon_K`: every frame of age
≥ K removed), which had been logged at K = 4, 8, 16, 32 only; this variant logs every K and runs
no per-frame probe (`vlm.kv_prune_influence_measures=[]`, `--influence-measures none`).

Counterfactual semantics are those of the leave-one-out arms, and they differ between the models:

* *LongNav-R1 (leaky)*: the visual key columns of every frame of age ≥ K are hidden from the
  replayed decision token. The surviving frames and the action text keep the K/V they were
  encoded with, i.e. computed while the old frames were visible. The curve is therefore what the
  decision token still reads *directly* from old visual slots, the decision-level twin of the
  eviction arm of the context-window ablation, not the cost of a model that never saw them.
* *Uni-NaVid*: the old frames' groups and every long-term slot are spliced out and the prompt is
  re-encoded, which is byte-for-byte the prompt a `K`-frame window builds
  (`trim_visual_history` in the context-window ablation: `feat_cache` cut to `K` rows, long-term
  cache dropped; `test_no_measure_builds_joint_probes_only`). This is a strict counterfactual.
* *LightNav-0* re-prefills its whole prompt every step, so both of its probes are strict
  counterfactuals, and both are run after every decision of one run
  (`eval_influence.py --measure recent_k`), the two arms of its context-window ablation:
  * **window** (label `LightNav-0`, solid): the prompt rebuilt as `CONTEXT_FRAMES=K` builds it
    (`eval_ctx.windowed_segments`: the layout of a fresh episode that started K steps ago, old
    blocks gone, nothing hidden), K = 1..32. There is no K = 0 prompt, so this curve starts at 1.
  * **old blocks kept** (label `LightNav-0 (old blocks kept)`, dashed): the full prompt with the
    `<|video_pad|>` columns of every older tubelet hidden on every layer; their `<t s>`
    timestamps and vision markers stay and positions do not move (the pruning study's `recency`
    layout). The 2-frame tubelet is the unit, so K = 2, 4, ..., 32, plus K = 1 (current tubelet
    rebuilt as `(t, t)`, every history block hidden) and K = 0 (no visual token). The odd K that
    occur while the fast tier holds an odd frame count (first 32 steps) are logged but not read.

  The action distribution is the one of the keep-one arm: the 256 `<act_l0_*>` codes after the
  probe's own free-run pointing prefix (and its fold onto the six executed-action classes, the
  `(6-bin)` rows of the CSVs, not drawn). The rebuilt prompts use a per-step copy of the
  episode's ViT tubelet cache, so the navigation path is untouched.

**Averaging.** Each `K` averages the decisions with at least `K` history frames, so every scored
decision has a frame to lose (the convention of `horizon.csv`); the figure draws the
episode-balanced mean ± 1 SEM across episodes. Because that decision set shrinks with `K` and
loses the short episodes, `recent_curve.csv` also carries the same means on one fixed set, the
decisions with at least 32 history frames (`fixed_*`), where differences between `K` are
within-decision (and, for LightNav-0's kept-blocks arm, a value at every K of its grid).
`flip_rate` is the share of decisions whose argmax moves off the full one (pooled);
`flip_balanced` ± `flip_balanced_sem` is the same share averaged over episodes like the KL, and
is what the `recent_frames_flip` figure draws (LightNav-0 on its six executed-action classes there,
so that a move between two codes that decode to the same action is not counted). The flip rate
does not depend on the softmax temperature; every KL here is at temperature 1, whereas Uni-NaVid
navigates at 0.2.

Arms: `r1_influence_recent` (`+experiment=paper_influence_recent`),
`r1_influence_recent_recompute` (the leak-free twin, all 10 episodes),
`uninavid_influence_recent` (`MEASURE=recent_k`, `--influence-horizons 2,...,32`) and
`lightnav_influence_recent` (`MEASURE=recent_k bash paper/frame_influence/run_lightnav.sh`, both
LightNav-0 arms). Same
deterministic / seeded trajectories as the leave-one-out arms, so `aggregate.py` re-measures the
shared K = 0, 1, 4, 8, 16, 32 against those runs per decision (`results/recent_reference.csv`).

```bash
MEASURE=recent_k nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &   # Uni-NaVid, LongNav-R1, LightNav-0
python paper/frame_influence/aggregate.py --measure recent_k
conda run -n vln python paper/frame_influence/plot.py --measure recent_k         # figures/recent_k.{pdf,png}
# any other view, on request (all five models unless --recent-models; --tag suffixes the file names):
conda run -n vln python paper/frame_influence/plot.py --measure recent_k \
    --only recent_frames,recent_frames_fixed,recent_frames_normalized,recent_frames_flip,recent_frames_box,recent_frames_box_split
```

**The figure (user decision 2026-11-20).** `figures/recent_k` is the
only figure kept for this variant: two panels, LongNav-R1 (leaky) and LightNav-0 (old blocks
kept; titled plainly "LightNav-0" and drawn solid green since 2026-11-21, as in `ctx_ablation`), drawn at
single-column size for a two-column paper (3.5 × 1.6 in, 8 pt text, `paper/figstyle.py`): the normalised KL of the section below as one box per even K across episodes (outlier
points not drawn) with the median across episodes as a line through them (every K; 100 % at
K = 0). Medians drawn (%), K = 1 / 2 / 4 / 8 / 16 / 24 / 32: LongNav-R1 30 / 15 / 2.4 / 1.9 /
1.4 / 0.9 / 1.1 (8 episodes), LightNav-0 kept 72 / 36 / 24 / 21 / 12 / 10 / 8.3 (10 episodes).
The line is the median, not the mean of the table below, so that one far-off episode cannot lift
it out of the boxes (the leak-free LongNav-R1 case below). The other views named in this section
(`recent_frames`, `_fixed`, `_normalized`, `_flip`, `_box`, `_box_split`, and the three-panel
variant with the leak-free arm) were removed from `figures/`; `plot.py --only` redraws them from
`results/`.

**Findings (runs of 2026-11-18, Uni-NaVid 14 min + LongNav-R1 33 min; `results/recent_curve.csv`,
`results/summary.json`).** Both arms on all 10 episodes; the LongNav-R1 trajectories are identical
to `r1_full`, and the K = 0, 1, 4, 8, 16, 32 values reproduce the leave-one-out runs per decision
(`recent_reference.csv`: LongNav-R1 exactly, max difference 0 over every shared probe; Uni-NaVid
median 0, worst 9.4e-3 from fp16 batch composition). Nats; episode-balanced mean over the
decisions with ≥ K history frames, and the same on the fixed set (decisions with ≥ 32 history
frames: 8 LongNav-R1 episodes / 791 decisions, 5 Uni-NaVid episodes / 519 decisions).

| K (frames kept) | 0 | 1 | 2 | 3 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| LongNav-R1 (leaky), balanced | 0.048 | 0.0124 | 0.0054 | 0.0015 | 0.0009 | 0.0006 | 0.0004 | 0.0005 | 0.0005 |
| LongNav-R1 (leaky), fixed set | 0.044 | 0.0123 | 0.0063 | 0.0016 | 0.0009 | 0.0007 | 0.0005 | 0.0005 | 0.0005 |
| LongNav-R1, argmax flip rate | 5.4 % | 1.8 % | 1.0 % | 0.4 % | 0.7 % | 0.2 % | 0.1 % | 0.2 % | 0.3 % |
| Uni-NaVid, balanced | 0.58 | 0.28 | 0.20 | 0.24 | 0.22 | 0.13 | 0.046 | 0.020 | 0.014 |
| Uni-NaVid, pooled | 0.63 | 0.23 | 0.157 | 0.155 | 0.140 | 0.075 | 0.042 | 0.027 | 0.018 |
| Uni-NaVid, fixed set | 0.66 | 0.19 | 0.135 | 0.121 | 0.119 | 0.063 | 0.042 | 0.027 | 0.014 |
| Uni-NaVid, argmax flip rate | 83 % | 30 % | 24 % | 24 % | 24 % | 17 % | 14 % | 12 % | 10 % |

* **The current frame alone closes about three quarters of the gap for both models**: K = 1
  leaves 26 % of the no-visual KL for LongNav-R1 (0.0124 of 0.048) and 29 % for Uni-NaVid on the
  fixed set (0.19 of 0.66).
* **LongNav-R1 (leaky)** then reaches its floor within three more frames: 0.0054 at K = 2, 0.0015
  at K = 3, 0.0009 at K = 4, and flat at 5–7e-4 from K = 5 to 32 (replay floor 3.0e-4 mean). The
  dense sweep adds two informative points (K = 2, 3) to what the K = 4, 8, 16, 32 bands already
  showed; the decision token reads directly from the visual slots of the last ~3 frames only.
  This is the leaky reading: the surviving K/V have absorbed the older frames, and the leak-free
  arm of the leave-one-out study (0.40 / 0.25 / 0.12 / 0.06 / 0.036 at K = 1 / 4 / 8 / 16 / 32)
  is the one comparable with Uni-NaVid.
* **Uni-NaVid** decays slowly and has not converged by K = 32. On the fixed set the KL falls
  0.19 → 0.12 over K = 1–4, 0.063 at K = 8, then roughly halves every 12–16 frames (0.042 at 16,
  0.027 at 24, 0.014 at 32); with 32 frames the argmax still differs from the full-context one at
  10 % of decisions (17 % at K = 8). Frames well beyond the last 8 keep moving the action
  distribution even though the closed-loop success rate of the context-window ablation is flat
  down to K = 8: nats and flips at a single decision do not map onto SR.
* **Composition effect, visible for Uni-NaVid only.** The balanced curve over "decisions with ≥ K
  history frames" is bumpy at small K (0.20 / 0.24 / 0.22 / 0.23 at K = 2–5, SEM ≈ 0.08) and has
  a shoulder at K ≈ 17–20 (0.046 at 16 → 0.017 at 20). Both come from the four 21–29-step
  episodes: each contributes a mean over the few decisions it has left at that K (0.2–0.7 nats in
  two of them at K ≤ 8), and that mean collapses towards 0 as K approaches the episode length,
  where only the episode's first frame or two are being removed. The pooled and fixed-set curves
  are smooth and monotone and agree with each other from K = 8 on, so the fixed-set curve
  (`recent_frames_fixed`) is the one to read Uni-NaVid's shape from; LongNav-R1's two
  curves coincide.
* Caveats: one-step measure on the full-context trajectory (no closed-loop compounding); the two
  models' counterfactuals differ as described above; 10 episodes, 5 of them in Uni-NaVid's fixed
  set, where the two long episodes (469 and 500 steps) hold 87 % of the decisions.

**LightNav-0 (run of 2026-11-19, 2 h 56 min; both arms probed after every decision of one run).**
10 episodes, 1696 decisions, SR .90 after the step-cap rule (one forced stop), every decision the
one of the keep-one run of 2026-10-26 (`recent_reference.csv`: same code at 1696/1696 decisions;
the kept-blocks K = 0 and K = 2 probes reproduce that run's no-visual and only-current probes,
worst difference 1.1e-2 nats, median 2e-3 and 6e-4). Replay floor exactly 0. Nats over the 256
codes unless marked; the fixed set is all 10 episodes (1376 / 1366 decisions).

| K (frames kept) | 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| window (old blocks gone), balanced | – | 5.89 | 4.98 | 3.28 | 2.61 | 1.73 | 1.31 | 1.16 |
| window, fixed set | – | 6.29 | 5.45 | 3.18 | 2.56 | 1.79 | 1.34 | 1.16 |
| old blocks kept, balanced | 9.94 | 6.72 | 3.30 | 2.48 | 2.08 | 1.35 | 1.04 | 0.82 |
| old blocks kept, fixed set | 10.32 | 7.63 | 3.55 | 2.39 | 1.95 | 1.24 | 1.02 | 0.82 |
| window, 6 classes, balanced | – | 2.56 | 1.90 | 0.90 | 0.72 | 0.52 | 0.40 | 0.44 |
| old blocks kept, 6 classes, balanced | 3.68 | 3.50 | 1.16 | 0.79 | 0.62 | 0.42 | 0.35 | 0.27 |

Share of decisions whose top action changes (`flip_balanced`, %; `recent_frames_flip`):

| K (frames kept) | 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| LongNav-R1 (leaky), 4 actions | 9.2 | 2.8 | 1.2 | 1.2 | 0.2 | 0.1 | 0.5 | 0.8 |
| Uni-NaVid, 4 actions | 82.7 | 29.2 | 22.6 | 25.8 | 19.4 | 8.5 | 8.2 | 10.6 |
| LightNav-0 window, 6 classes | – | 66.2 | 54.2 | 28.7 | 25.0 | 19.2 | 16.5 | 14.4 |
| LightNav-0 old blocks kept, 6 classes | 93.6 | 81.6 | 40.4 | 29.7 | 22.5 | 15.0 | 14.1 | 10.1 |
| LightNav-0 window, 256 codes | – | 84.9 | 85.3 | 75.5 | 68.7 | 54.6 | 46.3 | 40.8 |
| LightNav-0 old blocks kept, 256 codes | 98.2 | 94.2 | 79.6 | 70.2 | 62.2 | 49.7 | 42.3 | 33.2 |

* **LightNav-0 has not converged by K = 32 in either arm**: about 1 nat over the codes (1.16
  window, 0.82 kept) and a different executed-action class at 14 % / 10 % of decisions. The decay
  is slow and steady after the first frames (window 3.3 → 2.6 → 1.7 → 1.2 at K = 4 / 8 / 16 / 32).
* **The current tubelet is the big step, and it takes both frames.** With blocks kept, K = 0 → 2
  drops 9.9 → 3.3 nats and the class flip rate 94 % → 40 %; the single current frame (K = 1, the
  tubelet rebuilt as `(t, t)`) is still at 6.7 nats / 82 %. LightNav-0 needs about four frames to
  reach the flip rate Uni-NaVid has with its current frame alone (29 %).
* **Keeping the old blocks is closer to the full decision at every K ≥ 2.** Paired per decision
  (same decisions, window minus kept, episode-balanced ± SEM): +1.67 ± 0.34 nats at K = 2 (10 of
  10 episodes), then +0.74 ± 0.32 / +0.50 ± 0.27 / +0.44 ± 0.19 / +0.35 ± 0.16 at K = 4 / 8 / 16
  / 32 (6–8 of 10 episodes); class flips 54 % vs 40 % at K = 2 and 14 % vs 10 % at K = 32. The
  gap grows with the history: over the decisions with ≥ 64 history frames (5 episodes, 1118
  decisions; scratch re-aggregation, not a tracked result) the window arm is at 1.65 ± 0.23 nats
  and 19 % class flips at K = 32 against 1.03 ± 0.14 and 11 % with blocks kept. At K = 1 the two
  constructions are close (5.9 ± 0.7 vs 6.7 ± 0.4). Same direction as the closed-loop result of
  the context-window ablation (kept blocks flat down to K = 4, window cliff), far smaller here.
* **On executed actions LightNav-0 and Uni-NaVid coincide from K = 4 on** (29 / 25 / 19 / 14 %
  window and 30 / 23 / 15 / 10 % kept, against 26 / 19 / 9 / 11 % for Uni-NaVid at K = 4 / 8 /
  16 / 32). The two-orders gap between their KL curves is not a difference in reliance on old
  frames.

**Normalised view (`recent_frames_normalized`, `norm_balanced_mean`, added 2026-11-19).**
The log axis of `recent_frames` hides the first-frames drop for LightNav-0 (9.9 → 3.3 nats is its
largest step) and stretches the tail. On the fixed set, per episode, the KL at K over the
no-frame KL of the same decisions (share of the no-frame gap still open, %):

| K (frames kept) | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| LongNav-R1 (leaky) | 30 | 16 | 3.8 | 3.0 | 2.2 | 2.0 | 1.8 |
| Uni-NaVid | 29 | 21 | 19 | 10 | 6.6 | 4.3 | 2.4 |
| LightNav-0 window | 61 | 53 | 31 | 25 | 17 | 13 | 11 |
| LightNav-0 old blocks kept | 74 | 35 | 23 | 19 | 12 | 10 | 8.0 |

All three have the same shape, a steep fall over the first 2–4 frames and a slow tail: sharply
diminishing returns per frame (LightNav-0 kept: ~33 % of the gap per frame for the two current
frames, ~0.3 % per frame between K = 16 and 32), not zero influence beyond the first frames.
LightNav-0 has the heaviest tail (8–11 % open at K = 32). LongNav-R1's curve is the leaky probe
and overstates its recency (leak-free arm: 62 % of the history effect left at K = 4, 9 % at 32).

**LongNav-R1 leak-free (runs of 2026-11-19, 28 min for the nine short episodes + 42 min for the
500-step one; `ARM=r1_influence_recent_recompute bash paper/frame_influence/run_r1.sh`,
`+experiment=paper_influence_recent_recompute`).** The strict counterfactual for LongNav-R1: every
probe re-prefills the whole context with the visual tokens of every frame of age ≥ K deleted (at
most 34 re-prefills per step). All 10 episodes, 1099 decisions, trajectories identical to
`r1_full`; on the nine episodes of the leave-one-out leak-free arm, K = 0, 1, 4, 8, 16, 32
reproduce `r1_influence_recompute` exactly (difference 0 at every shared probe). Label
`LongNav-R1 (leak-free)`, dashed red. Fixed set: 8 episodes, 791 decisions (the leaky arm's).

| K (frames kept) | 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| leak-free, balanced (nats) | 0.48 | 0.43 | 0.37 | 0.25 | 0.13 | 0.065 | 0.047 | 0.032 |
| leak-free, normalised, median over episodes (%) | 100 | 103 | 92 | 72 | 36 | 13 | – | 4.4 |
| leak-free, normalised, 7 episodes without `wcojb4TFT35_0`, mean (%) | 100 | 100 | 92 | 67 | 33 | 14 | 8.1 | 5.1 |
| leak-free, normalised, mean over the 8 episodes (%, the plotted line) | 100 | 128 | 95 | 70 | 43 | 32 | 24 | 15 |
| leaky, normalised (%) | 100 | 30 | 16 | 3.8 | 3.0 | 2.2 | 2.0 | 1.8 |
| leak-free, top action changes (%) | 19.7 | 16.1 | 13.2 | 11.3 | 7.6 | 6.6 | 6.8 | 5.1 |

* **The leaky curve's steep drop is the leak.** Leak-free, the current frame alone closes none of
  the no-frame gap (about 100 % at K = 1), two frames 8 %, four frames a third; half the gap is
  closed only around K = 6 and 13–14 % is still open at K = 16. The leaky probe had 3.8 % left
  at K = 4.
* **LongNav-R1 is the least recency-biased of the three models on this measure**: at K = 4 it has
  about 70 % of the gap open against 19 % (Uni-NaVid) and 23 % (LightNav-0, blocks kept); the
  curves meet around K = 12–16, and in seven of its eight fixed-set episodes its tail at K = 32
  (0.5–18 %, median 4 %) is at or below LightNav-0's (8 %). The decision-level counterpart of the
  context-window ablation, where leak-free LongNav-R1 needs ≥ 16 frames. A claim of a sharp drop
  after the first few frames holds for Uni-NaVid and LightNav-0, not for LongNav-R1; "most of the
  visual influence lies within the last ~16 frames" holds for all three.
* **`wcojb4TFT35_0` (500 steps, the agent times out) is an outlier and drives the mean of the
  normalised curve.** Its no-frame KL is small (0.09 nats against 0.42–0.77 in the other seven)
  and its reduced-context decisions are *further* from the full one than the no-frame decision:
  317 % at K = 1, 115 % at K = 8, 159 % at K = 16, 83 % at K = 32. In a long failed episode the
  full decision is closer to the text-only one than to a decision that sees a few recent frames.
  One such episode among eight moves the mean of the per-episode ratios from 14 % to 32 % at
  K = 16 (SEM 18); the median and the ratio of the balanced means (106 / 92 / 68 / 34 / 16 / 6 %
  at K = 1 / 2 / 4 / 8 / 16 / 32) are unaffected. This is why the box
  figure's line is the median (2026-11-20). The absolute balanced KL barely moves with that episode.

Three-panel box figure (leaky, leak-free, LightNav-0 blocks kept; not kept in `figures/`):
`plot.py --measure recent_k --tag r1_both_lightnav_kept --recent-models "LongNav-R1,LongNav-R1 (leak-free),LightNav-0 (old
blocks kept)" --only recent_k`.

**Why the three KL curves sit at different heights.** Three measurement differences stack
(pooled, K = 32): (i) *output space*: 256 codes against 4 actions; folding LightNav-0 onto its
six classes cuts the KL 3–4× (1.13 → 0.28 kept) and the flip rate from 43 % to 12 %.
(ii) *How confidently the reduced-context decision disagrees*: a LightNav-0 flip costs 2.0–2.5
nats even on six classes, a Uni-NaVid flip 0.04 (its temperature-1 distribution is soft, entropy
0.79 of 1.39 nats); 83–85 % of LightNav-0's six-class KL comes from the flipped decisions, and
its median (0.012 kept, 0.034 window) is within a factor 2–4 of Uni-NaVid's (0.008).
(iii) *Counterfactual*: LongNav-R1's probe is the leaky one (two orders below its leak-free arm,
see the leave-one-out findings). Read the shapes within a model and the flip rates across models.

**Episode length confounds a fixed K.** "K frames kept" removes 1 frame at a decision with K
history frames and hundreds late in a long episode, and the KL grows with the number removed (at
K = 32, decisions losing 1–8 frames against > 128: Uni-NaVid 0.0008 vs 0.021, LightNav-0 window
0.25 vs 1.97, kept 0.34 vs 1.22). Short episodes therefore sit near zero at large K, which is
what stretches the boxes of `recent_frames_box_split` downwards (on a log axis) and bends the
balanced Uni-NaVid curve; the fixed set equalises the decisions across K, not the number of
frames each one loses. Restricting to decisions with ≥ 64 history frames (every K ≤ 32 then
removes ≥ 32) gives tight boxes and smooth curves but keeps only 5 / 3 / 5 episodes (LongNav-R1 /
Uni-NaVid / LightNav-0) of this set.

## Statistics

* Age curves: pooled mean / std / median per age with a 95% percentile bootstrap over
  *episodes* (clusters, 2000 resamples, seed 17), and the episode-balanced mean (mean over
  episodes of each episode's mean at that age). Both are reported because one episode
  (`wcojb4TFT35_0`, LongNav-R1 times out at 500 steps) holds ~83% of all LongNav-R1 pairs.
* Cumulative share within the `k` most recent history frames, `k ∈ {1, 4, 8, 16, 32}`: pooled
  mass share over the fixed set of decisions with ≥ 32 history frames, and the mean per-decision
  share over decisions with ≥ k history frames.
* Horizon: mean joint KL of the frames of age ≥ K next to the additive sum of their single-frame
  influences.
* Persistence (`longnav.analysis.influence.persistence`): Spearman ρ between a frame's influence at
  step `t` and at `t+k` over the history frames of `t`, mean over steps, raw and age-adjusted
  (raw ρ is inflated by the shared age decay). Age adjustment maps each influence to its
  percentile among all pooled values of the same age for that model, so every age is uniform by
  construction and what remains is "unusual for its age" (dividing by the per-age *mean*, as the
  old tool did, made most young frames fall below a heavy-tailed mean and produced a spurious
  reversed age gradient). Also top-quartile retention (chance 0.25) and revival counts (influence
  < 1e-3 for ≥ 5 steps, then > 1e-2). `influence_rank_heatmap` shows, for one episode, each frame's rank of raw influence within the
  decision (the diagonal is the current frame) among the frames above that decision's noise level (white at or below it; since most
  frames at a decision are noise, the level is twice the row's median influence, floored at the
  model's typical probe noise): recency is the red band on the diagonal, persistence would appear
  as vertical stripes.
* Covariates: mean influence by age bin × level of the action taken at frame `i`, the action
  chosen at `t`, the decision entropy, the distance to goal, and (LongNav-R1) the frame's
  surviving slot count; OLS of log10 influence on age-bin dummies + covariates with drop-one
  partial R².

## Running

```bash
# both arms, serially (Uni-NaVid ~30 min, LongNav-R1 ~4 h on one RTX 5090)
nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &     # log: paper_results_archive/frame_influence/runs/driver.log
# or individually
bash paper/frame_influence/run_uninavid.sh
conda activate longnav_vlm && bash paper/frame_influence/run_r1.sh
ARM=r1_influence_recompute bash paper/frame_influence/run_r1.sh     # leak-free arm, 9 short episodes (~2-3 h)

# aggregate (longnav_vlm) and plot (vln has matplotlib)
python paper/frame_influence/aggregate.py
conda run -n vln python paper/frame_influence/plot.py
```

Tests: `python tests/kl_influence_smoke.py`, `python tests/influence_recompute_smoke.py` (longnav_vlm, GPU) and
`PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python -m unittest baselines.tests.test_influence`.
A one-episode Uni-NaVid smoke (separate output directory):

```bash
bash baselines/run.sh eval \
  --labels "$PWD/src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json" \
  --expected-count 10 --limit 1 --sampling-seed 30 --frame-influence \
  --output "$PWD/dump/uninavid_influence_smoke"
```

The KL/keep-one probes remain supported diagnostics after cleanup; they are not selection
policies. The inherited `VLMWorker._kl_frame_probe` and replay helpers now live in
[diagnostics.py](../../src/longnav/utils/diagnostics.py).

## Outputs — all under `paper_results_archive/frame_influence/`

* `runs/r1_influence/`, `runs/r1_influence_recompute/`: `config.yaml`, `rollout/results_*`; the per-episode
  `rollout/<scene>/<label>.../sequence.json` (with `sup/kl_influence*`) stay on disk, gitignored.
  `runs/uninavid_influence/`: `manifest.json`, `summary.json`, `episodes/*.json`,
  `influence/<label>.jsonl` (one line per inference step, tracked), `influence/meta.json`;
  `traces/` gitignored.
* `results/influence.csv.gz` — the long join table; `steps.csv`; `age_curve.csv`, `age_bins.csv`,
  `cumulative_share.csv`, `horizon.csv`, `persistence.csv`, `covariates.csv`, `ols.json`,
  `flips.csv`, `summary.csv/json`, `table.tex`.
* `figures/influence_vs_age` (+ `influence_vs_age_box`, the same per-episode means as box plots),
  `figures/influence_horizon`, `figures/influence_rank_heatmap`
  (`.pdf` + `.png`): one figure per question — decay with age, joint influence beyond the last K
  frames, stability of a frame's importance over time (one representative episode per model).
  Cumulative shares, flip rates, covariates and lag correlations are numbers in `results/`, not
  figures.

## Findings (runs of 2026-09-07/08; `results/summary.json`, `results/leak.csv`)

Three arms: LongNav-R1 with evict-style probes (10 episodes), LongNav-R1 with leak-free probes
(the 9 episodes under 150 steps), Uni-NaVid (10 episodes). Both R1 arms reproduce the
context-window ablation's `r1_full` trajectories step for step. All influences in nats.

|  | R1, evict-style | R1, leak-free | Uni-NaVid |
|---|---|---|---|
| episodes / decisions / (t, i) pairs | 10 / 1099 / 150 301 | 9 / 599 / 25 551 | 10 / 650 / 123 044 |
| current frame removed (`age0`) | 0.023 | 0.088 | 0.355 |
| all history removed (`history_all`) | 0.009 | 0.403 | 0.232 |
| frames of age ≥ 4 / 8 / 16 / 32 removed | 0.0010 / 0.0008 / 0.0006 / 0.0006 | 0.249 / 0.125 / 0.063 / 0.036 | 0.140 / 0.075 / 0.042 / 0.018 |
| single-frame influence, age 1 / 2–3 / 4–7 / 8–15 | 0.0012 / 0.0010 / 0.0003 / 0.0003 | 0.052 / 0.019 / 0.0053 / 0.0013 | 0.219 / 0.016 / 0.0045 / 0.0011 |
| numerical floor (see Statistics) | 3.0e-4 | 4.4e-4 | 1.1e-5 |
| P(single-frame removal flips the argmax), age 1 / all | 0.9 % / 0.2 % | 12 % / 2.1 % | 21 % / 0.8 % |
| age-adjusted persistence ρ, lag 1–2 → 16 | 0.03 → 0.00 | 0.25 → 0.05 | 0.10 → 0.03 |
| OLS R² of log influence, age only / + covariates | 0.00 / 0.02 | 0.19 / 0.30 (entropy 0.09) | 0.55 / 0.56 |

**Q1 — How fast does a frame's influence decay with age?** (`influence_vs_age`) Steeply, and
identically for the two models once leakage is removed: the leak-free R1 curve lies on top of
Uni-NaVid's from age 1 to about 15 (0.05 → 0.001 nats), both dropping two orders of magnitude
in ten frames before reaching their floors. Uni-NaVid's previous frame is the outlier (0.22, a
fifth of all decisions change their argmax when it is removed); everything older behaves like
R1. The evict-style R1 curve sits 30× lower at age 1 and at its floor from age 4 on: hiding an
old frame's cache slots removes almost nothing, because later tokens' K/V already absorbed it.
Frames older than 64 steps in Uni-NaVid live in merged 1-token slots and are at the
reconstruction floor.

**Q2 — How much do the frames beyond the last K explain, jointly?** (`influence_horizon`)
More than their single-frame influences suggest, and more for R1 than for Uni-NaVid. Removing
every frame of age ≥ 4 costs R1 (leak-free) 0.25 nats and Uni-NaVid 0.14, against sums of the
single-frame influences of 0.05 and 0.04: old frames are individually redundant but jointly
informative (5× and 4×). Removing the whole history costs R1 0.40 nats, 4.6× its current frame,
whereas Uni-NaVid's history (0.23) is worth two thirds of its current frame — the decision-level
version of Experiment 1, where leak-free R1 needs ≥ 16 frames and Uni-NaVid degrades
gracefully. Even beyond 32 frames R1 keeps 0.036 (9 % of its history influence). The evict-style
R1 curve is two orders of magnitude below at every K; the paired gap (`leak.csv`: −0.39 at K=1,
−0.25 at K=4, −0.12 at K=8, −0.06 at K=16, −0.035 at K=32, all CIs excluding 0) is the
eviction leakage measured at the decision: at K=4, 99.6 % of what the old frames contribute has
already been written into surviving K/V.

**Q3 — Is a frame's importance stable enough for an online pruner to act on?**
(`influence_rank_heatmap`) No. Ranking frames within each decision shows the recency band on the
diagonal and no vertical stripes: a frame that ranks high at one decision does not at the next.
Age-adjusted rank correlations across steps are 0.03 (R1 evict), 0.10 (Uni-NaVid) and 0.25 (R1
leak-free) at the shortest lag, all decaying to ≤ 0.05 by lag 16, with top-quartile retention
0.25–0.40 against 0.25 chance; 4 revivals in 2572 dormant frames. Content covariates (action
at the frame, action chosen, distance to goal, surviving slot count) add nothing to age; only the
decision's entropy does, mechanically (flatter distributions move more in KL for the same
logit change). The leak-free R1 arm shows the most short-lived persistence (0.25 at lag 1),
consistent with its frames carrying resolved information for a few steps; it is gone by lag 8.

**Reading with Experiment 1.** The evict-style R1 probe is the decision-level twin of Experiment
1's evict arm (nearly flat down to K=4), the leak-free probe the twin of its recompute arm
(collapsing below K=16). Both experiments say the same thing in two coordinate systems: R1
depends on history more than Uni-NaVid does, but that dependence is carried by what later
tokens absorbed rather than by the old frames' own cache entries. For a KV-cache pruner that is
the operative fact: the entries it removes at decision time are nearly inert, which is why random
slot pruning at matched memory costs little and why eviction leaks.

Caveats: 10 (9) episodes; the 500-step `wcojb4TFT35_0` holds 83 % of the evict-style R1 pairs
(episode-balanced curves agree in shape) and is absent from the leak-free arm; Uni-NaVid's
trajectories are sampled (seeded) and differ from the context-window ablation's; Uni-NaVid
decides every second step, so its lags are 2, 4, 8, 16. The single-frame curves are resolved only
above each model's floor (R1 evict: age ≤ 3; R1 leak-free: age ≲ 15; Uni-NaVid: age ≲ 20, and
its 2e-4 plateau to age 63 is largely the structural cost of shortening the prompt by five
tokens, see `results/covariates.csv`).
