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

## Leave-one-in variant (`paper_results/frame_influence_keepone/`)

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

```bash
MEASURE=keep_one nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &   # Uni-NaVid (KEEP_ONE_MODE=mask|splice) then R1
python paper/frame_influence/aggregate.py --measure keep_one
conda run -n vln python paper/frame_influence/plot.py --measure keep_one
```

**Findings (runs of 2026-09-23, R1's 500-step episode added the same evening; `paper_results/frame_influence_keepone/results/summary.json`).**
Both models on all 10 episodes (R1 trajectories identical to `r1_full`; Uni-NaVid the same
seeded trajectories as the leave-one-out arm). Nats; age 0 = the current frame alone; single-frame
rows are episode-balanced means.

|  | R1, keep-one | Uni-NaVid, keep-one (masked) | R1, leave-one-out | Uni-NaVid, leave-one-out |
|---|---|---|---|---|
| age 0 | 0.054 (pooled 0.038; the 500-step episode holds 45 % of the decisions) | 0.422 | 0.023 | 0.355 |
| age 1 / 2 / 4 / 8 / 16 / 32 (balanced mean) | 0.031 / 0.021 / 0.0061 / 0.0013 / 0.0006 / 0.0004 | 0.161 / 0.162 / 0.170 / 0.153 / 0.137 / 0.100 | 0.0012 / — / 0.0003 / 0.0003 / — / — | 0.219 / — / 0.0045 / 0.0011 / — / — |
| merged 64+ slots (Uni-NaVid) | — | 0.026 | — | 2e-4 |
| floor | 3.8e-4 (replay) | 1e-5 (merged control) | 3.0e-4 | 1.1e-5 |
| age-adjusted persistence ρ, shortest lag | 0.12 | 0.97 | 0.03 | 0.10 |

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
nohup bash paper/frame_influence/run_all.sh > /dev/null 2>&1 &     # log: paper_results/frame_influence/runs/driver.log
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

## Outputs — all under `paper_results/frame_influence/`

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
