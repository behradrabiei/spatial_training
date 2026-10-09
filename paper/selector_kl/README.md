# Selector criteria vs keep-one-in KL (LongNav-R1, layer 0)

The flat token-pruning variant (`paper/token_prune/README.md`, "Flat variant") scores and deletes at
decoder layer 0 with no recency window and found that **no selection rule beats uniform random**
(F1) while the diversity criteria collapse (F2). This experiment asks the KL-side question for
LongNav-R1: do the criteria those selectors rank by — FastV's layer-0 attention, DivPrune's
diversity, DART's pivot-dissimilarity — track the tokens that actually move the decision, as
measured by the per-token keep-one-in KL of `paper/layer_token_info`? Tokens map to image patches,
so one decision is drawn as heatmaps over the frame; the statistic is aggregated over every
decision of ten short episodes. Decoder layer 0 only, keep-one-in only. LongNav-R1 is the headline
(maps + statistics); Uni-NaVid gets the same Spearman statistic (no maps: its history tokens are
pooled 2×2 groups, not patches).

## Definitions

**Regime.** Full episode context, argmax decoding, `ep_seed 17`, and the **sparse filter
disabled** (`sparse_threshold 2.0`, as `paper_token_info`): every patch of every frame is a token,
so the maps cover the whole 15×20 grid of a 640×480 frame instead of the post-filter subset. This
is off the checkpoint's training regime, so the trajectory differs from `ctx_ablation/r1_full`'s
(`aggregate.py` reports the per-episode difference; the flat sweep itself ran with the filter on).
Nothing is ever pruned (`kv_budget` unreachable); the probe reads the live cache after each
decision (`vlm.selector_probe`, `longnav.utils.selector_probe`). Episode lengths change without the
filter (see the arm table), so they were measured before choosing the set.

**Pool.** Every visual slot in the cache, the current frame included
(`kv_prune_candidate_scope=visual`, `kv_prune_recent_turns=0`, as in the flat arms). Each selector is
applied **one-shot to the whole unpruned pool** at every decision — the criterion itself, not the
flat run's cumulative state, in which earlier deletions shape later pools.

**Budgets.** `B(k) = round(k · mean visual count per turn)` for `k ∈ {4, 8, 16, 32}` frames' worth
(`kv_prune.frame_budget`); a budget binds only while the cache holds more than `k` frames.

**Per-token criteria** (current frame, raster order; `sup/selector_*` in `sequence.json`):

| method | continuous criterion (the maps and ρ) | budgeted selection (the mass-kept curves) |
|---|---|---|
| FastV (`attn`) | the decision token's layer-0 attention row, mean over heads (`kv_prune_score_layer=0`) | top-`B(k)` of that row |
| DivPrune | `divprune_nn` = the token's nearest-neighbour cosine distance in the pool on the embed DB (`1 − max_j cos`; the max-min objective, DivPrune's own initialisation score) | position in the greedy order (`divprune_select`, run to `B(32)`; −1 beyond) |
| DART | `dart_dis` = `1 − min_p cos` to the top-8 layer-0 key-norm pivots on the layer-0 output hidden states (a pivot claims the tokens least similar to it; pivots score 2.0, always kept) | membership at each budget (`VLMWorker._dart_select`; the sets are nested, `dart_nested` records it) |
| Random | none (control: minus the permutation position) | prefix of one permutation seeded like the flat random arm (`step_seed(17, label, step)`) |

For every method the token's **keep rank** is the smallest binding `k` at which it is kept (−1 =
never): the scale the four methods share.

**KL measures** — the leaky arm of `longnav.utils.token_info` at layer 0 (decision-token replay
against the live cache with the token's key column hidden; fp32 projection onto the four action
rows, `_ActionHead`):

* keep-one-in, **no visual history** — the token-level version of the information horizon's keep-one
  (`paper/info_horizon`: `KL(P_only_current ‖ P_no_visual)`):
  `keep_all[k] = KL(P_only k, every other visual token hidden ‖ P_no visual)`. Every history frame's
  visual tokens and the rest of the current frame are hidden; the past action text stays.

`3 + N` replays per decision (`N` = 300 tokens per frame with the filter off). `token_info`'s other
measures (`keep_cur`, history intact; `leave`) are selectable through `vlm.selector_probe_measures`
but are not part of the protocol.

**Statistics** (`src/longnav/analysis/selector.py`), per decision with `N ≥ 8` tokens and non-zero
KL mass:

* Spearman ρ between each continuous criterion and the KL over the frame's tokens (random: minus
  the permutation position, the control), plus ρ of each method's keep rank on the shared 5-level
  scale and of DivPrune's realised greedy order (`rho_divprune_order`);
* at every binding `k`, the **KL mass kept** = Σ keep-one-in KL over the tokens the method keeps
  at `k` / Σ over the frame, next to the **token share kept**; lift = their ratio (1 = random's
  expectation).

Aggregates: pooled mean with an episode-cluster bootstrap 95 % CI, the episode-balanced mean,
median, and the per-decision delta vs random with a cluster-bootstrap CI.

**Showcase decision.** Every budget binding (`n_frames > 32`) and the largest keep-one-in mass of
the frame (Σ over its tokens); `--showcase LABEL:STEP` overrides. Its RGB frame comes
from a deterministic re-drive (`showcase_frames.py`): the run's `video.mp4` frames carry
burned-in text and a lossy encode.

## Uni-NaVid arm

`--selector-probe` (`baselines/uninavid_hm3d/selector_probe.py`, `SelectorProbe`, modeled on the horizon
probe's `LayerProbe`): the native evaluator with the official seeded decoding of Exp 6
(`--sampling-seed 30`) and the full native history; after every inference (one per ~2 env steps) the
captured prefill is re-scored, decisions untouched (trajectories are bit-identical to
`info_horizon/runs/uninavid_probe`, which `aggregate.py` checks).

* **Tokens probed** = the current frame's columns as the horizon probe defines them: its 4-token
  history group + the 64 fine navigation tokens (68). Pool for the criteria = every visual column
  (history groups, long-term slots, the current frame). The flat Uni-NaVid pruner ranked history
  tokens only; here the criteria are applied to the current frame's tokens, as on LongNav-R1, because
  that is where the per-token KL lives.
* **Criteria**, all from one decoder-layer-0 forward on the captured prompt (the pruner's `_prepass`
  idiom): FastV = the last prompt token's layer-0 attention row, mean over heads; DivPrune =
  nearest-neighbour cosine distance on the LLM input embeddings; DART = `1 − min_p cos` to the top-8
  layer-0 key-norm pivots on the layer-0 output (pivots 2.0); random = position in a permutation of the
  pool seeded by `(17, label, step)`.
* **KL**: `keep_all[k] = KL(P_only k ‖ P_no visual)` by re-forwarding the prompt with every visual
  column except `k` closed (`layer_blind.hidden_mask`, every layer) — leak-free by construction since
  Uni-NaVid re-prefills. The action distribution is read through an fp32 projection of the last hidden
  state onto the four action-word rows (the fp16 head resolves ~3 significant digits per logit, the
  size of a single token's effect). 70 forwards per inference in batches of 12 (`--probe-batch`),
  1.1–1.9 s; the trace's `inference_seconds` / `cuda_peak_bytes` include the probe.
* Sidecar `selector/<label>.jsonl` (one line per inference: `keep_all, attn, divprune_nn, dart_dis,
  random_pos, is_nav, current_cols`, 68 each, plus `logp_full, logp_no_visual, kl_no_visual, n_visual,
  prompt_tokens, ...`); loader `analysis.selector.load_uninavid_selector`; no budget statistics (its
  flat budgets were history-pool budgets). Unit test `baselines/tests/test_selector_probe.py`.

## Arms

| arm | preset / flags | what |
|---|---|---|
| `uninavid_selector_probe` | `--sampling-seed 30 --selector-probe --probe-batch 12` (`run_uninavid.sh`) | Uni-NaVid on the same ten episodes (≈ 440 inferences; `wcojb4TFT35_0` runs to the 500-step cap under Uni-NaVid, 250 of them) |
| `r1_selector_probe` | `+experiment=paper_selector_probe` | ten episodes (`hm3d_v2_10_short.json`): the seed-17 ten minus `Nfvxx8J5NCo_2` plus `yr17PDCnDDW_0`. Lengths were measured first with the filter off and no probe (`runs/length_check/`): every seed-17 episode finishes in 21–96 steps except `Nfvxx8J5NCo_2` (298, fails), whose unfiltered cache makes the late replays cost minutes each, so it is replaced by the first `random.Random(17)` draw from the 71 other episodes of the 100 split that `r1_full` finishes under 150 steps (45 steps unfiltered). `wcojb4TFT35_0`, 500 steps with the filter on, takes 67 without it and stays in. 486 decisions; wandb project `longnav_paper_selector_nohist` |

## Run

```bash
conda activate longnav_vlm
bash paper/selector_kl/run.sh                     # R1 probe run (~2 h) -> Uni-NaVid probe run (~20 min) -> aggregate -> showcase frame -> plot
bash paper/selector_kl/run_uninavid.sh            # the Uni-NaVid arm alone (LABELS=... COUNT=1 ARM=... for a smoke)
PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python -m unittest baselines.tests.test_selector_probe
python paper/selector_kl/aggregate.py --allow-partial   # interim look while the run is going
/home/brabiei/miniconda3/envs/vln/bin/python paper/selector_kl/plot.py
python tests/selector_probe_smoke.py              # GPU smoke (first check CPU-only)
```

`paper_results/selector_kl/`: `runs/r1_selector_probe/` (config, launch log, `rollout/results_*`;
per-episode dirs gitignored), `runs/uninavid_selector_probe/` (manifest, `episodes/`, `selector/`;
traces gitignored), `runs/showcase/<label>/frames/` (gitignored), `results/`
(`per_token.parquet` — the tracked per-token data, `per_decision.csv`, `summary.csv`,
`table.tex`, `showcase.json`, `summary.json`), `figures/` (`showcase`, `spearman`).

**Figures.** `showcase`: one row — the observation, then the per-token keep-one-in KL, FastV's attention, DivPrune's
nearest-neighbour distance and DART's pivot dissimilarity, each smoothed onto the frame with its own colour scale and
its ρ vs the KL in the title; episode, decision, action and frame total go in the caption (printed by `plot.py`).
`spearman`: one panel per model, per-decision ρ of each criterion vs keep-one-in KL, box = quartiles, whiskers 5–95 %,
dot = pooled mean. The budgeted KL-share / token-share statistics stay in `results/summary.csv` and `table.tex`
(no figure).

## Status and findings

Built 2026-09-24. Two earlier attempts the same day were stopped and discarded: sparse-on on 10 episodes (with leave-one-out and a kept/dropped figure row), then sparse-off with the history-intact `keep_cur` measure (8 of 9 episodes; ρ ≈ 0 for FastV/DivPrune, −0.06 for DART — the same picture). The protocol above (no-history `keep_all`) was run on one episode first for sign-off, then the rest.

**Complete 2026-09-24 21:28** (10 episodes, 486 decisions, every one valid; consistency checks clean). Spearman ρ of
each criterion vs the no-history keep-one-in KL, pooled over decisions (balanced = mean of the per-episode means):

| criterion | ρ mean | median | balanced |
|---|---|---|---|
| Random (control) | −0.001 | −0.003 | −0.002 |
| FastV: layer-0 attention | −0.021 | −0.037 | −0.023 |
| DivPrune: nearest-neighbour distance | −0.062 | −0.065 | −0.066 |
| DART: pivot dissimilarity | −0.148 | −0.143 | −0.150 |

(DivPrune's realised greedy order gives the same ρ, −0.066. Bootstrap intervals are in `results/summary.csv`.)

**F1 — no criterion ranks the informative tokens.** FastV's layer-0 attention is indistinguishable from the random
control. Only ~4 % of the decision token's layer-0 attention falls on the current frame at all, and within
the frame it sits on the image's edge rows (rows 0–2 and 12–14 of 15 hold ~88 % of it; the same pattern on every frame,
map-to-map ρ 0.30 vs 0.14 for the KL maps): at layer 0 the row is a positional/sink pattern, not a content ranking.
This is the KL-side reading of the flat sweep's F1 (FastV ≈ random).

**F2 — the diversity criteria rank them backwards.** DivPrune and DART are significantly anti-correlated with the
keep-one-in KL on nearly every frame. The tokens that carry the decision on their own are the navigable surface ahead
(floor, path, doorway), which is uniform texture: exactly what a diversity criterion calls redundant and would drop
first, while it keeps the odd-looking patches (patterned borders, plants, furniture edges) that carry no decision
information. The flat sweep's collapse of DivPrune/DART (F2) has the same sign here at the token level.

Showcase: `wcojb4TFT35_0` decision 62 (forward; keep-one-in total 10 nats; ρ −0.12 / −0.08 / −0.20).

**Uni-NaVid arm (complete 2026-09-24 22:13; 10 episodes, 416 inferences, trajectories identical to
`info_horizon/runs/uninavid_probe`).** Same statistic over the current frame's 68 tokens:

| criterion | ρ mean | median | balanced |
|---|---|---|---|
| Random (control) | −0.006 | −0.006 | −0.014 |
| FastV: layer-0 attention | +0.183 | +0.189 | +0.079 |
| DivPrune: nearest-neighbour distance | −0.024 | −0.019 | −0.041 |
| DART: pivot dissimilarity | +0.008 | +0.011 | −0.069 |

**F3 — on Uni-NaVid, layer-0 attention does track the informative tokens, weakly.** FastV's ρ is
positive in every episode with more than one inference (0.02–0.28 per episode; 0.23 on the 64 fine
tokens alone, so it is not a group-vs-fine-token artefact) but small: the pooled mean is carried by the
250-inference `wcojb4TFT35_0` (ρ 0.25), the episode-balanced mean is 0.08. Unlike LongNav-R1, whose
layer-0 row is a positional/sink pattern, Uni-NaVid's LLaMA-7B layer-0 row over a short re-prefilled
prompt reads some content. This matches the flat sweep, where Uni-NaVid FastV was the one arm that sat
slightly above random at k = 16/32 without reaching significance. The diversity criteria are at zero on
Uni-NaVid (DivPrune −0.02, DART +0.01; balanced −0.04 / −0.07): no anti-correlation, because Uni-NaVid's
current frame is 64 fine tokens of one image whose informative and redundant regions are not as
cleanly separated as LongNav-R1's floor-versus-clutter split. Trajectories
differ from `r1_full`'s as expected with the filter off (`summary.json` lists them; `wcojb4TFT35_0` succeeds in 67
steps instead of failing at 500).
