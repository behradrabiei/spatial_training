# Experiment 7 — Hybrid pruning protocol vs PyramidDrop (+ the uniform protocol, Exp 7b)

**Question.** The three ablations say what a pruning protocol for these agents should look like:
history is only needed over a short horizon (Exp 1, the context window), visual tokens stop mattering
above a decoder layer (Exp 5/6, the information horizon: layer 18 of 28 in LongNav-R1), and below it
no selection rule beats a random draw (Exp 4/4b). This experiment puts the three together in one
protocol — **hybrid** — and compares it with the strongest published method of Exp 4, **PyramidDrop**,
run the way it is published: over the whole history, with no recency window.

Episodes, models, decoding, the prune substrate and the budget unit are those of
[`paper/token_prune`](../token_prune/README.md); only what differs is stated here.

## The budget

`k` frames' worth of history visual tokens **held by the whole model**: the number of history visual
tokens each decoder layer holds, averaged over all layers, equals `k ×` (mean tokens one frame costs the
model: ≈ 200 post-filter tokens for LongNav-R1, 4 for Uni-NaVid). `k ∈ {2, 4, 8, 16}`. A token kept on
every layer costs one unit, a token kept on a quarter of the layers a quarter — the measure is
proportional to the KV memory (and attention compute) spent on visual history.

Outside the budget, identically for both protocols: text/action tokens and the prompt (never pruned),
and the **current frame**, which every layer always sees in full (deletion is permanent and the hook
runs after the action is read, so pruning touches history only). Nothing is pruned while the scope holds
`≤ k` frames.

Meeting it. Each step a solver picks the constant ratio `λ` of the schedule — stage `s` keeps `λ` of
what stage `s−1` holds — so that the layer-weighted mean equals the budget (`kv_prune.model_budgets`,
`prune.model_stage_budgets`; the hybrid's random tail counts, clamped to the stage above it). Two
consequences of taking the budget seriously:

- **A stage at layer 0 in both protocols.** With layers 0–6 unpruned, no fixed whole-model budget is
  reachable: those layers alone hold the whole scope. The layer-0 stage is ranked by the layer-0
  attention row (as the flat FastV arm of Exp 4b) and deletes from every layer; on Uni-NaVid it is the
  flat variant's layer-0 pre-pass with row removal, in front of the column-hiding stages.
- **The schedule shrinks from what the cache holds now** (`kv_prune_schedule_anchor=pool`: stage 0 draws
  from its survivors + the entering frame), not from the original token count of the scope (Exp 4's
  convention). The latter is harmless under a window, but without one it grows all episode and the ratio
  collapses — 140 steps in at `k=8`, no-window PyramidDrop's bands would hold 26 / 4.9 / 0.9 / 0.17
  frames' worth, a baseline starved by the anchor rather than by its design.

Under permanent deletion a stationary stage 0 can only shed what enters, so the solved ratios are gentle
(`λ ≈ 0.75–0.95`, vs PyramidDrop's published fixed `λ = 0.5` for a one-shot prefill). Steady-state
holdings in frames' worth per layer band:

| k | `pyramiddrop` (0–6 / 7–13 / 14–20 / 21–27) | `hybrid` (0–6 / 7–13 / 14–19 / 20–27) |
|---|---|---|
| 2 | 2.9 / 2.2 / 1.6 / 1.2 | 3.1 / 2.3 / 1.7 / 1.0 |
| 4 | 5.1 / 4.3 / 3.6 / 3.0 | 6.0 / 5.1 / 4.4 / 1.0 |
| 8 | 9.3 / 8.4 / 7.6 / 6.8 | 11.7 / 10.7 / 9.9 / 1.0 |
| 16 | 17.4 / 16.4 / 15.5 / 14.7 | 22.9 / 21.9 / 21.0 / 1.0 |

So at a matched budget the hybrid's deep cut is what buys it more tokens below the horizon — the
hypothesis under test.

## Protocol 3 — `uniform` (Exp 7b, added 2026-09-25)

The three ablations taken literally: below the information horizon no selector beats random (Exp 4/4b)
and depth does not matter, above it visual tokens carry nothing (Exp 5/6). So: the same K=32 frame-cut
window, **one uniform random draw at layer 0** that deletes from every layer — decoder layers below the
cut all hold the *same* history slots (the flat variant's mechanism) — and a **visual-blind cut**: from
layer 18 (LongNav-R1, 28 layers; Exp 6 blind-from-18 keeps SR at .83 vs .84, SPL −.036) / layer 20
(Uni-NaVid, 32 layers, the same depth fraction; Exp 6 blind-from-20 .74/.404 vs .71/.382) no query may
attend to **any** visual token — history *and the current frame* — and those layers hold no visual slot.
The whole-model budget `k` (pool anchor, as above) is therefore spread over the layers below the cut:
layers 0–17 hold `28k/18 ≈ 1.56k` frames' worth on LongNav-R1, layers 0–19 `32k/20 = 1.6k` on Uni-NaVid,
and the blind layers 0. `k ∈ {16, 8, 4, 2}`, the same 100 episodes.

| k | LongNav-R1 layers 0–17 | 18–27 | Uni-NaVid layers 0–19 (tokens) | 20–31 |
|---|---|---|---|---|
| 16 | 24.9 | 0 | 25.6 (102) | 0 |
| 8 | 12.4 | 0 | 12.8 (51) | 0 |
| 4 | 6.2 | 0 | 6.4 (26) | 0 |
| 2 | 3.1 | 0 | 3.2 (13) | 0 |

(31 history frames are in scope, so even k = 16 prunes.) **Caveat:** unlike `hybrid` and `pyramiddrop`,
whose layers all see the current frame in full, the blind layers do not see the current frame either, so
the comparison bundles the window, the uniform selection and the blind cut; the hidden current frame is
the only thing the arm changes relative to a "history-only" variant (`kv_prune_tail_budget_frames=0`
without the blind, expressible but not run).

Mechanics. LongNav-R1: `kv_prune_visual_blind=true` **together with** `kv_prune_stages` makes the LAST
stage blind (`VLMWorker._blind_stage`): `ReindexState.blind_layers` = that stage's layers, so
`hide_columns` closes every visual column there during the forward (the current chunk's included), and
the stage's budget is an implicit tail of 0, so its layers drop every visual slot after the step;
`model_budgets` then solves λ over the single scheduled stage (`widths=[18]`, `tail=(10, 0)`). Stages
`[0, 18]` with `[random, random]`: stage 0 is one `random_select` over its survivors + the entering frame,
the master cache compacts with it, no decision row is captured. Uni-NaVid: `--prune-method uniform`
(`STAGES["uniform"] = ((0, 0), (None, 20))`, `BLIND_STAGE`): stage 0 is the input-level seeded draw of
`random` with rows removed (no layer-0 pre-pass forward; every layer below 20 sees the same shortened
prompt), stage 1 a random stage of budget 0 drawn when layer 20's prefill starts (the whole pool lands in
`deleted[1]`) whose column mask also covers the current frame's 4-token group and its 64 fine tokens
(`_blind_cols`); `held_tokens` = `20·n0/32`; the record's `n_kept`/`bits` describe stage 0. Requires
`--prune-budget-scope model`. `tests/vtp_hybrid_smoke.py` (checks 5–6: holdings, the hidden columns on
each blind layer, cache identical to the history-only variant but a different decision),
`baselines/tests/test_prune.py` (`UniformProtocolTests`).

## The two protocols

| | `pyramiddrop` (no window) | `hybrid` |
|---|---|---|
| history in scope | LongNav-R1: the whole episode. Uni-NaVid: 64 frames (the widest the pruner supports; the long-term tier starts at 65) | 32 frames (31 history + current, frame cut) in both models |
| LongNav-R1 (28 layers) | attention-ranked stages at layers 0 / 7 / 14 / 21 | attention-ranked stages at 0 / 7 / 14; **random** stage from layer 20: a uniform draw of **1 frame's worth** among the layer-14 survivors |
| Uni-NaVid (32 layers) | stages at 0 / 8 / 16 / 24 | attention stages at 0 / 8 / 16; **random** stage from layer 24 keeping 4 tokens |

An attention stage is ranked by the last text token's attention at the layer just below it (mean over
heads; the layer-0 stage by layer 0's own row). The random stage needs no score: its draw is seeded by
`(seed 17, episode label, step)` and is nested in the previous stage's survivors; its budget does not
depend on `k` and binds from the second history frame on. The cut layers follow each model's information
horizon (Exp 6): LongNav-R1's blind-from-L curve is a cliff at 18 (cut at 20); Uni-NaVid's recovers by 16
with KL still 0.013 at 24 (cut at 24).

## Arms

`<model>_<protocol>_mp<k>`: 2 models × 2 protocols × 4 budgets = **16 arms**, 100 HM3D-v2 episodes each
(`src/longnav/conf/episode_jsons/hm3d_v2_100.json`). No identity or reference arms are run; the unpruned
references are Exp 1's (`r1_full` .84/.468, `r1_evict_k32` .89/.474).

| arm | overrides |
|---|---|
| `r1_hybrid_mp<k>` | `+experiment=paper_hybrid 'vlm.kv_prune_stages=[0,7,14,20]' 'vlm.kv_prune_stage_importance=[attn,attn,attn,random]' vlm.kv_prune_tail_budget_frames=1 vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool vlm.kv_prune_budget_frames=<k>` |
| `r1_pyramiddrop_mp<k>` | `+experiment=paper_hybrid vlm.context_frames=null vlm.context_cut=header 'vlm.kv_prune_stages=[0,7,14,21]' vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool vlm.kv_prune_budget_frames=<k>` |
| `uninavid_hybrid_mp<k>` | `--context-frames 32 --prune-method hybrid --prune-k <k> --prune-budget-scope model --prune-schedule-anchor pool --sampling-seed 30` |
| `uninavid_pyramiddrop_mp<k>` | `--context-frames 64 --prune-method pyramiddrop --prune-k <k> --prune-budget-scope model --prune-schedule-anchor pool --sampling-seed 30` |
| `r1_uniform_mp<k>` (Exp 7b) | `+experiment=paper_hybrid 'vlm.kv_prune_stages=[0,18]' 'vlm.kv_prune_stage_importance=[random,random]' vlm.kv_prune_visual_blind=true vlm.kv_prune_budget_scope=model vlm.kv_prune_schedule_anchor=pool vlm.kv_prune_budget_frames=<k>` |
| `uninavid_uniform_mp<k>` (Exp 7b) | `--context-frames 32 --prune-method uniform --prune-k <k> --prune-budget-scope model --prune-schedule-anchor pool --sampling-seed 30` |

Uni-NaVid samples its actions (temperature 0.2); both arms seed the per-call sampler so the pair differs
by the protocol only (and is therefore not comparable with Exp 4's unseeded arms).

Knobs (LongNav-R1, `src/longnav/config_schema.py`): `vlm.kv_prune_stage_importance`,
`vlm.kv_prune_tail_budget_frames`, `vlm.kv_prune_budget_scope`, `vlm.kv_prune_schedule_anchor`;
`kv_prune_stages` may start at 0. Uni-NaVid: `--prune-method hybrid`, `--prune-budget-scope`,
`--prune-schedule-anchor`. All default to Exp 4's behaviour, which stays bit-identical.

## What is and is not matched

- **Matched:** the whole-model visual budget. `held_frames` in `results/summary.csv` is what each arm
  actually held by that measure, and `figures/budget_met` plots it against `k`; it sits on `k` once an
  episode holds more than `k` frames (episode means are slightly lower because of those first steps).
- **Not matched: text.** No-window PyramidDrop keeps every past turn's ~8 text/action slots on every
  layer (≈ 3.9k slots by step 500, vs ≈ 600 under the window), so its total resident KV (`kv_len_mean`,
  `figures/memory_tradeoff`) stays above the hybrid's at the same visual budget.
- Uni-NaVid rebuilds its prompt every inference. Its layer-0 stage removes rows, but the stages at 8+
  hide key columns, so its counts measure how many history tokens a query attends to, not resident memory.
- A random stage that keeps one frame's worth under permanent deletion is, in effect, a thinned sample of
  the last few frames; stages can sit a few slots under their budget when the per-frame count drifts.
- The sweep runs two arms at a time on one GPU: `vlm_latency_ms` is contaminated and is not a result.

## Earlier definitions of the budget (superseded, archived locally)

- `..._b<k>` arms, `results_last_stage/`, `figures_last_stage/` (historical reporting mode): `k`
  bounded only the **last scheduled stage** (PyramidDrop's own reading, as in Exp 4) and layers 0–6 were
  never pruned. By the whole-model measure those arms held 8.0 / 9.6 / 11.3 / 13.9 frames' worth (hybrid)
  and 25.7 / 29.8 / 30.9 / 41.2 (PyramidDrop) at nominal `k` = 1 / 4 / 8 / 16 on LongNav-R1 — nowhere near
  `k`, and not matched. SR/SPL were statistically tied at every `k`.
- `..._m<k>` arms (`runs_aborted/`, 21 episodes): whole-model budget anchored at the original token
  count; stopped for the PyramidDrop starvation described above.

The superseded run directories and outputs are preserved under
`.cleanup_archive/before-codebase-cleanup/obsolete/paper_results/hybrid_prune/` with
checksummed restoration manifests. Current reporting accepts only the retained model budget.
The `last_stage` budget calculation remains available for the windowed token-pruning studies.

## MP3D replicate

The hybrid/PyramidDrop comparison, extended to all 24 arms with uniform below, on MP3D ObjectNav val (`DATASET=mp3d`): 100 episodes drawn with seed 17 from the
2195-episode split (`src/longnav/conf/episode_jsons/mp3d_100_seed17.json`, the manifest of the MP3D
information-horizon replicate). LongNav-R1 runs with `+dataset=mp3d_val`; Uni-NaVid with
`baselines/uninavid_hm3d/config/objectnav_mp3d.yaml` and the MP3D dataset and scene paths. Protocols,
budgets, stages and seeds are unchanged. Data in `paper_results/hybrid_prune_mp3d/`, wandb project
`longnav_paper_hybrid_mp3d` (the arm names repeat, and wandb resume keys on project + run name);
`aggregate.py --dataset mp3d`, `plot.py --dataset mp3d`. The dashed reference in `protocol_curves` is
the unpruned full-context arm of `paper_results/info_horizon_mp3d` (`r1_ref` .52/.236, `uninavid_ref`
.38/.140).

**Findings (sweep complete 2026-09-22 00:34, 16 arms × 100 episodes, ~6 h with two workers, no failed arm;
`aggregate.py --dataset mp3d --strict`).** SR / SPL; **bold** = hybrid − pyramiddrop p < .05 (paired
sign-flip, uncorrected), † = survives Bonferroni over the 16 tests (p < .003):

| | k=16 | k=8 | k=4 | k=2 |
|---|---|---|---|---|
| LongNav-R1 `pyramiddrop` (no window) | .43 / .183 | .43 / .184 | .29 / .113 | .07 / .017 |
| LongNav-R1 `hybrid` | **.57**† / **.247**† | .53 / .207 | **.46**† / **.183**† | **.19** / .036 |
| Uni-NaVid `pyramiddrop` (64 frames) | .33 / .127 | .27 / .103 | .35 / .137 | .29 / .104 |
| Uni-NaVid `hybrid` | .36 / .132 | .37 / **.139** | .35 / .142 | .31 / .114 |

1. **The HM3D result replicates on LongNav-R1.** ΔSR +0.14 / +0.10 / +0.17 / +0.12 and ΔSPL +0.064 / +0.023 /
   +0.070 / +0.020 at k = 16 / 8 / 4 / 2 (`results/paired.csv`); 5 of the 8 tests p < .05, 4 after Bonferroni
   (k = 8 is the exception: SR p = .06, SPL p = .28). The hybrid at k ≥ 8 (.53–.57 / .21–.25) sits on or above
   the unpruned full-context reference (.52 / .236) while holding 7.5–13.4 frames' worth per layer; no-window
   PyramidDrop plateaus at .43 from k = 8 up — on MP3D more budget does not help it at all (k = 16 = k = 8).
2. **Budgets met, hybrid holds fewer tokens.** Per-step maxima 16.02 / 8.02 / 4.02 / 2.01 frames' worth for
   both protocols (`figures/budget_met`); episode-mean holdings 13.4 vs 14.2, 7.5 vs 7.6, 3.9 vs 3.9, 2.0 vs
   2.0 (hybrid vs pyramiddrop). Layers ≥ 20 of the hybrid hold ≈ 200–210 visual slots (one frame) at every k.
3. **Not a stuck-episode artifact.** On the 41 episodes the unpruned reference solves in ≤ 150 steps the gap
   is larger: SR 1.00 vs .78, .93 vs .73, .76 vs .54, .34 vs .10 at k = 16 / 8 / 4 / 2.
4. **Total memory.** The no-window arm again keeps every past turn's text (2.0k–3.8k slots vs ≈ 550–620
   under the window), so its resident KV is 1.5–4× the hybrid's at the same visual budget (`kv_len_mean` 4857
   vs 3196, 3939 vs 2017, 3945 vs 1402, 4240 vs 1047; `figures/memory_tradeoff`). MP3D episodes run longer
   (PyramidDrop 245–480 mean steps vs 167–439 for the hybrid), which is why its text slots exceed HM3D's.
5. **Uni-NaVid: tie at every budget**, as on HM3D (ΔSR +.03 / +.10 / .00 / +.02, p ≥ .07; the one p < .05 is
   SPL at k = 8, +.036, not surviving correction), and no dose response (SR .27–.37 across an 8× range of k).
   Both arms sit at the model's MP3D full-context level (.38 / .140).
6. **Budget sensitivity, MP3D.** Both LongNav-R1 protocols collapse at k = 2 (.19 / .07 SR, mean steps 439 /
   480), and the whole curve sits ≈ .3 SR below HM3D's (.86 → .57 at k = 16 for the hybrid), tracking the
   reference (.89 → .52): the protocol's relative gain transfers; the absolute level is the model's.

Same caveats as HM3D: gentle solved λ, "k frames' worth" is k × the mean post-filter tokens per frame in
scope (≈ 190–220 here), latency contaminated by the concurrent worker, Uni-NaVid counts are attention width.

## Running

```bash
conda activate longnav_vlm
python tests/vtp_hybrid_smoke.py                       # both arm types, CPU + GPU
PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python -m unittest baselines.tests.test_prune

# serial (all three protocols; ARMS="..." / K_LIST="..." narrow it)
bash paper/hybrid_prune/run_r1.sh && bash paper/hybrid_prune/run_uninavid.sh

# or two lock-coordinated workers on the one GPU (only ONE may hold Uni-NaVid arms)
nohup bash paper/hybrid_prune/run_worker.sh U uninavid_hybrid_mp8 uninavid_pyramiddrop_mp8 ... r1_pyramiddrop_mp2 ... &
EXTRA_OVERRIDES="resources.osm_gb=10" nohup bash paper/hybrid_prune/run_worker.sh R r1_hybrid_mp8 r1_pyramiddrop_mp8 ... &

# the MP3D replicate: prefix the same commands with DATASET=mp3d, and pass --dataset mp3d below
python paper/hybrid_prune/aggregate.py --status          # progress
python paper/hybrid_prune/aggregate.py --allow-partial   # interim
python paper/hybrid_prune/aggregate.py --strict          # final whole-model protocol
/home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot.py
# MP3D includes all three protocols: add --dataset mp3d to aggregate.py / plot.py
```

Finished arms are skipped; a partial arm resumes (LongNav-R1 through wandb — project
`longnav_paper_hybrid` — Uni-NaVid through
`--resume`). An arm that ends at 99/100 (iterator starvation) is filled by re-running its script.

## Outputs

`paper_results/hybrid_prune/`: `runs/<arm>/` (configs, logs, per-episode results), `results/`
(`per_episode.csv`, `summary.csv`, `paired.csv` = `<protocol>` − pyramiddrop per protocol (hybrid,
uniform), model and k with bootstrap CI and paired sign-flip p, `table.tex`), `figures/`
(`protocol_curves`, `budget_met`, `memory_tradeoff` — three lines each: PyramidDrop, Protocol (hybrid),
Uniform + blind — and `cost_k16*`: see the cost benchmark below). The same three figures are also written
once per protocol, each showing only PyramidDrop and that protocol labelled "Protocol":
`<figure>_hybrid` and `<figure>_uniform` (`plot.py`; `--no-split` skips them), in `figures/` and in the
MP3D replicate's `paper_results/hybrid_prune_mp3d/figures/`.

## Cost benchmark: memory and inference time at k = 16 (`figures/cost_k16_100`, `cost_k16_100_uniform`)

`plot_cost.py` shows what the protocol saves against the unpruned models over the course of an episode
(2026-09-22/23). The sweep's own `vlm_latency_ms` is not usable for this: its arms ran two at a time on one
GPU, and on the first 16 steps, where the protocol's cache is still identical to the unpruned one, the
protocol run clocked 68 ms per step against 46 ms alone. So the four arms were re-run by
`run_cost_bench.sh`, each **alone on the GPU**, on the 100-episode split
(`BENCH=cost_bench_100`; data in `runs/cost_bench_100/`, one run dir per arm plus its log):

| arm | run | what it is |
|---|---|---|
| `cost_r1_full_100` | `+experiment=paper_ctx`, no overrides | LongNav-R1 unpruned, whole episode in the cache (Exp 1's `r1_full`) |
| `cost_r1_hybrid_mp16_100` | `+experiment=paper_hybrid` + the `hybridmp` overrides, `kv_prune_budget_frames=16` | the protocol at k = 16 (`r1_hybrid_mp16`) |
| `cost_uninavid_full` | native evaluator, `--sampling-seed 30` | Uni-NaVid unpruned: its own per-frame cache (4 tokens per frame for 64 frames, 1-token merged long-term slots beyond) |
| `cost_uninavid_hybrid_mp16` | `--context-frames 32 --prune-method hybrid --prune-k 16 --prune-budget-scope model --prune-schedule-anchor pool --sampling-seed 30` | the protocol at k = 16 (`uninavid_hybrid_mp16`) |
| `cost_r1_uniform_mp16_100` | `+experiment=paper_hybrid` + the `uniformmp` overrides, `kv_prune_budget_frames=16` | the uniform protocol at k = 16 (`r1_uniform_mp16`); added 2026-09-26 |
| `cost_r1_pyramiddrop_mp16_100` | `+experiment=paper_hybrid` + the `pyramiddropmp` overrides, `kv_prune_budget_frames=16` | no-window PyramidDrop at k = 16 (`r1_pyramiddrop_mp16`); added 2026-09-26 |
| `cost_uninavid_uniform_mp16` | `--context-frames 32 --prune-method uniform` + the same budget flags | the uniform protocol (`uninavid_uniform_mp16`); added 2026-09-26 |
| `cost_uninavid_pyramiddrop_mp16` | `--context-frames 64 --prune-method pyramiddrop` + the same budget flags | PyramidDrop, 64 frames (`uninavid_pyramiddrop_mp16`); added 2026-09-26 |

Every arm reproduces its sweep arm's trajectories on all 100 episodes (episode lengths and outcomes
identical: LongNav-R1 is argmax, Uni-NaVid's sampler and the random draws are seeded), so the curves are the
paper's runs with clean timing. The four 2026-09-26 arms ran in a later session than the unpruned and
hybrid arms (2026-09-22/23). Session drift was checked where the two sessions run the same code on the same
cache: steps 1–15 median 46.0 ms (PyramidDrop, new) vs 46.8 ms (hybrid, old), a 1.7 % gap. If anything the new
session is slightly faster, which favours PyramidDrop against the old hybrid run. The run names carry the benchmark
suffix because wandb resume keys on (project, run name) and would otherwise skip episodes already timed by
another benchmark on the same labels (that bit the seed-17 run once). Two smaller benchmarks were run first
and are kept: `runs/cost_bench/` (17 episodes with ≥ 250 steps under either R1 arm,
`hm3d_v2_17_long.json`, figure `cost_k16`) and `runs/cost_bench_seed17/` (the 10 seed-17 episodes, figure
`cost_k16_seed17`); `runs/cost_bench_100/cost_uninavid_*_nomem/` are the first 100-episode Uni-NaVid runs,
before the evaluator logged memory and prompt length (same trajectories).

**What is measured, per step of the episode.** Every quantity is read from the per-episode traces
(`rollout/<scene>/<episode>/sequence.json` for LongNav-R1, `traces/<label>.jsonl` for Uni-NaVid):

| panel | LongNav-R1 | Uni-NaVid |
|---|---|---|
| cache / history | `kv_len`: resident KV slots, mean over the 28 decoder layers, text and visual (the protocol's layers hold different counts) | `prompt_tokens`: the tokens the LLM prefills on the call — instruction + the visual history it keeps; recorded by a forward pre-hook on decoder layer 0 that takes the **last** multi-token forward of the call, so the protocol's layer-0 scoring pre-pass (which sees the unpruned prompt) is not what is counted |
| memory | `vlm_mem_GB` = `torch.cuda.memory_allocated()` on the VLM actor after the step: weights (the 2B-class Qwen3-VL, 4.0 GB bf16 on disk, is the 4.5 GB floor) + KV cache + embed DB. One slot across 28 layers × 8 KV heads × 128 dims × 2 (K, V) in bf16 is 112 KB, so 55k slots = 5.9 GB, which is what the curve gains | `cuda_peak_bytes` = `torch.cuda.max_memory_allocated()` over the call after `reset_peak_memory_stats()`: fp16 7B weights (15.8 GB floor) + prompt KV + activations. The panel's y-axis starts at 15.5 GB so the cache is what is visible |
| inference time | `vlm_latency`: wall time of the actor's `infer_step` (new turn through the incremental cache, action read from one forward, no generation), prune hook included (the final `.cpu()` synchronises after it) | `inference_seconds`: wall time of the whole prediction per call -- prompt building, encoding the new frames, `model.generate` (full prompt re-prefilled, then ~4 action words decoded) and the pruner's bookkeeping; one call decides two env steps |

Uni-NaVid values are held over the two env steps each call decides. A curve is the **mean over the episodes
still running at that step** (the population shrinks as episodes finish; `MIN_ALIVE=1` draws it out to the
longest episode), the band the interquartile range across them, latency smoothed with a 9-step moving
average. **The bracket is paired** (fixed 2026-09-26): at its step it averages the unpruned and protocol arms
over the SAME episodes, those still running in both, and prints the ratio (the GB difference for the
memory panels, since the weights are a floor shared by both arms); the episode count n is printed by
`plot_cost.py` on the console and listed in the table below. The first
version divided each arm's own late-episode mean, which by step 499 rested on 6 unpruned and 7 protocol
episodes with only 3 in common: two different groups (for the uniform protocol it read 26.6× fewer KV
slots against 20.5× paired). The bracket's ends are the shared episodes' means, so they need not sit on the
curves. Two versions of each figure: the bracket at the last step with ≥ 1 shared episode (`cost_k16_100`,
`cost_k16_100_uniform`) and at the last step with ≥ 5 (`_min5`), where more episodes back it but the
unpruned cache has grown less. PyramidDrop (green) appears on the inference-time panels only, since it
meets the same visual budget; its ratio to the protocol, by the same paired rule, is printed on the console
and listed below, not drawn. Its total KV also carries every past action's text, which is not shown. The spread
across episodes in the LongNav-R1 panels is the sparse filter: among the six unpruned episodes that reach
step 500 the cache ranges from 37k to 82k slots (74–165 post-filter tokens per step), depending on how much
re-observed geometry the filter drops.

**Result (100 episodes; paired brackets, n = shared episodes).**

| | KV / prompt tokens | memory | time vs unpruned | time, PyramidDrop / protocol |
|---|---|---|---|---|
| LongNav-R1 hybrid, step 499 (n = 3) | 23.3× fewer | −5.9 GB | 5.3× faster | 1.41× (n = 5) |
| LongNav-R1 hybrid, step 336 (n = 5) | 17.7× | −5.1 GB | 4.6× | 1.41× (step 499, n = 5) |
| LongNav-R1 uniform, step 499 (n = 3) | 20.5× | −6.3 GB | 5.4× | 1.53× (n = 11) |
| LongNav-R1 uniform, step 336 (n = 5) | 15.7× | −5.3 GB | 4.7× | 1.53× (step 499, n = 11) |
| Uni-NaVid hybrid, step 498 (n = 4) | 1.8× | −0.28 GB | 1.2× | 0.96× (n = 4) |
| Uni-NaVid uniform, step 498 (n = 3) | 1.7× | −0.27 GB | 1.2× | 1.00× (n = 2; 1.01× at step 318, n = 5) |

Averaged over whole episodes, LongNav-R1 takes 82 ms per step unpruned, 56 ms under the hybrid, 53 ms under
the uniform protocol and 55 ms under PyramidDrop. Peak VLM memory is 13.3 GB unpruned, 5.0 GB under either
protocol and 5.7 GB under PyramidDrop. The two protocols cost about the same; the uniform one is marginally faster,
consistent with layers 18–27 holding no visual slot and no attention row being captured (not isolated). PyramidDrop matches them early in
an episode and falls behind as its window-free text history grows: its per-step time rises to ≈ 67 ms by
step 500, against ≈ 45–50 ms for the protocols. Uni-NaVid: 1.7–1.8× fewer prompt tokens (≈ 560 → 310 hybrid, 325 uniform), −0.3 GB,
1.2× faster per call (129 → 116–118 ms, mean over calls; PyramidDrop 114 ms). Its own compression already
bounds its history, so neither protocol nor PyramidDrop saves much there, and PyramidDrop is as fast as
either protocol. The uniform protocol's blind cut is a mask on Uni-NaVid, so it saves no compute. LongNav-R1 is faster per step than Uni-NaVid because it is
a 2B-class model in bf16 against 7B in fp16, feeds only the new turn into an incremental cache instead of
re-prefilling the prompt, and reads the action from one forward instead of decoding tokens; per env step
Uni-NaVid's ≈ 65 ms is close to LongNav-R1's early-episode 45–60 ms.

```bash
conda activate longnav_vlm
MANIFEST=src/longnav/conf/episode_jsons/hm3d_v2_100.json BENCH=cost_bench_100 bash paper/hybrid_prune/run_cost_bench.sh   # unpruned + hybrid, ~70 min
MANIFEST=src/longnav/conf/episode_jsons/hm3d_v2_100.json BENCH=cost_bench_100 ARMS="uniform pyramiddrop" bash paper/hybrid_prune/run_cost_bench.sh   # ~67 min
for P in hybrid uniform; do for E in last min5; do
  /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_cost.py --bench paper_results/hybrid_prune/runs/cost_bench_100 \
    --name cost_bench_100 --protocol $P --edge $E
done; done
```
`ARMS` picks the protocols (`full`, `hybrid`, `uniform`, `pyramiddrop`; default `full hybrid`); `SKIP_R1=1` /
`SKIP_UNINAVID=1` rerun one model family; the GPU must be otherwise idle for the timings to mean anything.
`plot_cost.py` prints every bracket's paired and unpaired value. The older benchmark figures (`cost_k16`,
`cost_k16_seed17`) were drawn with the first, unpaired bracket and have not been regenerated.

## Findings

Sweep complete 2026-09-20 08:35 (16 arms × 100 episodes, `aggregate.py --strict`, no failed arm). SR / SPL
at a whole-model budget of `k` frames' worth; **bold** = hybrid − pyramiddrop significant (paired sign-flip,
uncorrected p < .05; † = also survives Bonferroni over the 16 tests, p < .003):

| | k=16 | k=8 | k=4 | k=2 |
|---|---|---|---|---|
| LongNav-R1 `pyramiddrop` (no window) | .75 / .371 | .63 / .312 | .39 / .162 | .14 / .037 |
| LongNav-R1 `hybrid` | **.86** / **.437**† | **.77** / **.357** | **.61**† / **.266**† | **.36**† / **.094**† |
| Uni-NaVid `pyramiddrop` (64 frames) | .74 / .375 | .70 / .373 | .62 / .335 | .68 / .387 |
| Uni-NaVid `hybrid` | .72 / .400 | .72 / .402 | .65 / .352 | .67 / .376 |

1. **LongNav-R1: the hybrid beats no-window PyramidDrop at every budget**, ΔSR +0.11 / +0.14 / +0.22 / +0.22
   and ΔSPL +0.066 / +0.045 / +0.104 / +0.057 at k = 16 / 8 / 4 / 2 — all eight tests p < .05, five of them
   after Bonferroni (`results/paired.csv`). The gap widens as the budget shrinks; at k = 16 the hybrid
   (.86 / .437) is within noise of the unpruned K=32 window (.89 / .474) and full context (.84 / .468) while
   holding ≈ 13 frames' worth per layer instead of 31 or the whole episode.
2. **It is not a budget artifact.** Both protocols sit on the budget (`figures/budget_met`; per-step maxima
   16.02 / 8.02 / 4.02 / 2.01 frames' worth, episode means a little lower because an episode's first steps
   hold fewer than k frames), and the hybrid actually holds slightly FEWER tokens than PyramidDrop at every k
   (2782 vs 2986, 1572 vs 1663, 845 vs 903, 430 vs 463 visual tokens, layer mean). The gap is also not a
   stuck-episode effect: on the 76 episodes the unpruned model solves in ≤ 150 steps it is the same size or
   larger (SR .99 vs .87, .91 vs .74, .70 vs .50, .45 vs .16).
3. **What the comparison can and cannot attribute.** The two protocols differ in the window and in how the
   budget is spread over depth. At k = 2 the solved schedules are practically identical (3.1 / 2.3 / 1.8 /
   1.0 vs 3.1 / 2.4 / 1.8 / 1.3 frames' worth per band) and the hybrid still wins by +0.22 SR, so there the
   gain is **the window** — which tokens are held, not how many per layer: attention ranking without a window
   keeps tokens from anywhere in the episode and starves the recent frames (the Exp 4b result). At k ≥ 4 the
   deep random cut additionally lets the hybrid hold ≈ 15 % more below layer 20 (10.9 vs 9.4 frames' worth on
   layers 0–6 at k = 8; 18.8 vs 16.4 at k = 16) at no measurable cost above it (1.0 frame's worth on layers
   20–27 vs 7–14 for PyramidDrop). This experiment does not separate the two contributions; a
   `pyramiddrop` + K=32 window arm under the same budget would.
4. **Total memory.** The budget covers visual history only. No-window PyramidDrop also keeps every past
   turn's text (1.6k–3.7k slots, more the longer its episodes run) where the window holds ≈ 550, so its
   resident KV is 1.4–4× the hybrid's at the same visual budget (`kv_len_mean` 4558 vs 3316, 3672 vs 2122,
   3748 vs 1414, 4125 vs 1040; `figures/memory_tradeoff`).
5. **Uni-NaVid: no difference at any budget** (|ΔSR| ≤ 0.03, all p ≥ .19), and barely any dose response
   (SR .62–.74 across an 8× range of k): with 4 tokens per history frame and the current frame's 64 fine
   tokens untouched, this model does not depend on which history tokens survive — as in Exp 4. The protocol's
   benefit is specific to the agent whose decisions actually read the visual history.
6. **Budget sensitivity.** Both LongNav-R1 protocols degrade steeply below k = 8 (hybrid .77 → .61 → .36 SR);
   a whole-model budget of 2 frames' worth is too little for either. Under the first, last-stage definition
   the same nominal k looked flat because the unpruned layers 0–6 were silently holding 8–40 frames' worth.

Caveats. (a) The solved ratios are gentle (λ ≈ 0.75–0.95): under permanent deletion a stationary stage can
only shed what enters, so neither protocol has PyramidDrop's published λ = 0.5 shape; a fixed-λ variant is
untested. (b) "k frames' worth" is k × the mean post-filter tokens per frame in scope (≈ 205–220 here,
computed at inference); after an agent gets stuck the filter admits few new patches and the token budget
shrinks with it. (c) Rounding to whole tokens lets a step exceed the budget by < 0.2 %. (d) Latency is not
reported (two arms shared the GPU). (e) Uni-NaVid's counts are attention width, not resident memory.

## Findings — `uniform` (Exp 7b)

Sweep complete 2026-09-25 02:15 (8 arms × 100 episodes, ~1 h 50 min with two workers, no failed arm;
`aggregate.py --strict`, the 16 hybrid/pyramiddrop arms reused). SR / SPL; **bold** = uniform −
pyramiddrop p < .05 (paired sign-flip, uncorrected), † = survives Bonferroni over this protocol's 16
tests (p < .003):

| | k=16 | k=8 | k=4 | k=2 |
|---|---|---|---|---|
| LongNav-R1 `pyramiddrop` (no window) | .75 / .371 | .63 / .312 | .39 / .162 | .14 / .037 |
| LongNav-R1 `hybrid` | .86 / .437 | .77 / .357 | .61 / .266 | .36 / .094 |
| LongNav-R1 `uniform` | .80 / .396 | **.73** / **.349** | **.61**† / **.271**† | **.35**† / **.131**† |
| Uni-NaVid `pyramiddrop` (64 frames) | .74 / .375 | .70 / .373 | .62 / .335 | .68 / .387 |
| Uni-NaVid `hybrid` | .72 / .400 | .72 / .402 | .65 / .352 | .67 / .376 |
| Uni-NaVid `uniform` | .72 / .399 | .75 / .400 | **.74** / **.398** | .66 / .384 |

1. **LongNav-R1: uniform beats no-window PyramidDrop at k ≤ 8**, ΔSR +0.05 / +0.10 / +0.22 / +0.21 and
   ΔSPL +0.026 / +0.037 / +0.108 / +0.094 at k = 16 / 8 / 4 / 2 (`results/paired.csv`, `protocol=uniform`);
   six of the eight tests p < .05, four after Bonferroni (k = 4 and k = 2); at k = 16 the gain is not
   significant (SR p = .31, SPL p = .22). On the 76 episodes the unpruned model solves in ≤ 150 steps: SR
   .92 / .83 / .71 / .45 vs PyramidDrop's .87 / .74 / .50 / .16 (hybrid .99 / .91 / .70 / .45).
2. **Uniform ≈ hybrid at k ≤ 4, hybrid ahead at k = 16.** Paired uniform − hybrid on LongNav-R1: SR −0.06
   (p = .11) / −0.04 (.45) / 0.00 / −0.01; SPL −0.040 (p = .047) / −0.008 / +0.005 / +0.037 (p = .09);
   steps +33 (p = .006) / +28 (p = .055) / +6 / −12. So a protocol with **no attention score, no schedule
   and no visual token at all above layer 18** matches the attention-staged hybrid once the budget is
   4 frames' worth or less, and loses ≈ 0.04 SPL and 30 steps per episode at k = 16 / 8. That k = 16 gap
   is the size of the blind cut's own cost on the unpruned model (Exp 6, blind-from-18: SPL −0.036, steps
   +32 with SR intact), which is consistent with the cut — not the uniform selection — being what the
   arm pays for; this experiment does not separate the two (the history-only variant,
   `kv_prune_tail_budget_frames=0` without the blind, was not run).
3. **Budgets met, blind layers empty.** Per-step maxima 16.02 / 8.02 / 4.02 / 2.02 frames' worth
   (LongNav-R1) and 15.94 / 7.97 / 4.06 / 2.03 (Uni-NaVid, `held_tokens`); `sup/mean_kv_stage_visual_1`
   is 0 in every episode and Uni-NaVid's `stage_kept[1]` is 0 on every call. Episode-mean holdings
   13.2 / 7.4 / 3.9 / 2.0 (LongNav-R1) and 10.9 / 6.3 / 3.5 / 1.8 (Uni-NaVid) — slightly under the other
   two protocols' (`figures/budget_met`). Layers 0–17 held 20.8 / 11.7 / 6.2 / 3.2 frames' worth per layer
   on average (steady state 24.9 / 12.4 / 6.2 / 3.1; the first steps of an episode hold less), so the
   longest layer's cache is 6.2k / 3.5k / 2.1k / 1.4k slots while the layer mean (`kv_len_mean`, text
   included) is 3151 / 2008 / 1374 / 1014 — a little below the hybrid's 3316 / 2122 / 1414 / 1040 and
   1.4–4× below PyramidDrop's; peak VLM memory 4.99 / 4.78 / 4.67 / 4.62 GB.
4. **Uni-NaVid: a tie, as always**, except k = 4 where uniform is above both others (SR .74 vs .62 / .65;
   vs PyramidDrop +0.12 p = .026, SPL +0.063 p = .041; vs hybrid +0.09 p = .034 — none survives
   correction, and the curve has no dose response: SR .66–.75 across an 8× range of k, all at the model's
   full-context level .72 / .417). Its prompts shrink to 308 / 260 / 229 / 210 tokens (mean).
5. **What it says about the paper's story.** The ablations' recipe — recency window, random selection
   below the horizon, nothing above it — reproduces the hybrid's advantage over the published no-window
   method at every budget where that advantage is large (k ≤ 8), with a strictly simpler mechanism; the
   attention ranking and the 1-frame tail of the hybrid buy a measurable SPL margin only at the largest
   budget. The uniform arm also hides the current frame above the cut, so it is the stronger claim about
   the information horizon: the layers above 18 (LongNav-R1) / 20 (Uni-NaVid) can be blind to vision
   altogether at a cost of ≈ 0.04 SPL on LongNav-R1 and none on Uni-NaVid.

Same caveats as the hybrid comparison: latency contaminated (two arms per GPU; the k = 8 and k = 4 arms
also overlapped with a Uni-NaVid arm), "k frames' worth" is k × the mean post-filter tokens per frame in
scope, Uni-NaVid counts are attention width. Uni-NaVid's `--prune-seed` is 17 in every arm (the
`--sampling-seed 30` seeds the action sampler only).

### MP3D replicate of the uniform arm

The same 8 arms on MP3D val (`DATASET=mp3d`, `mp3d_100_seed17.json`; sweep complete 2026-09-25 12:44, ~2 h
55 min with two workers, no failed arm; one k = 16 episode that the first pass lost at the iterator's end was
filled by the worker's retry pass, and an independent rerun of it reproduced the row bit-for-bit).
`aggregate.py --dataset mp3d --strict` (all three protocols; the old `--protocols pyramiddrop hybrid`
narrowing is no longer needed), `plot.py --dataset mp3d`. SR / SPL; **bold** = uniform − pyramiddrop
p < .05 (paired sign-flip, uncorrected), † = survives Bonferroni over the 16 tests:

| | k=16 | k=8 | k=4 | k=2 |
|---|---|---|---|---|
| LongNav-R1 `pyramiddrop` (no window) | .43 / .183 | .43 / .184 | .29 / .113 | .07 / .017 |
| LongNav-R1 `hybrid` | .57 / .247 | .53 / .207 | .46 / .183 | .19 / .036 |
| LongNav-R1 `uniform` | .47 / .218 | .44 / .192 | .35 / .136 | **.20**† / **.063**† |
| Uni-NaVid `pyramiddrop` (64 frames) | .33 / .127 | .27 / .103 | .35 / .137 | .29 / .104 |
| Uni-NaVid `hybrid` | .36 / .132 | .37 / .139 | .35 / .142 | .31 / .114 |
| Uni-NaVid `uniform` | .38 / .151 | .34 / .137 | .36 / .119 | .32 / .129 |

1. **On MP3D the uniform arm does NOT replicate the hybrid's gain at k ≥ 4.** Uniform − pyramiddrop on
   LongNav-R1: ΔSR +0.04 / +0.01 / +0.06 / +0.13 and ΔSPL +0.034 / +0.008 / +0.023 / +0.047 at k = 16 / 8 /
   4 / 2 — only k = 2 is significant (SR p = .010, SPL p = .005, both survive correction). Paired uniform −
   hybrid: SR −0.10 (p = .042) / −0.09 (.11) / −0.11 (.049) / +0.01, SPL −0.030 / −0.015 / −0.047 (p = .06) /
   +0.027, steps +72 / +58 / +54 (all p ≤ .001) / −4. The hybrid solves the MP3D episodes ≈ 55–70 steps
   faster than the uniform arm at every k ≥ 4; on the 41 episodes the unpruned model solves in ≤ 150 steps,
   SR is 1.00 / .93 / .76 / .34 (hybrid) vs .85 / .76 / .56 / .34 (uniform) vs .78 / .73 / .54 / .10
   (PyramidDrop).
2. **Contrast with HM3D.** On HM3D the uniform arm tied the hybrid at k ≤ 4 and trailed only at k = 16; on
   MP3D it sits about halfway between the hybrid and PyramidDrop at k = 16 / 8 / 4 and equals the hybrid
   only at k = 2, where the two protocols hold practically the same slots (3.2 vs 3.1 / 2.3 / 1.8 / 1.0 frames'
   worth per band) and both are near failure. MP3D episodes are longer (mean 238–435 steps for the uniform
   arm vs 163–382 on HM3D), so the uniform arm's two costs weigh more: the blind cut takes the current
   frame away from layers 18–27 on every step, and the uniform draw keeps no preferentially-attended slot.
   Which of the two accounts for the gap is not separated here; the natural ablations are the history-only
   variant (`kv_prune_tail_budget_frames=0`, no blind) and a blind hybrid.
3. **Budgets met, blind layers empty**, as on HM3D: per-step maxima 16.02 / 8.02 / 4.02 / 2.02 frames' worth
   (LongNav-R1) and 15.94 / 7.97 / 4.06 / 2.03 (Uni-NaVid); layers 18–27 hold 0 visual slots in every
   episode; layers 0–17 hold 21.3 / 11.9 / 6.3 / 3.2 frames' worth per layer on average; `kv_len_mean` 2871 /
   1914 / 1293 / 1023 (below the hybrid's 3196 / 2017 / 1402 / 1047 and 1.7–4.1× below PyramidDrop's); peak
   VLM memory 4.98 / 4.77 / 4.67 / 4.62 GB.
4. **Uni-NaVid: tie at every k** (uniform − pyramiddrop SR +0.05 / +0.07 / +0.01 / +0.03, all p ≥ .19;
   uniform − hybrid |ΔSR| ≤ 0.03), no dose response (SR .32–.38), all at the model's MP3D full-context
   level (.38 / .140).
