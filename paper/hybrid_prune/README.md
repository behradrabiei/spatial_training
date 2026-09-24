# Experiment 7 — Hybrid pruning protocol vs PyramidDrop

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

## Earlier definitions of the budget (superseded, data kept)

- `..._b<k>` arms, `results_last_stage/`, `figures_last_stage/` (`aggregate.py --budget last_stage`): `k`
  bounded only the **last scheduled stage** (PyramidDrop's own reading, as in Exp 4) and layers 0–6 were
  never pruned. By the whole-model measure those arms held 8.0 / 9.6 / 11.3 / 13.9 frames' worth (hybrid)
  and 25.7 / 29.8 / 30.9 / 41.2 (PyramidDrop) at nominal `k` = 1 / 4 / 8 / 16 on LongNav-R1 — nowhere near
  `k`, and not matched. SR/SPL were statistically tied at every `k`.
- `..._m<k>` arms (`runs_aborted/`, 21 episodes): whole-model budget anchored at the original token
  count; stopped for the PyramidDrop starvation described above.

## MP3D replicate

The same 16 arms on MP3D ObjectNav val (`DATASET=mp3d`): 100 episodes drawn with seed 17 from the
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

# serial
bash paper/hybrid_prune/run_r1.sh && bash paper/hybrid_prune/run_uninavid.sh

# or two lock-coordinated workers on the one GPU (only ONE may hold Uni-NaVid arms)
nohup bash paper/hybrid_prune/run_worker.sh U uninavid_hybrid_mp8 uninavid_pyramiddrop_mp8 ... r1_pyramiddrop_mp2 ... &
EXTRA_OVERRIDES="resources.osm_gb=10" nohup bash paper/hybrid_prune/run_worker.sh R r1_hybrid_mp8 r1_pyramiddrop_mp8 ... &

# the MP3D replicate: prefix the same commands with DATASET=mp3d, and pass --dataset mp3d below
python paper/hybrid_prune/aggregate.py --status          # progress
python paper/hybrid_prune/aggregate.py --allow-partial   # interim
python paper/hybrid_prune/aggregate.py --strict          # final (--budget last_stage: the first run)
/home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot.py
```

Finished arms are skipped; a partial arm resumes (LongNav-R1 through wandb — project
`longnav_paper_hybrid` — Uni-NaVid through
`--resume`). An arm that ends at 99/100 (iterator starvation) is filled by re-running its script.

## Outputs

`paper_results/hybrid_prune/`: `runs/<arm>/` (configs, logs, per-episode results), `results/`
(`per_episode.csv`, `summary.csv`, `paired.csv` = hybrid − pyramiddrop per model and k with bootstrap CI
and paired sign-flip p, `table.tex`), `figures/` (`protocol_curves`, `budget_met`, `memory_tradeoff`).

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
