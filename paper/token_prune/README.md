# Visual-token pruning under a 32-frame window

Do the training-free visual-token pruning methods of the VLM literature (FastV, PyramidDrop,
DivPrune, DART) beat uniform-random pruning and the plain recency window when they are applied
to the *history* tokens of an ObjectNav VLA at a matched token budget? SR / SPL / steps vs the
history-token budget, for LongNav-R1 and Uni-NaVid, on the 100-episode HM3D-v2 val split, plus
which frame ages each method keeps.

## Definitions

**Window.** The context-window ablation's `K=32` arm: the current frame plus the **31** most
recent history frames (`K` counts the current frame, `paper/ctx_ablation/README.md`). Its
per-episode results are the *reference* of this experiment, and the ctx-ablation arms
`K ∈ {16, 8, 4, 1}` are the *recency* baseline. They are reused as-is, not rerun.

**Candidates `V_t`.** Every visual token of the history frames in the window. The prompt prefix,
every text/action token, the current frame and (Uni-NaVid) the separators and the current frame's
64 fine tokens are never candidates.

**Budget `B'`** — *what `k` history frames cost this model*, `k ∈ {16, 8, 4, 1}`:

| | LongNav-R1 | Uni-NaVid |
|---|---|---|
| tokens per history frame `n` | post-sparse-filter visual slots, content dependent (≈100–300; the per-arm mean is in `results/n_tokens.csv`) | 4, constant (`compress_type=grid:2`; no long-term bin under a 32-frame window) |
| `B'_t` | `round(k · V0_t / h_t)`, `h_t = min(t, 31)` history frames in the window, `V0_t` = their original post-sparse counts summed | `k · 4` = 64 / 32 / 16 / 4 |

The two models are never compared in absolute tokens. Nothing is pruned while the window holds
`≤ k` history frames, so every pruning arm coincides with the reference for its first `k` steps.

**The `K` offset.** The recency arm `K` holds `K−1` history frames (15 / 7 / 3 / 0), one fewer
than the `k` frames' worth a pruning arm keeps. Both are drawn on one axis, "history tokens kept
in frames' worth", at their true value (`frames_worth` in the tables): recency at 15/7/3/0, pruning
at 16/8/4/1, the reference at 31. The paired deltas in `summary.csv` are against the same-`k`
recency arm and against the reference.

**Permanent deletion (truncation protocol).** A history token the hook does not keep is removed
for good; the next step selects among the survivors plus the frame that just became history.
Survivors' K/V are never recomputed (the ctx ablation's evict protocol, same leakage as the
recency arms). For the layer-placed methods the deletion applies from decoder layer `L` onward:
layers `< L` keep the whole window.

*LongNav-R1.* The hook runs at the end of step `t`, after the action was read (equivalently
before step `t+1` decodes): the just-decided turn's frame is history by then. Deletion is a
physical slice of the KV cache (`context_window_mode=prune`, `kv_prune_candidate_scope=visual`,
`kv_prune_recent_turns=0`); survivors keep their original mRoPE positions. Layer-placed scores
are read in step `t`'s decision forward and act from step `t+1` — the one-step lag an incremental
cache imposes. **Caveat:** with FastV/DART/PyramidDrop the master cache (layers `< L`) and the
sparse-filter database keep every window token, so new patches are still deduplicated against
tokens the deep layers no longer hold; only Random/DivPrune (all layers) shrink the database and
let a deleted patch be re-admitted.

*Uni-NaVid.* The prompt is re-prefilled every inference. Layer-0 methods remove the deleted rows
from the prompt (an emptied frame's `<image_sep>` goes with it); layer-placed methods keep the
rows and hide their key columns from layer `L` onward, during prefill and decoding, with the
score read at layer `L−1` of the same prefill. Deletions persist across inferences.

## Arms

`r1_identity` / `uninavid_identity` run the hook with an unlimited budget and must reproduce the
reference (LongNav-R1 bit for bit; Uni-NaVid samples its actions, see Protocol). Then
2 models × 5 methods × 4 budgets = 40 arms, `<model>_<method>_b<k>`.

| method | rule (query = last text token, the decision position) | `L` (R1 28 layers / Uni-NaVid 32) | LongNav-R1 overrides on top of `+experiment=paper_prune` |
|---|---|---|---|
| `random` | uniform `B'` from the alive pool; generator reseeded from (seed, episode label, step) | 0 | `vlm.kv_prune_importance=random` |
| `fastv` | mean-over-heads attention of the last text token at layer 1 → keep top-`B'` (paper default `K=2`) | 2 | `vlm.kv_prune_importance=attn vlm.kv_prune_layer_start=2 vlm.kv_prune_score_layers=boundary` |
| `pyramiddrop` | same score read at the last layer of each stage; stages start at 7/14/21 (R1) and 8/16/24 (Uni-NaVid) with nested budgets `round(V0·λ^s)`, `λ = (B'/V0)^{1/3}`, so the last stage is exactly `B'` | staged | `vlm.kv_prune_importance=attn 'vlm.kv_prune_stages=[7,14,21]'` |
| `divprune` | greedy max–min under cosine distance on the post-projector token embeddings, initialised at the token with the largest nearest-neighbour distance, joint over the window | 0 | `vlm.kv_prune_importance=divprune` |
| `dart` | 8 pivots = largest L1 norm of the layer-1 keys across the window; cosine on the layer-2 input hidden states; per pivot (descending norm) keep the `(B'−8)//8` least-similar remaining tokens, remainder topped up round-robin; pivots kept | 2 | `vlm.kv_prune_importance=dart vlm.kv_prune_layer_start=2` |

Every non-identity R1 arm adds `vlm.kv_prune_budget_frames=<k>`; Uni-NaVid arms pass
`--prune-method <method> --prune-k <k>` (`--prune-method identity` for the identity arm).

## Protocol

- Split: `src/longnav/conf/episode_jsons/hm3d_v2_100.json`, identical for every arm; a final
  number uses an arm only if it holds all 100 episodes (`--allow-partial` for interim looks).
- LongNav-R1: `src/longnav/config/experiment/paper_prune.yaml` = the ctx-ablation preset with
  `context_frames=32 context_cut=frame` on the prune substrate (checkpoint `+checkpoint=longnav`,
  `sdpa`, bf16, argmax actions, 500-step cap, `ep_seed=17`, sparse filter on, no viz). Each arm
  adds only the overrides above; the resolved config is saved to `runs/<arm>/config.yaml`.
- Uni-NaVid: the ctx-ablation settings (official evaluator, seed 30, temperature 0.2, 2 actions
  per inference, 500 steps, `--context-frames 32`). The ctx-ablation arms did not seed the per-call
  sampler, so `uninavid_identity` reproduces `uninavid_k32` only up to the documented ±0.04 SR
  run-to-run noise; `aggregate.py --check-identity --identity-tolerance` accepts that.
- Keep-mask logs: every step, per history frame, the original token count, the alive count and a
  packed bit mask over the frame's tokens (LongNav-R1: `sup/vtp_*` in `sequence.json`, token order
  = raster patch index in `sup/vtp_raster`; Uni-NaVid: `prune/<label>.jsonl`). Rows are keyed
  `(model, episode, step, frame)` like `paper_results/frame_influence/results/influence.csv.gz`,
  so retention can be joined with the influence tables.
- Statistics: episode means with 95% percentile-bootstrap CIs (2000 resamples, seed 17); paired
  two-sided sign-flip permutation tests (10 000 resamples) on `episode_label` vs the same-`k`
  recency arm and vs the reference. Retention: per (arm, age) the mean over episodes of the
  episode's mean kept fraction, bootstrap CI over episodes.

## Running

```bash
# whole chain, one GPU: identity arms -> identity check -> 40 pruning arms (~14 h)
nohup bash paper/token_prune/run_all.sh > /dev/null 2>&1 &     # IDENTITY_TOLERANCE=1 to accept Uni-NaVid noise

# or per family (order: identity, k=8 for every method, then 4, 1, 16)
conda activate longnav_vlm
bash paper/token_prune/run_r1.sh          # ARMS="r1_identity r1_random_b8" bash ...  (~20-40 min per arm)
bash paper/token_prune/run_uninavid.sh    # ~11 min per arm

# progress / interim numbers and figures at any time
python paper/token_prune/aggregate.py --status
python paper/token_prune/aggregate.py --allow-partial
/home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py       # --k 8 for the retention figure

# final
python paper/token_prune/aggregate.py --strict
/home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py
```

Do not run the two model families concurrently on one GPU (Uni-NaVid-7B alone takes ~15 GB).
Both run scripts skip complete arms and resume partial ones (LongNav-R1 through wandb's
`task.run_name`, Uni-NaVid through `--resume`).

## Outputs — all under `paper_results/token_prune/` (tracked, per `paper_results/.gitignore`)

- `runs/<arm>/` — raw run output. LongNav-R1: `config.yaml`, `launch.log`, `rollout/results_*`
  (per-episode JSONL, incl. `sup/mean_vtp_n_new`); the per-episode `rollout/<scene>/…` dirs with
  `sequence.json` (keep masks) stay on disk but are gitignored. Uni-NaVid: `manifest.json`,
  `summary.json`, `episodes/*.json`, `prune/*.jsonl` + `prune/meta.json`, `launch.log`; `traces/`
  gitignored. `driver.log` is the chained launch log.
- `results/per_episode.{csv,parquet}` — one row per (arm, episode), the ctx-ablation recency and
  reference rows merged in (`method ∈ {recency, reference}`), `partial` flag.
- `results/summary.csv` — per arm: n, SR/SPL/steps with CIs, tokens per frame, tokens kept, paired
  deltas and p vs the same-`k` recency arm (`*_vs_recency`) and vs the reference (`*_vs_ref`).
- `results/table.tex` — booktabs, stars vs the same-`k` recency arm, † = partial arm.
- `results/n_tokens.csv` — `n`, mean tokens kept and kept fraction of the 31-frame window per arm.
- `results/masks.parquet` — one row per (arm, episode, step, history frame): tokens, kept, packed
  keep bits. `results/retention.csv` — kept fraction vs frame age per arm with CIs.
- `figures/budget_curves.{pdf,png}` — SR and SPL vs frames' worth, one column per model, the
  recency curve in black, the reference dashed; hollow markers = partial arms.
  `figures/retention_by_age.{pdf,png}` — kept fraction vs age at `k=8`, one panel per model.

## Findings (sweep complete 2026-09-09; 100 episodes per arm; SR / SPL / mean steps)

| method | LongNav-R1 k=1 | k=4 | k=8 | k=16 | Uni-NaVid k=1 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|---|---|
| recency (K−1 frames) | .00/.000/500 | .70/.282/261 | .73/.361/211 | .85/.420/146 | .48/.302/122 | .57/.289/110 | .68/.378/109 | .71/.383/123 |
| random | .18/.023/470 | .50/.209/309 | .69/.297/235 | .84/.389/158 | .61/.330/107 | .68/.358/102 | .73/.395/126 | .72/.371/126 |
| fastv | .04/.003/495 | .62/.196/299 | .77/.349/187 | .81/.405/149 | .61/.332/120 | .70/.403/116 | .71/.399/109 | .72/.389/125 |
| pyramiddrop | **.80/.394/169** | **.82/.421/146** | **.87/.450/124** | **.88/.461/125** | **.74/.368/112** | **.78/.447/107** | .76/.428/106 | .77/.415/108 |
| divprune | .04/.006/495 | .32/.047/442 | .67/.256/270 | .80/.396/174 | .63/.297/133 | .70/.384/118 | .75/.399/116 | .77/.400/124 |
| dart | .02/.002/498 | .36/.046/445 | .58/.154/348 | .75/.336/194 | .58/.305/133 | .74/.382/123 | .67/.361/143 | .74/.400/132 |
| reference (K=32) | .89/.474/112 | | | | .76/.430/99 | | | |

Identity arms reproduced the references exactly (0 mismatching episodes on both models).

**Q1 — does any method beat the recency window on LongNav-R1?** Only PyramidDrop, and it does so at
every budget (SPL vs same-k recency p < 0.001 at k ≤ 8, p = 0.06 at k = 16); at one frame's worth it
keeps SR 0.80 where the window and every other method collapse. Random, FastV, DivPrune and DART are
all *below* the recency curve at k ≤ 8 (SPL p ≤ 0.01 except FastV at k=8, p = 0.56) and only reach it
at k = 16. The published importance/duplication criteria do not beat uniform sampling: FastV ≈ random,
DivPrune and DART are clearly worse (both prefer old frames, see the retention figure, and starve the
recent ones the policy reads).

**Q2 — the PyramidDrop caveat.** It is not memory-matched: its layers 0–6 keep the whole window, so at
k = 1 it holds a layer-mean 2334 KV slots vs 865 for random (`results/n_tokens.csv`, `kv_len_mean`
in `summary.csv`), i.e. as much memory as random at k ≈ 6, where random already scores 0.6–0.7 SR.
Read together with the layer-horizon result (history is read below layer ~16–20), its advantage is
"the shallow layers see everything", not a better token selector.

**Q3 — Uni-NaVid.** Every method is on or above the recency curve at every budget (at k = 4 all five
beat recency on SPL, p ≤ 0.04); PyramidDrop matches the 32-frame reference from k = 4 on. Its 4-token
frames make the budget cheap to meet and the model was already flat in Exp 1, so pruning here costs
nothing and which tokens are kept hardly matters.

**Retention (k = 8).** Two families on both models: Random / FastV / PyramidDrop are recency-biased
survival curves (FastV drops everything older than ~18 steps on Uni-NaVid); DivPrune / DART flatten
retention across ages (~25 % of even the oldest frame survives) at the cost of the recent frames.

Caveats: permanent deletion means each step's selection acts on a pool earlier steps already shaped;
R1's layer-placed arms keep the sparse-filter DB intact for layers < L; DART used 8 pivots for both
models (6 % of Uni-NaVid's pool, above the paper's 2 % rule; degenerate at k = 1 where P is clipped
to 4).

## Flat variant — the selection rule alone (independent of time and of layer)

The experiment above confounds the selector with two other choices: the 32-frame window is itself a
recency prior, and layer placement decides how much the shallow layers still see (Q2: PyramidDrop's
win is "layers 0–6 see everything"). The flat variant removes both. Data: `paper_results/token_prune_flat/`.

| | LongNav-R1 | Uni-NaVid |
|---|---|---|
| context | **full episode** (`context_frames=null`; nothing is ever evicted by age) | `--context-frames 64`, the widest window the pruner supports: at 65 frames the model's own long-term tier starts (1-token slots merged in place), which is a time-dependent compression of its own and breaks the 4-token layout |
| budget `B'_t` | `round(k · V0_t / h_t)` with `h_t = t` — `k` × the running mean post-filter visual count over *every* history frame | `k · 4` |
| where | every method scores **and** deletes at decoder layer 0; all layers lose the token, so arms are memory-matched (`kv_len_mean` equal across methods at a given `k`) | same, by **row removal** for all four methods (deleted rows and an emptied frame's `<image_sep>` leave the prompt, positions compact) |

Methods (PyramidDrop is layer-staged by construction and is left out; its code is unchanged):

| method | rule at layer 0 | LongNav-R1 overrides on `+experiment=paper_prune_flat` | Uni-NaVid flags |
|---|---|---|---|
| `random` | as above | `vlm.kv_prune_importance=random` | `--prune-method random --prune-layer 0` |
| `divprune` | as above (input embeddings) | `vlm.kv_prune_importance=divprune` | `--prune-method divprune --prune-layer 0` |
| `fastv` | last-text-token attention **of decoder layer 0**, mean over heads → top-`B'` | `vlm.kv_prune_importance=attn vlm.kv_prune_score_layer=0` | `--prune-method fastv --prune-layer 0` |
| `dart` | pivots = L1 norm of the **layer-0 keys**, cosine on the **layer-0 output** hidden states (the layer-0 *input* is DivPrune's embedding DB) | `vlm.kv_prune_importance=dart` (`kv_prune_layer_start` stays 0) | `--prune-method dart --prune-layer 0` |

`k ∈ {32, 16, 8, 4}` → 2 models × 4 methods × 4 budgets = 32 arms + 2 identity arms, same arm
names as above under the flat root (LongNav-R1 logs to its own wandb project `longnav_paper_flat`,
because wandb resume keys on project + run name).

*LongNav-R1 mechanics.* Same prune substrate and one-step lag as above: the layer-0 decision row is
read in step `t`'s forward and the deletion acts from step `t+1`, on every layer. Text/action tokens
are never candidates and are no longer aged out, so the cache grows by the text of each turn
(~10 tokens). Since every layer is pruned, the sparse-filter DB shrinks with the cache for every
method (the caveat above about FastV/DART no longer applies). Keep masks are not logged
(`kv_prune_log_masks=false`: without a window every frame of the episode would be re-packed each
step), so there is no retention figure; `sup/mean_vtp_n_new` is still logged.

*Uni-NaVid mechanics.* `fastv` / `dart` run a **scoring pre-pass**: decoder layer 0 alone on the
assembled prompt (rows deleted by earlier inferences already absent), select, drop the fresh rows,
then the real prefill — so the budget holds inside the same inference and the four methods differ in
the selection rule only. Up to two frames enter per inference, so the pool is at most `B' + 8`.

```bash
VARIANT=flat nohup bash paper/token_prune/run_all.sh > /dev/null 2>&1 &     # 34 arms, one GPU
VARIANT=flat ARMS="r1_fastv_b8" bash paper/token_prune/run_r1.sh            # or per family / arm
# faster on one RTX 5090: two lock-coordinated workers (an R1 arm is ~6.5 GB / ~50% GPU, Uni-NaVid ~15 GB;
# only ONE worker may hold Uni-NaVid arms). Latency columns of concurrent arms are not comparable.
export VARIANT=flat EXTRA_OVERRIDES="resources.osm_gb=10"
nohup bash paper/token_prune/run_worker.sh A r1_divprune_b32 r1_divprune_b16 r1_dart_b32 ... > /dev/null 2>&1 &
nohup bash paper/token_prune/run_worker.sh B uninavid_random_b8 ... r1_random_b16 ...     > /dev/null 2>&1 &
python paper/token_prune/aggregate.py --variant flat --status              # progress
python paper/token_prune/aggregate.py --variant flat --strict              # results/ (deltas vs same-k random, vs identity)
/home/brabiei/miniconda3/envs/vln/bin/python paper/token_prune/plot.py --variant flat
```

Identity: `r1_identity` must reproduce the ctx ablation's `r1_full` episode for episode
(`aggregate.py --variant flat --check-identity`); `uninavid_identity` at 64 frames is a new unpruned
reference with no ctx-ablation counterpart. `figures/budget_curves.{pdf,png}`: SR and SPL vs frames'
worth, one plain line per method — no reference line, no recency curve, no error bars (the CIs are in
`results/summary.csv`). Tests: `tests/vtp_flat_smoke.py`, `baselines/tests/test_prune.py` (`FlatHookTests`).

### Flat-variant findings (sweep complete 2026-09-18; 100 episodes per arm; SR / SPL / mean steps)

| method | LongNav-R1 k=4 | k=8 | k=16 | k=32 | Uni-NaVid k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|---|
| random | .42/.163/361 | .59/.265/275 | .74/.368/196 | .82/.432/143 | .69/.346/112 | .71/.376/117 | .72/.394/117 | .69/.425/125 |
| fastv (layer 0) | .29/.125/399 | .64/.268/259 | .75/.377/183 | .82/.447/138 | .70/.391/107 | .72/.367/126 | .79/.400/120 | .76/.448/114 |
| divprune | .02/.003/495 | .13/.084/439 | .35/.210/335 | .62/.364/220 | .63/.292/158 | .70/.341/148 | .76/.411/119 | .74/.404/127 |
| dart (layer 0) | .02/.005/495 | .18/.077/436 | .41/.214/331 | .72/.382/191 | .69/.349/149 | .69/.338/166 | .74/.358/163 | .74/.369/122 |
| unpruned (identity) | .84/.468/111 (full context) | | | | .79/.429/110 (64 frames) | | | |

`r1_identity` reproduced the ctx ablation's `r1_full` exactly (0 mismatching episodes of 100).

**F1 — no selector beats uniform sampling.** On LongNav-R1 FastV's layer-0 attention is statistically
indistinguishable from random at k = 8/16/32 (SPL p = 0.88 / 0.63 / 0.32) and worse at k = 4 (SPL −0.038,
p = 0.035). On Uni-NaVid no method differs from random on SR or SPL at any budget (all p ≥ 0.05).

**F2 — the diversity criteria collapse without the window.** On LongNav-R1 DivPrune and DART are far below
random at every budget (SR −0.40 to −0.46 at k ≤ 8, −0.33/−0.39 at k = 16, still −0.10/−0.20 at k = 32; all
p ≤ 0.013). In Exp 4 the 32-frame window bounded how far back they could reach; with the whole episode as the
pool they spread the budget over old frames and starve the recent ones. Measured on a 10-episode rerun with
keep masks on (k = 8, decisions at step ≥ 64): DivPrune holds **85 %** of its budget in frames older than 31
steps, leaves 1.2 of its 8 frames' worth inside the last 31 frames and keeps 6 % of the four newest frames;
random holds 3 %, 7.7 of 8 and 76 %. DivPrune at k = 8 is thus effectively Exp 4's k ≈ 1 arm (SR .04 there). DART at layer 0 behaves like DivPrune, as expected from features one layer away from its embedding DB.

**F3 — budget, not selection, sets LongNav-R1's performance.** Random climbs SPL .163 → .265 → .368 → .432 and
at k = 32 is within noise of the unpruned full-context model on SR (−0.02, p = 0.72; SPL −0.036, p = 0.03);
FastV at k = 32 matches it on both (SPL −0.021, p = 0.24). 32 frames' worth of uniformly sampled history
tokens therefore buys roughly the full-context policy.

**F4 — Uni-NaVid is flat.** Every method at every budget sits within ≈0.1 SR of the 64-frame reference, and
from k = 16 no arm except DART on SPL differs from it significantly; DivPrune at k = 4 is the one clear loss
(SR −0.16, p = 0.001). With 4-token frames the budget is cheap and which tokens are kept hardly matters —
the same reading as Exp 4's Q3.

**Flat vs Exp 4 at the same k (paired, 100 episodes).** Uni-NaVid is unchanged (|ΔSR| ≤ 0.07, none significant),
so the layer-0 pre-pass and the 64-frame window introduced no shift. LongNav-R1 is lower: random −0.08/−0.10/−0.10
SR at k = 4/8/16 (p = .14/.008/.002), FastV −0.33/−0.13/−0.06, DivPrune −0.30/−0.54/−0.45, DART −0.34/−0.40/−0.34.
DivPrune/DART: the budget placement above. FastV: by design (layer-0 score, no unpruned shallow layers). Random:
NOT budget placement — its surviving tokens are as recent as under the window (3 % older than 31 frames) — so
the remaining difference is what else the full context keeps: every past turn's text/action tokens (~15 per turn,
up to ~7.5k slots of turns whose frames are gone; Exp 4 evicted text older than 31 turns), on top of the
reference shift (full context .84 vs 32-frame window .89 SR, n.s.). This is constant across the flat arms, so it
does not affect the ranking of methods, but flat and Exp 4 numbers at the same k are not interchangeable. A
text-only window would be needed to test it directly. Most added failures are "reached the goal, never issued
stop" (63 % of flat DivPrune k = 8 failures end within 1 m of the goal at the 500-step cap).

Audit (2026-09-19): all 34 saved configs/manifests match the protocol; interrupted-and-resumed arms hold only
complete episodes, 100 unique labels each; reruns of 10 episodes of `r1_random_b8`, `r1_divprune_b8` (mask log
on) and `r1_random_b16` (which had shared the GPU) reproduce the sweep's rows exactly (30/30).

Caveats: `kv_len_mean` differs across methods at a given k only through episode length
(text/action tokens, ~15 per turn, are never pruned); the visual budget is identical. The arms ran two at a
time on one GPU, so `vlm_latency_ms` is not comparable across arms or with Exp 4. LongNav-R1's DivPrune selector
is launch-bound (B' sequential GPU iterations per step: 33/64 ms per step at k = 4/8 alone, ≈0.8 s at k = 32
when the GPU is shared) — schedule those arms alone and last.

