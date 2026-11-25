# Visual-token pruning under a 32-frame window

Do the training-free visual-token pruning methods of the VLM literature (FastV, PyramidDrop,
DivPrune, DART) beat uniform-random pruning and the plain recency window when they are applied
to the *history* tokens of an ObjectNav VLA at a matched token budget? SR / SPL / steps vs the
history-token budget, for LongNav-R1 and Uni-NaVid, on the 100-episode HM3D-v2 val split, plus
which frame ages each method keeps.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`; plots use `vln`;
Uni-NaVid launchers use the isolated baseline runtime. To rebuild the retained tables and
figures without overwriting published outputs, follow the
[reproduction guide](../../paper_results/REPRODUCIBILITY.md). A Git clone alone is
insufficient: restore required local inputs from the checksummed backup first. Ordinary
launch logs and locks are ignored; protected traces, arrays, and images remain scientific inputs.

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
  `(model, episode, step, frame)` like `paper_results_archive/frame_influence/results/influence.csv.gz`,
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
bash paper/token_prune/run_r1.sh          # ~20-40 min per arm
# A subset: ARMS="r1_identity r1_random_b8" bash paper/token_prune/run_r1.sh
bash paper/token_prune/run_uninavid.sh    # ~11 min per arm

# progress / interim numbers and figures at any time
python paper/token_prune/aggregate.py --status
python paper/token_prune/aggregate.py --allow-partial
conda run -n vln python paper/token_prune/plot.py       # --k 8 for the retention figure

# final
python paper/token_prune/aggregate.py --strict
conda run -n vln python paper/token_prune/plot.py
```

Do not run the two model families concurrently on one GPU (Uni-NaVid-7B alone takes ~15 GB).
Both run scripts skip complete arms and resume partial ones (LongNav-R1 through wandb's
`task.run_name`, Uni-NaVid through `--resume`).

## Outputs — all under `paper_results_archive/token_prune/`

The windowed study's data was deleted on 2026-11-21 (the paper uses the flat variant, whose data
stays in `paper_results/token_prune_flat/`); the numbers quoted in this README are unchanged. The
run files it needs survive in the local backup: restore them as described in
[`paper_results_archive/README.md`](../../paper_results_archive/README.md), then
`aggregate.py --strict` followed by `plot.py` rebuilds `results/` and `figures/` next to the runs
(git-ignored there, about 2 minutes). New launches of the window variant also write here.

- `runs/<arm>/` — raw run output. LongNav-R1: `config.yaml`, `rollout/results_*`
  (per-episode JSONL, incl. `sup/mean_vtp_n_new`); the per-episode `rollout/<scene>/…` dirs with
  `sequence.json` (keep masks) stay on disk but are gitignored. Uni-NaVid: `manifest.json`,
  `summary.json`, `episodes/*.json`, `prune/*.jsonl` + `prune/meta.json`; `traces/`
  gitignored. New launches generate ignored `launch.log` / `driver.log` files; historical logs were archived.
- `results/per_episode.{csv,parquet}` — one row per (arm, episode), the ctx-ablation recency and
  reference rows merged in (`method ∈ {recency, reference}`), `partial` flag.
- `results/summary.csv` — per arm: n, SR/SPL/steps with CIs, tokens per frame, tokens kept, paired
  deltas and p vs the same-`k` recency arm (`*_vs_recency`) and vs the reference (`*_vs_ref`).
- `results/table.tex` — booktabs, stars vs the same-`k` recency arm, † = partial arm.
- `results/n_tokens.csv` — `n`, mean tokens kept and kept fraction of the 31-frame window per arm.
- `results/masks.parquet` — one row per (arm, episode, step, history frame): tokens, kept, packed
  keep bits. **Not in git** (149 MB; like every rebuilt output under `paper_results_archive/` it is
  ignored by the `.gitignore` there). Rebuilding it requires both Uni-NaVid `runs/*/prune/` records
  and LongNav-R1 per-episode `sequence.json` traces, including ignored inputs. Restore those
  inputs first, then use the isolated reproduction workflow above to aggregate before plotting
  `retention_by_age`. `results/retention.csv` — kept fraction vs frame age per arm with CIs.
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

| | LongNav-R1 | Uni-NaVid | LightNav-0 (2026-10-19) |
|---|---|---|---|
| context | **full episode** (`context_frames=null`; nothing is ever evicted by age) | `--context-frames 64`, the widest window the pruner supports: at 65 frames the model's own long-term tier starts (1-token slots merged in place), which is a time-dependent compression of its own and breaks the 4-token layout | **full episode**, the model's own SlowFast sampling of it (current pair at 112 tokens, 16 fast + 10 burst tubelets at 28, 13 span + 1 anchor tubelet at 8: 840 history tokens at the plateau); the current 2-frame tubelet is exempt, the anchor pair is a candidate |
| budget `B'_t` | `round(k · V0_t / h_t)` with `h_t = t` — `k` × the running mean post-filter visual count over *every* history frame | `k · 4` | `k · 14`: a fast-tier tubelet (2 frames, pool 2) holds 28 tokens, what the model spends on a recent history frame; `k = 32` is the whole fast tier (448 of 840) |
| where | every method scores **and** deletes at decoder layer 0; all layers lose the token, so arms are memory-matched (`kv_len_mean` equal across methods at a given `k`) | same, by **row removal** for all four methods (deleted rows and an emptied frame's `<image_sep>` leave the prompt, positions compact) | same, by **column masking** on every layer of the eager masked-attention kernel the information-horizon arms use (a hidden token influences nothing; survivors keep their mRoPE positions, as LongNav-R1's survivors keep theirs) |

Methods (PyramidDrop is layer-staged by construction and is left out; its code is unchanged):

| method | rule at layer 0 | LongNav-R1 overrides on `+experiment=paper_prune_flat` | Uni-NaVid flags | LightNav-0 knobs (`baselines/run_lightnav.sh`) |
|---|---|---|---|---|
| `random` | as above | `vlm.kv_prune_importance=random` | `--prune-method random --prune-layer 0` | `PRUNE_METHOD=random PRUNE_K=k` |
| `divprune` | as above (input embeddings) | `vlm.kv_prune_importance=divprune` | `--prune-method divprune --prune-layer 0` | `PRUNE_METHOD=divprune PRUNE_K=k` |
| `fastv` | last-text-token attention **of decoder layer 0**, mean over heads → top-`B'` | `vlm.kv_prune_importance=attn vlm.kv_prune_score_layer=0` | `--prune-method fastv --prune-layer 0` | `PRUNE_METHOD=fastv PRUNE_K=k` |
| `dart` | pivots = L1 norm of the **layer-0 keys**, cosine on the **layer-0 output** hidden states (the layer-0 *input* is DivPrune's embedding DB) | `vlm.kv_prune_importance=dart` (`kv_prune_layer_start` stays 0) | `--prune-method dart --prune-layer 0` | `PRUNE_METHOD=dart PRUNE_K=k` |
| `kl_keep_one` (2026-10-09) | **whole frames**: keep exactly the `k` pool frames with the largest keep-one-in KL, `KL(P(action \| only frame f's visual tokens) ‖ P(action \| no visual tokens))`; reference = every visual token hidden, the current frame's included (Exp 3's `keep_one` measure); ties → the newer frame | `vlm.kv_prune_importance=kl_keep_one` | `--prune-method kl_keep_one --prune-layer 0` | — |
| `recency` (2026-10-09) | **whole frames**: keep the `k` newest pool frames (the control for `kl_keep_one`: what age alone buys in this exact regime) | `vlm.kv_prune_importance=recency` | `--prune-method recency --prune-layer 0` | — |
| `entropy_keep_one` (2026-10-12) | **whole frames**: the same only-f probes as `kl_keep_one` (only frame f's visual tokens visible, the current frame's hidden too) scored by the entropy of `P(action \| only frame f)` instead of its KL to the no-visual reference; keep the `k` pool frames with the **lowest** entropy (the frames that alone make the decision most certain); no reference replay; ties → the newer frame | `vlm.kv_prune_importance=entropy_keep_one` | `--prune-method entropy_keep_one --prune-layer 0` | — |

`k ∈ {32, 16, 8, 4}` → 2 models × 4 methods × 4 budgets = 32 arms + 2 identity arms (plus the
3 whole-frame methods below: 24 more arms, 58 in all; plus LightNav-0's 16 token-budget arms and
its identity, 75 in all), same arm names as above under the flat root (LongNav-R1 logs to its own wandb project `longnav_paper_flat`,
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

*LightNav-0 mechanics (added 2026-10-19; `baselines/lightnav_hm3d/eval_prune.py`, upstream clone
untouched).* LightNav rebuilds its prompt every step (anchor, span, burst, fast tubelets oldest
first, then the current pair, one `<t seconds>` per 2-frame tubelet; 952 visual tokens at the
plateau) and prefills it inside vLLM, so the wrapper reuses the information-horizon kernel path:
eager vLLM, `FlashAttentionImpl.forward` replaced on every layer by the exact masked SDPA of
`eval_blind.py`, and the **pruned** `<|video_pad|>` columns hidden from every query on every
layer (timestamps, vision markers and the task text stay visible). Deletions are **permanent**
through a per-episode ledger keyed by tubelet identity `(frame0, frame1, pool, cell)`: each
step the candidates are the history tokens not in the ledger, the current tubelet is exempt,
and when they exceed `B' = 14k` the rule keeps `B'` and the rest join the ledger. The fast and
mid tiers pair consecutive frames, so the pairing flips parity every step: a tubelet recurs
every other step and the two parity classes are pruned independently (each holds `≤ B'`
visible tokens; the model sees one class per step), fast→mid keeps the pair and pool (same
identity), mid→long re-pools (2→4) and the span tier re-places its pairs as the episode grows,
so those 8-token tubelets (≤ 112 tokens, ~12% of the history) re-enter as fresh candidates.
Steps 0–1 have no history and are unpruned. `fastv` and `dart` need layer-0 quantities of the
current prompt and run a **scoring pre-pass** (a `generate` with `max_tokens=1`, ledger columns
hidden): at layer 0 of the prefill the post-softmax attention row of the last prompt token (the
`\n` after `assistant`, whose logits emit the first action token `<apos_*>`; mean over the 32
query heads, all visible causal keys) and the rotated keys (8 KV heads concatenated, L1 norm for
DART's 8 pivots), and at the input of layer 1 the residual stream (deepstack-0 injection
included — what LongNav-R1's layer-1 pre-hook captures); the real generate then hides the dropped
columns from layer 0 on, Uni-NaVid's pre-pass semantics (one extra prefill per pruning step,
~10% wall). `random` draws `randperm` seeded per (episode, step) with the shared formula;
`divprune` reads the LLM input embeddings of the visual tokens (the main channels of
`video_embeds`, pre-deepstack). The selection rules are copies of LongNav-R1's. Per inference
the wrapper logs `prune/<label>.jsonl` in Uni-NaVid's schema (one entry per history tubelet,
its newer frame as the frame id, `n_tokens` 28 / 8), so `masks.parquet` / `n_tokens.csv` hold
LightNav rows too. `lightnav_identity` is the information-horizon reference `lightnav_ref`
(nothing hidden, same kernel path), imported rather than rerun; the ctx-ablation windows
`lightnav_k<k>` (forced stops re-scored, see `paper/ctx_ablation`) are the `window` rows.

*Whole-frame arms (`kl_keep_one`, `recency`; added 2026-10-09; `entropy_keep_one` 2026-10-12).* These
test the **measure itself**: is the keep-one-in KL — the quantity the paper uses to define a
frame's information — a good selection criterion? (`entropy_keep_one` asks the same of the
decisiveness of the only-f decision: the KL decomposes as `KL(P_f ‖ Q) = CE(P_f, Q) − H(P_f)`, so
the entropy arm keeps the `−H` term and drops the cross-entropy to the no-visual prior `Q`.) The budget is `k` whole frames (the pool frames that keep their tokens; the
dropped frames lose every token). On LongNav-R1 the pool is every turn still holding a visual slot
(survivors + the frame just decided), so at a pruning step it is always `k+1` frames and one leaves;
`kl_keep_one` scores it on the pre-prune cache with the retained keep-one replay
(`PruningMixin._whole_frame_select` → `DiagnosticsMixin._keep_one_frame_kls`, the code of
`vlm.kv_prune_influence_measures=[keep_one]`): one decision replay with every visual column masked
(the no-visual reference) plus one per pool frame with only that frame's columns open — `k+2`
single-token replays per step, evict-style (the other cached tokens' K/V already absorbed the frames
that left, like every flat arm). On Uni-NaVid the pool is the history groups (≤ `k+2` frames), scored
in the layer-0 pre-pass by full-prompt forwards with `layer_blind.ColumnBlinder` closing every visual
column (history groups, current group, the 64 fine tokens) except the probed frame's group — the
paper's `keep_one_mask` probe — batched 8 per forward (`--probe-batch` fixed at 8), ≤ `ceil((k+3)/8)`
forwards per inference, leak-free by construction (no cache across inferences). `entropy_keep_one` runs
the same per-frame probes without the reference (R1: `pool` replays per step via
`DiagnosticsMixin._keep_one_frame_logps`, the loop `_keep_one_frame_kls` shares; Uni-NaVid:
`≤ ceil((k+2)/8)` forwards) and ranks by `−H`; its logged score is the entropy itself (lower =
kept). `recency` is the same code path with score = frame id and no replay. Per step the pool,
scores and survivors are logged (R1: `sup/kl_select` in `sequence.json`; Uni-NaVid: `score` / `n_probes` in
`prune/<label>.jsonl`), and `aggregate.py` summarises them in `results/kl_select.csv`: mean age and
score of the kept / dropped frames, how often the dropped set held the oldest pool frame (= the
recency rule's choice) and probes per step. The flat table/figure also carry the ctx-ablation
recency windows (`r1_evict_k{k}` / `uninavid_k{k}`, method `window`, drawn at `k-1` history frames)
— an approximate control only: those arms cut the text/action history with the frames
(`context_cut=frame`, evict mode), whereas every flat arm keeps all text. Token counts: whole-frame
arms hold `k` frames' own tokens rather than `k ×` the running mean, so `kv_len_mean` matches the
token-budget arms on average, not step by step.

```bash
VARIANT=flat nohup bash paper/token_prune/run_all.sh > /dev/null 2>&1 &     # 34 arms, one GPU
VARIANT=flat ARMS="r1_fastv_b8" bash paper/token_prune/run_r1.sh            # or per family / arm
# faster on one RTX 5090: two lock-coordinated workers (an R1 arm is ~6.5 GB / ~50% GPU, Uni-NaVid ~15 GB;
# only ONE worker may hold Uni-NaVid arms). Latency columns of concurrent arms are not comparable.
export VARIANT=flat EXTRA_OVERRIDES="resources.osm_gb=10"
nohup bash paper/token_prune/run_worker.sh A r1_divprune_b32 r1_divprune_b16 r1_dart_b32 > /dev/null 2>&1 &
nohup bash paper/token_prune/run_worker.sh B uninavid_random_b8 r1_random_b16     > /dev/null 2>&1 &
# LightNav-0 (isolated runtime, GPU to itself, ~1-1.2 h per navigating arm, ~4 h per arm at the 500-step cap)
VARIANT=flat nohup bash paper/token_prune/run_lightnav.sh > paper_results/token_prune_flat/runs/lightnav_driver.log 2>&1 &
python paper/token_prune/aggregate.py --variant flat --status              # progress
python paper/token_prune/aggregate.py --variant flat --strict              # results/ (deltas vs same-k random, vs identity)
conda run -n vln python paper/token_prune/plot.py --variant flat
```

Identity: `r1_identity` must reproduce the ctx ablation's `r1_full` episode for episode
(`aggregate.py --variant flat --check-identity`); `uninavid_identity` at 64 frames is a new unpruned
reference with no ctx-ablation counterpart; `lightnav_identity` (= `lightnav_ref`) is checked
against the ctx ablation's `lightnav_full` as a noise-level pair only (different kernel path, vLLM
not bit-reproducible). `figures/budget_curves.{pdf,png}` is the paper figure (since 2026-11-21): SR and SPL
vs frames' worth on the k grid 4 … 32, LongNav-R1 and LightNav-0 columns, random / FastV / DivPrune / DART as
plain lines and the unpruned identity arm as the dotted reference, no error bars (the CIs are in
`results/summary.csv`), drawn at single-column size (`paper/figstyle.py`). Uni-NaVid, the whole-frame arms,
the ctx-ablation windows and the layout-check arms below the k grid are not drawn; their numbers stay in
`results/summary.csv` and `table.tex`. `plot.py --variant flat --methods ...` draws other method sets.
`plot.py --variant flat --recency --name <name>` draws a reference view instead (not kept in `figures/`): x axis
from 0, the context-window ablation's recency curve as black squares — the series its own figure draws, read from
`paper_results/ctx_ablation/results/summary.csv` (`plot.py` `RECENCY_SOURCE`): LongNav-R1 KV eviction
`r1_evict_k{K}` at x = K − 1 history frames, LightNav-0 old blocks kept `lightnav_keep_k{K}` at x = K − 2
(its history is 14 (K − 2) tokens; K = 2 is the no-history run) — and the LightNav-0 Random line continued
through `random_b2` / `b1` / `b0` (k = 2 / 1 / 0; `FLAT_EXTRA`). The keep-one comparison figure
`budget_curves_keep_one.{pdf,png}` was removed on 2026-11-21 (redraw: `plot.py --variant flat --methods
random,fastv,kl_keep_one,entropy_keep_one,recency --name budget_curves_keep_one`).
Tests: `tests/vtp_flat_smoke.py` (incl. the whole-frame checks 5–7: stubbed-score selection on CPU,
and on the GPU the selectors' scores equal the `keep_one` diagnostic's KL row / the entropy of its
only-f log-prob rows on the same cache),
`baselines/tests/test_prune.py` (`FlatHookTests`, `WholeFrameTests`),
`baselines/tests/test_lightnav_prune.py` (layout, ledger, the four rules against naive references,
the scoring hook; client runtime) and the prune-log case of `baselines/tests/test_lightnav.py`.

### Flat-variant findings (sweep complete 2026-09-18; LightNav-0 arms 2026-10-23, see the section at the end; 100 episodes per arm; SR / SPL / mean steps)

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

### Keep-one KL as the criterion (whole-frame arms, sweep complete 2026-10-10; SR / SPL / mean steps)

| arm | LongNav-R1 k=4 | k=8 | k=16 | k=32 | Uni-NaVid k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|---|
| kl_keep_one | .49/.195/329 | **.78/.340/216** | .79/.390/173 | .82/.456/142 | .64/.314/165 | .67/.338/148 | .73/.339/170 | .80/.426/117 |
| recency (whole frames) | .51/.200/324 | .64/.314/239 | .73/.388/184 | .87/.462/128 | .55/.321/99 | .71/.394/128 | .74/.424/116 | .70/.391/122 |
| random (above) | .42/.163/361 | .59/.265/275 | .74/.368/196 | .82/.432/143 | .69/.346/112 | .71/.376/117 | .72/.394/117 | .69/.425/125 |
| window (Exp 1, k−1 history frames) | .70/.282/261 | .73/.361/211 | .85/.420/146 | .89/.474/112 | .57/.289/110 | .68/.378/109 | .71/.383/123 | .76/.430/99 |

Paired deltas (sign-flip, 100 episodes) are in `results/summary.csv` (`*_vs_random`, `*_vs_window`,
`*_vs_ref`); the selection statistics in `results/kl_select.csv`.

**F5 — the keep-one KL beats random at one budget on LongNav-R1, ties elsewhere.** At k = 8 the KL arm is far
above random (SR +0.19, p < 0.001; SPL +0.075, p = 0.011; −59 steps) and above FastV (+0.14 SR, p = 0.005);
at k = 4 it beats FastV (+0.20 SR, p < 0.001) and random on steps only (−33, p = 0.035; SR +0.07, n.s.); at
k = 16 / 32 it ties random (SR +0.05 / 0.00, SPL +0.02 both n.s.). Against the exact whole-frame recency control
it wins only at k = 8 (SR +0.14, p = 0.002; SPL +0.026, p = 0.040) and ties at k = 4 / 16 / 32 (SR −0.02 / +0.06
/ −0.05, all n.s.; at k = 32 recency is 13 steps shorter, p = 0.02). The selection log explains the proximity:
the KL arm drops the oldest pool frame on 33 % (k = 32) to 60 % (k = 4) of its pruning steps and keeps frames
whose mean age is within a frame of recency's (16.3 vs 15.5 at k = 32, 1.9 vs 1.5 at k = 4) — on LongNav-R1 the
keep-one KL is a recency rule with content-driven exceptions (its values decay .06 → 4e-4 over ages 0–12, as in
Exp 3), and the exceptions pay off at k = 8 only.

**F6 — on Uni-NaVid the keep-one KL ranks by content, and that costs efficiency.** The KL arm keeps old frames
(mean age kept 21–24 at every k vs recency's 2.5–16; the oldest pool frame is dropped on only 4–8 % of
pruning steps) and it is at best even with random on SR (+0.11 at k = 32, p = 0.02, the one gain; −0.05 … +0.01
elsewhere) while taking +32 … +54 more steps at k ≤ 16 (all p ≤ 0.045) with SPL −0.03 … −0.06 (vs recency at
k = 16: SPL −0.085, p = 0.007). Uni-NaVid's age-flat keep-one values (.13–.26 at every age, cf. Exp 3) mean the
rule hoards whichever frames look informative on their own, and the policy wanders longer. Whole-frame recency
is itself no better than random there (SR −0.14 at k = 4, p = 0.021; ties above).

**F7 — the full text history is what separates the flat random arm from Exp 4 (the open question of F3).**
The whole-frame `recency` arm holds exactly the frames of the ctx-ablation window with one more
(k kept + the current one vs K = k frames) and differs only in keeping every past turn's text/action tokens
(header cut, nothing aged out); it is *below* that window on LongNav-R1 at every k ≤ 16: SR .51 vs .70 (k = 4),
.64 vs .73 (k = 8), .73 vs .85 (k = 16), .87 vs .89 (k = 32), SPL −0.09 / −0.05 / −0.03 / −0.01. Unbounded
action-text history hurts this model by about 0.1 SR at small frame budgets; the flat arms share that handicap,
so their ranking stands but their absolute numbers sit below Exp 4's. On Uni-NaVid the two coincide (±0.06 SR).

Reading: the keep-one KL is a faithful measure of what a frame contributes on its own, but as an online
selection criterion it buys little — on LongNav-R1 it mostly re-derives age, on Uni-NaVid it keeps old
frames the policy then over-weights. It costs k+2 decision replays per step (R1) or ⌈(k+3)/8⌉ full forwards
per inference (Uni-NaVid), against none for random/recency. Together with Exp 8 (the heuristics do not track
the KL) this closes the selector question from both ends: the heuristics are not the measure, and the measure
is not a good selector.

### Keep-one entropy as the criterion (whole-frame arms, sweep complete 2026-10-13; SR / SPL / mean steps)

Motivation: `kl_keep_one`'s one gain (R1, k = 8) could come from either term of
`KL(P_f ‖ Q) = CE(P_f, Q) − H(P_f)`: a frame that *shifts* the decision away from the no-visual
prior `Q`, or one that merely makes it *certain*. `entropy_keep_one` keeps the `k` frames with the
lowest `H(P_f)` — a confidence criterion with no reference replay (one replay cheaper per step).

*Offline check before the sweep* (entropy of the only-f log-probs already stored by the Exp 3
keep-one rerun, `paper_results_archive/frame_influence_keepone/results/influence.csv.gz`; full-context
pool, 10 episodes per model; R1 1089 steps / 150k frame rows, Uni-NaVid 640 / 123k):

| | LongNav-R1 | Uni-NaVid |
|---|---|---|
| within-step Spearman ρ(H, KL) | −0.02 (median −0.08) | −0.67 (median −0.72) |
| k lowest-H frames == k highest-KL frames, k = 4/8/16/32 | 21 / 12 / 11 / 11 % | 13 / 10 / 6 / 1 % |
| Jaccard of the two kept sets | .42 / .40 / .43 / .47 | .45 / .55 / .67 / .70 |
| mean H by age, 1 → ≥ 64 | .37 → .41 → **.27** | 1.14 → 1.25 |
| mean KL by age, 1 → ≥ 64 | .020 → .0002 | .18 → .03 |
| argmax of the only-f decision, lowest-H quartile | 86 % "left" (= the no-visual prior) | 98 % "forward" |

On Uni-NaVid the no-visual prior is near-uniform (H ≥ 0.5, mean 1.1–1.25 of ln 4 = 1.386), so
within a step `KL ≈ const − H` and the two rankings align, though the kept sets still differ
(entropy tilts newer). On LongNav-R1 the prior is itself a confident "left", so a low entropy
mostly marks a frame that *agrees* with the prior rather than informs it; the oldest frames
(KL ≈ 2e-4) are the most "decisive". Prediction: on R1 the entropy arm hoards prior-confirming,
uninformative frames and lands below `kl_keep_one`; on Uni-NaVid it tracks `kl_keep_one` with a
recency tilt. (The online pool is `k+1` frames, not the episode, so these overlaps are indicative
only.)

| arm | LongNav-R1 k=4 | k=8 | k=16 | k=32 | Uni-NaVid k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|---|
| entropy_keep_one | .40/.152/364 | .57/.229/298 | .64/.289/253 | .71/.363/199 | .60/.289/189 | .71/.337/175 | .64/.325/168 | .75/.394/130 |
| kl_keep_one (above) | .49/.195/329 | .78/.340/216 | .79/.390/173 | .82/.456/142 | .64/.314/165 | .67/.338/148 | .73/.339/170 | .80/.426/117 |
| recency (whole frames) | .51/.200/324 | .64/.314/239 | .73/.388/184 | .87/.462/128 | .55/.321/99 | .71/.394/128 | .74/.424/116 | .70/.391/122 |
| random | .42/.163/361 | .59/.265/275 | .74/.368/196 | .82/.432/143 | .69/.346/112 | .71/.376/117 | .72/.394/117 | .69/.425/125 |

Paired deltas vs random / window / identity are in `results/summary.csv`; the entropy-vs-KL and
entropy-vs-recency deltas below are paired sign-flip tests over the same 100 episodes; selection
statistics in `results/kl_select.csv`.

**F8 — on LongNav-R1 the entropy criterion is the worst whole-frame rule, and the KL arm's k = 8 gain
is entirely the cross-entropy term.** The entropy arm ties random at k ≤ 8 (SR −0.02, n.s.) and is
below it at k = 16 / 32 (SR −0.10, p = 0.03 / −0.11, p = 0.008; SPL −0.08 / −0.07, p = 0.001; +57
steps, p < 0.001). Against `kl_keep_one` on the same probes it loses at every k: SR −0.09 (n.s.) /
−0.21 (p = 0.001) / −0.15 (p = 0.002) / −0.11 (p = 0.018), SPL −0.04 … −0.11, +36 … +82 steps; against
whole-frame recency SR −0.07 … −0.16 and SPL −0.05 … −0.10 (p ≤ 0.004 on SPL at k ≥ 8). At k = 8,
where the KL arm was +0.19 SR over random, the entropy arm is −0.02: of `KL = CE(P_f, Q) − H(P_f)`,
the `−H` term carries none of the gain — what paid off was selecting frames that *move* the decision
away from the no-visual prior, not frames that make it certain.

**F9 — the mechanism is the one the offline check predicted: decisiveness marks agreement with the
prior, so the rule hoards old frames and discards the fresh view.** Mean age of the kept frames is
26 / 27 / 49 / 60 at k = 4 / 8 / 16 / 32 (KL arm 1.9 … 16.3, recency ≤ 16.5); the oldest pool frame is
dropped on only 24 / 13 / 7 / 5 % of pruning steps (KL 60 … 33 %) and the just-decided frame survives
on 50 … 57 % of them (KL ≈ 100 %). The kept frames' mean entropy is .43–.47 vs .55 for the dropped —
a small, stable margin that the prior-confirming old frames win (their only-f decision is the
confident "left" of the no-visual prior). The policy then wanders: at k = 8, 22 of its 43 failures
end within 1 m of the goal at the 500-step cap (random 17 of 41, KL 8 of 22).

**F10 — on Uni-NaVid it buys nothing and costs steps.** SR is within noise of random at every k
(−0.09 … +0.06, all p ≥ 0.11) and of the KL arm (−0.09 … +0.04, n.s.), but at k ≤ 16 the arm takes
+51 … +77 more steps than random (p < 0.001) with SPL −0.04 … −0.07 (vs recency at k = 16: SPL −0.10,
p = 0.001); only at k = 32 is it even on steps. The selection log matches the KL arm's: kept frames
21–25 steps old, the oldest dropped on 6–9 % of steps, the newest kept on 22 … 76 % — the same hoarding
as F6, minus the KL arm's one SR gain at k = 32 (+0.11 → +0.06, n.s.).

Reading: confidence is not information. A frame that alone makes the policy certain is, on
LongNav-R1, mostly a frame that agrees with what the policy would do blind; the KL's reference term
is what distinguishes informative frames from merely decisive ones, and the whole-frame selector
question stays closed — now from a third direction. Cost: `pool` replays per step on LongNav-R1
(one fewer than the KL arm), `⌈(k+2)/8⌉` forwards per inference on Uni-NaVid; the R1 k = 32 arm
nevertheless took 9.1 h against the KL arm's 6.4 h because its episodes are longer.

Audit (2026-09-19): all 34 saved configs/manifests match the protocol; interrupted-and-resumed arms hold only
complete episodes, 100 unique labels each; reruns of 10 episodes of `r1_random_b8`, `r1_divprune_b8` (mask log
on) and `r1_random_b16` (which had shared the GPU) reproduce the sweep's rows exactly (30/30).

Caveats: `kv_len_mean` differs across methods at a given k only through episode length
(text/action tokens, ~15 per turn, are never pruned); the visual budget is identical. The arms ran two at a
time on one GPU, so `vlm_latency_ms` is not comparable across arms or with Exp 4. LongNav-R1's DivPrune selector
is launch-bound (B' sequential GPU iterations per step: 33/64 ms per step at k = 4/8 alone, ≈0.8 s at k = 32
when the GPU is shared) — schedule those arms alone and last.


### LightNav-0 arms (sweep complete 2026-10-23; 100 episodes per arm; SR / SPL / mean steps; forced stops at the 500-step cap scored as failures)

| method | k=4 (56 tokens) | k=8 (112) | k=16 (224) | k=32 (448 = fast tier) |
|---|---|---|---|---|
| random | .81/.450/160 | .82/.506/140 | .82/.466/153 | .82/.460/139 |
| fastv (layer 0, pre-pass) | .77/.440/164 | .80/.446/170 | .81/.446/162 | .83/.451/147 |
| divprune | .77/.447/185 | .80/.464/149 | .80/.473/133 | .78/.454/140 |
| dart (layer 0, pre-pass) | .80/.443/171 | .80/.450/163 | .78/.433/144 | .85/.469/152 |
| window (ctx ablation, k frames) | .34/.188/376 | .68/.365/236 | .80/.444/189 | .82/.467/177 |
| unpruned (`lightnav_identity` = `lightnav_ref`) | .84/.458/139 | | | |

`lightnav_identity` vs the compiled `lightnav_full`: dSR +0.04 (p = 0.39), dSPL +0.008 (p = 0.67), 95 of 100
episodes differ in some metric — the eager masked-SDPA kernel is not bit-identical to the compiled path (known
from the information-horizon arm), hence the non-exact identity entry. Budget audit over all 16 arms (~230k
inference steps): on every pruning step the visible history equals the budget exactly, the ledger is monotone,
7 forwards per step plus exactly one pre-pass forward on FastV/DART pruning steps.

**F11 — on LightNav-0 no selector beats random, and random is flat down to 7 % of the history.** Random holds
SR .81–.82 at every budget (vs the unpruned .84: −0.02 … −0.03, all p ≥ 0.54) and its only cost at k = 4 is
+21 steps (p = 0.08). No method differs from random on SR at any k (|ΔSR| ≤ 0.04, all p ≥ 0.34). On SPL the
selectors are below random at k = 8 (FastV −0.060, p = 0.017; DART −0.056, p = 0.059; DivPrune −0.042,
p = 0.079) and tied elsewhere; the k = 8 random SPL (.506) is the one arm above the reference (+0.048,
p = 0.053). Unlike LongNav-R1 (F2), DivPrune and DART do not collapse without a window: the worst arm of the
grid is DivPrune at k = 32 (SR .78, −0.06 vs reference, p = 0.07).

**F12 — token pruning at matched budget beats the frame window by a wide margin.** At k = 8 random is +0.14 SR
and +0.14 SPL above the 8-frame window (both p ≤ 0.004, −96 steps); at k = 4 it is +0.47 SR / +0.26 SPL
(p < 0.001, −216 steps; the window has 65 forced stops, random 9). All three selectors show the same gap
(k = 8: +0.12 SR, p ≤ 0.027; k = 4: +0.43 … +0.46, p < 0.001). At k = 16/32 the window ties on SR/SPL but takes
+38 … +49 more steps than the reference (p ≤ 0.008). Same reading as F3 on LongNav-R1: what the policy needs from its history is
spread over the whole episode, and a few tokens from everywhere beat every token from the last k frames.

**What survives under permanent deletion (keep masks, step ≥ 120, kept fraction by tier).** With a ledger, random
pruning is a geometric decay: a token survives each of its candidate steps with probability
1 − (drop / candidates), so at k = 8 (34 of 146 dropped per step) a fast-tier tubelet retains ≈ 1 % of its tokens
by the time it ages into the mid tier. Measured: random keeps 20 % of the fast tier, 0 % of the mid tier and 19 %
of the span tier at k = 8 (10 / 0 / 10 % at k = 4; 66 / 32 / 57 % at k = 32); FastV 21 / 6 / 0 %, DART 19 / 8 / 3 %,
DivPrune 16 / 10 / 14 % at k = 8. The span tier survives under random only because its pairs are re-placed as the
episode grows and re-enter as fresh identities (~12 % of history tokens). So the random arm's content is the
newest fast-tier pairs plus a thin, periodically refreshed sketch of the whole episode, and that is enough for
the reference SR; FastV and DART concentrate on the fast tier and forgo the span tier entirely, with no gain.

Caveats: text (timestamps, `<|vision_start|>`/`<|vision_end|>`, task) is never pruned, so even at k = 4 the model
sees the full temporal skeleton; the window arms remove whole tubelets including their timestamps — part of
F12's gap may be that skeleton (same protocol difference as F7 on LongNav-R1; a text-only window would test it).
The two parity classes of the fast tier carry independent ledgers (README mechanics paragraph); the FastV/DART
pre-pass doubles the prefill on pruning steps (≈ +0.19 s/step at the 1.2k-token plateau), so `seconds` in the
prune logs is not comparable across methods. Wall: 1–1.9 h per arm, 16 arms over 2026-10-19 … 10-23 with two
pauses for the GPU.

### Layout check (audit 2026-10-23; random at k = 2 / 1 / 0, the window-content control `recency_b{2,6,14,30}` of 2026-10-25/27 and its K = 1 point `recency_b0_cur1`, 100 episodes; the random arms extend the figure's Random line, the rest are summary.csv only)

F12 looked too good next to the context-window ablation (LightNav-0 window K = 8 → SR .68, K = 4 → .34, while
random pruning holds .81–.82 down to k = 4), so the pruning path was audited for a logical bug before the numbers
were trusted. **No bug was found.** Verified: the masked kernel gathers K/V through the block table and removes
the hidden columns on all 36 layers, for the prefill and the six decode forwards alike (the hidden set is held
for the whole generate call); vLLM writes K/V before the patched attention runs; eager mode, no CUDA graphs, no
chunked prefill, one request per forward are asserted; vLLM's prefix cache is on but LightNav salts every video
hash with the clock, and vLLM keys every block overlapping a video placeholder on that hash, so neither
consecutive steps nor the FastV/DART pre-pass and its real generate can share a visual block (only the text
blocks before the first `<|video_pad|>` can hit, and they never attend to visual tokens; the pre-pass also asserts
that no candidate lies inside the cached offset). The logged keep masks are the tensor handed to the kernel;
across the 15,979 steps of `random_b4` the visible history never exceeds 56 tokens and equals 56 at every step
≥ 100, the current pair is always the remaining 112, and log lines equal inference calls. Both the prune arms
and the window rows hold the 100 manifest episodes and are re-scored for forced stops by the same loader.

What differs is the protocol, in three ways. (i) **The window rewrites the prompt skeleton; pruning keeps it.**
`eval_ctx.py` lays the newest K frames out as a *fresh K-step episode*: older segments vanish with their
`<t seconds>` timestamps and vision markers, the anchor pair (frames 0/1) is gone, and the oldest visible
timestamp reads (K − 1)/4 s. Pruning hides pad columns only, so all ~41 timestamps stay and the model still
reads e.g. "37 seconds" of elapsed time at step 150 — the prompt itself says "You may be at the beginning,
middle, or end of the task". (ii) **Budgets are not matched at the same k.** Window K counts the two current
frames and holds K − 2 history frames at pool 2 = 14 (K − 2) tokens (anchor/mid/long tiers are empty for
K ≤ 34): K = 4 is 28 history tokens (= prune k = 2), K = 8 is 84 (= k = 6); the aggregator plots the window at
K − 1. (iii) **Kernel path:** the window arms ran on upstream's compiled FlashAttention (`lightnav_full` .80, 13
forced stops), the prune arms on the eager masked SDPA (`lightnav_identity` .84, 5 forced stops; paired ΔSR
+.04, p = .39) — noise-level, but it favours the prune side by a few points at k ≥ 16.

The existing data already shows which of these drives F12. Splitting every arm into episodes the agent ended
itself and episodes forced to stop at the 500-step cap:

| arm | forced stops | SR among self-stopped | oracle SR | steps (self-stopped) |
|---|---|---|---|---|
| window `lightnav_full` / k32 / k16 / k8 / k4 | 13 / 16 / 19 / 27 / 65 | .92 / .98 / .99 / .93 / .97 | .89 / .90 / .92 / .85 / .67 | 109 / 115 / 115 / 139 / 147 |
| prune identity / random b32 / b16 / b8 / b4 | 5 / 7 / 8 / 9 / 9 | .88 / .88 / .89 / .90 / .89 | .87 / .86 / .87 / .86 / .86 | 120 / 112 / 123 / 105 / 126 |

Under the window, whenever the agent chooses to stop it succeeds (.93–.99, above the unpruned .88); the SR cliff
is entirely episodes that never emit stop (65 of 100 at K = 4, 15 of them parked within 1 m of the goal), and
forced stops already rise at K = 32/16 where SR is intact. The pruned arms keep 5–11 forced stops at every
budget. So the window ablation on LightNav-0 measures the stop decision under a prompt that says the episode just
began, not the visual memory; F12's "+.47 SR at k = 4" is mostly that skeleton (the same protocol difference as
F7 on LongNav-R1, now with the mechanism visible). What random k = 4 actually keeps at the plateau is ~45 tokens
scattered over 7–8 fast-tier tubelets plus ~11 over 4 long-tier tubelets (median age 14 steps, mid tier ~0): the
model navigates on fragments of ~12 low-res history tubelets, the full-resolution current pair and the complete
timestamp skeleton.

Direct test (run 2026-10-23, `ARMS="lightnav_random_b2 lightnav_random_b1 lightnav_random_b0"`, same wrapper,
`k = 0` = every history token hidden, current pair + text skeleton only): `random_b2` holds the window K = 4 arm's
28 history tokens with the skeleton kept; `random_b1` 14; `random_b0` none. Paired against `lightnav_k4`
(content match for b2, an upper bound on history for b1/b0), `random_b4` and the identity arm; since 2026-10-28
the left end of the LightNav-0 Random line in `figures/budget_curves.*` (x = 2 / 1 / 0), out of `table.tex`. Smoke (episode 0, k = 0): success at 481 steps vs 63 unpruned — the agent wanders without
history but still stops.

| arm (history tokens) | SR | SPL | steps | forced stops | SR among self-stopped | ΔSR vs window K=4 (p) | ΔSR vs random k=4 (p) | ΔSR vs identity (p) |
|---|---|---|---|---|---|---|---|---|
| window `lightnav_k4` (28, fresh-episode skeleton) | .34 | .188 | 376 | 65 | .97 | — | −.47 (<.001) | −.50 (<.001) |
| `random_b4` (56) | .81 | .450 | 160 | 9 | .89 | +.47 (<.001) | — | −.03 (.54) |
| `random_b2` (28) | .82 | .463 | 165 | 13 | .94 | +.48 (<.001) | +.01 (1.0) | −.02 (.80) |
| `random_b1` (14) | .77 | .428 | 178 | 16 | .92 | +.43 (<.001) | −.04 (.50) | −.07 (.18) |
| `random_b0` (0, no visual history) | .59 | .332 | 286 | 38 | .95 | +.25 (<.001) | −.22 (<.001) | −.25 (<.001) |

**F13 — the window's K = 4 collapse is the prompt layout, not the visual memory.** With exactly the window's 28
history tokens but the full timestamp skeleton, random is at .82 (ΔSR +.48 vs the window, p < .001; indistinguishable
from random k = 4 and from the unpruned .84). Fourteen tokens still give .77. Even with *no* visual history at all —
the current pair, the task and 41 empty timestamped segments — the model scores .59, +.25 above a window that
holds 28 history tokens (p < .001), and once more the deficit is forced stops (38 of 100; among self-stopped episodes
it succeeds .95, oracle success .80). So visual history is worth about −.25 SR / −.13 SPL / +147 steps on
LightNav-0 (b0 vs identity, all p ≤ .001) and 14–28 random tokens of it recover nearly all of that; the K = 4
window's −.50 is mostly the model being told the episode just began. Read the LightNav-0 rows of the
context-window ablation as a stop-decision result, and F12 as "pruning keeps the skeleton that the window removes"
rather than as a memory-efficiency gain of the same size. Keep-mask audit of the three arms: 0 over-budget steps,
visible history == budget and current pair == 112 at every step ≥ 100 (62,821 steps). Wall: b2 72 min, b1 77 min,
b0 130 min (2026-10-23 16:38 → 21:18).

**Window-content control (run 2026-10-25, `ARMS="lightnav_recency_b2"`, `--prune-method recency`).** The user asked
for the K = 4 window's exact visual content with the old blocks' timestamps kept: the model sees image tokens of
the four newest frames only — the current pair at full resolution (112 tokens) and the previous pair at the
model's own 2× pooling (28 tokens; LightNav-0 never shows a history pair at more than 28) — and every older block
is blanked but keeps its `<t seconds>` tag and vision markers. `recency` keeps the newest candidates by tubelet
age; at k = 2 that is exactly the newest history tubelet, frames (t − 3, t − 2) at pool 2, the pair the window
shows (verified on upstream's layout at every step ≥ 34; before step 34 the sampler pads the history on even steps,
so the newest history tubelet is (t − 2, t − 2) there — 1,493 of 19,924 steps). Keep-mask audit: 0 violations,
visible history == 28 and current pair == 112 at every step ≥ 4. Paired against the K = 4 window (same content),
`random_b2` (same count) and identity; out of the figure and `table.tex`. 100 episodes, 118 min.

| arm (image tokens visible) | SR | SPL | steps | forced stops | SR among self-stopped | ΔSR vs window K=4 (p) | ΔSR vs random k=2 (p) | ΔSR vs identity (p) |
|---|---|---|---|---|---|---|---|---|
| window `lightnav_k4` (4 newest frames, old blocks deleted) | .34 | .188 | 376 | 65 | .97 | — | −.48 (<.001) | −.50 (<.001) |
| `recency_b2` (4 newest frames, old blocks kept) | .79 | .431 | 203 | 18 | .96 | +.45 (<.001) | −.03 (.55) | −.05 (.30) |
| `random_b2` (28 history tokens spread over time) | .82 | .463 | 165 | 13 | .94 | +.48 (<.001) | — | −.02 (.80) |
| `random_b0` (current pair only) | .59 | .332 | 286 | 38 | .95 | +.25 (<.001) | −.23 (<.001) | −.25 (<.001) |

**F14 — same four frames, blocks kept: .79 vs .34.** Holding the window's visual content fixed and restoring only
the empty old blocks with their timestamps recovers +.45 SR (p < .001), cuts forced stops from 65 to 18 and steps
from 376 to 203, and brings SPL from .19 to .43. The arm is not significantly different from `random_b2` or from
the unpruned model on SR or SPL, but it does take more steps (+38 vs `random_b2`, p < .001; +64 vs identity) and
keeps 18 forced stops against 13 and 5: seeing only the newest history pair costs some efficiency that 28 tokens
spread over the episode do not. So the window's K = 4 collapse is the deleted skeleton — the model's sense of
elapsed time — almost entirely, with a small genuine memory cost on top. F13 stands, and the context-window
ablation's LightNav-0 rows should be read as a stop-decision result.

The same control at the other window sizes (`lightnav_recency_b6` / `b14` / `b30` = window `K` = 8 / 16 / 32,
run 2026-10-27; `k = K − 2` is the window's history-frame count, verified equal to the newest tubelets of the full
layout at every `K`): SR / SPL / steps / forced stops .84 / .475 / 164 / 11, .79 / .439 / 174 / 14,
.81 / .445 / 155 / 10 against the windows' .68 / .365 / 236 / 27, .80 / .444 / 189 / 19, .82 / .467 / 177 / 16.
Keep-mask audit: 0 violations. The control is flat from `K = 32` to `K = 4`; the window's `K = 8` loss (−.12 SR
vs full, p = .02) is also the skeleton (keep − window +.16 SR, p = .0002). These four arms are the dashed
"LightNav-0 (old blocks kept)" line of the context-window figure (`paper/ctx_ablation`, which reads them from
this study's run directory as `lightnav_keep_k{K}`); here they are `summary.csv`-only rows paired against the
window at the same `K` and the nearest random grid arm.

**The K = 1 point (`lightnav_recency_b0_cur1`, run 2026-10-27 11:10 → 15:00, `PRUNE_CURRENT_FRAMES=1`).** The
K = 1 window shows one frame: upstream's step-0 layout, the current frame doubled into a single pool-1 tubelet.
The control reproduces that content with the skeleton kept: `--prune-k 0` hides every history image token and
`--prune-current-frames 1` replaces the current pair (t − 1, t) with (t, t) in the sampler's layout (the two
frames of a tubelet share one 112-token block, so column masking alone cannot drop frame t − 1), every older
block blanked but keeping its `<t seconds>` tag and markers. Keep-mask audit: 100 episodes, 45,123 step
records, 0 violations (no history token visible, 112 current tokens, no tubelet containing t or t − 1, 7
forwards per step, no pre-pass). Result: SR .02 / SPL .014 / 452 steps / 88 forced stops against the window's
.00 / .000 / 500 / 100 (ΔSR +.02, p = .51; Δsteps −48, p = .001) and `random_b0`'s .59 / .332 / 286 / 38
(ΔSR −.57, p < .001). The model stops on its own in 12 episodes, 2 of them at the goal (oracle success 5/100).
So the skeleton rescues the window from K = 4 up but not at K = 1: with a single frame the model has no motion
cue at all, and the whole K = 1 collapse is the missing second current frame — `random_b0`, which keeps the
full-resolution current pair and nothing else, is .57 SR above it. Read the LightNav-0 window curve as two
different failures: K = 1 is the loss of the current pair, K = 4 / 8 is the loss of elapsed time. This arm is
the dashed line's K = 1 point in the context-window figure (`lightnav_keep_k1`), and `random_b0` is its K = 2
point (`lightnav_keep_k2`: the current pair with no history; there is no K = 2 window arm); paired here against
`lightnav_k1` (merged into `summary.csv` as method `window_lo`, out of the figure) and `random_b0`.
