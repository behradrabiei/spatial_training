# Layer-wise visual-token information of the current observation

Qualitative figure in the spirit of Fig. 7 of *When Token Pruning is Worse than Random*
(arXiv 2512.07580): at one decision of one episode, how much does each visual token of the
**current observation** contribute to LongNav-R1's action, as a function of the decoder layer
the token is removed from? (a) the distribution across tokens per layer; (b) the per-patch map
over the image at three layers. Measured two ways (keep-one-in, leave-one-out) and under two
counterfactual semantics (leaky, leak-free), i.e. four headline figures plus two supplementary
ones.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`; plots use `vln`;
Uni-NaVid launchers use the isolated baseline runtime. To rebuild the retained tables and
figures without overwriting published outputs, follow the
[reproduction guide](../../paper_results/REPRODUCIBILITY.md). A Git clone alone is
insufficient: restore required local inputs from the checksummed backup first. Ordinary
launch logs and locks are ignored; protected traces, arrays, and images remain scientific inputs.

## Definitions

**Episode and step.** One HM3D-v2 val episode from the seed-17 manifest
(`src/longnav/conf/episode_jsons/hm3d_v2_10_seed17.json`), driven for 10 steps with deterministic
(argmax) decoding and the full episode in context. The probed decision is the **10th** (frame
index 9, nine history frames in the cache). All 10 manifest episodes are driven (`survey`); the
figure's episode is the one whose 10th observation moves the decision most — the largest
leak-free KL of hiding the whole current frame — among those that reach step 10 without
stopping (`runs/survey.csv`, `runs/selected.json`; `--episode` overrides).

**Tokens.** The current frame's visual tokens. The sparse filter is **disabled for this figure**
(`vlm.sparse_threshold=2.0`: cosine similarity never exceeds 1, so no patch is deduplicated) so
the map covers the full merged token grid — 640×480 → 15×20 = 300 tokens. `use_sparse` stays on
(both substrates need its bookkeeping). Caveat: the checkpoint's SFT base is a `sparse_03drop`
model, so 300 tokens per frame is off its training regime and the trajectory differs from the
paper's `r1_full` runs; the figure is qualitative.

**Measures.** For decoder layer `i ∈ 0..27` and token `k`, with every hiding applied on layers
`≥ i` only (layers `< i` untouched, as in the paper's Eq. 2–6):

| measure | definition | file key |
|---|---|---|
| leave-one-out | KL( P_full ‖ P_{−k} ) | `leave` |
| keep-one-in, current-frame reference (headline) | KL( P_{only k of the current frame; history intact} ‖ P_{current frame hidden} ) | `keep_cur` |
| keep-one-in, all-visual reference (supplementary, the paper's) | KL( P_{only k; every other visual token incl. history hidden} ‖ P_{no visual} ) | `keep_all` |

`P` is the action distribution over `{stop, forward, left, right}` at the decision token
(temperature 1), KL in nats with the expectation under the first argument — the convention of
`paper/frame_influence` and of `vlm.kv_prune_log_keep_one`. The paper uses the probability
difference of the ground-truth answer; there is no ground truth here, so the change of the
whole action distribution is used, as in every other experiment of this branch.

**Arms (counterfactual semantics).**

* **leaky** (`mask`, `paper_results/.../token_info_mask.npz`): the worker runs on the prune
  substrate (`context_window_mode=prune`, unreachable budget, nothing pruned). Hiding = an
  additive `−inf` on the hidden tokens' key columns for the **replayed decision token only**
  (`VLMWorker._decision_replay` with `extra_layers=range(i, 28)`), the per-token version of the
  frame-level `_score_layer_frame_influence` and the same probe as Exp 3's evict-style arm. The
  same-chunk markup tokens and everything encoded after `k` keep what they absorbed from it —
  the leakage the context-window ablation measured.
* **leak-free** (`recompute`, `token_info_recompute.npz`): stock evict-mode worker; every probe
  re-prefills prefix + all 10 turns from their stored post-filter embeddings against an empty
  cache while the hidden columns are invisible to **every** query on layers `≥ i` (a forward
  pre-hook on each layer's attention swaps in the mask; `longnav.utils.token_info._ColumnHider`).
  Survivors keep their absolute mRoPE positions. This is the paper's `H^i ⊙ M` procedure with
  the tokens removed rather than zeroed. For `keep_all` the history's visual columns are hidden
  from every query here but only from the decision query in the leaky arm — that contrast is the
  point of having both arms.

**Cost.** Per layer 2 references + 3·N probes; 28 layers × 902 ≈ 25k probes per arm. Leaky:
single-token replays, minutes. Leak-free: full ~3.3k-token prefills, tens of minutes.

## Reading the decision: fp32 head

The model reads its action from a bf16 `lm_head`. At the decision's logit magnitude one bf16
ulp is ~1e-3 nats of KL, and removing a single visual token usually moves the logits by less
than that, so KLs read from the bf16 head collapse onto a handful of discrete levels (in the
first run, leave-one-out took 9 distinct values across 300 tokens, all multiples of ~7e-4). Every
probe therefore projects the decision token's final-norm hidden state onto the four action rows
of the head in float32 (`token_info._ActionHead`; the hidden state itself stays bf16). The base
uses the same reading, so the argmax agrees with the live decision except at near-ties (the
probability differs by ~1e-2, the bf16 rounding of the head); `fp32_head=False` restores the
bf16 reading, which reproduces the live decision exactly and is what the smoke pins against the
frame-level diagnostic. The survey score in `runs/survey.csv` was computed with the bf16 reading
(it only ranks episodes; the ranking is unaffected at that resolution).

## Numerical floors

* Leaky arm: the replayed base reproduces the live decision exactly (same cache, same kernel).
* Leak-free arm: a from-scratch bf16 re-prefill differs from the live incremental decision by
  up to ~2e-2 in probability (the floor documented in `tests/influence_recompute_smoke.py`), and
  two prefills of *different lengths* differ by the same order. Every probe of the arm is a
  same-length forward through the same masked-sdpa kernel path (the base included), so that
  floor never enters a measured KL: probes differ only by the hidden columns, and a hook that
  hides nothing reproduces the base bit-for-bit (`tests/token_info_smoke.py`).
* Zero KLs are drawn at 1e-6 on the log axis of panel (a).

## Figures (`paper_results/layer_token_info/figures/<episode>/`)

Layer indices in this protocol, CLI arguments, and saved arrays are zero-based. The token-map
figures display one-based labels: `--layers 1 12 18` is labelled 2 / 13 / 19.

`token_info_<leaky|leakfree>_<keepone|leaveone|keepone_allvisual>.{pdf,png}`, one row: (a) box
plot of the per-token KL per layer (log y, whiskers span the full range so the few hot tokens of
a sparse map are visible; the dotted lines mark the three layers of (b)); (b) the raw current
observation, then the same frame in greyscale with the per-token map of three layers overlaid —
bilinearly upsampled + Gaussian-blurred by half a cell (the paper's look; values are clipped to
the colour scale before blurring so one extreme token cannot bleed into a large blob). One
colour scale across the three layers (99.5th percentile), green at zero through yellow to red
(the upper half of `jet`); the overlay's opacity follows the value (transparent at zero, 0.75 at
the top), so the grey scene shows wherever a token carries nothing and every colour belongs to
the map. Each panel title carries the total over tokens, because a layer with a few hot tokens
can look stronger than a layer with more information spread over many. Unless `--layers a b c`
is given the three layers are chosen automatically: the information horizon is the largest drop
between consecutive layers of the layer strength (mean of the top 5 % of tokens); two of the
most mutually dissimilar normalised maps before it (greedy farthest-point seeded at the layer
with the largest spatial coefficient of variation, ≥ 6 layers apart) plus the first layer past
it, so the panel shows the structure and its vanishing.
`results/<episode>/layer_summary.csv` has per arm / measure / layer the mean, median, max and
across-token CV, plus the two joint references (KL of hiding the whole current frame / every
visual token from that layer on).

## No-visual-history study (`paper_results/layer_token_info/nohist/`, 2026-09-17)

The same figure for more scenes and timesteps, restricted to the question "what does the
**current frame alone** tell the agent": every history frame's visual tokens are hidden on
layers `≥ i` in every forward of both measures (the text of the history — the past actions —
stays), and panel (b) always shows layers **1 / 12 / 18**.

| measure | definition | file key | figure |
|---|---|---|---|
| keep-one-in | KL( P_{only k; every other visual token hidden} ‖ P_{no visual} ) | `keep_all` | `t<step>_<arm>_keepone_allvisual` |
| leave-one-out | KL( P_{history visuals hidden} ‖ P_{history visuals hidden, −k} ) | `leave_nohist` | `t<step>_<arm>_leaveone_allvisual` |

`keep_all` is exactly the `keepone_allvisual` measure of the headline study (re-running it on
q5QZSEeHe5g_0, decision 10, layer 12 reproduces the stored values bit-for-bit); `leave_nohist` is new
(`token_info.NOHIST_MEASURES`, `probe_*_arm(measures=...)`; its per-layer reference is saved as
`text_hist_logp` / `text_hist_kl` = KL of hiding the history's visual tokens). On the first frame
of an episode the two coincide with `keep_cur` / `leave` bit-for-bit (`tests/token_info_smoke.py`).

Leaky and leak-free still differ here even though only current-frame tokens are ablated: the
current frame **is** in the cache when the action is read (the decision token closes the chunk
that carries the frame), so in the leaky arm the later visual tokens and the markup of the same
chunk keep what they absorbed from `k`, and the past action tokens keep what they absorbed from
the hidden history frames. The leak-free arm hides the columns from every query.

Protocol: `probe_episode.py --study nohist survey --max-steps 40 --step-scores` drives the ten
seed-17 episodes for up to 40 steps (an episode that stops earlier is kept) and logs at every
step the leak-free KL of hiding the whole current frame (`step_scores_leakfree_kl` in
`trajectory.json`); the first 10 actions reproduce the headline survey. Three scenes, two
decisions each — the decision whose observation moves the action most in an early (5–15) and a
late (16–30) window: `Nfvxx8J5NCo_1` (tv_monitor) 7 / 26, `h1zeeAwLh9Z_0` (bed) 8 / 21,
`zt1RVoi7PcG_0` (chair) 9 / 16. `probe --steps` probes after the listed decisions (1-based) in
one replay and writes `runs/<label>/token_info_<arm>_t<step>.npz`.

```bash
bash paper/layer_token_info/run_nohist.sh                       # drive -> leaky -> leak-free -> plots
PROBES="q5QZSEeHe5g_0:10 q5QZSEeHe5g_0:18" bash paper/layer_token_info/run_nohist.sh
```

Cost per (scene, decision), all 28 layers: 1 + 28·(3 + 2·300) ≈ 16.9k forwards. Leaky ~15 min.
Leak-free grows with the context (one full re-prefill per probe): ~25 min at decision 8, ~1 h
at 16, ~1.5 h at 21, ~2 h at 26, ~2.5 h at 30. `LAYERS="1 12 18"` probes only the panel layers
(9× cheaper; the box plot then has three boxes).

## Information horizon figure (`figures/information_horizon.{pdf,png}`, the paper's Fig. 8)

`horizon.py` draws four panels against the first layer `L` without visual tokens: current-frame
keep-one-in KL, leave-one-out KL, and the agent's SR / SPL when no query may read any visual
token from `L` on (`vlm.kv_prune_visual_blind`, 100 episodes). Panels show means; 95 %
episode-bootstrap CIs are saved in `results/horizon.csv`. Dashed horizontal lines mark the
unblinded reference, also drawn at `L = none`; this historical figure uses zero-based layers. The information curve is **frame-level**: the current-frame keep-one-in KL of the
10-episode keep-one diagnostic (`dump/longnav_eval/keep_one_influence_w32_b10000000`,
`vlm.kv_prune_log_keep_one`, every 4th step, probe layers 0/8/16/20/24/28, native sparse regime,
32 past frames + current in context — the same regime as the blind arms). The purple line marks the horizon at
layer 18, the layer at which this experiment's per-token maps collapse; the frame-level probes
only bracket it between 16 and 20. `--with-token-curve` overlays the 2-episode per-token mean
(leaky arm) as a dotted line; it is off by default so the figure stays frame-level. Blind-arm results
are copied from `dump/longnav_eval/hm3d_v2_100_w32_vblind*` and `hm3d_v2_100_win32` into
`runs/` (config + `rollout/results_*`); new arms are launched with
`VISUAL_BLIND=true IMPORTANCE=random PREFIX=hm3d_v2_100 LAYER_START=<L> bash tools/run_kv_prune_ablation.sh`
and copied the same way. Numbers in `results/horizon.csv`.

Consistency of the sources: the blind arms (L = 0, 4, 8, 12, 16, 20, 24) all come from the same
launcher with identical configs apart from `kv_prune_layer_start` (checked by diffing
`config.yaml`; the later runs only add new flags at their defaults). The keep-one diagnostic runs
the same header-cut window (32 past frames + current), sparse filter, seeds and deterministic decoding on the
prune substrate with nothing pruned (`kv_budget` unreachable rather than `null`), scored every
4th step on the 10-episode seed-17 subset; the reference `hm3d_v2_100_win32` is the stock
evict-mode window. The information line is frame-level (the whole current frame is kept /
hidden); no token-level quantity enters the figure unless `--with-token-curve` is given.

## Status and findings (2026-09-14)

Probed episodes (both arms, all 28 layers, 300 tokens each): `a8BtkwhxdRV_2` (the survey's pick;
agent spinning in place, sofa not in view, action left) and `q5QZSEeHe5g_0` (runner-up; sofa in
view through a doorway, action forward at 0.97 — the recommended headline figure). The other
eight manifest episodes have frames and trajectories only.

* **Horizon.** In every arm, measure and episode the per-token maps collapse between decoder
  layers 17 and 18 (per-token maximum drops 3–4 orders of magnitude; the map turns to noise).
* **Where.** Before the horizon the information sits on the goal-relevant structure: the sofa
  and the floor path to it (q5), the doorway edge (a8).
* **Leaky vs leak-free.** Leaky keep-one-in is flat across layers 0–17. Leak-free keep-one-in
  is not flat: on q5 it rises from ~0.05 to ~0.9 nats/token by the mid layers (a lone token is
  worth little until it has absorbed its neighbourhood), on a8 it decays instead; the shape
  before the horizon is episode-specific, the cliff is not. Joint current-frame KL: leaky 0.034
  vs leak-free 0.23 at layer 0 on a8, converging by ~layer 12.
* **Leave-one-out** is ~1e-5 nats for almost every token (redundancy), with a handful of sofa /
  doorway tokens at ~2e-3, and zero from 18.
* **Fig. 8 analogue** (`results/horizon.csv`): frame-level information 0.045 nats at L = 0/8/16,
  0.0007 at 20; SR with all visual tokens removed from L on: 0.00 at L ≤ 12, 0.35 at 16, 0.85 at
  20 and 24 vs 0.88 unblinded. Layers 18–27 can drop every visual token at no cost.

This historical windowed horizon retains its original coarse layer grid. The later
[full-context information-horizon study](../info_horizon/README.md) supplies the separate
blind-rollout and leak-free probe comparison. The windowed figure still reads
`dump/longnav_eval/keep_one_influence_w32_b10000000`; this protected input must remain
available alongside the `paper_results/layer_token_info/` inputs.

## How to run

```bash
conda activate longnav_vlm
bash paper/layer_token_info/run.sh                       # survey -> probes -> plot
EPISODE=VBzV5z6i1WS_1 bash paper/layer_token_info/run.sh # a chosen episode
LAYERS="0 4 8 12 16 20 24 27" bash paper/layer_token_info/run.sh # subset of layers while iterating
```

`probe_episode.py survey` boots one habitat sim (Ray, `vln` env) and the LongNav-R1 worker
in-process (`+checkpoint=longnav +dataset=hm3d_v2_val +experiment=paper_token_info
+resources=single`, composed programmatically), saves each episode's frames and actions under
`runs/<label>/` (frames gitignored) and the survey score; `probe --arm {recompute,mask}` re-feeds
the saved frames to a fresh worker per arm, checks the decisions reproduce the recorded ones
(argmax must match; the max probability gap is printed), and runs the probes after the 10th step.
`plot.py` (vln python) draws the figures, `horizon.py` the information-horizon figure. GPU smoke: `python tests/token_info_smoke.py`.

Layout: `paper_results/layer_token_info/runs/config.yaml`, `runs/survey.csv`, `runs/selected.json`,
`runs/<label>/{trajectory.json, token_info_mask.npz, token_info_recompute.npz, frames/}`,
`results/<episode>/layer_summary.csv`, `figures/<episode>/`.
