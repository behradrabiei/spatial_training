# Information horizon (full context, leak-free)

At which decoder layer does the current observation's visual information enter the action
decision, and from which layer on can the visual tokens be dropped at no cost? Two models
(LongNav-R1, Uni-NaVid), full episode context, on the HM3D-v2 val split. This is the
full-context, leak-free rerun of the Fig.-8-style figure in `paper/layer_token_info/horizon.py`,
whose data came from a window of 32 past frames plus the current frame and whose KL curve
was the leaky keep-one diagnostic.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`; plots use `vln`;
Uni-NaVid launchers use the isolated baseline runtime. To rebuild the retained tables and
figures without overwriting published outputs, follow the
[reproduction guide](../../paper_results/REPRODUCIBILITY.md). A Git clone alone is
insufficient: restore required local inputs from the checksummed backup first. Ordinary
launch logs and locks are ignored; protected traces, arrays, and images remain scientific inputs.

## Definitions

**Hiding from layer `L`** — the visual tokens' key columns are closed for *every* query on
decoder layers `>= L` (additive `-inf` / `finfo.min` on the attention mask); layers below `L`
read them as usual and positions never change. `L = n_layers` ("none", 28 for LongNav-R1, 32
for Uni-NaVid, 36 for LightNav-0) hides nothing and is 0 / the unblinded run by construction.

*LightNav-0.* The third model (Qwen3-VL-4B, 36 text decoder layers, `baselines/LightNav-0`)
has the SR/SPL arm only, no KL probe. Its visual tokens are the `<|video_pad|>` positions of
the prompt; the `<|vision_start|>` / `<|vision_end|>` markers and the `<t seconds>` timestamp
text stay visible, like the separators of the other two models. The model decodes inside vLLM,
whose FlashAttention kernel takes no mask and whose compiled CUDA-graph decode runs no Python
hook, so the wrapper `baselines/lightnav_hm3d/eval_blind.py` runs vLLM eager and replaces the
attention kernel of **every** decoder layer with an exact masked attention (torch SDPA over the
K/V gathered from the paged cache, causal mask, GQA); layers `>= L` additionally drop the
`<|video_pad|>` columns on prefill and decode alike. Qwen3-VL also injects ViT features into
the visual positions after decoder layers 0-2 ("deepstack"); those positions are the hidden
columns, so for `L >= 4` the semantics are those of the other two models. The upstream clone is
not edited.

**KL curves** (current frame, one value per step and layer):
* keep-one-in `= KL(P_only_current || P_no_visual)` — what the current frame alone says when it
  is the only visual input from `L` on;
* leave-one-out `= KL(P_full || P_current_hidden)` — what it adds on top of the history.
`P` is the distribution over the four actions at the decision token, temperature 1, `KL(a||b)`
with the expectation under `a`, nats. LongNav-R1 reads it through an fp32 projection onto the
action rows (`token_info._ActionHead`); Uni-NaVid renormalises the four action-word logits
(`influence.action_logp`). Averaged over every step of the 10 seed-17 episodes
(`hm3d_v2_10_seed17.json`, the Exp 3 set); steps whose current frame kept no visual token after the sparse
filter are skipped.

**Blind SR/SPL** — the live rollout with *all* visual tokens (history and current frame) hidden
from `L` on, 100 episodes, SR / SPL vs the unblinded reference.

## Blind-from-layer KL (2026-11-23)

The depth analogue of the recent-window measure of
[`paper/frame_influence`](../frame_influence/README.md) (`recent_k`: KL(full || K most recent
frames)), requested by the user to replace the keep-one-in / leave-one-out curves:

`Blind_t(L) = KL( P(action | full context) || P(action | every visual token hidden from layer L on) )`

with the expectation under the full-context decision, T = 1, nats. History and current frame are
hidden together, so it is the KL counterpart of the blind rollouts above, on the unblinded
trajectory. `L = 0` is KL(full || no visual), the recent-window study's `K = 0` point; `L = n_layers`
hides nothing and is 0. Probed after every decision of the 10 seed-17 episodes; no leak-free
LongNav-R1 arm and no Uni-NaVid arm (user).

| arm | model | layers | mechanism |
|---|---|---|---|
| `r1_probe_leaky_dense` | LongNav-R1 | every layer 0..27 | `r1_probe_leaky` with `vlm.horizon_probe_blind=true`: the leaky decision replay already computes the all-visual-hidden distribution as the keep-one reference, so the measure costs no forward. Logged per step and layer: `sup/horizon_blind`, the three probe distributions `sup/horizon_logp_{blind,only_cur,minus_cur}` (4 action log-probs each; `sup/horizon_base_logp` is the full one) and the keep-one-in / leave-one-out values. A step whose current frame kept no visual token is probed with the all-visual set alone. 4 actions. |
| `lightnav_probe` | LightNav-0 | every second layer 0..34 | `eval_influence.py --measure blind_layer` (`INFLUENCE=1 INFLUENCE_MEASURE=blind_layer` on `baselines/run_lightnav.sh`): unblinded navigation on the eager masked kernel (= `lightnav_ref`) and, after every decision, one full re-prefill per layer with every `<|video_pad|>` column closed on layers `>= L` -- the forward `eval_blind.py --blind-layer L` runs. Leak-free by construction. 256 `<act_l0>` codes, plus the fold onto the six executed-action bins; every probe's 256 log-probs are stored (`logp_f16`). The nothing-hidden replay is the noise floor and the `L = 36` point. |

LongNav-R1's curve is the *leaky* probe (the columns are hidden from the decision query only);
LightNav-0's is not. Raw KL heights are not comparable across the two models (4 actions vs 256
codes): compare shapes, the curve normalised by its `L = 0` value, or the executed-action flip
rate, all of which `aggregate_blind.py` writes.

```bash
ARMS=r1_probe_leaky_dense bash paper/info_horizon/run_r1.sh     # ~1.3 h
ARMS=lightnav_probe bash paper/info_horizon/run_lightnav.sh     # ~1.3 h; PROBE_LAYERS="0 4 8" picks other layers
python paper/info_horizon/aggregate_blind.py                    # results/blind_{curve,episode}.csv, blind_steps.csv.gz, blind_summary.json
```

**Results (2026-11-23; zero-based `L`, pooled mean over decisions).** LongNav-R1 (1,099 decisions) is a
step: .033 nats for `L` = 0..14, .027 at 16-17, .0014 at 18 and below .001 after; the decision
changes on 5.6 % of steps up to `L` = 15 and on under 0.5 % from 18. LightNav-0 (1,696 decisions) is
a ramp: about 10 nats for `L` = 0..14, 8.4 / 7.4 / 5.5 / 2.6 / 1.2 / .32 at 16 / 18 / 20 / 22 / 24 / 26
and about .1 from 28; its executed action changes on 94 % of steps at `L` = 0, 38 % at 20, 15 % at 24,
6 % at 26 and 1 % from 28. Both checks below pass (LightNav-0 reproduces the recent-window `K = 0`
values exactly; LongNav-R1 to .001 nats in the median, same chosen action on every decision).

`blind_summary.json` holds two reference checks: the shallowest layer against the recent-window
runs' `K = 0` on the same decisions, and the dense run's keep-one-in / leave-one-out against the
sparse `r1_probe_leaky` at the layers both probed. These runs and tables are kept locally
(`paper_results/.gitignore`). No figure yet. Smokes: `tests/horizon_probe_smoke.py` section 6 (GPU),
`baselines.tests.test_lightnav_influence` (CPU: the layer gate and its count check).

## Arms

| arm | model | mechanism | leak-free? |
|---|---|---|---|
| `r1_probe_leaky` | LongNav-R1 | decision-token replay against the live KV cache; the columns are hidden from that one query only (`vlm.horizon_probe_layers` in `context_window_mode=prune`) | no — later tokens' K/V already absorbed the frame |
| `r1_probe` | LongNav-R1 | hooked full re-prefill of prefix + every turn from the stored post-filter embeds, columns invisible to every query (`context_window_mode=evict`, `token_info._ColumnHider`) | yes |
| `r1_blind_l<L>` | LongNav-R1 | `vlm.kv_prune_visual_blind` from layer `L` (prune substrate, no budget, full context); reference = `paper_results/ctx_ablation/runs/r1_full` (evict mode; the prune substrate with nothing pruned is bit-identical, `tests/vtp_prune_smoke.py`) | yes, by construction |
| `uninavid_probe` | Uni-NaVid | after each inference the captured prompt is re-forwarded with the current frame's 4 history tokens + 64 fine tokens hidden from `L` on (`layer_blind.LayerProbe`); decisions untouched | yes (Uni-NaVid re-prefills every inference) |
| `uninavid_blind_l<L>` | Uni-NaVid | every visual column (history groups, long-term slots, fine tokens) hidden from `L` on during the live rollout (`--blind-layer L`); reference = `uninavid_ref` | yes |
| `lightnav_ref` | LightNav-0 | `BLIND_LAYER=36`: nothing hidden, but the same eager vLLM + masked-SDPA path as the blind arms (the compiled FlashAttention run `ctx_ablation/runs/lightnav_full` is *not* the reference; `aggregate.py` reports the paired agreement of the two in `summary.json`) | — |
| `lightnav_blind_l<L>` | LightNav-0 | every `<|video_pad|>` column hidden from `L` on during the live rollout (`BLIND_LAYER=L` on `baselines/run_lightnav.sh full`, wrapper `lightnav_hm3d/eval_blind.py`); reference = `lightnav_ref` | yes (greedy; one request per step) |

Layer grids: LongNav-R1 `L ∈ {8,12,14,16,18,20,24}` + none; Uni-NaVid
`L ∈ {8,12,16,18,20,24,28}` + none; LightNav-0 `L ∈ {12,16,18,20,22,24,26,28,30}` + none
(chosen 2026-10-16; 22 and 26 added 2026-10-18 to bracket the cliff, 28 and 30 on 2026-11-23 to fill
the gap between 26 and the unblinded reference). These are the live blind-rollout grids;
probe grids also include other layers. Unmeasured blind-rollout layers have no reported SR/SPL. The leak-free `r1_probe` arm is
optional (`ARMS=r1_probe bash paper/info_horizon/run_r1.sh`) and drawn only if present. Uni-NaVid arms use `--sampling-seed 30` (as Exp 3; the token-pruning sweeps did not reseed each call),
so `uninavid_ref` and `uninavid_probe` share trajectories with `frame_influence/uninavid_influence`.
Separators and special tokens are never hidden in either model.

Layer indices in the protocol, CLI, and CSVs are zero-based. The plotted axes use one-based
labels; the unblinded endpoint is at 29 / 33 / 37 for LongNav-R1 / Uni-NaVid / LightNav-0 (larger
hollow marker; the last tick, labelled `L+1`, sits at the deepest drawn model's endpoint).
Figures show means; confidence intervals remain in `results/horizon.csv`.

**Figure format (2026-11-23, RSS draft).** The figures draw LongNav-R1 and LightNav-0 only: Uni-NaVid
is no longer a paper baseline (user decision); its arms, rows in `results/horizon.csv` and
`table.tex` are unchanged, and `plot.py --models LongNav-R1 Uni-NaVid LightNav-0` draws it again.
The KL figure is `information_kl_blind` (blind-from-layer KL, LongNav-R1 leaky probe, pooled mean over the
1,099 probe decisions; see the section above). The current-frame KL figures `information_kl`,
`information_kl_keep` and `information_kl_leave` were removed from `figures/` on 2026-11-23 (user); their
data stay in `results/horizon.csv` and `plot.py --only information_kl information_kl_keep
information_kl_leave` redraws them (LongNav-R1 leaky probe alone: LightNav-0 has no current-frame probe).
All figures are drawn at single-column size through `paper/figstyle.py` (3.5 in wide, 1.5 in tall
for the two-panel figures and 1.9 in for the standalone panels, 8 pt labels, 7 pt ticks and legend,
metric on the y axis instead of a panel title, legend across the top, TrueType fonts in the PDF).
In the SR/SPL panels each model's unblinded reference is a dashed horizontal line; the models'
dashes interleave so that references at the same height (both models' SR is .84) stay visible.

## Protocol

100-episode HM3D-v2 val split (`hm3d_v2_100.json`) for SR/SPL, the 10-episode seed-17 subset
for the KL probes; `+checkpoint=longnav`, sdpa, bf16, argmax decoding, 500-step cap,
`ep_seed=17`, no guards, no visualisation, sparse filter on (0.95) — the presets
`src/longnav/config/experiment/paper_horizon.yaml` (blind arms) and `paper_horizon_probe.yaml`
(probe arms) pin everything; each run saves its resolved `config.yaml`. Uni-NaVid: official
decoding (temperature 0.2, two actions per inference) with the sampling seed. LightNav-0: the
comparable protocol of the context-window ablation (`baselines/run_lightnav.sh full`: success
radius 1.0 m, the model's own 480x270 / 120° camera, velocity actions, greedy decoding, 500-step
cap), `VLN_VLLM_ENFORCE_EAGER=1`; each arm's `run.json` records `blind_layer`, `enforce_eager`
and `client_entry`. vLLM is not bit-reproducible run to run even unmodified (the one-episode
smoke and the 100-episode run differ by a step on the same episode), so `lightnav_ref` vs
`lightnav_full` is a noise-level check, not an identity. LightNav-0 arms are loaded with
`results_io.load_lightnav_run`, which scores episodes ended by upstream's forced STOP at the
500-step cap as failures, the rule the other two models are under (see
`paper/ctx_ablation/README.md`); in the blind arms this removes the capped agents that happened
to sit inside the 1.0 m radius (2 episodes at `L=12`), nothing in `lightnav_ref`.

What is exact and what is not:
* KL at `L = n_layers` is exactly 0 (no forward issued); the blind arm at `L = n_layers` is the
  reference.
* Probe arms never move the live decision (`tests/horizon_probe_smoke.py`; Uni-NaVid probe
  traces equal the seeded reference's).
* The leak-free LongNav-R1 leave-one-out at `L = 0` is the masking analogue of Exp 3's `age0`
  (which *deletes* the tokens): the same quantity up to the bf16 rebuild floor (~2e-2 in
  probability), not bit-identical.

## MP3D replicate

The same protocol on MP3D ObjectNav val (`DATASET=mp3d`): SR/SPL on 100 episodes drawn with
seed 17 from the 2195-episode split (`mp3d_100_seed17.json`), KL probes on 10 of those
(`mp3d_10_seed17.json`, seed 17 again). LongNav-R1 gets its own unblinded full-context reference arm `r1_ref`
(`+experiment=paper_ctx` protocol, `+dataset=mp3d_val`); Uni-NaVid uses
`baselines/uninavid_hm3d/config/objectnav_mp3d.yaml` with the MP3D dataset and scene paths.
Data in `paper_results_archive/info_horizon_mp3d/` (**archived 2026-11-23 at the user's request**: the paper does
not use the MP3D replicate; the launchers, `aggregate.py --dataset mp3d` and `plot.py --dataset mp3d` point
there, and its `results/` and `figures/` are local, git-ignored copies that rebuild from `runs/`).

## Running

```bash
conda activate longnav_vlm
bash paper/info_horizon/run_uninavid.sh            # uninavid_ref, uninavid_blind_l*, uninavid_probe (~3 h)
DATASET=mp3d bash paper/info_horizon/run_uninavid.sh   # MP3D; the same prefix works with run_r1.sh
bash paper/info_horizon/run_r1.sh                  # r1_probe_leaky, r1_blind_l* (~8-12 h); ARMS=r1_probe for the leak-free probe
bash paper/info_horizon/run_lightnav.sh            # lightnav_ref, lightnav_blind_l* (HM3D only; eager vLLM; ~1 h per arm that navigates, ~3.9 h per collapsed arm whose episodes all run to the 500-step cap; the 6-arm sweep of 2026-10-16/17 took 17.5 h)
python paper/info_horizon/aggregate.py             # results/horizon.csv, probe_steps.csv, table.tex
python paper/info_horizon/aggregate_blind.py       # results/blind_curve.csv etc. (blind-from-layer KL)
conda run -n vln python paper/info_horizon/plot.py   # figures/information_perf.{pdf,png}: blind-rollout SR / SPL vs L
# Also writes figures/information_kl_blind{,_lightnav,_norm}.{pdf,png}: blind-from-layer KL vs L (probe episodes), one figure per model in nats and both models normalised
# LongNav-R1 and LightNav-0 by default; add --models LongNav-R1 Uni-NaVid LightNav-0 for the three-model view

# Aggregate and plot the archived MP3D replicate explicitly (paper_results_archive/info_horizon_mp3d/)
python paper/info_horizon/aggregate.py --dataset mp3d
conda run -n vln python paper/info_horizon/plot.py --dataset mp3d
```
(plus the single-panel figures `information_perf_{sr,spl}`)
The R1 and Uni-NaVid scripts skip complete arms and resume unfinished ones; `ARMS="r1_probe r1_blind_l18"` selects a LongNav-R1 subset.
LightNav arms skip when complete but cannot resume (append-only `results.jsonl`): move a partial arm away and rerun.
`aggregate.py --allow-partial` for interim looks. Smokes: `python tests/horizon_probe_smoke.py`
(GPU), `PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python
-m unittest baselines.tests.test_layer_blind` (CPU), `PYTHONPATH=baselines/LightNav-0/src
baselines/.envs/lightnav/bin/python -m unittest baselines.tests.test_lightnav_blind` (CPU: the
masked kernel against a naive attention, the hook's layer gating and guards) and
`BLIND_LAYER=36 bash baselines/run_lightnav.sh smoke` (GPU, one episode through the hooked engine).

## Outputs — all under `paper_results/info_horizon/` (MP3D: `paper_results_archive/info_horizon_mp3d/`)

```
runs/<arm>/                  config.yaml, rollout/results_* (LongNav-R1) or
                             manifest.json, summary.json, episodes/*.json, horizon/*.jsonl (Uni-NaVid) or
                             run.json, raw/results.jsonl, episodes/*.json, summary.json, launch.log (LightNav-0);
                             per-episode rollout dirs (sequence.json), traces, and horizon/ are gitignored but required
results/horizon.csv          one row per (curve, layer): KL means + CIs, SR/SPL + CIs + paired deltas
results/probe_steps.csv      every (curve, episode, step, layer) probe value
results/table.tex, summary.json
results/blind_*              blind-from-layer KL tables (aggregate_blind.py; local, gitignored)
figures/information_kl_blind.{pdf,png}   blind-from-layer KL vs L, LongNav-R1 (probe episodes; local, gitignored)
figures/information_kl_blind_lightnav.{pdf,png}   the same for LightNav-0
figures/information_kl_blind_norm.{pdf,png}   both models, each curve divided by its own first-layer (no visual input) value
figures/information_perf.{pdf,png}   SR / SPL of the blind rollouts vs L (100 episodes)
figures/information_perf_{sr,spl}.{pdf,png}   each panel on its own
```
