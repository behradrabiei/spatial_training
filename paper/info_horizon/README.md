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
`L ∈ {8,12,16,18,20,24,28}` + none; LightNav-0 `L ∈ {12,16,18,20,22,24,26}` + none
(chosen 2026-10-16; 22 and 26 added 2026-10-18 to bracket the cliff). These are the live blind-rollout grids;
probe grids also include other layers. Unmeasured blind-rollout layers have no reported SR/SPL. The leak-free `r1_probe` arm is
optional (`ARMS=r1_probe bash paper/info_horizon/run_r1.sh`) and drawn only if present. Uni-NaVid arms use `--sampling-seed 30` (as Exp 3; the token-pruning sweeps did not reseed each call),
so `uninavid_ref` and `uninavid_probe` share trajectories with `frame_influence/uninavid_influence`.
Separators and special tokens are never hidden in either model.

Layer indices in the protocol, CLI, and CSVs are zero-based. The plotted axes use one-based
labels; the unblinded endpoint is at 29 / 33 / 37 for LongNav-R1 / Uni-NaVid / LightNav-0 (the
last tick, labelled `L+1`, sits at 37).
Figures show means; confidence intervals remain in `results/horizon.csv`.

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
Data in `paper_results/info_horizon_mp3d/`; `aggregate.py --dataset mp3d`, `plot.py --dataset mp3d`.

## Running

```bash
conda activate longnav_vlm
bash paper/info_horizon/run_uninavid.sh            # uninavid_ref, uninavid_blind_l*, uninavid_probe (~3 h)
DATASET=mp3d bash paper/info_horizon/run_uninavid.sh   # MP3D; the same prefix works with run_r1.sh
bash paper/info_horizon/run_r1.sh                  # r1_probe_leaky, r1_blind_l* (~8-12 h); ARMS=r1_probe for the leak-free probe
bash paper/info_horizon/run_lightnav.sh            # lightnav_ref, lightnav_blind_l* (HM3D only; eager vLLM; ~1 h per arm that navigates, ~3.9 h per collapsed arm whose episodes all run to the 500-step cap; the 6-arm sweep of 2026-10-16/17 took 17.5 h)
python paper/info_horizon/aggregate.py             # results/horizon.csv, probe_steps.csv, table.tex
conda run -n vln python paper/info_horizon/plot.py   # figures/information_kl.{pdf,png}     keep-one-in / leave-one-out KL vs L (probe episodes)
# Also writes figures/information_perf.{pdf,png}: blind-rollout SR / SPL vs L

# Aggregate and plot the MP3D replicate explicitly
python paper/info_horizon/aggregate.py --dataset mp3d
conda run -n vln python paper/info_horizon/plot.py --dataset mp3d
```
(plus the single-panel figures `information_kl_{keep,leave}` and `information_perf_{sr,spl}`)
The R1 and Uni-NaVid scripts skip complete arms and resume unfinished ones; `ARMS="r1_probe r1_blind_l18"` selects a LongNav-R1 subset.
LightNav arms skip when complete but cannot resume (append-only `results.jsonl`): move a partial arm away and rerun.
`aggregate.py --allow-partial` for interim looks. Smokes: `python tests/horizon_probe_smoke.py`
(GPU), `PYTHONPATH=baselines:baselines/Uni-NaVid baselines/.envs/uninavid-hm3d/bin/python
-m unittest baselines.tests.test_layer_blind` (CPU), `PYTHONPATH=baselines/LightNav-0/src
baselines/.envs/lightnav/bin/python -m unittest baselines.tests.test_lightnav_blind` (CPU: the
masked kernel against a naive attention, the hook's layer gating and guards) and
`BLIND_LAYER=36 bash baselines/run_lightnav.sh smoke` (GPU, one episode through the hooked engine).

## Outputs — all under `paper_results/info_horizon/`

```
runs/<arm>/                  config.yaml, rollout/results_* (LongNav-R1) or
                             manifest.json, summary.json, episodes/*.json, horizon/*.jsonl (Uni-NaVid) or
                             run.json, raw/results.jsonl, episodes/*.json, summary.json, launch.log (LightNav-0);
                             per-episode rollout dirs (sequence.json), traces, and horizon/ are gitignored but required
results/horizon.csv          one row per (curve, layer): KL means + CIs, SR/SPL + CIs + paired deltas
results/probe_steps.csv      every (curve, episode, step, layer) probe value
results/table.tex, summary.json
figures/information_kl.{pdf,png}     keep-one-in / leave-one-out KL vs L (probe episodes)
figures/information_perf.{pdf,png}   SR / SPL of the blind rollouts vs L (100 episodes)
figures/information_{kl_keep,kl_leave,perf_sr,perf_spl}.{pdf,png}   each panel on its own
```
