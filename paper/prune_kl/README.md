# Pruning KL: full context vs pruned context

How far does a token-pruning method move the action distribution away from the full-context one,
at each frame budget? The third KL analysis of the paper, after the one along time
([recent-window variant](../frame_influence/README.md): full context vs the K most recent frames)
and the one along depth ([blind-from-layer KL](../info_horizon/README.md): full context vs no
visual token from layer L on). Here the reduced context is what a pruning method of the
[flat token-pruning study](../token_prune/README.md) holds: Random, FastV, DivPrune, DART at
k = 4, 8, 16, 32 frames' worth of history tokens. Models: LongNav-R1 and LightNav-0, on the 10
seed-17 HM3D-v2 episodes of the other two KL analyses.

Run commands from the repository root. LongNav-R1 uses `longnav_vlm`, LightNav-0 its isolated
runtime (`baselines/run_lightnav.sh`), the plot uses `vln`. Everything under
`paper_results/prune_kl/` is kept locally (git-ignored).

## Definition

    Prune_t(M, k) = KL( P(action | full context) || P(action | the visual tokens M holds at budget k) )

at every decision `t` of the **full-context trajectory**, expectation under the full-context
decision, T = 1, nats. The agent always executes the full-context action; the pruned decision is
a side probe, so every (method, k) is scored on the same decisions.

**Cumulative pruned set (shadow pruner).** The flat arms delete permanently: what a method holds
at step `t` depends on every selection it made before. One *shadow pruner* per (method, k)
therefore runs along the trajectory with the flat arm's own rule and its own record of dropped
tokens; nothing is deleted from the live context.

| | LongNav-R1 | LightNav-0 |
|---|---|---|
| rule shadowed | `paper_prune_flat` (`_apply_kv_prune`): pool = survivors + the frame just decided on, text never a candidate | `eval_prune.py`: ledger of dropped `(f0, f1, pool, cell)` tokens, pool = history tokens not in the ledger, current tubelet exempt |
| budget | `frame_budget`: k x the running mean visual tokens per frame | k x 14 tokens |
| timing | select **after** the decision (acts from `t+1`): the probe at `t` hides the drops made through `t-1`, frame `t` fully visible | select **before** the decision of the same step |
| probe | leaky decision replay: the cached decision token replayed against the full KV cache with the dropped slots at `-inf` on every layer (as the recent-window measure) | the whole prompt re-run with the dropped `<|video_pad|>` columns closed on every layer (leak-free by construction) |
| reference | the replayed full decision (a shadow that dropped nothing scores exactly 0) | the live decision (replay floor measured: `replay.kl`) |
| distribution | the 4 action ids, bf16 head | the 256 `<act_l0>` codes; `kl_exec` folds them onto the 6 executed-action bins |
| selectors | `random_select` (private generator, `step_seed(17, label, step)`), FastV = layer-0 decision row, `divprune_select`, `_dart_select` | `eval_prune.select_keep` (random seeded on the episode's index in the 100-episode runs; FastV / DART in a one-token pre-pass with the shadow's ledger hidden) |

**Averaging.** The figure averages over the decisions where the budget *binds* (some method's
shadow hides a token: the same decisions for the four methods at a given k). An episode no longer
than the budget contributes nothing at that k. `kl_all_*` in the CSV is the mean over every
decision (non-binding ones count as 0); `kl_norm_*` divides each episode's mean by its mean
no-visual KL on the same decisions (`novis_*`, the measure's ceiling: KL(full || no visual token)).

**What the shadow is not.** It follows the full-context trajectory, so it is what the method
*would* hold on that observation stream, not the state of the live pruned run (whose trajectory
diverges). For LongNav-R1 three inputs of the live arm cannot be reproduced on an unpruned cache:
the sparse filter of later frames saw the full embed DB, the FastV row is a softmax over every slot
rather than the survivors, and DART's layer-0 outputs attended to the full cache. Random and
DivPrune are exact (GPU smoke: equal to live flat workers at every step); FastV and DART are exact
through the first binding step. LongNav-R1's probe is leaky (surviving text and action slots keep
the K/V they computed with every frame visible) and LightNav-0's is not, and the action spaces
differ (4 actions vs 256 codes): **curve heights are not comparable across models**, only the
ordering of the methods and the trend over k within a model.

## Running

```bash
ARM=r1_prune_kl bash paper/prune_kl/run_r1.sh        # ~26 min, resumes through wandb
bash paper/prune_kl/run_lightnav.sh                  # ~2 h, needs the GPU alone, cannot resume
python paper/prune_kl/aggregate.py                   # longnav_vlm
/home/brabiei/miniconda3/envs/vln/bin/python paper/prune_kl/plot.py
```

Smoke runs go to an ignored directory: `OUT_ROOT_OVERRIDE=... MANIFEST=... ARM=...` on either
launcher (`task.wandb_project=<own project>` for LongNav-R1; `GATE_RECENCY=1` adds the recency
check arms for LightNav-0), then `aggregate.py --runs ... --out ... --manifest ... --arm MODEL=ARM
--allow-partial`.

## Code

| Piece | Where |
|---|---|
| LongNav-R1 shadow pruners and probe | `src/longnav/utils/prune_kl_probe.py` (`ShadowPruner`, `prune_kl_probe`), `vlm.prune_kl_probe` / `prune_kl_methods` / `prune_kl_budgets`, preset `+experiment=paper_prune_kl`; rows in `sequence.json` as `sup/prune_kl_*` |
| LightNav-0 measure | `baselines/lightnav_hm3d/eval_influence.py --measure prune_kl` (`shadow_step`, `--prune-kl-arms`, `--prune-kl-seed-episodes`); `INFLUENCE_MEASURE=prune_kl` in `baselines/run_lightnav.sh`; records in `influence/<label>.jsonl` |
| Loaders and statistics | `src/longnav/analysis/prune_kl.py` |
| Tests | `tests/prune_kl_smoke.py` (GPU; in `tests/run_inference_checks.py`), `tests/kv_prune_selectors.py`, `tests/influence_loaders_unit.py`, `baselines/tests/test_lightnav_influence.py` |

## Outputs

`paper_results/prune_kl/runs/{r1_prune_kl,lightnav_prune_kl}`, then under `results/`:
`prune_kl_steps.csv.gz` (every decision x method x k), `prune_kl_episode.csv`, `prune_kl_curve.csv`
(per model x method x k: pooled mean + episode-bootstrap CI, episode-balanced mean +- SEM, median,
flip rates, normalised KL, no-visual ceiling, all-decisions means), `prune_kl_summary.json`
(completeness and the checks below). Figure: `figures/prune_kl.{pdf,png}` -- pooled mean KL against
the frame budget, one line per method, one panel per model (own y axis), single column.
`plot.py --value <column> --name <basename>` draws another statistic without touching it.

## Checks (in `prune_kl_summary.json`)

* both runs against the recent-window runs of `paper_results/frame_influence_recent`: same
  decisions, and for LongNav-R1 bit-equal live probabilities, replayed full decision and no-visual KL;
* LightNav-0 random shadow against the masks the live `token_prune_flat/runs/lightnav_random_b<k>`
  runs logged: the tubelet layout depends on the step alone and the draw on (seed, episode, step),
  so the keep bits must be equal on every step both runs reached;
* recency check arms (smoke runs only) against the recent-window run's old-blocks-kept probes.

## Findings (2026-11-24, 10 episodes, all checks pass)

Pooled mean KL (nats) over the decisions where the budget binds; SR of the live flat run
(`paper_results/token_prune_flat`, 100 episodes) beside it.

**LongNav-R1** (1099 decisions; no-visual KL .032; live-vs-replay gap .0003 mean)

| k | Random | FastV | DivPrune | DART | recent window (k+1 frames) | SR Random / FastV / DivPrune / DART |
|---|---|---|---|---|---|---|
| 4 | .0012 | .0025 | .0045 | .0057 | .0009 | .42 / .29 / .02 / .02 |
| 8 | .0009 | .0010 | .0035 | .0041 | .0007 | .59 / .64 / .13 / .18 |
| 16 | .0008 | .0006 | .0019 | .0027 | .0007 | .74 / .75 / .35 / .41 |
| 32 | .0006 | .0005 | .0013 | .0015 | .0006 (K = 32) | .82 / .82 / .62 / .72 |

**LightNav-0** (1696 decisions; no-visual KL 10.3; replay exact)

| k | Random | FastV | DivPrune | DART | recent window, old blocks kept (k+2 frames) | SR Random / FastV / DivPrune / DART |
|---|---|---|---|---|---|---|
| 4 | 1.98 | 2.46 | 3.27 | 2.19 | 2.33 | .81 / .77 / .77 / .80 |
| 8 | 1.84 | 1.91 | 2.90 | 1.74 | 1.96 | .82 / .80 / .80 / .80 |
| 16 | 1.31 | 1.47 | 1.93 | 1.26 | 1.54 | .82 / .81 / .80 / .78 |
| 32 | 0.59 | 0.84 | 0.97 | 0.79 | 1.14 (K = 32) | .82 / .83 / .78 / .85 |

The recent-window column is the measure of `paper_results/frame_influence_recent` on the same
decisions (LightNav-0: the 1356-1496 decisions both runs probed, on which Random scores
1.94 / 1.83 / 1.35 / 0.59); at k = 32 the largest probed window, K = 32, stands in.

1. **No method is closer to the full-context decision than Random.** On LongNav-R1 FastV ties
   Random from k = 8 (above it at k = 4, in 9 of 10 episodes); DivPrune and DART are 2-5x higher at
   every budget (above Random in 9-10 of 10 episodes at k <= 16). On LightNav-0 DivPrune is the
   highest at every budget (above Random in 8-10 of 10 episodes); Random, FastV and DART lie within
   each other's intervals.
2. **On LongNav-R1 the KL ranks the methods as the closed-loop runs do**: Random ~ FastV well
   above DivPrune ~ DART in SR, and well below in KL, including FastV's deficit at k = 4. It does
   not predict the size of the loss: the worst arm (DART, k = 4) moves the decision by .006 nats
   (15 % of the no-visual KL, 1 % of actions changed) on the full-context trajectory, and its live
   run reaches SR .02. The probe is leaky and scores one decision; the live run compounds.
3. **On LightNav-0 the KL is large and SR does not follow it.** Even at k = 32 Random changes the
   top code on 29 % of decisions (8 % of executed actions; 64 % and 24 % at k = 4), yet every arm
   of the flat study holds SR .77-.85. As in the recent-window analysis, the 256-code distribution
   is far from converged at 32 frames' worth of history while navigation is unaffected.
4. **Against the recency window at the same budget**: on LongNav-R1 keeping the newest frames
   ties the best pruning method at every budget (below Random at k <= 8, level with Random and
   FastV from k = 16; under the leaky probe the recent frames carry the decision); on
   LightNav-0 Random is at or below the old-blocks-kept window at every budget.
5. KL falls monotonically with the budget for every method and model. Random and FastV on
   LongNav-R1 at k >= 16 are within 2-3x the live-vs-replay numerical gap.
