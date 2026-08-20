# KV-budget pruning results (2026-08-19)

**Question:** the context-window ablation showed 32 frames ≈ full history, competitive down
to 8. Can the 32-frame window be compressed to ~4-frame *memory cost* while keeping
performance? (Metric: actual resident KV slots, never nominal frames.)

**Mechanism:** `vlm.context_window_mode='prune'` (`src/longnav/utils/kv_prune.py`) — a hard
KV token budget (`vlm.kv_budget`) enforced every step on the reindex (pre-rotation)
substrate: lowest-importance unprotected slots are sliced out of the cache, the per-slot
mRoPE table, the slot metadata, and the sparse embed DB together. Survivors keep their
original positions (no renumbering). Protected: prompt prefix + last
`kv_prune_recent_turns` (2) turns. Importance modes: `attn` (EMA of the decision token's
attention row, captured free inside `pre_rope_attention_forward`) or `random`;
granularity `slot` or `turn` (whole-keyframe selection); optional `kv_prune_merge`
(CaM-style: fold dropped visual slots into their nearest kept slot by embed cosine).
Smoke: `tests/kv_prune_smoke.py`. Sweeps: `tools/run_kv_prune_ablation.sh`; analysis:
`tools/compare_runs.py`, `tools/plot_kv_budget_tradeoff.py`.

Budgets matched to measured window memory (reindex relogs with the new `sup/mean_kv_len`
metric): ~153 tokens/turn after sparse filtering, ~574-token pinned prefix →
B4=1187, B8=1799, B16=3022 slots. KV cost ≈ 0.109 MB/slot (28 layers × 8 KV heads ×
128 dim × 2 × bf16).

## Results — 100-ep HM3D-v2 split, deterministic, viz off

| run | SR | SPL | steps | mean kv | vs win32 KV | lat ms |
|---|---|---|---|---|---|---|
| win32 evict (reference) | 0.880 | 0.481 | 121 | ~5474 | 1× | 57.0 |
| full history | 0.840 | 0.468 | 111 | (grows) | — | 82.1 |
| **random-prune B16** | **0.860** | **0.427** | 152 | 2769 | **2.0×↓** | 50.4 |
| win16 evict | 0.820 | 0.419 | 152 | ~3024 | 1.8×↓ | 50.2 |
| **random-prune B8** | **0.810** | **0.406** | 171 | 1736 | **3.2×↓** | 46.3 |
| win8 evict | 0.790 | 0.383 | 184 | ~1799 | 3.0×↓ | 45.5 |
| **random-prune B4** | **0.730** | **0.324** | 224 | 1172 | **4.7×↓** | 44.2 |
| random-prune B4, no window pool | 0.707 | 0.329 | 227 | 1172 | 4.7×↓ | 44.2 |
| win4 evict | 0.700 | 0.309 | 239 | ~1187 | 4.6×↓ | 42.9 |
| keyframe-attn B8 (turn granularity) | 0.730 | 0.337 | 209 | 1641 | 3.3×↓ | 47.3 |
| slot-attn B8 (TOVA/H2O-style EMA) | 0.650 | 0.328 | 222 | 1745 | 3.1×↓ | 47.4 |
| keyframe-attn B4 | 0.650 | 0.289 | 265 | 1044 | 5.2×↓ | 45.2 |
| slot-attn B4 | 0.630 | 0.289 | 254 | 1172 | 4.7×↓ | 44.9 |
| attn B4 + merge (CaM-style) | 0.580 | 0.250 | 273 | 1173 | 4.7×↓ | 48.3 |
| reindex win4 (renumbered) | 0.450 | 0.236 | 308 | 1187 | 4.6×↓ | 43.5 |

Figure: `dump/longnav_eval/hm3d_v2_100_kv_budget_tradeoff.png`. Evict-baseline kv is the
fitted schedule (574 + 153·N); prune arms are measured. Peak process memory 4.6 GB at B4
vs 5.4 GB at win32 (13.3 GB worst-case at full); per-step latency −22% vs win32.

## Findings

1. **Random pruning of the sparse-deduped cache is the best compressor tried.** It sits
   above the recency-window curve at every matched budget (+0.02..+0.04 SR, individually
   n.s. at n=100 but consistent in direction on SR, SPL, and steps at all three budgets),
   and B16-random matches win32's *success rate* at half the KV. Why: the sparse filter
   already deduplicates patches, so cache content is near-uniformly informative — uniform
   sampling preserves spatial coverage. It also hard-caps memory (kv_max = budget exactly),
   which windows do not (win32's kv peaked at 8780 slots on one scene).
2. **Attention-salience selection is *worse than random*** (B4: 0.630 vs 0.730, paired
   p=0.06; same ordering at B8). The decision row concentrates on currently-relevant
   content — redundant with the protected recent frames — and discards the diffuse
   coverage needed later. Keyframe (turn-granular) selection closes part of the gap at B8
   (0.730) but never reaches random.
3. **KV merging hurts** (0.580/0.250): importance-weighted averaging of K/V across
   distinct geometry blurs it, even pre-rotation.
4. **Positional holes beat renumbering at extreme compression:** reindex win4 collapses to
   0.450 SR vs evict win4's 0.700 — the evict≈reindex parity (win8–64) breaks at win4.
   The prune mode keeps original positions, which this validates.
5. **The 32-frame pool is optional for random pruning** (no-window ≈ windowed): per-step
   budget re-sampling already imposes geometric attrition on old slots.
6. **A hard information bottleneck remains.** No arm reaches win32 SPL at B4/B8 memory
   (B8-random concedes −0.076 SPL, p=0.002). Consistent with the recompute-leakage
   result: leak-free win4 SR is 0.43, so a large share of every compressed arm's
   performance rides on eviction leakage through surviving K/V, and SPL keeps improving
   with genuinely visible context.

## Practical recommendation

`context_window_mode=prune, kv_prune_importance=random, kv_budget=1800–3000` gives
0.81–0.86 SR at 2–3.2× less KV than win32 with lower latency; use B≈1187 only when memory
is the binding constraint. Learned/trained compression (fine-tuning under the budget) is
the natural next step; selection heuristics alone appear tapped out.

## Reproduction

```bash
# baselines with kv metrics
MODE=reindex PREFIX=hm3d_v2_100_reindexv2win WINDOWS="32 4" bash tools/run_context_window_ablation.sh
# main arms
BUDGETS="1187 1799 3022" WINDOW=32 IMPORTANCE=random PREFIX=hm3d_v2_100_prunerand bash tools/run_kv_prune_ablation.sh
BUDGETS="1187 1799" WINDOW=32 bash tools/run_kv_prune_ablation.sh                      # attn slots
GRANULARITY=turn BUDGETS="1187 1799" WINDOW=32 PREFIX=hm3d_v2_100_pruneturn bash tools/run_kv_prune_ablation.sh
MERGE=true BUDGETS="1187" WINDOW=32 PREFIX=hm3d_v2_100_prunemerge bash tools/run_kv_prune_ablation.sh
# analysis
python tools/compare_runs.py <runs...> --baseline <run>
/home/brabiei/miniconda3/envs/vln/bin/python tools/plot_kv_budget_tradeoff.py
```
