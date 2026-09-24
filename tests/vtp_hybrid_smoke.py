"""Smoke test for the two protocols of paper/hybrid_prune.

Hybrid: K-frame window (frame cut) + PyramidDrop-like attention stages below the information
horizon + a random last stage with its own, aggressive budget (kv_prune_stage_importance,
kv_prune_tail_budget_frames). PyramidDrop without a window: kv_prune_stages on the full episode
context. Four things have to hold:

  1. Hybrid stages (CPU) -- nested slot sets and per-layer cache lengths; budgets are the
     geometric schedule over the attention stages plus min(tail, B') for the random stage; every
     stage holds min(budget, available) visual slots; the tail binds from the second frame on,
     before the schedule does; the random stage needs no decision row, ignores the rows it is not
     given and depends on (seed, episode label, step) only.
  2. Staged pruning without a window (CPU) -- budgets follow the running mean over every history
     frame, text is never pruned, the master cache never shrinks, the log is the scalar only.
  3. Whole-model budget (CPU) -- kv_prune_budget_scope='model' with a stage at layer 0 and the
     schedule anchored at what the cache holds (kv_prune_schedule_anchor='pool'): the
     history visual slots AVERAGED OVER ALL LAYERS meet k frames' worth (never above it, close to
     it once pruning binds), the master cache compacts with stage 0, stages stay nested; both
     with the window + random tail (hybrid) and without a window (PyramidDrop).
  4. Validation -- the constructor refuses malformed stage knobs.
  5. GPU -- every arm runs with finite probabilities, per-layer lengths equal to each stage's
     alive count, a synced embed DB and the rows of its attention stages only; the whole-model
     arms hold at most k frames' worth.

Run inside longnav_vlm:
    python tests/vtp_hybrid_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_frames_smoke import LEAD, PREFIX  # noqa: E402
from ctx_recompute_smoke import make_images, release  # noqa: E402
from kv_prune_smoke import MODEL_ID, N_STEPS  # noqa: E402
from vtp_prune_smoke import COMMON, feed, prune_worker, run_logged, visual_chunk  # noqa: E402
from longnav.utils.kv_prune import frame_budget, model_budgets, pyramid_budgets  # noqa: E402
from longnav.utils.vlm_worker import VLMWorker  # noqa: E402

STAGES = [range(2, 4), range(4, 6), range(6, 8)]
SELECTORS = ["attn", "attn", "random"]
COUNTS = [6, 9, 5, 8, 7, 10, 6]


def hybrid_worker(window, k, tail, label="scene_0"):
    worker = prune_worker(window, n_layers=8, stages=STAGES, importance="attn",
                          kv_prune_budget_frames=float(k), kv_prune_stages=[2, 4, 6],
                          kv_prune_stage_importance=list(SELECTORS),
                          kv_prune_tail_budget_frames=float(tail), kv_prune_seed_per_step=True,
                          _episode_label=label)
    # Rows are captured for the attention stages only (what _score_layer_range installs).
    worker._reindex_state.configure(pruned_layers=None, score_layers=(1, 3), n_layers=8, stages=STAGES)
    return worker


def drive_hybrid(worker, window, k, tail, row_seed, failures=None, verbose=False):
    """Feed COUNTS; returns the deepest stage's global ids per turn."""
    gen = torch.Generator().manual_seed(row_seed)
    next_id, visual_ids, tails = 0, set(), []
    for turn, n_vis in enumerate(COUNTS):
        n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
        ids = torch.arange(next_id, next_id + n)
        next_id += n
        new_vis = set(ids[vis].tolist())
        visual_ids |= new_vis
        st = worker._reindex_state
        layers = worker.past_key_values.layers if worker.past_key_values is not None else None
        before = [set() if layers is None else set(layers[r.start].ids()) for r in STAGES]
        master = st.master_len() + n
        rows = {1: torch.rand(master, generator=gen), 3: torch.rand(master, generator=gen)}
        feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD, rows=rows)
        ps = worker._kv_prune_state
        first_keep = max(0, ps.n_turns - window)
        h = ps.n_turns - first_keep
        total = sum(COUNTS[first_keep:ps.n_turns])
        cap = None if h <= k else int(round(k * total / h))
        sched = [None, None] if cap is None else pyramid_budgets(total, cap, 2)
        tail_cap = None if h <= tail else int(round(tail * total / h))
        if tail_cap is not None and sched[-1] is not None:
            tail_cap = min(tail_cap, sched[-1])
        budgets = sched + [tail_cap]
        layers = worker.past_key_values.layers
        lengths = worker._kv_lengths()
        alive = [st.stage_alive(s) for s in range(3)]
        vis_alive = [int((ps.is_visual & a).sum()) for a in alive]
        upstream = set(layers[0].ids())  # the master cache: the window's tokens
        expect = []
        for s, rng in enumerate(STAGES):
            available = (before[s] | new_vis) & upstream & visual_ids
            expect.append(len(available) if budgets[s] is None else min(budgets[s], len(available)))
            upstream = set(layers[rng.start].ids())
        tails.append(sorted(set(layers[6].ids()) & visual_ids))
        if verbose:
            print(f"  turn {turn}: {h} frames, budgets {budgets}, visual alive per stage {vis_alive} "
                  f"(expect {expect}), layer lengths {lengths}")
        if failures is None:
            continue
        if any(bool((alive[s + 1] & ~alive[s]).any()) for s in range(2)):
            failures.append(f"turn {turn}: a deeper stage holds a slot a shallower one dropped")
        want = [st.master_len()] * 2 + [int(a.sum()) for a in alive for _ in range(2)]
        if lengths != want:
            failures.append(f"turn {turn}: layer lengths {lengths}, expected {want}")
        if vis_alive != expect:
            failures.append(f"turn {turn}: stage visual counts {vis_alive}, expected {expect} (budgets {budgets})")
        if worker._prune_stage_visual != vis_alive:
            failures.append(f"turn {turn}: logged stage counts {worker._prune_stage_visual} vs {vis_alive}")
        if turn == 1 and (vis_alive[1] != sum(COUNTS[:2]) or vis_alive[2] >= vis_alive[1]):
            failures.append(f"turn 1: the tail must bind before the schedule does, got {vis_alive}")
        if frame_budget(ps, first_keep, float(tail)) != (None if h <= tail else int(round(tail * total / h))):
            failures.append(f"turn {turn}: frame_budget disagrees with the tail budget")
    return tails


# --- check 1: hybrid stages (CPU) --------------------------------------------------------
def check_hybrid(window=4, k=2, tail=1):
    failures = []
    base = drive_hybrid(hybrid_worker(window, k, tail), window, k, tail, row_seed=3,
                        failures=failures, verbose=True)
    # The random stage draws among the attention stages' survivors, so it is compared where those
    # agree: same rows + same label must reproduce; another label must not.
    again = drive_hybrid(hybrid_worker(window, k, tail), window, k, tail, row_seed=3)
    other = drive_hybrid(hybrid_worker(window, k, tail, label="scene_1"), window, k, tail, row_seed=3)
    print(f"  random stage: reproducible {again == base}, label-dependent {other != base}")
    if again != base:
        failures.append("the random stage is not reproducible under the same seed / label / rows")
    if other == base:
        failures.append("the random stage ignores the episode label")
    # k == tail: the tail budget equals the schedule's last stage, so it never drops anything more.
    worker = hybrid_worker(window, 1, 1)
    drive_hybrid(worker, window, 1, 1, row_seed=5)
    st = worker._reindex_state
    if not torch.equal(st.stage_alive(1), st.stage_alive(2)):
        failures.append("k = tail: the random stage dropped slots the last attention stage holds")
    return failures


# --- check 2: staged pruning without a window (CPU) ---------------------------------------
def check_staged_flat(k=2):
    failures = []
    worker = prune_worker(None, n_layers=8, stages=STAGES, importance="attn",
                          kv_prune_budget_frames=float(k), kv_prune_stages=[2, 4, 6],
                          kv_prune_log_masks=False)
    worker.context_cut = "header"
    gen = torch.Generator().manual_seed(7)
    next_id = 0
    for turn, n_vis in enumerate(COUNTS):
        n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
        ids = torch.arange(next_id, next_id + n)
        next_id += n
        st = worker._reindex_state
        master = st.master_len() + n
        rows = {l: torch.rand(master, generator=gen) for l in (1, 3, 5)}
        feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD, rows=rows)
        ps = worker._kv_prune_state
        h, total = turn + 1, sum(COUNTS[:turn + 1])
        budgets = [total] * 3 if h <= k else pyramid_budgets(total, int(round(k * total / h)), 3)
        alive = [st.stage_alive(s) for s in range(3)]
        vis_alive = [int((ps.is_visual & a).sum()) for a in alive]
        text_alive = [int((~ps.is_visual & a).sum()) for a in alive]
        exp_text = PREFIX + sum((0 if t == 0 else LEAD) + 4 for t in range(h))
        print(f"  turn {turn}: {h} frames, budgets {budgets}, visual alive {vis_alive}, master {st.master_len()}")
        # Permanent deletion under a shrinking schedule can leave a stage a few slots short.
        if any(v > b for v, b in zip(vis_alive, budgets)) or vis_alive != sorted(vis_alive, reverse=True):
            failures.append(f"turn {turn}: stage visual counts {vis_alive} exceed budgets {budgets}")
        if vis_alive[-1] != budgets[-1] and h > k and turn == k:
            failures.append(f"turn {turn}: first pruning step keeps {vis_alive[-1]}, budget {budgets[-1]}")
        if text_alive != [exp_text] * 3:
            failures.append(f"turn {turn}: text slots {text_alive}, expected {exp_text} on every stage")
        if st.master_len() != next_id or worker._kv_lengths()[:2] != [next_id] * 2:
            failures.append(f"turn {turn}: the master cache shrank ({st.master_len()} of {next_id})")
        if worker._vtp_log != {"mean/vtp_n_new": n_vis}:
            failures.append(f"turn {turn}: mask-less log is {worker._vtp_log}")
        if worker.past_image_embeds[0].shape[0] != total:
            failures.append(f"turn {turn}: embed DB holds {worker.past_image_embeds[0].shape[0]} of {total}")
    return failures


# --- check 3: whole-model budget with a stage at layer 0 (CPU) -----------------------------
MODEL_STAGES = [range(0, 2), range(2, 4), range(4, 6), range(6, 8)]
LONG_COUNTS = [6, 9, 5, 8, 7, 10, 6, 9, 8, 7, 6, 10]


def check_model_budget(k=2):
    failures = []
    arms = {"hybrid": dict(window=5, tail=1.0, selectors=["attn", "attn", "attn", "random"], rows=(0, 1, 3)),
            "pyramiddrop_flat": dict(window=None, tail=None, selectors=None, rows=(0, 1, 3, 5))}
    for name, cfg in arms.items():
        worker = prune_worker(cfg["window"], n_layers=8, stages=MODEL_STAGES, importance="attn",
                              kv_prune_budget_frames=float(k), kv_prune_stages=[0, 2, 4, 6],
                              kv_prune_stage_importance=cfg["selectors"], kv_prune_tail_budget_frames=cfg["tail"],
                              kv_prune_budget_scope="model", kv_prune_schedule_anchor="pool",
                              kv_prune_seed_per_step=True, kv_prune_log_masks=False, _episode_label="scene_0")
        if cfg["window"] is None:
            worker.context_cut = "header"
        worker._reindex_state.configure(pruned_layers=None, score_layers=cfg["rows"], n_layers=8,
                                        stages=MODEL_STAGES)
        gen = torch.Generator().manual_seed(11)
        next_id, worst = 0, 0.0
        for turn, n_vis in enumerate(LONG_COUNTS):
            n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
            ids = torch.arange(next_id, next_id + n)
            next_id += n
            st = worker._reindex_state
            master = st.master_len() + n
            feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD,
                 rows={l: torch.rand(master, generator=gen) for l in cfg["rows"]})
            ps = worker._kv_prune_state
            first_keep = 0 if cfg["window"] is None else max(0, ps.n_turns - cfg["window"])
            scope = LONG_COUNTS[first_keep:ps.n_turns]
            h, total = len(scope), sum(scope)
            cap = None if h <= k else int(round(k * total / h))
            alive = [st.stage_alive(s) for s in range(4)]
            vis_alive = [int((ps.is_visual & a).sum()) for a in alive]
            held = sum(2 * v for v in vis_alive) / 8
            lengths = worker._kv_lengths()
            print(f"  {name} turn {turn}: {h} frames in scope, budget {cap}, visual per stage {vis_alive}, "
                  f"whole-model mean {held:.1f} ({worker._prune_visual_model_frames:.2f} frames)")
            if any(bool((alive[s + 1] & ~alive[s]).any()) for s in range(3)):
                failures.append(f"{name} turn {turn}: a deeper stage holds a slot a shallower one dropped")
            if lengths != [int(a.sum()) for a in alive for _ in range(2)] or lengths[0] != st.master_len():
                failures.append(f"{name} turn {turn}: layer lengths {lengths} vs master {st.master_len()}")
            if not bool(alive[0].all()) or worker.past_image_embeds[0].shape[0] != vis_alive[0]:
                failures.append(f"{name} turn {turn}: the master cache did not compact with stage 0")
            # Text is never a candidate: every stage holds the prefix + each in-scope turn's text
            # (minus the boundary lead the frame cut retires).
            exp_text = PREFIX + sum((0 if t == 0 else LEAD) + 4 for t in range(first_keep, ps.n_turns)) \
                - (LEAD if first_keep > 0 else 0)
            text_alive = [int((~ps.is_visual & a).sum()) for a in alive]
            if text_alive != [exp_text] * 4:
                failures.append(f"{name} turn {turn}: text slots {text_alive}, expected {exp_text} on every stage")
            if abs(worker._prune_visual_model - held) > 1e-9:
                failures.append(f"{name} turn {turn}: logged whole-model mean {worker._prune_visual_model} vs {held}")
            if cap is not None:
                # Rounding moves each stage by < 1 slot; permanent deletion can only underfill.
                if held > cap + 1:
                    failures.append(f"{name} turn {turn}: whole-model mean {held:.1f} exceeds the budget {cap}")
                worst = max(worst, (cap - held) / cap)
        print(f"  {name}: largest shortfall below the budget {100 * worst:.0f}%")
        # Anchored at what the cache holds, the schedule is stationary: late in the episode the deepest
        # scheduled stage still holds a real share of the budget (anchored at the ORIGINAL count of a
        # no-window scope it decays towards nothing as the episode grows -- 2 of 15 slots by turn 11).
        deepest = vis_alive[-1] if cfg["tail"] is None else vis_alive[-2]
        if deepest < 0.4 * cap:
            failures.append(f"{name}: the deepest scheduled stage holds {deepest} of a {cap}-slot budget")
        if worst > 0.25:
            failures.append(f"{name}: the schedule leaves {100 * worst:.0f}% of the whole-model budget unused")
    # A tail larger than the last scheduled stage is clamped to it; the solver must spend that slack.
    t = model_budgets(6200, 400, [7, 7, 6], tail=(8, 200))
    if t[-1] >= 200 or abs((7 * t[0] + 7 * t[1] + 6 * t[2] + 8 * min(200, t[-1])) / 28 - 400) > 1:
        failures.append(f"model_budgets under-uses a small budget when the tail is clamped: {t}")
    b = model_budgets(6200, 1600, [7, 7, 6], tail=(8, 200))
    if abs((7 * b[0] + 7 * b[1] + 6 * b[2] + 8 * 200) / 28 - 1600) > 1 or b != sorted(b, reverse=True):
        failures.append(f"model_budgets schedule {b} misses a 1600-slot whole-model budget")
    if model_budgets(1000, 1600, [7] * 4) != [1000] * 4 or model_budgets(20000, 1600, [7] * 3, n_unpruned=7) != [0] * 3:
        failures.append("model_budgets edge cases (under budget / unreachable)")
    return failures


# --- check 4: constructor refusals ---------------------------------------------------------
def check_validation():
    failures = []
    base = dict(load_model=False, use_sparse=True, context_frames=32, kv_prune_importance="attn",
                kv_prune_budget_frames=8.0, **COMMON)
    bad = {
        "selector count": dict(kv_prune_stages=[7, 14, 20], kv_prune_stage_importance=["attn", "random"]),
        "selector name": dict(kv_prune_stages=[7, 14, 20], kv_prune_stage_importance=["attn", "attn", "dart"]),
        "tail above k": dict(kv_prune_stages=[7, 14, 20], kv_prune_tail_budget_frames=9.0),
        "tail without a budget": dict(kv_prune_stages=[7, 14, 20], kv_prune_tail_budget_frames=1.0,
                                      kv_prune_budget_frames=None),
        "tail, one stage": dict(kv_prune_stages=[20], kv_prune_tail_budget_frames=1.0),
        "selectors without stages": dict(kv_prune_stage_importance=["random"]),
        "scope without stages": dict(kv_prune_budget_scope="model"),
        "scope name": dict(kv_prune_stages=[0, 7, 14, 20], kv_prune_budget_scope="layers"),
        "anchor name": dict(kv_prune_stages=[0, 7, 14, 20], kv_prune_schedule_anchor="cache"),
        "anchor without stages": dict(kv_prune_schedule_anchor="pool"),
        "model scope without a budget": dict(kv_prune_stages=[0, 7, 14, 20], kv_prune_budget_scope="model",
                                             kv_prune_budget_frames=None),
        "tail without stages": dict(kv_prune_tail_budget_frames=1.0),
    }
    for name, cfg in bad.items():
        try:
            VLMWorker(model_id=MODEL_ID, **{**base, **cfg})
        except ValueError as exc:
            ok = any(knob in str(exc) for knob in ("kv_prune_stage_importance", "kv_prune_tail_budget_frames",
                                                    "kv_prune_budget_scope", "kv_prune_schedule_anchor"))
            print(f"  {name}: refused ({'ours' if ok else 'OTHER: ' + str(exc)[:60]})")
            if not ok:
                failures.append(f"{name}: refused for another reason: {exc}")
        else:
            failures.append(f"{name}: accepted")
    return failures


# --- check 5: GPU --------------------------------------------------------------------------
FLAT = {**COMMON, "context_cut": "header", "kv_prune_log_masks": False}
ARMS = {
    "hybrid_model": dict(context_frames=5, kv_prune_budget_frames=2.0, kv_prune_stages=[0, 7, 14, 20],
                         kv_prune_stage_importance=["attn", "attn", "attn", "random"],
                         kv_prune_tail_budget_frames=1.0, kv_prune_budget_scope="model",
                         kv_prune_schedule_anchor="pool", **COMMON),
    "pyramiddrop_flat_model": dict(context_frames=None, kv_prune_budget_frames=2.0,
                                   kv_prune_stages=[0, 7, 14, 21], kv_prune_budget_scope="model",
                                   kv_prune_schedule_anchor="pool", **FLAT),
    "hybrid": dict(context_frames=4, kv_prune_budget_frames=2.0, kv_prune_stages=[7, 14, 20],
                   kv_prune_stage_importance=["attn", "attn", "random"], kv_prune_tail_budget_frames=1.0,
                   **COMMON),
    "pyramiddrop_flat": dict(context_frames=None, kv_prune_budget_frames=1.0, kv_prune_stages=[7, 14, 21],
                             **{**COMMON, "context_cut": "header", "kv_prune_log_masks": False}),
}


def check_gpu(images):
    failures = []
    for name, cfg in ARMS.items():
        worker = VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
                           kv_prune_importance="attn", **cfg)
        probs, _, lengths = run_logged(worker, images)
        ps, st = worker._kv_prune_state, worker._reindex_state
        lens, master = lengths[-1], st.master_len()
        stage_vis = worker._prune_stage_visual
        frames = worker._prune_visual_model_frames
        print(f"  {name}: layers 0/7/14/{st.stages[-1].start} hold {lens[0]}/{lens[7]}/{lens[14]}/"
              f"{lens[st.stages[-1].start]}, visual per stage {stage_vis}, rows at {sorted(st.layer_rows)}, "
              f"whole-model {frames:.2f} frames")
        if any(not np.all(np.isfinite(p)) for p in probs):
            failures.append(f"{name}: non-finite probabilities")
        for l, n in enumerate(lens):
            stage = st.layer_stage(l)
            want = master if stage is None else int(st.stage_alive(stage).sum())
            if n != want:
                failures.append(f"{name}: layer {l} holds {n}, expected {want}")
        if not lens[0] > lens[-1] or stage_vis != sorted(stage_vis, reverse=True) \
                or len(stage_vis) != len(cfg["kv_prune_stages"]):
            failures.append(f"{name}: stages are not nested in size ({lens[0]} .. {lens[-1]}, {stage_vis})")
        if worker.past_image_embeds[0].shape[0] != int(ps.is_visual.sum()):
            failures.append(f"{name}: embed DB desynced")
        want_rows = sorted({max(l - 1, 0) for l, m in zip(
            cfg["kv_prune_stages"], cfg.get("kv_prune_stage_importance") or ["attn"] * 9) if m == "attn"})
        if sorted(st.layer_rows) != want_rows:
            failures.append(f"{name}: decision rows captured at {sorted(st.layer_rows)}, expected {want_rows}")
        if name.endswith("_model"):
            if lens[0] != master or not frames <= 2.0 + 0.05:
                failures.append(f"{name}: holds {frames:.2f} frames' worth over the whole model (budget 2), "
                                f"layer 0 {lens[0]} vs master {master}")
        if name == "hybrid":
            h = min(ps.n_turns, 3)  # context_frames=4 holds 3 past turns
            tail = int(round(sum(ps.visual_per_turn[-h:]) / h))
            if not 0 < stage_vis[-1] <= tail:
                failures.append(f"hybrid: the random stage holds {stage_vis[-1]} visual slots, budget {tail}")
        release(worker)
    return failures


def main():
    failures = []
    print("check 1: hybrid stages (attention schedule + random tail)")
    failures += check_hybrid()
    print("check 2: staged pruning without a window")
    failures += check_staged_flat()
    print("check 3: whole-model budget with a stage at layer 0")
    failures += check_model_budget()
    print("check 4: constructor refusals")
    failures += check_validation()
    if torch.cuda.is_available() and "--cpu" not in sys.argv:
        print("check 5: GPU arms")
        failures += check_gpu(make_images(N_STEPS))
    else:
        print("check 5 skipped (no GPU)")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: hybrid stages, staged flat, whole-model budget, refusals, GPU arms")


if __name__ == "__main__":
    main()
