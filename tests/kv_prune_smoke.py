"""Smoke test for context_window_mode='prune' (importance-based KV-slot pruning).

'prune' enforces a hard KV token budget on the reindex (pre-rotation) substrate: each
step the lowest-importance unprotected slots are sliced out of the cache, the per-slot
position table, the slot metadata, and the sparse embed db together. Importance is an EMA
of the decision token's attention row, captured inside pre_rope_attention_forward.

Five things have to hold:

  1. Forced-schedule parity -- with an effectively infinite budget, prune's window drop
     (turn-id based, since budget drops break evict's contiguous-cut arithmetic) must
     retire exactly the same tokens on exactly the same steps as evict, and survivors
     must keep their ORIGINAL positions (holes expected -- prune never renumbers) with
     worker.offset never rebased. CPU only, no model.

  2. Budget bookkeeping -- with an active budget and injected scores: the cache, position
     table, metadata, and embed db stay 1:1; the prefix and the last kv_prune_recent_turns
     turns always survive; every kept unprotected slot outranks every dropped one; the db
     rows track the visual survivors across repeated prunes. CPU only, no model.

  3. Merge math -- merge_dropped_visual folds a dropped slot into its nearest kept slot
     as an importance-weighted average and moves its importance mass. CPU only.

  4. Fidelity -- with a budget nothing ever exceeds, prune must reproduce the stock
     forward (capture armed but inert), same tolerance as the reindex smoke.

  5. Non-vacuity -- with a budget that actually prunes, the cache must stay at the
     budget and the decision must move well beyond the arithmetic floor.

Checks 1-3 run anywhere; 4 and 5 need the GPU and the HF model. Run inside longnav_vlm:
    python tests/kv_prune_smoke.py
"""
import gc
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.kv_prune import KVPruneState, merge_dropped_visual
from longnav.utils.pre_rope import ReindexState
from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
WIDE_BUDGET = 10 ** 9
NARROW_BUDGET = 280       # ~70 slots/turn on 240x320 noise frames -> prunes from ~step 4
FIDELITY_TOL = 2e-2
DIVERGENCE_MARGIN = 3.0

START = [
    {"role": "user", "content": [{"type": "text", "text": "Find the chair."}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
TURN = [
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]


def make_images(n, h=240, w=320):
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def build_prune(budget, window=None, merge=False):
    return VLMWorker(
        model_id=MODEL_ID,
        attn_impl="sdpa",
        dtype="bfloat16",
        use_sparse=True,
        context_window=window,
        context_window_mode="prune",
        kv_budget=budget,
        kv_prune_merge=merge,
    )


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def run_probs(worker, images):
    out = []
    for idx, image in enumerate(images):
        probs, _, _ = worker.infer_probs(START if idx == 0 else TURN, [image])
        out.append(np.asarray(probs))
    return out


# --- checks 1-2: bookkeeping on synthetic turns, CPU only ----------------------------
# Every token is its own global id AND its own mRoPE position, so cache slices, the
# position table, and the embed db can all be compared as plain lists of integers.
PARITY_PREFIX = 5
PARITY_HEADER = 3
PARITY_WINDOW = 2
PARITY_TURNS = [11, 7, 9, 6, 8, 7, 10]


class FakeLayer:
    def __init__(self, ids):
        self.keys = ids.reshape(1, 1, -1, 1)
        self.values = self.keys.clone()

    def ids(self):
        return self.keys.reshape(-1).tolist()


class FakeCache:
    def __init__(self, ids):
        self.layers = [FakeLayer(ids)]

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def bare_worker(mode, window, budget=WIDE_BUDGET, importance="random", recent_turns=2,
                granularity="slot"):
    """A VLMWorker with just enough state to drive the eviction/prune paths, no model."""
    worker = VLMWorker.__new__(VLMWorker)
    worker.kv_prune_granularity = granularity
    worker.context_window = window
    worker.context_window_mode = mode
    worker.prefix_ids = list(range(PARITY_HEADER))
    worker._prefix_len = PARITY_PREFIX
    worker.use_sparse = False
    worker.past_image_embeds = None
    worker.attn_probe = None
    worker.past_key_values = None
    worker._frame_records = []
    worker._abs_bounds = []
    worker._vis_counts = []
    worker._dropped = 0
    worker._n_evicted = 0
    worker._window_recs = []
    worker._prefix_rec = None
    worker._boundary_rec = None
    worker._inference_ctx = torch.no_grad
    worker._reindex_state = ReindexState() if mode in ("reindex", "prune") else None
    worker._kv_prune_state = KVPruneState() if mode == "prune" else None
    worker.kv_budget = budget
    worker.kv_prune_recent_turns = recent_turns
    worker.kv_prune_ema_beta = 0.0
    worker.kv_prune_merge = False
    worker.kv_prune_importance = importance
    worker.offset = 0
    return worker


def feed_turn(worker, ids):
    """Append one synthetic pure-text turn (global positions == token ids) and prune."""
    live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
    worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
    if worker._reindex_state is not None:
        pos = torch.arange(worker.offset, worker.offset + len(ids))
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += len(ids)
    if worker.context_window_mode == "prune":
        worker._apply_kv_prune()
    else:
        worker._apply_context_window()
    return worker.past_key_values.layers[0].ids()


def check_forced_schedule_parity():
    """Prune with an infinite budget must reproduce evict's window schedule exactly."""
    evict = bare_worker("evict", PARITY_WINDOW)
    prune = bare_worker("prune", PARITY_WINDOW)
    failures = []

    next_id, total = 0, 0
    for step, length in enumerate(PARITY_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        total += length
        a = feed_turn(evict, ids)
        b = feed_turn(prune, ids)
        print(f"  step {step}: {len(a)} tokens retained ({'match' if a == b else 'MISMATCH'})")
        if a != b:
            failures.append(f"step {step}: evict retains {a} but prune retains {b}")

        st = prune._reindex_state
        positions = st.pos_table[0, 0].tolist()
        if positions != b:
            failures.append(f"step {step}: survivors' positions {positions} are not their "
                            f"original ids {b}; prune must never renumber")
        if prune.offset != total:
            failures.append(f"step {step}: offset {prune.offset} was rebased (expected {total})")
        ps = prune._kv_prune_state
        if not (len(b) == st.pos_table.shape[-1] == ps.turn_id.numel()
                == ps.importance.numel() == ps.is_visual.numel()):
            failures.append(f"step {step}: cache({len(b)}) / table({st.pos_table.shape[-1]}) / "
                            f"metadata({ps.turn_id.numel()}) lengths desynced")

    if not evict._n_evicted:
        failures.append("the window never evicted, so nothing was tested")
    for field in ("_dropped", "_n_evicted"):
        got = {m: getattr(w, field) for m, w in (("evict", evict), ("prune", prune))}
        if got["evict"] != got["prune"]:
            failures.append(f"{field} diverged: {got}")
    if prune._kv_prune_state.n_budget_dropped:
        failures.append("an infinite budget dropped slots; forced/budget accounting is mixed up")
    return failures


BUDGET_TURNS = [12, 9, 10, 8, 11]
BUDGET = 22
RECENT = 1


def check_budget_bookkeeping():
    """Injected scores: protection, ranking, and db/table/metadata sync under real prunes."""
    torch.manual_seed(0)
    worker = bare_worker("prune", window=None, budget=BUDGET, importance="attn",
                         recent_turns=RECENT)
    worker.use_sparse = True
    worker.language_model = SimpleNamespace(visual_pos_masks=None)
    worker.past_image_embeds = None
    failures = []

    next_id = 0
    vis_by_id = {}
    prev_kept = []
    for step, length in enumerate(BUDGET_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        # Odd token ids are 'visual'; their db row carries the id so drift is visible.
        vis_row = (ids % 2).bool()
        for i, v in zip(ids.tolist(), vis_row.tolist()):
            vis_by_id[i] = v
        worker.language_model.visual_pos_masks = vis_row.reshape(1, -1)
        db_new = ids[vis_row].float().reshape(-1, 1)
        if worker.past_image_embeds is None:
            worker.past_image_embeds = [db_new]
        else:
            worker.past_image_embeds[0] = torch.cat([worker.past_image_embeds[0], db_new])

        live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
        worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
        pos = torch.arange(worker.offset, worker.offset + length)
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += length
        # Injected decision row: importance == a fixed pseudo-random score per token id,
        # so ranking is deterministic and checkable across steps (beta=0 -> no EMA mixing).
        kv_len = worker.past_key_values.get_seq_length()
        all_ids = torch.tensor(worker.past_key_values.layers[0].ids())
        score = torch.sin(all_ids.float() * 12.9898) * 0.5 + 0.5
        worker._reindex_state.score_sum = score
        worker._reindex_state.score_layers = 1
        worker._apply_kv_prune()

        kept = worker.past_key_values.layers[0].ids()
        ps = worker._kv_prune_state
        st = worker._reindex_state
        n_turns = step + 1
        # 1:1 sync
        if not (len(kept) == st.pos_table.shape[-1] == ps.turn_id.numel()):
            failures.append(f"step {step}: cache/table/metadata desynced")
        if st.pos_table[0, 0].tolist() != kept:
            failures.append(f"step {step}: positions no longer match survivors' original ids")
        # budget respected (floor is far below BUDGET here)
        if len(kept) > max(BUDGET, PARITY_PREFIX):
            failures.append(f"step {step}: cache holds {len(kept)} slots over budget {BUDGET}")
        # prefix + recent turns survive
        prefix_ids_expected = list(range(PARITY_PREFIX))
        if kept[:PARITY_PREFIX] != prefix_ids_expected:
            failures.append(f"step {step}: prefix slots {prefix_ids_expected} not intact: {kept[:8]}")
        recent_lo = sum(BUDGET_TURNS[:n_turns - RECENT])
        recent_expected = list(range(recent_lo, next_id))
        if [i for i in kept if i >= recent_lo] != recent_expected:
            failures.append(f"step {step}: recent-turn slots incomplete")
        # ranking: THIS step's drops must all rank below every kept unprotected slot
        # (a slot protected at drop time can outlive a higher-scored slot dropped earlier,
        # so the greedy property only holds per step)
        dropped_now = sorted(set(prev_kept) | set(ids.tolist()))
        dropped_now = [i for i in dropped_now if i not in set(kept)]
        unprotected_kept = [i for i in kept if i >= PARITY_PREFIX and i < recent_lo]
        if dropped_now and unprotected_kept:
            def score_of(i):
                return float(torch.sin(torch.tensor(float(i)) * 12.9898) * 0.5 + 0.5)
            worst_kept = min(score_of(i) for i in unprotected_kept)
            best_dropped = max(score_of(i) for i in dropped_now)
            if best_dropped > worst_kept + 1e-6:
                failures.append(f"step {step}: dropped a higher-importance slot "
                                f"({best_dropped:.4f}) while keeping {worst_kept:.4f}")
        prev_kept = kept
        # db rows track visual survivors exactly
        db = worker.past_image_embeds[0].reshape(-1).long().tolist()
        vis_kept = [i for i in kept if vis_by_id[i]]
        if db != vis_kept:
            failures.append(f"step {step}: embed db {db} != visual survivors {vis_kept}")
        print(f"  step {step}: {len(kept)} slots kept, db rows {len(db)}, "
              f"budget_dropped so far {ps.n_budget_dropped}")

    if not worker._kv_prune_state.n_budget_dropped:
        failures.append("the budget never dropped anything, so nothing was tested")
    return failures


def check_turn_granularity():
    """granularity='turn': whole turns drop as units; survivors' successors keep a header."""
    torch.manual_seed(0)
    worker = bare_worker("prune", window=None, budget=BUDGET, importance="attn",
                         recent_turns=RECENT, granularity="turn")
    failures = []

    next_id = 0
    turn_slots = {}
    for step, length in enumerate(BUDGET_TURNS):
        ids = torch.arange(next_id, next_id + length)
        turn_slots[step] = ids.tolist()
        next_id += length
        live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
        worker.past_key_values = FakeCache(torch.tensor(live + ids.tolist()))
        pos = torch.arange(worker.offset, worker.offset + length)
        worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        worker.offset += length
        all_ids = torch.tensor(worker.past_key_values.layers[0].ids())
        worker._reindex_state.score_sum = torch.sin(all_ids.float() * 12.9898) * 0.5 + 0.5
        worker._reindex_state.score_layers = 1
        worker._apply_kv_prune()

        kept = worker.past_key_values.layers[0].ids()
        kept_set = set(kept)
        n_turns = step + 1
        # every turn is either intact, absent, or exactly a trailing header stub
        stubs = 0
        state_by_turn = {}
        for t in range(n_turns):
            slots = turn_slots[t]
            body = [i for i in slots if i in kept_set and i >= PARITY_PREFIX] if t == 0 else \
                   [i for i in slots if i in kept_set]
            if t == 0:
                if [i for i in slots if i in kept_set and i < PARITY_PREFIX] != list(range(PARITY_PREFIX)):
                    failures.append(f"step {step}: prefix broken")
            full = [i for i in slots if i >= PARITY_PREFIX] if t == 0 else slots
            if body == full:
                state_by_turn[t] = "intact"
            elif not body:
                state_by_turn[t] = "absent"
            elif body == full[-PARITY_HEADER:]:
                state_by_turn[t] = "stub"
                stubs += 1
            else:
                failures.append(f"step {step}: turn {t} partially pruned in turn mode: {body}")
        # a stub is only legitimate right before a surviving turn
        for t, s in state_by_turn.items():
            if s == "stub" and t + 1 < n_turns and state_by_turn.get(t + 1) == "absent":
                failures.append(f"step {step}: stale header stub for turn {t} "
                                f"(successor absent)")
        # recent turns intact
        for t in range(max(0, n_turns - RECENT), n_turns):
            if state_by_turn.get(t) != "intact":
                failures.append(f"step {step}: recent turn {t} is {state_by_turn.get(t)}")
        # budget within one turn's slack + header stubs
        if len(kept) > BUDGET + max(BUDGET_TURNS) + stubs * PARITY_HEADER:
            failures.append(f"step {step}: {len(kept)} kept far exceeds budget {BUDGET}")
        print(f"  step {step}: {len(kept)} slots kept, turn states "
              f"{[state_by_turn[t] for t in range(n_turns)]}")

    if not worker._kv_prune_state.n_budget_dropped:
        failures.append("turn-granular budget never dropped anything")
    return failures


def check_merge_math():
    """One dropped slot folds into its nearest kept slot as a weighted average."""
    failures = []
    state = KVPruneState()
    state.turn_id = torch.zeros(4, dtype=torch.long)
    state.is_visual = torch.tensor([True, True, True, True])
    state.importance = torch.tensor([1.0, 3.0, 1.0, 1.0])
    keep = torch.tensor([True, False, True, True])
    budget_dropped = torch.tensor([False, True, False, False])
    # db: slot 1 is nearest to slot 2 (identical embed), far from 0 and 3
    db = [torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])]
    layer = FakeLayer(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    merge_dropped_visual([layer], keep, budget_dropped, state, db, torch.device("cpu"))

    got = layer.keys.reshape(-1).tolist()
    # slot 2 <- (1*30 + 3*20) / 4 = 22.5; others untouched
    expected = [10.0, 20.0, 22.5, 40.0]
    if not np.allclose(got, expected):
        failures.append(f"merged keys {got} != expected {expected}")
    if not np.allclose(layer.values.reshape(-1).tolist(), expected):
        failures.append("values did not follow the same merge")
    if abs(float(state.importance[2]) - 4.0) > 1e-6:
        failures.append(f"target importance {float(state.importance[2])} != 4.0 "
                        "(should absorb the dropped mass)")
    return failures


# --- checks 4-5: real model, GPU ------------------------------------------------------

def check_fidelity(images):
    """With a budget nothing exceeds, prune (capture armed) must not move the decision."""
    plain = VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True)
    baseline = run_probs(plain, images)
    release(plain)

    worker = build_prune(WIDE_BUDGET)
    pruned = run_probs(worker, images)
    ps = worker._kv_prune_state
    table_len = worker._reindex_state.pos_table.shape[-1]
    kv_len = worker.past_key_values.get_seq_length()
    n_dropped = ps.n_budget_dropped
    release(worker)

    failures = []
    if n_dropped:
        failures.append(f"an infinite budget dropped {n_dropped} slots")
    if table_len != kv_len:
        failures.append(f"position table ({table_len}) desynced from cache ({kv_len})")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, pruned)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_stock - p_prune| = {delta:.5f}")
    floor = max(deltas)
    if floor > FIDELITY_TOL:
        failures.append(f"with nothing pruned the capture/pre-rotation path moved the "
                        f"action distribution by {floor:.5f} (> {FIDELITY_TOL})")
    return failures, floor, baseline


def check_divergence(images, floor, baseline, merge):
    tag = "prune+merge" if merge else "prune"
    worker = build_prune(NARROW_BUDGET, merge=merge)
    probs = run_probs(worker, images)
    ps = worker._kv_prune_state
    kv_len = worker.past_key_values.get_seq_length()
    table_len = worker._reindex_state.pos_table.shape[-1]
    db_rows = worker.past_image_embeds[0].shape[0]
    n_vis = int(ps.is_visual.sum())
    release(worker)

    failures = []
    if not ps.n_budget_dropped:
        failures.append(f"[{tag}] budget {NARROW_BUDGET} never pruned; raise N_STEPS")
    if kv_len > NARROW_BUDGET:
        failures.append(f"[{tag}] cache holds {kv_len} slots over budget {NARROW_BUDGET}")
    if table_len != kv_len:
        failures.append(f"[{tag}] position table ({table_len}) desynced from cache ({kv_len})")
    if db_rows != n_vis:
        failures.append(f"[{tag}] embed db ({db_rows}) desynced from visual slots ({n_vis})")
    if any(not np.all(np.isfinite(p)) for p in probs):
        failures.append(f"[{tag}] non-finite action probabilities")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, probs)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_stock - p_{tag}| = {delta:.5f}")
    signal = max(deltas)
    print(f"  [{tag}] signal={signal:.5f} vs floor={floor:.5f} "
          f"({signal / max(floor, 1e-9):.1f}x), final kv_len={kv_len}, "
          f"budget_dropped={ps.n_budget_dropped}")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"[{tag}] differs from stock by only {signal:.5f}, within "
                        f"{DIVERGENCE_MARGIN}x of the {floor:.5f} floor -- pruning is inert")
    return failures


def main():
    failures = []

    print(f"=== forced-schedule parity: window={PARITY_WINDOW}, budget=inf (no model) ===")
    failures += check_forced_schedule_parity()

    print(f"=== budget bookkeeping: budget={BUDGET}, recent={RECENT} (no model) ===")
    failures += check_budget_bookkeeping()

    print(f"=== turn granularity: budget={BUDGET}, recent={RECENT} (no model) ===")
    failures += check_turn_granularity()

    print("=== merge math (no model) ===")
    failures += check_merge_math()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    images = make_images(N_STEPS)
    print(f"=== fidelity: budget={WIDE_BUDGET} (never prunes), {N_STEPS} steps ===")
    fidelity_failures, floor, baseline = check_fidelity(images)
    failures += fidelity_failures

    print(f"=== divergence: budget={NARROW_BUDGET}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor, baseline, merge=False)

    print(f"=== divergence (merge arm): budget={NARROW_BUDGET}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor, baseline, merge=True)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"\nPASS: prune reproduces evict's window schedule, keeps cache/table/metadata/db "
          f"in sync under real prunes, the merge math is exact, the capture path is inert "
          f"(within {floor:.5f}), and an active budget both holds the cache at the budget "
          f"and moves the decision.")


if __name__ == "__main__":
    main()
