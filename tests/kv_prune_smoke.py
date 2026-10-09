"""Cache eviction and slot-budget pruning invariants."""
import gc
from prune_fixtures import *
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.kv_prune import KVPruneState
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


# --- checks 1-2: bookkeeping on synthetic turns, CPU only ----------------------------
# Every token is its own global id AND its own mRoPE position, so cache slices, the
# position table, and the embed db can all be compared as plain lists of integers.
PARITY_PREFIX = 5
PARITY_HEADER = 3
PARITY_WINDOW = 2
PARITY_TURNS = [11, 7, 9, 6, 8, 7, 10]


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


def check_sparse_history_window():
    """Pre-forward sparse history excludes turns due to leave the cache this step."""
    failures = []

    worker = VLMWorker.__new__(VLMWorker)
    worker.context_window = 2
    worker.context_window_mode = "evict"
    worker.use_sparse = True
    worker.past_image_embeds = [torch.tensor([[10.0], [20.0], [21.0]])]
    worker._abs_bounds = [10, 20, 30]
    worker._vis_counts = [[2], [1], [2]]
    worker._n_evicted = 1
    view = worker._windowed_sparse_history()
    if not torch.equal(view[0], torch.tensor([[20.0], [21.0]])):
        failures.append(f"evict sparse view kept out-of-window rows: {view[0].flatten().tolist()}")
    if not torch.equal(worker.past_image_embeds[0], torch.tensor([[10.0], [20.0], [21.0]])):
        failures.append("windowed sparse view mutated the canonical embedding DB")

    ps = KVPruneState()
    ps.n_turns = 3
    ps.turn_id = torch.tensor([1, 1, 2, 2, 2])
    ps.is_visual = torch.tensor([True, False, True, True, False])
    worker.context_window_mode = "prune"
    worker._kv_prune_state = ps
    prune_view = worker._windowed_sparse_history()
    if not torch.equal(prune_view[0], torch.tensor([[20.0], [21.0]])):
        failures.append(f"prune sparse view kept out-of-window rows: {prune_view[0].flatten().tolist()}")
    if not torch.equal(worker.past_image_embeds[0], torch.tensor([[10.0], [20.0], [21.0]])):
        failures.append("prune sparse view mutated the canonical embedding DB")
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


def check_divergence(images, floor, baseline):
    tag = "prune"
    worker = build_prune(NARROW_BUDGET)
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
    failures = check_forced_schedule_parity() + check_budget_bookkeeping() + check_sparse_history_window()
    import sys
    if "--cpu-only" not in sys.argv:
        images = make_images(N_STEPS)
        checks, floor, baseline = check_fidelity(images)
        failures += checks + check_divergence(images, floor, baseline)
    if failures:
        raise AssertionError("\n".join(failures))
    print("PASS: cache scheduling, budgets, sparse history, and inference fidelity")


if __name__ == "__main__":
    main()
