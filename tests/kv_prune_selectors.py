"""CPU tests for the common partition, the seeded random selector and the whole-frame ranking."""
import torch
from longnav.utils.kv_prune import (
    KVPruneState, build_prune_partition, build_selected_keep, random_select, topk_frames,
)

def make_state(rows, seed=17):
    state = KVPruneState(seed=seed)
    for row in rows:
        state.append(len(row), torch.tensor(row, dtype=torch.bool))
    state.importance = torch.zeros(state.turn_id.numel())
    return state

def test_partition():
    state = make_state([
        [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0],
        [1, 0, 1, 0], [1, 1, 0, 0],
    ])
    probe = build_prune_partition(state, prefix_len=2, context_window=3, holdback_n=1,
                                  budget=10**6, recent_turns=1,
                                  candidate_scope="visual")
    fixed = int(probe.protected.sum())
    part = build_prune_partition(state, prefix_len=2, context_window=3, holdback_n=1,
                                 budget=fixed + 3, recent_turns=1,
                                 candidate_scope="visual")
    assert part.capacity == 3
    assert not bool((part.candidates & ~state.is_visual).any())
    assert bool((part.protected & ~state.is_visual & part.base_keep).equal(
        (~state.is_visual) & part.base_keep))
    recent = state.turn_id == 4
    assert bool(part.protected[recent].all())
    assert bool(part.protected[:2].all())
    keep = build_selected_keep(part, torch.nonzero(part.candidates).flatten()[:3])
    all_part = build_prune_partition(
        state, prefix_len=2, context_window=3, holdback_n=1,
        budget=10**6, recent_turns=1, candidate_scope="all")
    assert bool((all_part.candidates & ~state.is_visual).any()), \
        "historical in-window text must be budget-prunable in all-slot mode"
    assert bool(all_part.protected[:2].all())
    assert bool(all_part.protected[recent].all())
    assert int(keep.sum()) == fixed + 3
    try:
        build_prune_partition(state, 2, 3, 1, fixed - 1, 1, "visual")
    except RuntimeError as exc:
        assert "protected cache floor" in str(exc)
    else:
        raise AssertionError("an infeasible protected floor did not raise")

def test_deterministic_random():
    candidates = torch.arange(30)
    a = KVPruneState(seed=17)
    b = KVPruneState(seed=17)
    c = KVPruneState(seed=18)
    sa = random_select(candidates, 9, a.generator)
    sb = random_select(candidates, 9, b.generator)
    sc = random_select(candidates, 9, c.generator)
    assert torch.equal(sa, sb)
    assert not torch.equal(sa, sc)

def test_topk_frames():
    """Whole-frame rule (kl_keep_one / recency): largest score first, ties to the NEWER frame,
    kept ids returned sorted, the whole pool when k covers it."""
    assert topk_frames([3, 7, 9, 12], [0.5, 0.1, 0.5, 0.0], 1) == [9]
    assert topk_frames([3, 7, 9, 12], [0.5, 0.1, 0.5, 0.0], 2) == [3, 9]
    assert topk_frames([12, 3, 9], [1.0, 2.0, 1.5], 2) == [3, 9]
    assert topk_frames([3, 7], [0.0, 0.0], 5) == [3, 7]
    assert topk_frames([], [], 2) == []
    assert topk_frames([4, 5, 6], [4, 5, 6], 2) == [5, 6]  # recency: score = frame id
    # entropy_keep_one ranks by the NEGATED entropy: the lowest-entropy frames survive, ties to the newer
    H = [0.30, 0.05, 0.30, 1.38, 0.05]
    assert topk_frames([0, 1, 2, 3, 4], [-h for h in H], 2) == [1, 4]
    assert topk_frames([0, 1, 2, 3, 4], [-h for h in H], 3) == [1, 2, 4]

def test_shadow_pruner_matches_slicing():
    """paper/prune_kl: a ShadowPruner (an `alive` mask over a state that is never sliced) holds,
    after every step, exactly the slots a physically sliced state holds under the same rule --
    for the seeded random draw (position-based) and for a score ranking (identity-based)."""
    from longnav.utils.kv_prune import frame_budget, step_seed
    from longnav.utils.prune_kl_probe import NONE, ShadowPruner

    kwargs = dict(prefix_len=2, context_window=None, holdback_n=1, recent_turns=0,
                  candidate_scope="visual", frame_cut=False)
    counts = [6, 9, 5, 0, 8, 7, 10, 4]
    score = torch.rand(10_000, generator=torch.Generator().manual_seed(3))  # per global slot id
    for method in ("random", "fastv"):
        for k in (1, 2, 3):
            full, ref = KVPruneState(seed=17), KVPruneState(seed=17)
            shadow = ShadowPruner(method, k)
            ref_ids = torch.zeros(0, dtype=torch.long)  # global ids the sliced state holds
            n_total = 0
            for turn, n_vis in enumerate(counts):
                row = torch.tensor([0, 0] + [1] * n_vis + [0, 0, 0], dtype=torch.bool)
                for state in (full, ref):
                    state.append(row.numel(), row)
                    state.fold_scores(torch.zeros(state.turn_id.numel()), 0.0)
                ref_ids = torch.cat([ref_ids, torch.arange(n_total, n_total + row.numel())])
                n_total += row.numel()
                shadow.extend(n_total)
                hidden = shadow.hidden(full)
                assert not bool((hidden & (full.turn_id == turn)).any()), "the newest frame is never hidden"
                assert torch.equal(torch.nonzero(~hidden).flatten(), ref_ids), \
                    "the probe's visible set is the sliced state before this step's selection"

                def select(ids, cand, capacity):
                    if method == "random":
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(step_seed(17, "ep", turn))
                        return random_select(cand, capacity, gen)
                    return cand[torch.argsort(score[ids[cand]], descending=True, stable=True)[:capacity]]

                dropped = shadow.select(full, kwargs, lambda m, cand, cap: select(torch.arange(n_total), cand, cap))
                cap = frame_budget(ref, 0, k)
                part = build_prune_partition(ref, budget=None, capacity=cap, **kwargs)
                cand = torch.nonzero(part.candidates).flatten()
                selected = cand if part.capacity == cand.numel() else select(ref_ids, cand, part.capacity)
                keep = build_selected_keep(part, selected)
                assert dropped == int((~keep).sum())
                ref_ids = ref_ids[keep]
                ref.slice(keep)
                assert torch.equal(torch.nonzero(shadow.alive).flatten(), ref_ids), (method, k, turn)
                assert shadow.cap == (NONE if cap is None else int(part.capacity))
                assert bool(shadow.alive[~full.is_visual].all()), "text is never dropped"
            assert int((shadow.alive & full.is_visual).sum()) == frame_budget(full, 0, k)

if __name__ == "__main__":
    test_partition()
    test_deterministic_random()
    test_topk_frames()
    test_shadow_pruner_matches_slicing()
    print("PASS")
