"""CPU tests for the common partition and seeded random selector."""
import torch
from longnav.utils.kv_prune import KVPruneState, build_prune_partition, build_selected_keep, random_select

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

if __name__ == "__main__":
    test_partition()
    test_deterministic_random()
    print("PASS")
