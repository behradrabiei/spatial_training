"""CPU-only tests for exact visual KV-pruning partitions and selectors."""
import torch

from longnav.utils.kv_prune import (
    KVPruneState,
    build_prune_partition,
    build_selected_keep,
    capped_largest_remainder,
    farthest_point_select,
    grid_coords_from_pos_table,
    grid_stratified_select,
    mixed_diversity_select,
    random_select,
    stratified_select,
    voxel_cell_ids,
    voxel_dedup_select,
    voxel_stratified_select,
)
from longnav.utils.voxel_utils import VOXEL_NONE


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


def test_stratified_redistribution():
    state = make_state([[1], [1, 1, 1, 1], [1, 1, 1, 1, 1]])
    mask = torch.ones(10, dtype=torch.bool)
    selected = stratified_select(state, mask, 6)
    counts = torch.bincount(state.turn_id[selected], minlength=3)
    assert counts.tolist() == [1, 2, 3]
    assert selected.unique().numel() == 6


def test_diversity_and_hybrid_pool():
    state = make_state([[1, 1, 1, 1]])
    db = [torch.tensor([[1.0, 0.0], [0.995, 0.1], [-1.0, 0.0], [0.0, 1.0]])]
    candidates = torch.tensor([0, 1, 2])
    anchors = torch.tensor([False, False, False, True])
    selected = farthest_point_select(state, db, candidates, 2, anchor_mask=anchors)
    assert 2 in selected.tolist(), "FPS should first take the point opposite the anchor"
    assert set(selected.tolist()) != {0, 1}, "FPS retained the redundant pair"

    # Fisher's top-2 are deliberately redundant. A 2B pool exposes the opposite point,
    # and FPS must use it in the final B selection.
    fisher_ranked = torch.tensor([0, 1, 2, 3])
    hybrid = farthest_point_select(state, db, fisher_ranked[:4], 2)
    assert set(hybrid.tolist()) != {0, 1}
    assert set(hybrid.tolist()).issubset(set(fisher_ranked[:4].tolist()))
    mixed_state = make_state([[1, 1, 1, 1, 0, 0]])
    mixed_db = [db[0].clone()]
    mixed_scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.9])
    mixed = mixed_diversity_select(
        mixed_state, mixed_db, torch.arange(6), 3,
        nonvisual_scores=mixed_scores)
    assert int(mixed_state.is_visual[mixed].sum()) == 2
    assert int((~mixed_state.is_visual[mixed]).sum()) == 1
    assert 5 in mixed.tolist(), "hybrid must Fisher-rank the nonvisual quota"



def test_proportional_quotas():
    quotas = capped_largest_remainder([10.0, 1.0], [2, 10], 6)
    assert quotas.tolist() == [2, 4]
    assert int(quotas.sum()) == 6
    equal = capped_largest_remainder([0.0, 0.0, 0.0], [1, 4, 5], 6)
    assert int(equal.sum()) == 6 and equal[0] == 1
    assert int(equal[1:].max() - equal[1:].min()) <= 1


def make_grid_state(n_turns, gh, gw, keep_frac=0.6, seed=0, state_seed=17, n_text=3,
                    drop_row0=False):
    """State plus a get_rope_index-style pos_table for `n_turns` single-image turns.

    Each turn is n_text text tokens, one gh x gw image with a random subset of tokens
    dropped (the sparse filter), then n_text more text tokens. Image tokens get
    (t, h, w) = base + (0, row, col); text tokens get t == h == w == position. Returns
    (state, pos_table, rc) where rc lists every visual slot's true (row, col).
    """
    rng = torch.Generator().manual_seed(seed)
    state = KVPruneState(seed=state_seed)
    cols, rc_all, pos = [], [], 0
    for _ in range(n_turns):
        vis_rows = []
        text = torch.arange(pos, pos + n_text)
        cols.append(torch.stack([text, text, text]))
        vis_rows += [False] * n_text
        pos += n_text
        base = pos
        keep = torch.rand(gh * gw, generator=rng) < keep_frac
        keep[-1] = True
        if drop_row0:
            keep[:gw] = False
        idx = torch.nonzero(keep).flatten()
        r, c = idx // gw, idx % gw
        cols.append(torch.stack([torch.full_like(r, base), base + r, base + c]))
        vis_rows += [True] * idx.numel()
        rc_all.append(torch.stack([r, c], dim=1))
        pos = base + max(gh, gw)
        text = torch.arange(pos, pos + n_text)
        cols.append(torch.stack([text, text, text]))
        vis_rows += [False] * n_text
        pos += n_text
        state.append(len(vis_rows), torch.tensor(vis_rows))
    state.importance = torch.zeros(state.turn_id.numel())
    return state, torch.cat(cols, dim=1).unsqueeze(1), torch.cat(rc_all)


def _min_pairwise(points):
    if points.shape[0] < 2:
        return None
    d = torch.cdist(points.float(), points.float())
    d.fill_diagonal_(float("inf"))
    return float(d.min())


def _frame_spread(state, coords, selected):
    vals = []
    for t in torch.unique(state.turn_id[selected]).tolist():
        pts = coords[selected[(state.turn_id[selected] == t) & state.is_visual[selected]]]
        m = _min_pairwise(pts)
        if m is not None:
            vals.append(m)
    return sum(vals) / len(vals)


def test_grid_coords():
    for drop_row0 in (False, True):
        state, pos_table, rc = make_grid_state(3, 5, 7, drop_row0=drop_row0)
        coords = grid_coords_from_pos_table(pos_table)
        assert torch.equal(coords[state.is_visual], rc), "grid coords must be (h - t, w - t)"
        assert bool((coords[~state.is_visual] == 0).all())


def test_grid_quota_parity():
    state, pos_table, _ = make_grid_state(4, 6, 8)
    mask = state.turn_id < 3
    cand = torch.nonzero(mask).flatten()
    ref = KVPruneState(seed=17)
    draw = random_select(cand, 40, ref.generator)
    sel = grid_stratified_select(state, pos_table, mask, 40, state.generator,
                                 state.aux_generator)
    assert sel.numel() == 40 and sel.unique().numel() == 40 and bool(mask[sel].all())
    vis_d, vis_s = draw[state.is_visual[draw]], sel[state.is_visual[sel]]
    assert torch.equal(torch.bincount(state.turn_id[vis_d], minlength=4),
                       torch.bincount(state.turn_id[vis_s], minlength=4)), \
        "grid must keep random's per-frame visual quotas"
    assert torch.equal(draw[~state.is_visual[draw]].sort().values,
                       sel[~state.is_visual[sel]].sort().values), \
        "grid must keep random's text picks"
    assert torch.equal(state.generator.get_state(), ref.generator.get_state()), \
        "grid consumed the main generator stream beyond random's one draw"
    # keep-all: quota == pool for every frame returns the whole pool
    full = grid_stratified_select(state, pos_table, mask, cand.numel(), state.generator,
                                  state.aux_generator)
    assert set(full.tolist()) == set(cand.tolist())


def test_grid_evenness():
    grid_spread, rand_spread = [], []
    for seed in range(20):
        state, pos_table, _ = make_grid_state(3, 8, 10, seed=seed, state_seed=100 + seed)
        coords = grid_coords_from_pos_table(pos_table)
        mask = state.turn_id < 2
        ref = KVPruneState(seed=100 + seed)
        draw = random_select(torch.nonzero(mask).flatten(), 30, ref.generator)
        sel = grid_stratified_select(state, pos_table, mask, 30, state.generator,
                                     state.aux_generator)
        rand_spread.append(_frame_spread(state, coords, draw))
        grid_spread.append(_frame_spread(state, coords, sel))
    g, r = sum(grid_spread) / 20, sum(rand_spread) / 20
    assert g > r, f"grid picks are not more evenly spread than random ({g:.2f} vs {r:.2f})"


def test_grid_determinism():
    picks = []
    for state_seed in (17, 17, 18):
        state, pos_table, _ = make_grid_state(3, 6, 8, state_seed=state_seed)
        mask = state.turn_id < 2
        picks.append(grid_stratified_select(state, pos_table, mask, 25, state.generator,
                                            state.aux_generator))
    assert torch.equal(picks[0], picks[1])
    assert not torch.equal(picks[0], picks[2])


NONE = (VOXEL_NONE, VOXEL_NONE, VOXEL_NONE)
A, B, C = (0, 0, 0), (1, 0, 0), (2, 0, 5)
A_HIGH = (0, 3, 0)  # same (x, z) as A, different height
# Per turn: a list of slots; None = text slot, a tuple = the visual slot's voxel id.
VOXEL_TURNS = [
    [None, A, B, None],
    [None, A, C, NONE, None],
    [A_HIGH, B],
]


def make_voxel_state(turns=VOXEL_TURNS, seed=17):
    state = KVPruneState(seed=seed)
    for turn in turns:
        vis = torch.tensor([s is not None for s in turn])
        vox = torch.tensor([s if s is not None else NONE for s in turn], dtype=torch.long)
        state.append(len(turn), vis, vox)
    state.importance = torch.zeros(state.turn_id.numel())
    return state


def _slots(state, *cells):
    """Cache slots (in cache order) whose voxel is one of `cells`."""
    want = torch.tensor(cells, dtype=torch.long)
    hit = (state.voxel.unsqueeze(1) == want.unsqueeze(0)).all(-1).any(-1) & state.is_visual
    return torch.nonzero(hit).flatten()


def test_voxel_cell_ids():
    state = make_voxel_state()
    every = torch.arange(state.turn_id.numel())
    flat = voxel_cell_ids(state, every, scale=1, two_d=True)
    assert bool((flat[~state.is_visual] == -1).all()), "text slots must have no cell"
    assert int(flat[_slots(state, NONE)]) == -1, "patches without depth must have no cell"
    a_cells = flat[_slots(state, A, A_HIGH)]
    assert a_cells.unique().numel() == 1, "2-D cells must ignore height"
    three_d = voxel_cell_ids(state, every, scale=1, two_d=False)
    assert three_d[_slots(state, A, A_HIGH)].unique().numel() == 2, "3-D cells must keep height"
    coarse = voxel_cell_ids(state, every, scale=2, two_d=True)
    assert coarse[_slots(state, A, B)].unique().numel() == 1, "scale=2 must merge x=0 and x=1"
    assert flat[_slots(state, A, B)].unique().numel() == 2


def test_voxel_dedup():
    state = make_voxel_state()
    visual = torch.nonzero(state.is_visual).flatten()
    newest = {int(_slots(state, A, A_HIGH)[-1]), int(_slots(state, B)[-1]),
              int(_slots(state, C)[0]), int(_slots(state, NONE)[0])}
    sel = voxel_dedup_select(state, visual, 4, cap=1, scale=1, two_d=True,
                             generator=state.generator)
    assert set(sel.tolist()) == newest, "cap=1 must keep exactly the newest slot per cell"
    topped = voxel_dedup_select(state, visual, 6, cap=1, scale=1, two_d=True,
                                generator=state.generator)
    assert topped.numel() == 6 and topped.unique().numel() == 6
    assert newest.issubset(set(topped.tolist())), "survivors must all be kept before top-up"
    cap2 = voxel_dedup_select(state, visual, 6, cap=2, scale=1, two_d=True,
                              generator=state.generator)
    assert set(cap2.tolist()) == set(visual.tolist()) - {int(_slots(state, A)[0])}, \
        "cap=2 must drop only the oldest of A's three copies"
    # text share is count-proportional and sampled, never deduped away
    every = torch.arange(state.turn_id.numel())
    mixed = voxel_dedup_select(state, every, 8, cap=1, scale=1, two_d=True,
                               generator=state.generator)
    n_text = int((~state.is_visual[mixed]).sum())
    assert mixed.numel() == 8 and n_text == round(8 * 4 / 11), f"text share {n_text} != 3"
    # determinism under the seed
    s1 = voxel_dedup_select(make_voxel_state(), visual, 6, generator=make_voxel_state().generator)
    s2 = voxel_dedup_select(make_voxel_state(), visual, 6, generator=make_voxel_state().generator)
    assert torch.equal(s1, s2)


def test_voxel_strat():
    state = make_voxel_state()
    visual = torch.nonzero(state.is_visual).flatten()
    cells = voxel_cell_ids(state, visual, 1, True)
    sel = voxel_stratified_select(state, visual, 4, scale=1, two_d=True,
                                  generator=state.generator)
    assert sel.numel() == 4
    sel_cells = cells[torch.isin(visual, sel)]
    assert sel_cells[sel_cells >= 0].unique().numel() == 3 and int((sel_cells < 0).sum()) == 1, \
        "capacity 4 over cells A/B/C + one lone slot must take one of each"
    six = voxel_stratified_select(state, visual, 6, scale=1, two_d=True,
                                  generator=state.generator)
    six_cells = cells[torch.isin(visual, six)]
    per_cell = torch.bincount(six_cells[six_cells >= 0], minlength=3)
    assert per_cell.tolist() == [2, 2, 1] and int((six_cells < 0).sum()) == 1, \
        f"equal share over counts [3, 2, 1, 1] at capacity 6 must be [2, 2, 1, 1], got {per_cell.tolist()}"
    a_slots = _slots(state, A, A_HIGH)
    kept_a = [s for s in six.tolist() if s in a_slots.tolist()]
    assert int(a_slots[0]) not in kept_a and len(kept_a) == 2, "A keeps its two newest copies"


def main():
    test_partition()
    test_deterministic_random()
    test_stratified_redistribution()
    test_diversity_and_hybrid_pool()
    test_proportional_quotas()
    test_grid_coords()
    test_grid_quota_parity()
    test_grid_evenness()
    test_grid_determinism()
    test_voxel_cell_ids()
    test_voxel_dedup()
    test_voxel_strat()
    print("PASS: exact visual partition, deterministic random, stratification, FPS, KL quotas, "
          "hybrid pool, grid coords/parity/evenness/determinism, voxel cells/dedup/strat")


if __name__ == "__main__":
    main()
