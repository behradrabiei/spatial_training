"""Smoke checks for layer-selective KV pruning (vlm.kv_prune_layer_start / kv_prune_layer_end).

The budget may apply to a range of decoder layers only; the other layers keep the window-only
(master) cache. These checks pin the bookkeeping that makes that safe and that uniform pruning
is untouched.

CPU (no model):
  1. mask_for_layer: as-is / gather / structural rebuild for bool (sdpa) and float (eager)
     masks, compared with masks built by transformers' own factories; misuse raises.
  2. fold_decision_row scatters per-layer decision rows into master coordinates.
  3. Uniform parity: a full layer range that is NOT collapsed to the uniform fast path keeps
     the same slots as the single-cache worker on every step.
  4. Layer-selective bookkeeping on a 3-layer fake cache: unpruned layers == master ==
     position table == metadata == embed DB, pruned layers == master[alive], budget respected,
     window drops reach every layer, protected slots stay alive, dead slots are not recounted,
     this step's drops rank below every kept candidate.
  5. Empty pruned range: the window still applies, the budget never does.
GPU (real checkpoint, sdpa unless noted):
  6. (0, None) vs (0, 28) bit-identical; wide budget at layer_start=14 inert vs stock; narrow
     budget at layer_start=14 diverges with layers < 14 == master and layers >= 14 <= budget;
     band (0, 14) under sdpa and eager; layer_start == n_layers == wide budget.
  7. Decision replay at layer_start=14: zero bias reproduces the decision, every reference is
     restored, an ablation confined to an empty layer range is a no-op, kl / fisher complete.
  8. Layer-influence rows: one finite KL per probe layer, the n_layers probe is 0.

    conda activate longnav_vlm
    python tests/kv_prune_layer_smoke.py
"""
import gc
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.kv_prune import accumulate_decision_row, fold_decision_row
from longnav.utils.pre_rope import ReindexState, hide_columns, mask_for_layer
from longnav.utils.vlm_worker import VLMWorker

from kv_prune_smoke import (  # noqa: E402  (tests/ is run as a script directory)
    FIDELITY_TOL, DIVERGENCE_MARGIN, MODEL_ID, N_STEPS, NARROW_BUDGET, PARITY_PREFIX,
    PARITY_TURNS, PARITY_WINDOW, START, TURN, WIDE_BUDGET, FakeCache, bare_worker, feed_turn,
    make_images, release, run_probs,
)

REPLAY_TOL = 2e-2
N_LAYERS = 3
LAYER_RANGE = range(1, 3)
LAYER_TURNS = [12, 9, 10, 8, 11, 9, 10]
LAYER_BUDGET = 22
RECENT = 1
GPU_LAYER_START = 14


def fails(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (RuntimeError, ValueError):
        return True
    return False


# --- 1. mask_for_layer ------------------------------------------------------------------

def _reference_masks(n_past, q):
    from transformers.masking_utils import eager_mask, sdpa_mask
    cache_position = torch.arange(n_past, n_past + q)
    bool_mask = sdpa_mask(batch_size=1, cache_position=cache_position, kv_length=n_past + q,
                          allow_is_causal_skip=False)
    float_mask = eager_mask(batch_size=1, cache_position=cache_position, kv_length=n_past + q,
                            dtype=torch.float32)
    return bool_mask, float_mask


def check_mask_for_layer():
    failures = []
    q, n_master_old, alive_old = 5, 11, [0, 2, 3, 7, 10]
    master_len = n_master_old + q
    idx = torch.tensor(alive_old + list(range(n_master_old, master_len)))
    n_keys = idx.numel()
    for label, ref_full, ref_layer in zip(("bool", "float"),
                                          _reference_masks(n_master_old, q),
                                          _reference_masks(len(alive_old), q)):
        # as-is: identity object when the width already matches
        same = mask_for_layer(ref_full, None, master_len, q, master_len)
        if same is not ref_full:
            failures.append(f"[{label}] a matching mask was copied instead of passed through")
        # gather: master-width mask -> this layer's columns
        got = mask_for_layer(ref_full, idx, n_keys, q, master_len)
        if not torch.equal(got, ref_layer):
            failures.append(f"[{label}] gathered mask differs from the reference built for the layer")
        # structural rebuild: mask built from a shorter (pruned) layer 0 -> a longer layer
        short_ref = _reference_masks(len(alive_old), q)[0 if label == "bool" else 1]
        rebuilt = mask_for_layer(short_ref, None, master_len, q, master_len + 99)
        if not torch.equal(rebuilt, ref_full):
            failures.append(f"[{label}] structurally rebuilt mask differs from the reference")
        # a past block that is not all-visible cannot be rebuilt
        bad = short_ref.clone()
        if label == "bool":
            bad[..., 0] = False
        else:
            bad[..., 0] = torch.finfo(torch.float32).min
        if not fails(mask_for_layer, bad, None, master_len, q, master_len + 99):
            failures.append(f"[{label}] rebuilt a mask whose past block hides a slot")
    # None is only legal for a single query (or FlashAttention, which is causal without a mask)
    if not fails(mask_for_layer, None, idx, n_keys, q, master_len, "sdpa"):
        failures.append("a multi-token chunk was handed to sdpa without a mask")
    if mask_for_layer(None, idx, n_keys, 1, master_len, "sdpa") is not None:
        failures.append("a one-token replay without a mask was not passed through")
    if mask_for_layer(None, idx, n_keys, q, master_len, "flash_attention_2") is not None:
        failures.append("flash_attention_2 must stay mask-free")
    # wrong query rows or an impossible width fail loudly
    ref_full = _reference_masks(n_master_old, q)[0]
    if not fails(mask_for_layer, ref_full, None, master_len + 3, q + 1, master_len + 99):
        failures.append("a mask with the wrong number of query rows was accepted")
    print(f"  bool/float gather + rebuild match transformers' factories; misuse raises")
    return failures


def check_hide_columns():
    """Visual-blind mask: hidden columns are unattendable for every query row, others intact."""
    failures = []
    q, n_past = 3, 5
    n_keys = n_past + q
    hide = torch.tensor([1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
    bool_ref, float_ref = _reference_masks(n_past, q)
    got_b = hide_columns(bool_ref, hide, q, n_keys)
    if bool(got_b[..., hide].any()) or not torch.equal(got_b[..., ~hide], bool_ref[..., ~hide]):
        failures.append("bool mask: hidden columns visible or other columns changed")
    got_f = hide_columns(float_ref, hide, q, n_keys)
    if not bool((got_f[..., hide] == torch.finfo(torch.float32).min).all()) \
            or not torch.equal(got_f[..., ~hide], float_ref[..., ~hide]):
        failures.append("float mask: hidden columns not at the dtype minimum or others changed")
    got_n = hide_columns(None, hide, q, n_keys)
    if not torch.equal(got_n, bool_ref & ~hide.view(1, 1, 1, -1)):
        failures.append("None mask: rebuilt causal mask differs from the reference")
    if got_b is bool_ref or got_f is float_ref:
        failures.append("hide_columns mutated the shared mask in place")
    if not fails(hide_columns, bool_ref, hide[:-1], q, n_keys - 1):
        failures.append("a width mismatch was accepted")
    print("  hidden columns unattendable for bool / float / None masks; width mismatch raises")
    return failures


# --- 2. per-layer score scatter -----------------------------------------------------------

def check_score_scatter():
    failures = []
    st = ReindexState()
    st.configure(pruned_layers=LAYER_RANGE, score_layers=range(0, N_LAYERS), n_layers=N_LAYERS)
    st.append(torch.arange(6).reshape(1, 1, -1).expand(3, 1, -1).clone())
    alive = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.bool)
    st.set_alive(alive)
    st.arm_capture()
    r0 = torch.arange(6, dtype=torch.float32) + 1          # unpruned layer: master width
    r1 = torch.tensor([10.0, 20.0, 30.0, 40.0])            # pruned layers: alive width
    r2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    fold_decision_row(st, r0, 0)
    fold_decision_row(st, r1, 1)
    fold_decision_row(st, r2, 2)
    expected = r0.clone()
    expected[alive] = (r0[alive] + r1 + r2) / 3
    if not torch.allclose(st.mean_scores(), expected):
        failures.append(f"scattered mean {st.mean_scores().tolist()} != {expected.tolist()}")
    if st.score_count.tolist() != [3, 3, 1, 3, 1, 3]:
        failures.append(f"per-slot fold counts {st.score_count.tolist()} != [3, 3, 1, 3, 1, 3]")
    # a layer outside score_layer_set is skipped by the attention hook's guard
    st2 = ReindexState()
    st2.configure(pruned_layers=LAYER_RANGE, score_layers=LAYER_RANGE, n_layers=N_LAYERS)
    if st2.scores_layer(0) or not st2.scores_layer(1):
        failures.append("score_layer_set membership is wrong")
    # uniform: accumulate_decision_row equals a manual head-mean softmax and the int count
    torch.manual_seed(0)
    uni = ReindexState()
    uni.append(torch.arange(9).reshape(1, 1, -1).expand(3, 1, -1).clone())
    uni.arm_capture()
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 9, 8)
    accumulate_decision_row(uni, q, k, 0.5, 0)
    accumulate_decision_row(uni, q, k, 0.5, 1)
    qh = q[0, :, -1].reshape(2, 2, 8)
    manual = torch.softmax(torch.einsum("grd,gkd->grk", qh, k[0]) * 0.5, -1).mean((0, 1))
    if uni.score_layers != 2 or uni.score_count is not None or not torch.allclose(uni.mean_scores(), manual, atol=1e-6):
        failures.append("uniform accumulate_decision_row changed behaviour")
    print(f"  scatter over {N_LAYERS} layers reproduces the per-slot mean; uniform path unchanged")
    return failures


# --- 3-5. bookkeeping on a fake multi-layer cache ------------------------------------------

def score_of(ids):
    ids = torch.as_tensor(ids, dtype=torch.float32)
    return torch.sin(ids * 12.9898) * 0.5 + 0.5


def layer_worker(n_layers, layer_range, budget, window=None, recent_turns=RECENT,
                 importance="attn", collapse=True):
    worker = bare_worker("prune", window, budget=budget, importance=importance,
                         recent_turns=recent_turns, n_layers=n_layers, layer_range=layer_range)
    if not collapse and layer_range is not None:
        # Force the general layer-selective path even though the range covers every layer
        # (configure would collapse it to the uniform fast path).
        worker._reindex_state.pruned_layers = layer_range
        worker._reindex_state.score_layer_set = None
    worker.use_sparse = True
    worker.language_model = SimpleNamespace(visual_pos_masks=None)
    worker.past_image_embeds = None
    return worker


def feed_layers(worker, ids, vis_row):
    """Append one turn to every layer's own cache, inject the decision rows, prune."""
    n_layers = worker._n_fake_layers
    if worker.past_key_values is None:
        per_layer = [ids.clone() for _ in range(n_layers)]
    else:
        per_layer = [torch.tensor(layer.ids() + ids.tolist())
                     for layer in worker.past_key_values.layers]
    worker.past_key_values = FakeCache(per_layer)
    worker.language_model.visual_pos_masks = vis_row.reshape(1, -1)
    db_new = ids[vis_row].float().reshape(-1, 1)
    if worker.past_image_embeds is None:
        worker.past_image_embeds = [db_new]
    else:
        worker.past_image_embeds[0] = torch.cat([worker.past_image_embeds[0], db_new])
    st = worker._reindex_state
    pos = torch.arange(worker.offset, worker.offset + len(ids))
    st.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
    worker.offset += len(ids)
    # Injected decision rows, exactly as the attention hook would fold them: every scored
    # layer contributes a row over the slots it holds (positions == ids in these fakes).
    st.arm_capture()
    master_ids = st.pos_table[0, 0]
    for layer_idx in range(n_layers):
        if not st.scores_layer(layer_idx):
            continue
        idx = st.layer_index(layer_idx)
        fold_decision_row(st, score_of(master_ids if idx is None else master_ids[idx]), layer_idx)
    worker._apply_kv_prune()
    st.capture_scores = False


def run_sequence(worker, turns, prefix=PARITY_PREFIX):
    """Feed `turns` and return per-step per-layer kept ids (odd ids are 'visual')."""
    history, next_id = [], 0
    for length in turns:
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        feed_layers(worker, ids, (ids % 2).bool())
        history.append([layer.ids() for layer in worker.past_key_values.layers])
    return history


def check_uniform_parity():
    """A full range on the general path keeps exactly what the single-cache worker keeps."""
    failures = []
    single = layer_worker(1, None, LAYER_BUDGET)
    explicit = layer_worker(N_LAYERS, range(0, N_LAYERS), LAYER_BUDGET, collapse=False)
    collapsed = layer_worker(N_LAYERS, range(0, N_LAYERS), LAYER_BUDGET)
    if collapsed._reindex_state.layer_selective:
        failures.append("configure did not collapse a full layer range to the uniform path")
    if not explicit._reindex_state.layer_selective or not explicit._reindex_state.all_layers_pruned:
        failures.append("the explicit full range is not on the general layer-selective path")
    ref = run_sequence(single, LAYER_TURNS)
    for name, worker in (("explicit", explicit), ("collapsed", collapsed)):
        got = run_sequence(worker, LAYER_TURNS)
        for step, (a, b) in enumerate(zip(ref, got)):
            if any(layer != a[0] for layer in b):
                failures.append(f"[{name}] step {step}: layers {b} != uniform {a[0]}")
        st, ps = worker._reindex_state, worker._kv_prune_state
        if not (st.pos_table.shape[-1] == ps.turn_id.numel() == len(got[-1][0])):
            failures.append(f"[{name}] master table/metadata desynced from the cache")
        if ps.n_budget_dropped != single._kv_prune_state.n_budget_dropped:
            failures.append(f"[{name}] budget_dropped {ps.n_budget_dropped} != "
                            f"{single._kv_prune_state.n_budget_dropped}")
    print(f"  explicit and collapsed full ranges match the single cache on {len(ref)} steps, "
          f"budget_dropped={single._kv_prune_state.n_budget_dropped}")
    return failures


def check_layer_selective(window=None):
    tag = f"window={window}"
    failures = []
    worker = layer_worker(N_LAYERS, LAYER_RANGE, LAYER_BUDGET, window=window)
    st, ps = worker._reindex_state, worker._kv_prune_state
    next_id, prev_pruned, prev_dropped, saw_ragged = 0, [], 0, False
    for step, length in enumerate(LAYER_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        feed_layers(worker, ids, (ids % 2).bool())
        layers = [layer.ids() for layer in worker.past_key_values.layers]
        master = st.pos_table[0, 0].tolist()
        alive = st.alive.tolist()
        alive_ids = [m for m, a in zip(master, alive) if a]
        n_turns = step + 1

        if layers[0] != master or len(master) != ps.turn_id.numel() != len(alive):
            failures.append(f"[{tag}] step {step}: unpruned layer {layers[0]} != master {master}")
        for l in LAYER_RANGE:
            if layers[l] != alive_ids:
                failures.append(f"[{tag}] step {step}: pruned layer {l} {layers[l]} != alive {alive_ids}")
        if len(alive_ids) > max(LAYER_BUDGET, PARITY_PREFIX):
            failures.append(f"[{tag}] step {step}: pruned layers hold {len(alive_ids)} > budget")
        for l, kept in enumerate(layers):
            if kept[:PARITY_PREFIX] != list(range(PARITY_PREFIX)):
                failures.append(f"[{tag}] step {step}: layer {l} lost the prefix")
            recent_lo = sum(LAYER_TURNS[:n_turns - RECENT])
            if [i for i in kept if i >= recent_lo] != list(range(recent_lo, next_id)):
                failures.append(f"[{tag}] step {step}: layer {l} lost a recent-turn slot")
        if window is not None and n_turns > window:
            first_keep = n_turns - window
            oldest_allowed = sum(LAYER_TURNS[:first_keep]) - len(worker.prefix_ids)
            for l, kept in enumerate(layers):
                stale = [i for i in kept if PARITY_PREFIX <= i < oldest_allowed]
                if stale:
                    failures.append(f"[{tag}] step {step}: layer {l} kept window-evicted slots {stale}")
        db = worker.past_image_embeds[0].reshape(-1).long().tolist()
        if db != [i for i in master if i % 2]:
            failures.append(f"[{tag}] step {step}: embed DB {db} is not the master's visual slots")
        # dead slots (dropped from the pruned layers earlier) must not be recounted
        pool = set(prev_pruned) | set(ids.tolist())
        gone = pool - set(alive_ids)
        gone_from_master = gone - set(master)
        newly_budget_dropped = gone - gone_from_master
        if ps.n_budget_dropped - prev_dropped != len(newly_budget_dropped):
            failures.append(f"[{tag}] step {step}: budget_dropped grew by "
                            f"{ps.n_budget_dropped - prev_dropped}, expected {len(newly_budget_dropped)}")
        # this step's drops rank below every kept unprotected alive slot (per-step greedy)
        recent_lo = sum(LAYER_TURNS[:n_turns - RECENT])
        kept_cand = [i for i in alive_ids if PARITY_PREFIX <= i < recent_lo]
        if newly_budget_dropped and kept_cand:
            if float(score_of(list(newly_budget_dropped)).max()) > float(score_of(kept_cand).min()) + 1e-6:
                failures.append(f"[{tag}] step {step}: dropped a higher-scored slot than one kept")
        saw_ragged |= len(layers[0]) > len(layers[1])
        prev_pruned, prev_dropped = alive_ids, ps.n_budget_dropped
        print(f"  [{tag}] step {step}: master {len(master)} / pruned {len(alive_ids)} slots, "
              f"budget_dropped {ps.n_budget_dropped}")
    if not ps.n_budget_dropped:
        failures.append(f"[{tag}] the budget never dropped anything")
    if not saw_ragged:
        failures.append(f"[{tag}] the unpruned layer never held more than the pruned ones")
    if window is not None and not worker._n_evicted:
        failures.append(f"[{tag}] the window never evicted")
    return failures


def check_empty_range():
    """layer_start == n_layers: the window schedule of evict, no budget drops, all layers equal."""
    failures = []
    evict = bare_worker("evict", PARITY_WINDOW)
    control = layer_worker(N_LAYERS, range(N_LAYERS, N_LAYERS), budget=3, window=PARITY_WINDOW,
                           recent_turns=2)
    control.use_sparse = False
    control.language_model = None
    if not control._reindex_state.no_pruned_layers:
        failures.append("an empty layer range is not recognised as 'no pruned layers'")
    next_id = 0
    for step, length in enumerate(PARITY_TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        a = feed_turn(evict, ids)
        per_layer = [ids.clone() for _ in range(N_LAYERS)] if control.past_key_values is None else \
            [torch.tensor(layer.ids() + ids.tolist()) for layer in control.past_key_values.layers]
        control.past_key_values = FakeCache(per_layer)
        pos = torch.arange(control.offset, control.offset + length)
        control._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
        control.offset += length
        control._apply_kv_prune()
        layers = [layer.ids() for layer in control.past_key_values.layers]
        if any(kept != a for kept in layers):
            failures.append(f"step {step}: control layers {layers} != evict {a}")
    if control._kv_prune_state.n_budget_dropped:
        failures.append("an empty pruned range dropped slots on the budget")
    if not evict._n_evicted:
        failures.append("the window never evicted, so nothing was tested")
    print(f"  empty range followed evict's schedule for {len(PARITY_TURNS)} steps with budget=3")
    return failures


# --- 6-8. real model ----------------------------------------------------------------------

def build_layer_worker(budget, layer_start=0, layer_end=None, importance="random",
                       attn_impl="sdpa", window=None, **kwargs):
    return VLMWorker(
        model_id=MODEL_ID, attn_impl=attn_impl, dtype="bfloat16", use_sparse=True,
        context_window=window, context_window_mode="prune", kv_budget=budget,
        kv_prune_importance=importance, kv_prune_layer_start=layer_start,
        kv_prune_layer_end=layer_end, kv_prune_seed=17, **kwargs,
    )


def lengths_ok(worker, budget, pruned, tag):
    """Layers in `pruned` hold alive (<= budget) slots, the others the master cache."""
    failures = []
    st = worker._reindex_state
    lengths = worker._kv_lengths()
    master = st.master_len()
    n_alive = int(st.alive.sum()) if st.alive is not None else master
    for l, n in enumerate(lengths):
        if l in pruned:
            if n != n_alive or n > budget:
                failures.append(f"[{tag}] layer {l} holds {n} (alive {n_alive}, budget {budget})")
        elif n != master:
            failures.append(f"[{tag}] unpruned layer {l} holds {n} != master {master}")
    ps = worker._kv_prune_state
    if not (master == ps.turn_id.numel() == st.pos_table.shape[-1]):
        failures.append(f"[{tag}] master {master} / metadata {ps.turn_id.numel()} / table desynced")
    if worker.past_image_embeds[0].shape[0] != int(ps.is_visual.sum()):
        failures.append(f"[{tag}] embed DB desynced from the master's visual slots")
    return failures


def check_gpu_arms(images):
    failures = []
    # (a) an explicit full range is the uniform path, bit for bit
    a = build_layer_worker(NARROW_BUDGET)
    n_layers = len(a.language_model.layers)
    probs_a, lens_a = run_probs(a, images), a._kv_lengths()
    release(a)
    b = build_layer_worker(NARROW_BUDGET, 0, n_layers)
    probs_b, lens_b = run_probs(b, images), b._kv_lengths()
    if b._reindex_state.layer_selective:
        failures.append("(0, n_layers) was not collapsed to the uniform path")
    release(b)
    if lens_a != lens_b or any(not np.array_equal(x, y) for x, y in zip(probs_a, probs_b)):
        failures.append("(0, None) and (0, n_layers) differ")
    print(f"  (0, None) == (0, {n_layers}): identical probs, final kv_len {lens_a[0]}")

    # (b) fidelity floor: wide budget at layer_start=14 vs stock
    plain = VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True)
    baseline = run_probs(plain, images)
    release(plain)
    wide = build_layer_worker(WIDE_BUDGET, GPU_LAYER_START)
    probs = run_probs(wide, images)
    failures += lengths_ok(wide, WIDE_BUDGET, range(GPU_LAYER_START, n_layers), "wide l14")
    if wide._kv_prune_state.n_budget_dropped:
        failures.append("a wide budget dropped slots at layer_start=14")
    release(wide)
    floor = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs))
    print(f"  wide budget, layer_start={GPU_LAYER_START}: max |p_stock - p| = {floor:.5f}")
    if floor > FIDELITY_TOL:
        failures.append(f"layer-selective path with nothing pruned moved the decision by {floor:.5f}")

    # (c) narrow budget at layer_start=14: ragged layers, real divergence
    narrow = build_layer_worker(NARROW_BUDGET, GPU_LAYER_START)
    probs = run_probs(narrow, images)
    failures += lengths_ok(narrow, NARROW_BUDGET, range(GPU_LAYER_START, n_layers), "narrow l14")
    dropped = narrow._kv_prune_state.n_budget_dropped
    lens = narrow._kv_lengths()
    release(narrow)
    signal = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs))
    print(f"  narrow budget, layer_start={GPU_LAYER_START}: signal={signal:.5f} vs floor={floor:.5f}, "
          f"layers 0/{GPU_LAYER_START}/{n_layers - 1} hold {lens[0]}/{lens[GPU_LAYER_START]}/{lens[-1]}, "
          f"budget_dropped={dropped}")
    if not dropped or lens[0] <= lens[-1]:
        failures.append("narrow budget at layer_start=14 never made the layers ragged; raise N_STEPS")
    if any(not np.all(np.isfinite(p)) for p in probs):
        failures.append("non-finite probabilities under layer-selective pruning")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"layer-selective pruning is inert ({signal:.5f} vs floor {floor:.5f})")

    # (d) band (0, 14): layer 0 pruned, deeper layers not -> the structural mask rebuild
    for impl in ("sdpa", "eager"):
        band = build_layer_worker(NARROW_BUDGET, 0, GPU_LAYER_START, attn_impl=impl)
        probs = run_probs(band, images)
        failures += lengths_ok(band, NARROW_BUDGET, range(0, GPU_LAYER_START), f"band {impl}")
        lens = band._kv_lengths()
        release(band)
        if any(not np.all(np.isfinite(p)) for p in probs) or lens[0] >= lens[-1]:
            failures.append(f"band (0, {GPU_LAYER_START}) under {impl}: lengths {lens[0]}/{lens[-1]}")
        print(f"  band (0, {GPU_LAYER_START}) under {impl}: layers 0/{n_layers - 1} hold {lens[0]}/{lens[-1]}")

    # (e) layer_start == n_layers: the budget never applies
    none = build_layer_worker(NARROW_BUDGET, n_layers)
    probs = run_probs(none, images)
    dropped, lens = none._kv_prune_state.n_budget_dropped, none._kv_lengths()
    release(none)
    delta = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs))
    print(f"  layer_start={n_layers}: delta vs stock {delta:.5f}, kv_len {lens[0]}, dropped {dropped}")
    if dropped or len(set(lens)) != 1 or delta > FIDELITY_TOL:
        failures.append(f"layer_start={n_layers} still pruned (dropped={dropped}, delta={delta:.5f})")
    return failures


def check_gpu_replay(images):
    failures = []
    worker = build_layer_worker(NARROW_BUDGET, GPU_LAYER_START, importance="fisher")
    n_layers = len(worker.language_model.layers)
    for step, image in enumerate(images):
        original, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
    st, ps = worker._reindex_state, worker._kv_prune_state
    if not ps.n_budget_dropped:
        failures.append("replay check ran on an unpruned cache")
    keys = [layer.keys for layer in worker.past_key_values.layers]
    values = [layer.values for layer in worker.past_key_values.layers]
    snap = (st.pos_table, st.alive.clone(), st.alive_index.clone())
    master = st.master_len()
    empty = torch.zeros(master, dtype=torch.bool)

    base = worker._decision_replay(worker._replay_base_bias(empty))
    base_probs = torch.softmax(base.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    delta = float(np.abs(base_probs - original).max())
    if delta > REPLAY_TOL:
        failures.append(f"unmasked replay delta {delta:.5f} exceeds {REPLAY_TOL}")
    if not all(layer.keys is k and layer.values is v
               for layer, k, v in zip(worker.past_key_values.layers, keys, values)):
        failures.append("replay did not restore the cache references")
    if st.pos_table is not snap[0] or not torch.equal(st.alive, snap[1]) \
            or not torch.equal(st.alive_index, snap[2]) or st.replay_extra is not None:
        failures.append("replay did not restore the reindex state")

    term = worker._replay_ablation_term(ps.is_visual)
    noop = worker._decision_replay(worker._replay_base_bias(empty), extra=term,
                                   extra_layers=range(n_layers, n_layers))
    noop_probs = torch.softmax(noop.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    if float(np.abs(noop_probs - base_probs).max()) > 1e-4:
        failures.append("an ablation over an empty layer range changed the decision")
    hit = worker._decision_replay(worker._replay_base_bias(empty), extra=term,
                                  extra_layers=range(0, n_layers))
    hit_probs = torch.softmax(hit.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    if float(np.abs(hit_probs - base_probs).max()) <= 1e-5:
        failures.append("hiding every visual slot on every layer did not move the decision")
    deep = worker._decision_replay(worker._replay_base_bias(empty), extra=term,
                                   extra_layers=range(GPU_LAYER_START, n_layers))
    deep_probs = torch.softmax(deep.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    print(f"  replay delta={delta:.5f}; hiding visual slots on layers >=0 / >={GPU_LAYER_START} "
          f"/ >={n_layers} moves the decision by "
          f"{float(np.abs(hit_probs - base_probs).max()):.4f} / "
          f"{float(np.abs(deep_probs - base_probs).max()):.4f} / "
          f"{float(np.abs(noop_probs - base_probs).max()):.6f}")

    fisher = worker._score_fisher_slots(empty)
    if fisher.numel() != master or not bool(torch.isfinite(fisher).all()):
        failures.append("Fisher scores are not master-length and finite")
    dead = ~st.alive
    if bool(dead.any()) and float(fisher[dead].abs().max()) > 0:
        failures.append("Fisher assigned a nonzero score to a slot the pruned layers do not hold")
    release(worker)

    for method in ("kl", "fisher"):
        worker = build_layer_worker(NARROW_BUDGET, GPU_LAYER_START, importance=method)
        replays = 0
        for step, image in enumerate(images):
            probs, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
            replays += worker._prune_replay_count
            if not np.all(np.isfinite(probs)):
                failures.append(f"[{method}] non-finite probabilities")
        failures += lengths_ok(worker, NARROW_BUDGET, range(GPU_LAYER_START, n_layers), method)
        if not worker._kv_prune_state.n_budget_dropped or not replays:
            failures.append(f"[{method}] never pruned or never replayed")
        print(f"  {method} at layer_start={GPU_LAYER_START}: kv layers 0/{n_layers - 1} hold "
              f"{worker._kv_lengths()[0]}/{worker._kv_lengths()[-1]}, replays={replays}")
        release(worker)
    return failures


def check_gpu_visual_blind(images):
    """kv_prune_visual_blind: no-op at n_layers, image-blind at 0, ragged and finite at 20."""
    failures = []
    plain = VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True)
    n_layers = len(plain.language_model.layers)
    baseline = run_probs(plain, images)
    release(plain)

    noop = build_layer_worker(None, n_layers, kv_prune_visual_blind=True)
    probs = run_probs(noop, images)
    delta = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs))
    n_vis = int(noop._kv_prune_state.is_visual.sum())
    lens = noop._kv_lengths()
    release(noop)
    print(f"  blind from layer {n_layers}: delta vs stock {delta:.5f}, visual slots {n_vis}, kv {lens[0]}")
    if delta > FIDELITY_TOL or n_vis == 0 or len(set(lens)) != 1:
        failures.append(f"blind from {n_layers} is not a no-op (delta {delta:.5f}, visual {n_vis})")

    rng = np.random.RandomState(7)
    other = [Image.fromarray(rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in images]
    blind = build_layer_worker(None, 0, kv_prune_visual_blind=True)
    probs_a = run_probs(blind, images)
    vis_after = int(blind._kv_prune_state.is_visual.sum())
    db_none = blind.past_image_embeds is None
    lens = blind._kv_lengths()
    blind.reset()
    probs_b = run_probs(blind, other)
    release(blind)
    image_delta = max(float(np.abs(x - y).max()) for x, y in zip(probs_a, probs_b))
    stock_delta = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs_a))
    print(f"  blind from layer 0: |p(images A) - p(images B)| = {image_delta:.6f}, "
          f"vs stock {stock_delta:.4f}, visual slots left {vis_after}, kv {lens[0]}")
    if image_delta > 1e-3:
        failures.append(f"a fully visual-blind model still depends on the image ({image_delta:.5f})")
    if vis_after or not db_none or len(set(lens)) != 1:
        failures.append("blind from 0 left visual slots or an embed DB behind")
    if stock_delta < 1e-3:
        failures.append("blind from 0 equals stock -- the mask is not applied")

    mid = build_layer_worker(None, 20, kv_prune_visual_blind=True)
    probs = run_probs(mid, images)
    st, ps = mid._reindex_state, mid._kv_prune_state
    lens = mid._kv_lengths()
    alive_vis = int((ps.is_visual & st.alive).sum())
    master = st.master_len()
    release(mid)
    delta = max(float(np.abs(x - y).max()) for x, y in zip(baseline, probs))
    print(f"  blind from layer 20: delta vs stock {delta:.5f}; layers 0/19/20/27 hold "
          f"{lens[0]}/{lens[19]}/{lens[20]}/{lens[27]} (master {master}, visual alive {alive_vis})")
    if any(not np.all(np.isfinite(p)) for p in probs):
        failures.append("blind from 20: non-finite probabilities")
    if alive_vis or any(lens[l] != master for l in range(20)) or any(lens[l] != lens[20] for l in range(20, n_layers)):
        failures.append(f"blind from 20: wrong per-layer composition {lens[0]}/{lens[20]}, visual alive {alive_vis}")
    if lens[20] >= master:
        failures.append("blind from 20: the blind layers did not shrink")
    return failures


def check_gpu_keep_one(images):
    """keep_one selector prunes to budget (uniform and layer-selective); the keep-one /
    leave-one diagnostic rows are well-formed and agree on a single-frame cache."""
    failures = []
    for layer_start in (0, GPU_LAYER_START):
        worker = build_layer_worker(NARROW_BUDGET, layer_start, importance="keep_one")
        n_layers = len(worker.language_model.layers)
        replays = 0
        for step, image in enumerate(images):
            probs, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
            replays += worker._prune_replay_count
            if not np.all(np.isfinite(probs)):
                failures.append(f"[keep_one l{layer_start}] non-finite probabilities")
        pruned = range(layer_start, n_layers)
        failures += lengths_ok(worker, NARROW_BUDGET, pruned, f"keep_one l{layer_start}")
        if not worker._kv_prune_state.n_budget_dropped or not replays:
            failures.append(f"[keep_one l{layer_start}] never pruned or never replayed")
        print(f"  keep_one at layer_start={layer_start}: kv layers 0/{n_layers - 1} hold "
              f"{worker._kv_lengths()[0]}/{worker._kv_lengths()[-1]}, replays={replays}")
        release(worker)

    starts = (0, GPU_LAYER_START, 28)
    worker = build_layer_worker(WIDE_BUDGET, kv_prune_log_keep_one=True,
                                kv_prune_layer_influence_starts=starts)
    for step, image in enumerate(images[:3]):
        worker.infer_probs(START if step == 0 else TURN, [image])
        keep, leave = worker._keep_one_rows, worker._leave_one_rows
        ok = (len(keep) == len(leave) == len(starts)
              and all(len(r) == step + 1 for r in keep + leave)
              and all(np.isfinite(v) and v >= 0 for r in keep + leave for v in r))
        if not ok:
            failures.append(f"step {step}: malformed keep/leave rows {keep} / {leave}")
            continue
        if max(keep[-1]) > 1e-6 or max(leave[-1]) > 1e-6:
            failures.append(f"step {step}: probe {starts[-1]} hides nothing but scored "
                            f"{keep[-1]} / {leave[-1]}")
        if step == 0 and abs(keep[0][0] - leave[0][0]) > 1e-4:
            # One cached frame: 'only f' is the full cache and 'without f' is text-only,
            # so both measures reduce to KL(P_full || P_text) -- up to the KL direction.
            failures.append(f"single-frame keep-one {keep[0][0]:.5f} != leave-one {leave[0][0]:.5f}")
        print(f"  step {step}: keep-one {[[f'{v:.4f}' for v in r] for r in keep[:2]]} "
              f"leave-one {[[f'{v:.4f}' for v in r] for r in leave[:2]]} (probes {starts[:2]}, "
              f"replays {worker._prune_replay_count})")
    release(worker)
    return failures


def check_gpu_layer_influence(images):
    failures = []
    starts = (0, GPU_LAYER_START, 28)
    worker = build_layer_worker(WIDE_BUDGET, kv_prune_log_layer_influence=True,
                                kv_prune_layer_influence_starts=starts)
    n_layers = len(worker.language_model.layers)
    for step, image in enumerate(images[:3]):
        worker.infer_probs(START if step == 0 else TURN, [image])
        for name in ("hist", "all"):
            row = getattr(worker, f"_layer_influence_{name}")
            if len(row) != len(starts) or not all(np.isfinite(v) and v >= 0 for v in row):
                failures.append(f"step {step}: {name} row {row} malformed")
            elif row[-1] > 1e-6:
                failures.append(f"step {step}: hiding nothing (probe {n_layers}) gave KL {row[-1]}")
        print(f"  step {step}: hist {[f'{v:.4f}' for v in worker._layer_influence_hist]} "
              f"all {[f'{v:.4f}' for v in worker._layer_influence_all]} "
              f"(probes {starts}, replays {worker._prune_replay_count})")
    if worker._layer_influence_all[0] <= 1e-6:
        failures.append("hiding every visual slot from layer 0 on had no effect")
    release(worker)
    return failures


def main():
    failures = []
    print("=== 1. mask_for_layer / hide_columns (no model) ===")
    failures += check_mask_for_layer()
    failures += check_hide_columns()
    print("=== 2. per-layer score scatter (no model) ===")
    failures += check_score_scatter()
    print(f"=== 3. uniform parity: {N_LAYERS} layers, full range (no model) ===")
    failures += check_uniform_parity()
    print(f"=== 4. layer-selective bookkeeping: layers {list(LAYER_RANGE)} of {N_LAYERS}, "
          f"budget {LAYER_BUDGET} (no model) ===")
    failures += check_layer_selective(window=None)
    failures += check_layer_selective(window=3)
    print("=== 5. empty pruned range (no model) ===")
    failures += check_empty_range()
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    images = make_images(N_STEPS)
    print(f"=== 6. real model: parity, fidelity, divergence, band, empty range ({N_STEPS} steps) ===")
    failures += check_gpu_arms(images)
    print("=== 7. real model: decision replay under layer-selective pruning ===")
    failures += check_gpu_replay(images)
    print("=== 8. real model: layer-influence diagnostic ===")
    failures += check_gpu_layer_influence(images)
    print("=== 9. real model: visual-blind arm ===")
    failures += check_gpu_visual_blind(images)
    print("=== 10. real model: keep_one selector and keep-one / leave-one diagnostic ===")
    failures += check_gpu_keep_one(images)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: layer-selective pruning keeps unpruned layers on the master cache and pruned "
          "layers on the budget, uniform pruning is untouched, masks/positions/replays follow "
          "each layer's own slot axis, and the layer-influence rows are well-formed.")


if __name__ == "__main__":
    main()
