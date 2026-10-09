"""Smoke test for the flat variant of paper/token_prune (+experiment=paper_prune_flat).

The flat arms drop the recency window (full episode context) and place every method at decoder
layer 0: the token is scored from layer 0 and deleted from every layer. Four things have to hold:

  1. Frame budget without a window -- the budget is k x the running mean visual count over every
     history frame, text is never touched, and with the mask log off only the scalar
     mean/vtp_n_new is carried (CPU, no model).
  2. Identity -- prune mode with no window and no budget reproduces the plain full-context
     (evict mode, nothing ever evicted) probabilities bit for bit (GPU).
  3. FastV at layer 0 -- kv_prune_score_layer=0 folds exactly one decision row, the survivors
     are the top-B' of that row among the visual slots, and the choice differs from the
     all-layer mean (GPU).
  4. Every flat method runs at k=1 with equal cache lengths on all layers (memory-matched),
     synced side DBs, finite probabilities and a decision that moves; DART reads layer 0 (GPU).

Run inside longnav_vlm:
    python tests/vtp_flat_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_frames_smoke import LEAD, PREFIX  # noqa: E402
from ctx_recompute_smoke import make_images, release, run_probs  # noqa: E402
from prune_fixtures import MODEL_ID, N_STEPS, START, TURN  # noqa: E402
from vtp_prune_smoke import feed, prune_worker, visual_chunk  # noqa: E402
from longnav.utils.kv_prune import KVPruneState, frame_budget  # noqa: E402
from longnav.utils.vlm_worker import VLMWorker  # noqa: E402

COMMON = dict(context_cut="header", context_window_mode="prune", kv_prune_candidate_scope="visual",
              kv_prune_recent_turns=0, kv_prune_ema_beta=0.0, kv_prune_seed=17,
              kv_prune_seed_per_step=True, kv_prune_log_masks=False)
METHODS = {
    "random": dict(kv_prune_importance="random"),
    "divprune": dict(kv_prune_importance="divprune"),
    "fastv": dict(kv_prune_importance="attn", kv_prune_score_layer=0),
    "dart": dict(kv_prune_importance="dart"),
}


# --- check 1: frame budget with no window (CPU) ------------------------------------------
def check_frame_budget(k=2):
    failures = []
    worker = prune_worker(None, kv_prune_budget_frames=float(k), kv_prune_log_masks=False)
    worker.context_cut = "header"
    counts = [6, 9, 5, 8, 7, 10]
    next_id = 0
    for turn, n_vis in enumerate(counts):
        n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
        ids = torch.arange(next_id, next_id + n)
        next_id += n
        feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD)
        ps, log = worker._kv_prune_state, worker._vtp_log
        h = turn + 1
        seen = counts[:h]
        expect_vis = sum(seen) if h <= k else int(round(k * sum(seen) / h))
        exp_text = PREFIX + sum((0 if t == 0 else LEAD) + 4 for t in range(h))
        n_vis_alive, n_text = int(ps.is_visual.sum()), int((~ps.is_visual).sum())
        print(f"  turn {turn}: {h} frames, visual alive {n_vis_alive} (expect {expect_vis}), text {n_text}")
        if n_vis_alive != expect_vis:
            failures.append(f"turn {turn}: {n_vis_alive} visual slots alive, expected {expect_vis}")
        if n_text != exp_text:
            failures.append(f"turn {turn}: {n_text} text slots, expected {exp_text} (text must never be pruned)")
        if frame_budget(ps, 0, float(k)) != (None if h <= k else expect_vis):
            failures.append(f"turn {turn}: frame_budget disagrees with the running mean")
        if log != {"mean/vtp_n_new": n_vis}:
            failures.append(f"turn {turn}: mask-less log is {log}")
        if worker.past_image_embeds[0].shape[0] != n_vis_alive:
            failures.append(f"turn {turn}: embed DB desynced")
    return failures


# --- GPU checks -----------------------------------------------------------------------
def build(**kwargs):
    return VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True, **kwargs)


def slot_keys(pos_table, mask):
    """Original mRoPE position triples of the masked slots: a stable per-slot identity."""
    pos = pos_table.reshape(3, -1)[:, mask.to(pos_table.device)]
    return {tuple(int(v) for v in col) for col in pos.T.tolist()}


def run_shadowed(worker, images):
    """Drive the episode; at every selection record the scores the selector saw, the slots it
    chose among (identified by their mRoPE positions) and what survived the prune."""
    records, current = [], {}
    orig_fold, orig_apply = KVPruneState.fold_scores, worker._apply_kv_prune

    def fold(ps, scores, beta):
        st = worker._reindex_state
        current.update(scores=scores.detach().float().cpu().clone(), is_visual=ps.is_visual.clone(),
                       pos_table=st.pos_table.clone(), folded=st.score_layers)
        return orig_fold(ps, scores, beta)

    def apply():
        current.clear()
        orig_apply()
        if current:
            ps, st = worker._kv_prune_state, worker._reindex_state
            records.append(dict(current, survivors=slot_keys(st.pos_table, ps.is_visual)))

    KVPruneState.fold_scores = fold
    worker._apply_kv_prune = apply
    try:
        for idx, image in enumerate(images):
            worker.infer_probs(START if idx == 0 else TURN, [image])
    finally:
        KVPruneState.fold_scores = orig_fold
    return records


def check_identity(images):
    failures = []
    full = build(context_cut="header", context_window_mode="evict")
    base = run_probs(full, images)
    base_len = full._kv_lengths()[0]
    release(full)
    ident = build(**COMMON)
    probs = run_probs(ident, images)
    lens = ident._kv_lengths()
    release(ident)
    exact = all(np.array_equal(a, b) for a, b in zip(base, probs))
    floor = max(float(np.abs(a - b).max()) for a, b in zip(base, probs))
    print(f"  identity vs full context: {'bit-identical' if exact else f'max |dp| = {floor:.2e}'}, "
          f"kv_len {lens[0]} vs {base_len}")
    if not exact:
        failures.append(f"identity arm differs from the full-context model (max |dp| {floor:.2e})")
    if lens[0] != base_len or len(set(lens)) != 1:
        failures.append(f"identity arm holds {lens}, full context {base_len}")
    return failures, base


def check_fastv_layer0(images):
    failures = []
    worker = build(kv_prune_budget_frames=1.0, **COMMON, **METHODS["fastv"])
    st = worker._reindex_state
    if st.score_layer_set != range(0, 1) or st.layer_selective:
        failures.append(f"score layers {st.score_layer_set!r}, layer_selective={st.layer_selective}")
    records = run_shadowed(worker, images)
    pruned_steps = 0
    for rec in records:
        if rec["folded"] != 1:
            failures.append(f"{rec['folded']} decision rows folded, expected layer 0 only")
        cand = torch.nonzero(rec["is_visual"]).flatten()
        n_keep = len(rec["survivors"])
        if n_keep == cand.numel():
            continue
        pruned_steps += 1
        top = cand[torch.argsort(rec["scores"][cand], descending=True, stable=True)[:n_keep]]
        mask = torch.zeros_like(rec["is_visual"])
        mask[top] = True
        expect = slot_keys(rec["pos_table"], mask)
        if expect != rec["survivors"]:
            failures.append(f"survivors are not the top-{n_keep} of the layer-0 row "
                            f"({len(expect ^ rec['survivors'])} slots differ)")
    print(f"  fastv@0: {pruned_steps} pruning steps checked against the layer-0 row")
    if not pruned_steps:
        failures.append("fastv@0 never pruned")
    kept0 = records[-1]["survivors"]
    release(worker)

    mean = build(kv_prune_budget_frames=1.0, **COMMON, kv_prune_importance="attn")
    if mean._reindex_state.score_layer_set is not None:
        failures.append("plain attn no longer averages every layer")
    records = run_shadowed(mean, images)
    if records and records[-1]["folded"] < 2:
        failures.append("plain attn folded a single layer")
    differ = len(kept0 ^ records[-1]["survivors"])
    print(f"  layer-0 row vs all-layer mean: {differ} surviving slots differ")
    if not differ:
        failures.append("layer-0 scoring selects exactly what the all-layer mean selects")
    release(mean)
    return failures


def check_methods(images, base):
    failures = []
    for name, cfg in METHODS.items():
        worker = build(kv_prune_budget_frames=1.0, **COMMON, **cfg)
        try:
            p = run_probs(worker, images)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: raised {exc!r}")
            release(worker)
            continue
        ps, st = worker._kv_prune_state, worker._reindex_state
        lens = worker._kv_lengths()
        n_vis = int(ps.is_visual.sum())
        expect = frame_budget(ps, 0, 1.0)
        signal = max(float(np.abs(a - b).max()) for a, b in zip(base, p))
        print(f"  {name}: signal {signal:.4f}, dropped {ps.n_budget_dropped}, visual alive {n_vis} "
              f"(budget {expect}), layers hold {lens[0]}..{lens[-1]}")
        if any(not np.all(np.isfinite(x)) for x in p):
            failures.append(f"{name}: non-finite probabilities")
        if n_vis != expect:
            failures.append(f"{name}: {n_vis} visual slots alive, budget {expect}")
        if len(set(lens)) != 1 or st.layer_selective:
            failures.append(f"{name}: layers are not memory-matched {lens}")
        if not ps.n_budget_dropped or signal <= 0:
            failures.append(f"{name}: nothing pruned or the decision never moved")
        if worker.past_image_embeds[0].shape[0] != n_vis:
            failures.append(f"{name}: embed DB desynced")
        if name == "dart" and worker._hidden_db.shape[0] != n_vis:
            failures.append("dart: hidden DB desynced")
        release(worker)
    return failures


def main():
    failures = []
    print("check 1: frame budget without a window")
    failures += check_frame_budget()
    if torch.cuda.is_available() and "--cpu" not in sys.argv:
        images = make_images(N_STEPS)
        print("check 2: identity vs the full-context model")
        f, base = check_identity(images)
        failures += f
        print("check 3: FastV scored at layer 0")
        failures += check_fastv_layer0(images)
        print("check 4: flat methods at k=1")
        failures += check_methods(images, base)
    else:
        print("checks 2-4 skipped (no GPU)")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: window-free frame budget, identity, layer-0 FastV, flat methods")


if __name__ == "__main__":
    main()
