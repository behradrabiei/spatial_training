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
     synced side DBs, finite probabilities and a decision that moves; DART reads layer 0 (GPU);
     the whole-frame arms leave exactly one frame's slots.
  5. Whole-frame selectors (kl_keep_one / entropy_keep_one / recency, Exp 4b) -- exactly k frames
     survive once the pool exceeds k, ranked by a stubbed keep-one KL table / the entropy of a
     stubbed only-f log-prob table (lowest kept) / by frame id with ties to the newer frame, text
     and the random generator untouched, the sup/kl_select record well-formed (CPU).
  6. kl_keep_one on the model -- the selector's frame scores equal the keep_one diagnostic's row
     on the same cache, the survivors are that row's top-k frames, the selector costs exactly
     1 + pool replays per pruning step and none below, the diagnostic does not move its decision,
     and random/identity never enter the whole-frame path (GPU).
  7. entropy_keep_one on the model -- the selector's scores equal the entropy of the keep_one
     diagnostic's only-f log-prob rows, the survivors are the pool's k lowest-entropy frames, the
     selector costs exactly pool replays per pruning step (no reference) and none below, and the
     diagnostic does not move its decision (GPU).

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
    "kl_keep_one": dict(kv_prune_importance="kl_keep_one"),
    "entropy_keep_one": dict(kv_prune_importance="entropy_keep_one"),
    "recency": dict(kv_prune_importance="recency"),
}
WHOLE_FRAME = ("kl_keep_one", "entropy_keep_one", "recency")
REPLAY = ("kl_keep_one", "entropy_keep_one")
KL_TOL = 1e-5  # selector score vs the diagnostic's keep-one KL / entropy (same replays; the log rounds to 6 dp)


def entropy(logp):
    """Entropy of a log-prob row, the selector's arithmetic (fp32)."""
    lp = torch.as_tensor(logp, dtype=torch.float32)
    return max(float(-(lp.exp() * lp).sum()), 0.0)


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


# --- check 5: whole-frame selectors (CPU) ---------------------------------------------------
def check_whole_frame(k=2):
    failures = []
    table = {0: 0.05, 1: 0.5, 2: 0.01, 3: 0.3, 4: 0.3, 5: 0.02}  # frame -> stubbed keep-one KL
    probs = {0: [0.97, 0.01, 0.01, 0.01], 1: [0.25, 0.25, 0.25, 0.25], 2: [0.49, 0.49, 0.01, 0.01],
             3: [0.8, 0.1, 0.05, 0.05], 4: [0.8, 0.1, 0.05, 0.05], 5: [0.9, 0.05, 0.03, 0.02]}
    logp_table = {f: torch.log(torch.tensor(p)) for f, p in probs.items()}  # stubbed only-f decisions
    H = {f: entropy(lp) for f, lp in logp_table.items()}  # frames 3/4 tie -> the newer (4) ranks first
    counts = [6, 9, 5, 8, 7, 10]
    for importance in WHOLE_FRAME:
        worker = prune_worker(None, importance=importance, kv_prune_budget_frames=float(k), kv_prune_log_masks=False)
        worker.context_cut = "header"
        calls = []
        worker._replay_base_bias = lambda forced: None

        def fake_kls(base_bias, frames, layers, vis_logp=None):
            calls.append(list(frames))
            return None, {f: (table[f], None) for f in frames}

        def fake_logps(base_bias, frames, layers):
            calls.append(list(frames))
            return {f: logp_table[f] for f in frames}

        worker._keep_one_frame_kls = fake_kls
        worker._keep_one_frame_logps = fake_logps
        gen_state = worker._kv_prune_state.generator.get_state().clone()
        alive, next_id = set(), 0
        for turn, n_vis in enumerate(counts):
            n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
            ids = torch.arange(next_id, next_id + n)
            next_id += n
            feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD)
            ps, log = worker._kv_prune_state, worker._kl_select_log
            pool = sorted(alive | {turn})
            if len(pool) <= k:
                expect = set(pool)
            elif importance == "kl_keep_one":
                expect = set(sorted(pool, key=lambda f: (-table[f], -f))[:k])
            elif importance == "entropy_keep_one":
                expect = set(sorted(pool, key=lambda f: (H[f], -f))[:k])
            else:
                expect = set(pool[-k:])
            got = set(ps.turn_id[ps.is_visual].tolist())
            print(f"  {importance} turn {turn}: pool {pool} -> alive {sorted(got)} (expect {sorted(expect)})")
            if got != expect:
                failures.append(f"{importance} turn {turn}: alive frames {sorted(got)}, expected {sorted(expect)}")
            alive = got
            exp_text = PREFIX + sum((0 if t == 0 else LEAD) + 4 for t in range(turn + 1))
            if int((~ps.is_visual).sum()) != exp_text:
                failures.append(f"{importance} turn {turn}: text slots {int((~ps.is_visual).sum())}, expected {exp_text}")
            if worker.past_image_embeds[0].shape[0] != int(ps.is_visual.sum()):
                failures.append(f"{importance} turn {turn}: embed DB desynced")
            if len(pool) <= k:
                if log != {} or (importance in REPLAY and calls and calls[-1] == pool and turn < k):
                    failures.append(f"{importance} turn {turn}: pool within budget but log {log}")
                continue
            score = {f: (table[f] if importance == "kl_keep_one" else H[f] if importance == "entropy_keep_one"
                         else float(f)) for f in pool}
            n_replays = {"kl_keep_one": 1 + len(pool), "entropy_keep_one": len(pool)}.get(importance, 0)
            if log.get("frames") != pool or log.get("kept") != [int(f in expect) for f in pool] \
                    or any(abs(s - score[f]) > 1e-6 for s, f in zip(log.get("score", []), pool)) \
                    or log.get("n_replays") != n_replays or log.get("measure") != importance:
                failures.append(f"{importance} turn {turn}: malformed kl_select log {log}")
            if importance in REPLAY and calls[-1] != pool:
                failures.append(f"{importance} turn {turn}: scored {calls[-1]}, pool is {pool}")
            imp = ps.importance[ps.is_visual]
            want = torch.tensor([score[f] for f in ps.turn_id[ps.is_visual].tolist()])
            if not torch.allclose(imp, want):
                failures.append(f"{importance} turn {turn}: importance does not carry the frame score")
        if not torch.equal(worker._kv_prune_state.generator.get_state(), gen_state):
            failures.append(f"{importance}: the random generator was consumed")
        if importance == "recency" and calls:
            failures.append("recency replayed something")
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
        frames_alive = len(set(ps.turn_id[ps.is_visual].tolist()))
        # Token-budget arms keep k x the running mean; whole-frame arms keep exactly k frames.
        expect = frames_alive if name in WHOLE_FRAME else frame_budget(ps, 0, 1.0)
        signal = max(float(np.abs(a - b).max()) for a, b in zip(base, p))
        print(f"  {name}: signal {signal:.4f}, dropped {ps.n_budget_dropped}, visual alive {n_vis} "
              f"(budget {expect}), frames alive {frames_alive}, layers hold {lens[0]}..{lens[-1]}")
        if any(not np.all(np.isfinite(x)) for x in p):
            failures.append(f"{name}: non-finite probabilities")
        if name in WHOLE_FRAME:
            if frames_alive != 1:
                failures.append(f"{name}: {frames_alive} frames hold visual slots, expected exactly 1")
        elif n_vis != expect:
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


def shadow_select(worker, images):
    """Drive the episode recording, after every prune, the whole-frame log, the diagnostic's
    keep-one row, the frames still holding visual slots and the replay count of that step."""
    records = []
    orig_apply = worker._apply_kv_prune

    def apply():
        orig_apply()
        ps = worker._kv_prune_state
        records.append(dict(log=dict(worker._kl_select_log), diag=list(worker._kl_influence_keep_one),
                            diag_logp=[list(r) for r in worker._kl_influence_keep_one_logp],
                            alive=sorted(set(ps.turn_id[ps.is_visual].tolist())),
                            replays=worker._prune_replay_count, n_turns=ps.n_turns,
                            counts=list(ps.visual_per_turn)))

    worker._apply_kv_prune = apply
    probs = run_probs(worker, images)
    return records, probs


def check_replay_selector(images, importance, k=2):
    """kl_keep_one / entropy_keep_one on the model vs the keep_one diagnostic on the same cache:
    the selector's scores are the diagnostic's KL row / the entropy of its only-f log-prob rows,
    the survivors the pool's top-k (KL) / bottom-k (entropy), the replays 1 + pool / pool."""
    failures = []
    from longnav.utils.kv_prune import topk_frames

    with_diag = build(kv_prune_budget_frames=float(k), **COMMON, kv_prune_importance=importance,
                      kv_prune_log_influence=True, kv_prune_influence_measures=("keep_one",))
    recs, p_diag = shadow_select(with_diag, images)
    release(with_diag)
    alone = build(kv_prune_budget_frames=float(k), **COMMON, kv_prune_importance=importance)
    recs_alone, p_alone = shadow_select(alone, images)
    release(alone)
    pruned_steps, max_diff = 0, 0.0
    for rec, rec_a in zip(recs, recs_alone):
        log, alive = rec["log"], rec["alive"]
        pool = [f for f in range(rec["n_turns"]) if rec["counts"][f] > 0 and (f in alive or f == rec["n_turns"] - 1
                                                                                or f in log.get("frames", []))]
        if len(pool) <= k:
            if log or rec_a["replays"]:
                failures.append(f"step {rec['n_turns'] - 1}: pool {pool} within budget but log {log} / "
                                f"{rec_a['replays']} selector replays")
            continue
        pruned_steps += 1
        frames = log.get("frames", [])
        if sorted(frames) != pool:
            failures.append(f"step {rec['n_turns'] - 1}: scored {frames}, pool is {pool}")
        if importance == "kl_keep_one":
            expect = {f: rec["diag"][f] for f in frames}
            n_replays = 1 + len(frames)
        else:
            expect = {f: entropy(rec["diag_logp"][f]) for f in frames}
            n_replays = len(frames)
        diff = max(abs(s - expect[f]) for s, f in zip(log["score"], frames))
        max_diff = max(max_diff, diff)
        if diff > KL_TOL:
            failures.append(f"step {rec['n_turns'] - 1}: selector score differs from the keep_one row by {diff:.2e}")
        sign = -1.0 if importance == "entropy_keep_one" else 1.0
        want = topk_frames(frames, [sign * expect[f] for f in frames], k)
        if alive != want or [int(f in want) for f in frames] != log["kept"]:
            failures.append(f"step {rec['n_turns'] - 1}: survivors {alive}, expected {want}")
        if log["n_replays"] != n_replays or rec_a["replays"] != n_replays:
            failures.append(f"step {rec['n_turns'] - 1}: {log['n_replays']} logged / {rec_a['replays']} counted "
                            f"replays, expected {n_replays}")
        if rec_a["log"] != log:
            failures.append(f"step {rec['n_turns'] - 1}: the diagnostic changed the selection log")
    moved = max(float(np.abs(a - b).max()) for a, b in zip(p_diag, p_alone))
    print(f"  {importance}: {pruned_steps} pruning steps, max |score - keep_one row| = {max_diff:.2e}, "
          f"diagnostic on/off max |dp| = {moved:.2e}")
    if not pruned_steps:
        failures.append(f"{importance} never pruned")
    if moved > 1e-4:
        failures.append(f"the keep_one diagnostic moved the {importance} decision by {moved:.2e}")
    return failures


def check_kl_keep_one(images, k=2):
    failures = check_replay_selector(images, "kl_keep_one", k)
    for name in ("random", "identity"):
        cfg = METHODS["random"] if name == "random" else {}
        budget = dict(kv_prune_budget_frames=1.0) if name == "random" else {}
        worker = build(**budget, **COMMON, **cfg)
        recs, _ = shadow_select(worker, images)
        release(worker)
        if any(r["log"] or r["replays"] for r in recs):
            failures.append(f"{name}: entered the whole-frame path (log/replays non-empty)")
    return failures


def main():
    failures = []
    print("check 1: frame budget without a window")
    failures += check_frame_budget()
    print("check 5: whole-frame selectors (stubbed scores)")
    failures += check_whole_frame()
    if torch.cuda.is_available() and "--cpu" not in sys.argv:
        images = make_images(N_STEPS)
        print("check 2: identity vs the full-context model")
        f, base = check_identity(images)
        failures += f
        print("check 3: FastV scored at layer 0")
        failures += check_fastv_layer0(images)
        print("check 4: flat methods at k=1")
        failures += check_methods(images, base)
        print("check 6: kl_keep_one vs the keep_one diagnostic")
        failures += check_kl_keep_one(images)
        print("check 7: entropy_keep_one vs the keep_one diagnostic's only-f rows")
        failures += check_replay_selector(images, "entropy_keep_one")
    else:
        print("checks 2-4, 6-7 skipped (no GPU)")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: window-free frame budget, identity, layer-0 FastV, flat methods, whole-frame selectors")


if __name__ == "__main__":
    main()
