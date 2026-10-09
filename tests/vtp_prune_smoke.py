"""Smoke test for the visual-token pruning arms of paper/token_prune.

The arms run context_window_mode='prune' with context_cut='frame', a frame-relative budget
(kv_prune_budget_frames), visual-only candidates that include the just-decided turn
(candidate_scope='visual', recent_turns=0) and the per-frame keep-mask log. Five things have
to hold:

  1. Frame-cut parity -- with no budget, prune retires exactly the tokens the evict frame
     cut retires, step for step, for K = 1, 2, 3 (CPU, no model).
  2. Frame budget -- once the window holds more than k frames, exactly k x (mean original
     visual count) candidates survive, text is untouched, and the mask log decodes to the
     surviving counts (CPU).
  3. Staged pruning -- PyramidDrop stages hold nested slot sets under the geometric budget
     schedule, each ranked by its own layer's decision row (CPU).
  4. Selectors -- DivPrune and DART agree with brute-force references; per-step seeding
     makes the random draw independent of the trajectory prefix (CPU).
  5. GPU -- the identity arm reproduces the evict frame-cut arm bit for bit; every method
     runs at k=1 with consistent per-layer cache lengths, a well-formed log, finite
     probabilities and a decision that actually moves.

Run inside longnav_vlm:
    python tests/vtp_prune_smoke.py
"""
import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_frames_smoke import LEAD, PREFIX, TURNS, drive, parity_worker  # noqa: E402
from ctx_recompute_smoke import make_images, release, run_probs  # noqa: E402
from prune_fixtures import MODEL_ID, N_STEPS, FakeCache, bare_worker  # noqa: E402
from longnav.utils.kv_prune import (  # noqa: E402
    dart_select, divprune_select, pyramid_budgets, step_seed,
)
from longnav.utils.vlm_worker import VLMWorker  # noqa: E402

CONTEXT_FRAMES = 3  # small window so the 6-step smoke prunes on most steps
COMMON = dict(context_cut="frame", context_window_mode="prune", kv_prune_candidate_scope="visual",
              kv_prune_recent_turns=0, kv_prune_ema_beta=0.0, kv_prune_seed=17,
              kv_prune_seed_per_step=True, kv_prune_log_masks=True)
METHODS = {
    "random": dict(kv_prune_importance="random"),
    "divprune": dict(kv_prune_importance="divprune"),
    "fastv": dict(kv_prune_importance="attn", kv_prune_layer_start=2, kv_prune_score_layers="boundary"),
    "dart": dict(kv_prune_importance="dart", kv_prune_layer_start=2),
    "pyramiddrop": dict(kv_prune_importance="attn", kv_prune_stages=[7, 14, 21]),
}


def prune_worker(window, n_layers=1, layer_range=None, stages=None, importance="random", **kwargs):
    """bare_worker in the token_prune configuration (frame cut, visual candidates, no budget)."""
    worker = bare_worker("prune", window, budget=None, importance=importance, recent_turns=0,
                         n_layers=n_layers, layer_range=layer_range)
    worker.context_cut = "frame"
    worker.kv_prune_candidate_scope = "visual"
    worker.kv_prune_log_masks = True
    worker.prefix_ids = list(range(4))
    worker._prefix_len = PREFIX
    if stages is not None:
        st = worker._reindex_state
        st.configure(pruned_layers=None, score_layers=tuple(sorted({max(r.start - 1, 0) for r in stages})),
                     n_layers=n_layers, stages=stages)
    for key, value in kwargs.items():
        setattr(worker, key, value)
    return worker


def feed(worker, ids, vis_row=None, lead=0, rows=None):
    """Append one chunk (global ids) to every layer, then prune; returns layer-0 ids."""
    live = worker.past_key_values.layers[0].ids() if worker.past_key_values is not None else []
    n_layers = worker._n_fake_layers
    if worker.past_key_values is None:
        per_layer = [list(ids.tolist()) for _ in range(n_layers)]
    else:
        per_layer = [layer.ids() + ids.tolist() for layer in worker.past_key_values.layers]
    worker.past_key_values = FakeCache([torch.tensor(x) for x in per_layer], n_layers)
    st = worker._reindex_state
    pos = torch.arange(worker.offset, worker.offset + len(ids))
    st.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
    worker.offset += len(ids)
    if vis_row is not None:
        worker.use_sparse = True
        worker.language_model = SimpleNamespace(visual_pos_masks=vis_row.reshape(1, -1),
                                                vis_keep_mask=torch.ones(int(vis_row.sum()), dtype=torch.bool))
        n_vis = int(vis_row.sum())
        emb = F.normalize(torch.randn(n_vis, 8), dim=-1)
        worker.past_image_embeds = [emb] if worker.past_image_embeds is None \
            else [torch.cat([worker.past_image_embeds[0], emb])]
    worker._turn_leads.append(lead)
    if rows is not None:
        st.arm_capture()
        st.layer_rows = {l: r.clone() for l, r in rows.items()}
    worker._apply_kv_prune()
    return worker.past_key_values.layers[0].ids()


def decode_log(log):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(log["vtp_bits"]), dtype=np.uint8))
    out, pos = [], 0
    for n in log["vtp_ntok"]:
        out.append(int(bits[pos:pos + n].sum()))
        pos += n
    return out


# --- check 1: frame-cut parity vs evict (CPU) -------------------------------------------
def drive_prune(worker):
    retained, next_id = [], 0
    for turn, length in enumerate(TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        lead = 0 if turn == 0 else LEAD
        if worker._strip_next_lead:
            ids = ids[lead:]
            worker._strip_next_lead = False
            lead = 0
        retained.append(list(feed(worker, ids, lead=lead)))
    return retained


def check_frame_cut_parity():
    failures = []
    for K in (1, 2, 3):
        evict = parity_worker("evict", K - 1, "frame")
        expect, _ = drive(evict, "frame")
        got = drive_prune(prune_worker(K - 1))
        for step, (a, b) in enumerate(zip(expect, got)):
            print(f"  K={K} step {step}: {len(a)} tokens ({'match' if a == b else 'MISMATCH'})")
            if a != b:
                failures.append(f"K={K} step {step}: evict retains {a}, prune retains {b}")
    return failures


# --- check 2: frame-relative budget and the mask log (CPU) ------------------------------
def visual_chunk(n_text_lead, n_vis, n_text_tail=4):
    n = n_text_lead + n_vis + n_text_tail
    vis = torch.zeros(n, dtype=torch.bool)
    vis[n_text_lead:n_text_lead + n_vis] = True
    return n, vis


def check_frame_budget(k=1, window=3):
    failures = []
    worker = prune_worker(window, kv_prune_budget_frames=float(k))
    counts = [6, 9, 5, 8, 7, 10]
    next_id = 0
    for turn, n_vis in enumerate(counts):
        n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
        ids = torch.arange(next_id, next_id + n)
        next_id += n
        feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD)
        ps, log = worker._kv_prune_state, worker._vtp_log
        first_keep = max(0, ps.n_turns - window)
        h = ps.n_turns - first_keep
        window_counts = counts[first_keep:ps.n_turns]
        n_vis_alive = int(ps.is_visual.sum())
        n_text = int((~ps.is_visual).sum())
        # text: prefix + every window turn's text minus the boundary lead
        exp_text = PREFIX + sum((0 if t == 0 else LEAD) + 4 for t in range(first_keep, ps.n_turns)) \
            - (LEAD if first_keep > 0 else 0)
        expect_vis = sum(window_counts) if h <= k else int(round(k * sum(window_counts) / h))
        print(f"  turn {turn}: window {first_keep}..{turn} ({h} frames), visual alive {n_vis_alive} "
              f"(expect {expect_vis}), text {n_text}, log kept {log['vtp_kept']}")
        if n_vis_alive != expect_vis:
            failures.append(f"turn {turn}: {n_vis_alive} visual slots alive, expected {expect_vis}")
        if n_text != exp_text:
            failures.append(f"turn {turn}: {n_text} text slots, expected {exp_text}")
        if log["vtp_frames"] != list(range(first_keep, ps.n_turns)) or log["vtp_ntok"] != window_counts:
            failures.append(f"turn {turn}: log frames/ntok {log['vtp_frames']}/{log['vtp_ntok']}")
        if decode_log(log) != log["vtp_kept"] or sum(log["vtp_kept"]) != n_vis_alive:
            failures.append(f"turn {turn}: packed bits decode to {decode_log(log)}, kept {log['vtp_kept']}")
        if log["mean/vtp_n_new"] != n_vis or len(log["vtp_raster"]) != n_vis:
            failures.append(f"turn {turn}: entering-frame fields wrong")
        if worker.past_image_embeds[0].shape[0] != n_vis_alive:
            failures.append(f"turn {turn}: embed DB desynced")
    return failures


# --- check 3: PyramidDrop stages (CPU) --------------------------------------------------
def check_staged(k=1, window=3, n_layers=6):
    failures = []
    stages = [range(2, 4), range(4, 6)]
    worker = prune_worker(window, n_layers=n_layers, stages=stages, importance="attn",
                          kv_prune_budget_frames=float(k), kv_prune_stages=[2, 4])
    counts = [6, 9, 5, 8, 7]
    next_id = 0
    gen = torch.Generator().manual_seed(3)
    for turn, n_vis in enumerate(counts):
        n, vis = visual_chunk(PREFIX if turn == 0 else LEAD, n_vis)
        ids = torch.arange(next_id, next_id + n)
        next_id += n
        st = worker._reindex_state
        master = st.master_len() + n
        rows = {1: torch.rand(master, generator=gen), 3: torch.rand(master, generator=gen)}
        feed(worker, ids, vis_row=vis, lead=0 if turn == 0 else LEAD, rows=rows)
        ps = worker._kv_prune_state
        first_keep = max(0, ps.n_turns - window)
        h = ps.n_turns - first_keep
        lengths = worker._kv_lengths()
        a0, a1 = st.stage_alive(0), st.stage_alive(1)
        if bool((a1 & ~a0).any()):
            failures.append(f"turn {turn}: stage 1 holds a slot stage 0 dropped")
        if lengths[:2] != [st.master_len()] * 2 or lengths[2:4] != [int(a0.sum())] * 2 \
                or lengths[4:] != [int(a1.sum())] * 2:
            failures.append(f"turn {turn}: layer lengths {lengths} vs master {st.master_len()}, "
                            f"alive {int(a0.sum())}/{int(a1.sum())}")
        vis_alive = [int((ps.is_visual & a).sum()) for a in (a0, a1)]
        total = sum(counts[first_keep:ps.n_turns])
        if h <= k:
            expect = [total, total]
        else:
            expect = pyramid_budgets(total, int(round(k * total / h)), 2)
        print(f"  turn {turn}: {h} frames, visual alive per stage {vis_alive} (expect {expect}), "
              f"layer lengths {lengths}")
        if vis_alive != expect:
            failures.append(f"turn {turn}: stage visual counts {vis_alive}, expected {expect}")
        if worker._vtp_log["vtp_kept"] and sum(worker._vtp_log["vtp_kept"]) != vis_alive[1]:
            failures.append(f"turn {turn}: log reports {sum(worker._vtp_log['vtp_kept'])} at the deepest stage")
    if pyramid_budgets(100, 12, 3)[-1] != 12 or pyramid_budgets(100, 12, 3) != sorted(
            pyramid_budgets(100, 12, 3), reverse=True):
        failures.append(f"pyramid_budgets schedule {pyramid_budgets(100, 12, 3)} is not monotone to 12")
    return failures


# --- check 4: selectors vs brute force, per-step seeding (CPU) ---------------------------
def ref_divprune(x, capacity):
    x = F.normalize(x.float(), dim=-1)
    d = 1.0 - x @ x.T
    d.fill_diagonal_(torch.inf)
    chosen = [int(d.min(dim=1).values.argmax())]
    while len(chosen) < capacity:
        best, best_val = None, -1.0
        for i in range(x.shape[0]):
            if i in chosen:
                continue
            val = min(float(d[i, j]) for j in chosen)
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return chosen


def ref_dart(norms, hidden, capacity, n_pivots):
    h = F.normalize(hidden.float(), dim=-1)
    order = sorted(range(len(norms)), key=lambda i: -float(norms[i]))
    pivots = order[:n_pivots]
    remaining = [i for i in range(len(norms)) if i not in pivots]
    kept = list(pivots)
    quota, extra = divmod(capacity - n_pivots, n_pivots)
    for rank, p in enumerate(pivots):
        q = quota + (1 if rank < extra else 0)
        remaining.sort(key=lambda i: float(h[i] @ h[p]))
        kept += remaining[:q]
        remaining = remaining[q:]
    return sorted(kept)


def check_selectors():
    failures = []
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(40, 16, generator=gen)
    cand = torch.arange(100, 140)
    got = divprune_select(x, cand, 10).tolist()
    ref = [100 + i for i in ref_divprune(x, 10)]
    print(f"  divprune: {got == ref} ({got[:5]}...)")
    if got != ref:
        failures.append(f"divprune {got} != reference {ref}")
    if divprune_select(x, cand, 40).tolist() != cand.tolist() or divprune_select(x, cand, 0).numel():
        failures.append("divprune edge cases (capacity == n, capacity == 0)")
    norms = torch.rand(40, generator=gen)
    hidden = torch.randn(40, 12, generator=gen)
    got = sorted(dart_select(norms, hidden, cand, 14, n_pivots=4).tolist())
    ref = [100 + i for i in ref_dart(norms, hidden, 14, 4)]
    print(f"  dart: {got == ref} ({got[:6]}...)")
    if got != ref:
        failures.append(f"dart {got} != reference {ref}")
    for cap in (3, 7, 40):
        n = dart_select(norms, hidden, cand, cap, n_pivots=8).numel()
        if n != cap:
            failures.append(f"dart retained {n} of {cap}")
    # per-step seeding: the draw at a step ignores how many draws came before it
    a = torch.Generator().manual_seed(step_seed(17, "scene_3", 5))
    b = torch.Generator().manual_seed(step_seed(17, "scene_3", 4))
    torch.rand(3, generator=b)
    b.manual_seed(step_seed(17, "scene_3", 5))
    if not torch.equal(torch.randperm(50, generator=a), torch.randperm(50, generator=b)):
        failures.append("per-step seed is not prefix-independent")
    if step_seed(17, "scene_3", 5) == step_seed(17, "scene_4", 5) or \
            step_seed(17, "scene_3", 5) == step_seed(17, "scene_3", 6):
        failures.append("per-step seed does not separate labels / steps")
    return failures


# --- check 5: GPU ---------------------------------------------------------------------
def build(**kwargs):
    return VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
                     context_frames=CONTEXT_FRAMES, **kwargs)


def run_logged(worker, images):
    probs, logs, lengths = [], [], []
    for idx, image in enumerate(images):
        from prune_fixtures import START, TURN
        p, _, _ = worker.infer_probs(START if idx == 0 else TURN, [image])
        probs.append(np.asarray(p))
        logs.append(dict(worker._vtp_log))
        lengths.append(worker._kv_lengths())
    return probs, logs, lengths


def check_gpu(images):
    failures = []
    evict = build(context_cut="frame", context_window_mode="evict")
    base = run_probs(evict, images)
    base_len = evict._kv_lengths()[0]
    release(evict)
    ident = build(**COMMON)
    probs, logs, lengths = run_logged(ident, images)
    n_layers = len(ident.language_model.layers)
    release(ident)
    exact = all(np.array_equal(a, b) for a, b in zip(base, probs))
    floor = max(float(np.abs(a - b).max()) for a, b in zip(base, probs))
    print(f"  identity vs evict frame cut: {'bit-identical' if exact else f'max |dp| = {floor:.2e}'}, "
          f"final kv_len {lengths[-1][0]} vs {base_len}")
    if not exact:
        failures.append(f"identity arm differs from evict (max |dp| {floor:.2e})")
    if lengths[-1][0] != base_len:
        failures.append(f"identity arm holds {lengths[-1][0]} slots, evict {base_len}")
    if any(decode_log(l) != l["vtp_kept"] or l["vtp_kept"] != l["vtp_ntok"] for l in logs):
        failures.append("identity mask log is not all-ones")

    for name, cfg in METHODS.items():
        worker = build(kv_prune_budget_frames=1.0, **COMMON, **cfg)
        try:
            p, logs, lengths = run_logged(worker, images)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: raised {exc!r}")
            release(worker)
            continue
        ps, st = worker._kv_prune_state, worker._reindex_state
        signal = max(float(np.abs(a - b).max()) for a, b in zip(base, p))
        dropped = ps.n_budget_dropped
        last = logs[-1]
        h = len(last["vtp_frames"])
        expect = sum(last["vtp_ntok"]) if h <= 1 else int(round(sum(last["vtp_ntok"]) / h))
        lens = lengths[-1]
        print(f"  {name}: signal {signal:.4f}, dropped {dropped}, last window kept {last['vtp_kept']} "
              f"of {last['vtp_ntok']} (expect {expect}), layers 0/{n_layers - 1} hold {lens[0]}/{lens[-1]}")
        if any(not np.all(np.isfinite(x)) for x in p):
            failures.append(f"{name}: non-finite probabilities")
        if sum(last["vtp_kept"]) != expect or decode_log(last) != last["vtp_kept"]:
            failures.append(f"{name}: kept {last['vtp_kept']} vs expected {expect}")
        if not dropped or signal <= 0:
            failures.append(f"{name}: nothing pruned or the decision never moved")
        pruned = st.pruned_layers
        if pruned is not None:
            master = st.master_len()
            for l, n in enumerate(lens):
                stage = st.layer_stage(l)
                want = master if stage is None else int(st.stage_alive(stage).sum())
                if n != want:
                    failures.append(f"{name}: layer {l} holds {n}, expected {want}")
            if lens[0] <= lens[-1]:
                failures.append(f"{name}: pruned layers are not shorter than the unpruned ones")
        elif len(set(lens)) != 1:
            failures.append(f"{name}: uniform arm has ragged layers {lens}")
        if worker.past_image_embeds[0].shape[0] != int(ps.is_visual.sum()):
            failures.append(f"{name}: embed DB desynced")
        if name == "dart" and worker._hidden_db.shape[0] != int(ps.is_visual.sum()):
            failures.append("dart: hidden DB desynced")
        release(worker)
    return failures


def main():
    failures = []
    print("check 1: frame-cut parity (prune vs evict)")
    failures += check_frame_cut_parity()
    print("check 2: frame-relative budget and mask log")
    failures += check_frame_budget()
    print("check 3: PyramidDrop stages")
    failures += check_staged()
    print("check 4: selectors and seeding")
    failures += check_selectors()
    if torch.cuda.is_available() and "--cpu" not in sys.argv:
        print("check 5: GPU identity and methods")
        failures += check_gpu(make_images(N_STEPS))
    else:
        print("check 5 skipped (no GPU)")
    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: frame-cut parity, frame budget + mask log, staged nesting, selectors, GPU arms")


if __name__ == "__main__":
    main()
