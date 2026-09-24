"""Smoke test for vlm.context_frames / vlm.context_cut='frame'.

The paper's context-window ablation defines "context size K" as: when the action is read,
the agent sees the current frame and the K-1 frames before it -- and nothing older, not
even the action token that closed the oldest evicted turn. context_frames=K sets that up
(internally context_window = K-1) and context_cut='frame' moves the eviction cut from the
held-back assistant header to the oldest retained '<|im_start|>user' opener.

Four things have to hold:

  1. Token parity under the frame cut -- evict and recompute retire exactly the same tokens
     on the same steps for K in {1, 2, 3}; every cut lands on a chunk's user opener; K=1
     leaves the cache at the pinned prefix and strips the next chunk's lead on entry.
     Synthetic records, CPU only, no model.

  2. Legacy guard -- context_cut='header' (the default) still reproduces the retained-token
     counts pinned by tests/ctx_recompute_smoke.py, so existing runs are untouched.

  3. Frames actually visible -- with the real model and the sparse filter on (the regime the
     runs use), reconstructing the cache from the worker's own bookkeeping shows exactly K
     frames at decision time and K-1 after eviction, the window opening on '<|im_start|>user',
     and the reconstruction's length equal to the live cache length.

  4. Built-in control -- at K=1 the cache is prefix-only after every step and the prefix K/V
     never saw a frame, so evict and recompute must agree to bf16 noise.

Checks 1-2 run anywhere; 3-4 need the GPU and the HF model. Run inside longnav_vlm:
    python tests/ctx_frames_smoke.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_recompute_smoke import (  # noqa: E402
    FIDELITY_TOL, MODEL_ID, START, TURN, FakeCache, fake_turn, make_images, release,
)
from longnav.utils.vlm_worker import VLMWorker  # noqa: E402

PREFIX = 5          # pinned prompt tokens
HEADER = 4          # len(prefix_ids): '<|im_start|>assistant\n**'
LEAD = 4            # action token + len(postfix_ids): 'forward' '**' '<|im_end|>' '\n'
# Chunk lengths. Turn 0 carries the prefix; every later chunk carries LEAD tokens of the
# previous reply before its own '<|im_start|>user' opener.
TURNS = [11, 9, 12, 8, 10, 9, 13]
N_STEPS = 6


def parity_worker(mode, window, cut):
    """A VLMWorker with just enough state to drive _apply_context_window, and no model."""
    worker = VLMWorker.__new__(VLMWorker)
    worker.context_window = window
    worker.context_window_mode = mode
    worker.context_cut = cut
    worker.prefix_ids = list(range(HEADER))
    worker.postfix_ids = list(range(LEAD - 1))
    worker._prefix_len = PREFIX
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
    worker._turn_leads = []
    worker._strip_next_lead = False
    worker._inference_ctx = torch.no_grad
    if mode == "recompute":
        def rebuild():
            merged = worker._window_forward_inputs()
            worker.past_key_values = FakeCache(merged["inputs_embeds"].reshape(-1))
        worker._rebuild_window_cache = rebuild
    return worker


def drive(worker, cut):
    """Feed the synthetic chunks the way infer_step would, returning retained ids per step."""
    live, retained, next_id = [], [], 0
    chunk_starts = []  # global id of each chunk's user opener
    for turn, length in enumerate(TURNS):
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        lead = 0 if turn == 0 else LEAD
        chunk_starts.append(int(ids[lead]) if turn else PREFIX)
        if cut == "frame":
            if worker._strip_next_lead:
                ids = ids[lead:]
                worker._strip_next_lead = False
                lead = 0
            worker._turn_leads.append(lead)
        if worker.context_window_mode == "evict":
            live += ids.tolist()
            worker.past_key_values = FakeCache(torch.tensor(live))
        else:
            worker._record_window_turn(fake_turn(ids))
            merged = worker._window_forward_inputs()
            worker.past_key_values = FakeCache(merged["inputs_embeds"].reshape(-1))
        worker._apply_context_window()
        live = worker.past_key_values.layers[0].ids()
        retained.append(list(live))
    return retained, chunk_starts


def check_frame_parity():
    failures = []
    for K in (1, 2, 3):
        window = K - 1
        workers = {m: parity_worker(m, window, "frame") for m in ("evict", "recompute")}
        out = {m: drive(w, "frame") for m, w in workers.items()}
        starts = out["evict"][1]
        for step, (a, b) in enumerate(zip(out["evict"][0], out["recompute"][0])):
            print(f"  K={K} step {step}: {len(a)} tokens retained ({'match' if a == b else 'MISMATCH'})")
            if a != b:
                failures.append(f"K={K} step {step}: evict retains {a} but recompute retains {b}")
            # After the eviction at step t the cache holds the prefix plus turns
            # [t-K+2 .. t] (K-1 turns), the oldest opening on its user opener.
            first_keep = max(0, step + 1 - window)
            if first_keep == 0:
                expect = list(range(sum(TURNS[:step + 1])))
            elif first_keep > step:  # K=1: nothing but the prefix survives
                expect = list(range(PREFIX))
            else:
                expect = list(range(PREFIX)) + list(range(starts[first_keep], sum(TURNS[:step + 1])))
            if a != expect:
                failures.append(f"K={K} step {step}: retained {a}, expected {expect}")
        for mode, w in workers.items():
            if K == 1 and not w._strip_next_lead:
                failures.append(f"K=1 [{mode}]: the next chunk's lead is not scheduled for stripping")
            if K > 1 and w._strip_next_lead:
                failures.append(f"K={K} [{mode}]: lead strip armed although the lead was cached")
        for field in ("_dropped", "_n_evicted"):
            got = {m: getattr(w, field) for m, w in workers.items()}
            if got["evict"] != got["recompute"]:
                failures.append(f"K={K} {field} diverged: {got}")
    return failures


def check_header_legacy():
    """Same fixture as tests/ctx_recompute_smoke.py; the counts there are the contract."""
    expect = [11, 18, 24, 23, 22, 23, 25]
    worker = parity_worker("evict", 2, "header")
    worker.prefix_ids = list(range(3))
    legacy_turns = [11, 7, 9, 6, 8, 7, 10]
    live, counts, next_id = [], [], 0
    for length in legacy_turns:
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        live += ids.tolist()
        worker.past_key_values = FakeCache(torch.tensor(live))
        worker._apply_context_window()
        live = worker.past_key_values.layers[0].ids()
        counts.append(len(live))
    print(f"  header cut retained counts: {counts}")
    return [] if counts == expect else [f"header cut retained {counts}, expected {expect}"]


# --- GPU checks -------------------------------------------------------------------------

def build(K, mode):
    return VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
                     context_frames=K, context_cut="frame", context_window_mode=mode)


def retained_ids(worker):
    """The token ids the evicting cache holds, rebuilt from the worker's own bookkeeping."""
    ids = worker.cumulative_inputs["input_ids"][0]
    sparse = ids[worker.seq_keep_mask.bool()]
    prefix = worker._prefix_len
    return torch.cat([sparse[:prefix], sparse[prefix + worker._dropped:]])


def check_visible_frames(images, K=2):
    failures = []
    worker = build(K, "evict")
    tok = worker.processor.tokenizer
    vision_start = tok.convert_tokens_to_ids("<|vision_start|>")
    im_start, user = worker.prefix_ids[0], tok.convert_tokens_to_ids("user")
    frames_after_prev = 0
    for step, image in enumerate(images):
        worker.infer_probs(START if step == 0 else TURN, [image])
        held = retained_ids(worker)
        n_frames = int((held == vision_start).sum())
        cache_len = worker.past_key_values.get_seq_length()
        visible = frames_after_prev + 1
        opener = held[worker._prefix_len: worker._prefix_len + 2].tolist()
        print(f"  step {step}: cache={cache_len} reconstructed={len(held)} frames_after={n_frames} "
              f"visible_at_decision={visible} opener={tok.decode(opener)!r}")
        if len(held) != cache_len:
            failures.append(f"step {step}: reconstruction has {len(held)} tokens, cache {cache_len}")
        if visible != min(step + 1, K):
            failures.append(f"step {step}: {visible} frames visible at decision, expected {min(step + 1, K)}")
        if n_frames != min(step + 1, K - 1):
            failures.append(f"step {step}: {n_frames} frames after eviction, expected {min(step + 1, K - 1)}")
        if worker._n_evicted and opener != [im_start, user]:
            failures.append(f"step {step}: window opens on {tok.decode(opener)!r}, not '<|im_start|>user'")
        frames_after_prev = n_frames
    release(worker)
    return failures


def check_single_frame_control(images):
    failures, runs = [], {}
    for mode in ("evict", "recompute"):
        worker = build(1, mode)
        probs = []
        for step, image in enumerate(images):
            p, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
            probs.append(np.asarray(p))
            if worker.past_key_values.get_seq_length() != worker._prefix_len:
                failures.append(f"[{mode}] step {step}: cache holds "
                                f"{worker.past_key_values.get_seq_length()} tokens after eviction, "
                                f"expected the {worker._prefix_len}-token prefix only")
        runs[mode] = probs
        release(worker)
    deltas = [float(np.abs(a - b).max()) for a, b in zip(runs["evict"], runs["recompute"])]
    for step, d in enumerate(deltas):
        print(f"  step {step}: max |p_evict - p_recompute| = {d:.5f}")
    if max(deltas) > FIDELITY_TOL:
        failures.append(f"K=1 evict and recompute differ by {max(deltas):.5f} (> {FIDELITY_TOL}); "
                        f"they should be the same computation")
    return failures


def main():
    failures = []
    print("[1/4] frame-cut token parity (CPU)")
    failures += check_frame_parity()
    print("[2/4] header-cut legacy counts (CPU)")
    failures += check_header_legacy()
    if failures or not torch.cuda.is_available():
        if not torch.cuda.is_available():
            print("no GPU; skipping checks 3-4")
    else:
        images = make_images(N_STEPS)
        print("[3/4] frames visible with the real model (K=2)")
        failures += check_visible_frames(images)
        print("[4/4] K=1 evict vs recompute control")
        failures += check_single_frame_control(images)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
