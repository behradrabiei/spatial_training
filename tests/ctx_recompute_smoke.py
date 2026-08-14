"""Smoke test for context_window_mode='recompute'.

Evicting K/V from the cache frees memory but does not remove information: a token that
survives in the window had its layer>=1 keys and values computed from a residual stream
that attended over the whole episode, so evicted frames still reach the decision through
it. The 'recompute' mode rebuilds the window's K/V against a cache holding only
[pinned prefix + window], which is what makes the ablation measure memory rather than
summarization.

Three things have to hold:

  1. Token parity -- the two modes must retire exactly the same tokens on exactly the same
     steps, or the ablation would be comparing two different contexts rather than two ways
     of computing K/V over one. Checked on synthetic records, CPU only, no model.

  2. Fidelity -- replaying stored embeds must reproduce what the incremental path built.
     Measured behaviourally, on actions rather than on cache tensors: one big prefill and
     a run of chunked prefills reduce in different orders, and Qwen's massive activations
     (layer-0 keys peak near 450) turn bf16 rounding into absolute K/V gaps of several
     units even when the inputs are bit-identical. What has to hold is that the decision
     does not move. Forcing a rebuild on every step at a window wider than the run keeps
     the token set identical to the full-context arm, so whatever is left is pure
     arithmetic noise -- the floor the real experiment has to clear.

  3. Non-vacuity -- with a window that does evict, recompute must diverge from evict by
     much more than that floor. If it did not, the rebuild would be inert (or the ablation
     would be measuring rounding), and the experiment would return a null result for the
     wrong reason.

Check 1 runs anywhere; checks 2 and 3 need the GPU and the HF model (no habitat, ray or
dataset). Run inside longnav_vlm:
    python tests/ctx_recompute_smoke.py
"""
import gc
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
NARROW_WINDOW = 2   # evicts from step 2 onwards
WIDE_WINDOW = 64    # never evicts over N_STEPS, so a rebuild spans the full history
FIDELITY_TOL = 2e-2      # how far bf16 reduction order alone may move an action probability
DIVERGENCE_MARGIN = 3.0  # leakage must beat that floor by this factor to be a real signal

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
    """Fixed pseudo-random frames, so every configuration sees byte-identical input."""
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def build(window, mode):
    return VLMWorker(
        model_id=MODEL_ID,
        attn_impl="sdpa",
        dtype="bfloat16",
        use_sparse=True,
        context_window=window,
        context_window_mode=mode,
    )


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def run_probs(worker, images):
    """Action probabilities per step."""
    out = []
    for idx, image in enumerate(images):
        probs, _, _ = worker.infer_probs(START if idx == 0 else TURN, [image])
        out.append(np.asarray(probs))
    return out


# --- check 1: token parity, CPU only -------------------------------------------------
# Synthetic turns. Every token is its own global id, so what the cache holds can be
# compared between the two modes as a plain list of integers.
PARITY_PREFIX = 5      # pinned prompt tokens
PARITY_HEADER = 3      # len(prefix_ids), the assistant header held back at the boundary
PARITY_WINDOW = 2
PARITY_TURNS = [11, 7, 9, 6, 8, 7, 10]  # turn 0's length includes the prefix


class FakeLayer:
    """One cache layer whose 'keys' carry global token ids, so the real slice is visible."""

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


def parity_worker(mode):
    """A VLMWorker with just enough state to drive _apply_context_window, and no model."""
    worker = VLMWorker.__new__(VLMWorker)
    worker.context_window = PARITY_WINDOW
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
    if mode == "recompute":
        # Stand in for the forward, but compose the window with the real code so the
        # prefix/boundary/turn assembly is what is actually under test here.
        def rebuild():
            merged = worker._window_forward_inputs()
            worker.past_key_values = FakeCache(merged["inputs_embeds"].reshape(-1))
        worker._rebuild_window_cache = rebuild
    return worker


def fake_turn(ids):
    """An embed record whose embeddings are the tokens' global ids."""
    return SimpleNamespace(
        inputs_embeds=ids.reshape(1, -1, 1),
        position_ids=ids.reshape(1, 1, -1).expand(3, 1, -1).clone(),
        visual_pos_masks=torch.zeros(1, len(ids), dtype=torch.bool),
        deepstack_visual_embeds=None,
    )


def check_token_parity():
    workers = {mode: parity_worker(mode) for mode in ("evict", "recompute")}
    live = {mode: [] for mode in workers}  # ids the evict-mode cache holds, tracked directly
    retained = {mode: [] for mode in workers}

    next_id = 0
    for length in PARITY_TURNS:
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        for mode, worker in workers.items():
            if mode == "evict":
                live[mode] += ids.tolist()
                worker.past_key_values = FakeCache(torch.tensor(live[mode]))
            else:
                worker._record_window_turn(fake_turn(ids))
                merged = worker._window_forward_inputs()
                worker.past_key_values = FakeCache(merged["inputs_embeds"].reshape(-1))
            worker._apply_context_window()
            live[mode] = worker.past_key_values.layers[0].ids()
            retained[mode].append(list(live[mode]))

    failures = []
    if not workers["evict"]._n_evicted:
        failures.append("the window never evicted, so token parity was not tested; "
                        "lower PARITY_WINDOW or add turns")
    for idx, (a, b) in enumerate(zip(retained["evict"], retained["recompute"])):
        print(f"  step {idx}: {len(a)} tokens retained ({'match' if a == b else 'MISMATCH'})")
        if a != b:
            failures.append(f"step {idx}: evict retains {a} but recompute retains {b}")
    # _dropped feeds the next step's _abs_bounts arithmetic, so it has to advance in
    # lockstep too -- a mode that forgets it drifts silently after the first eviction.
    for field in ("_dropped", "_n_evicted"):
        got = {mode: getattr(w, field) for mode, w in workers.items()}
        if got["evict"] != got["recompute"]:
            failures.append(f"{field} diverged: {got}")
    return failures


def check_fidelity(images):
    """Rebuilding over the full history must not move the decision. Returns (failures, floor)."""
    plain = build(WIDE_WINDOW, "evict")
    baseline = run_probs(plain, images)
    release(plain)

    worker = build(WIDE_WINDOW, "recompute")
    # At WIDE_WINDOW nothing is evicted, so _apply_context_window never rebuilds on its
    # own. Forcing it keeps the token set identical to the baseline and isolates the cost
    # of replaying the history as one prefill instead of many.
    evict = worker._apply_context_window

    def evict_then_rebuild():
        evict()
        worker._rebuild_window_cache()

    worker._apply_context_window = evict_then_rebuild
    rebuilt = run_probs(worker, images)
    evicted = worker._n_evicted
    release(worker)

    failures = []
    if evicted:
        failures.append(f"the window evicted after {N_STEPS} steps, so this compares two "
                        f"different token sets; raise WIDE_WINDOW")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, rebuilt)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_full - p_rebuilt| = {delta:.5f}")
    floor = max(deltas)
    if floor > FIDELITY_TOL:
        failures.append(f"rebuilding the same tokens moved the action distribution by {floor:.5f}, "
                        f"more than arithmetic noise should account for ({FIDELITY_TOL}); the "
                        f"replay is not reproducing the incremental forward")
    return failures, floor


def check_divergence(images, floor):
    """With a window that evicts, recompute must diverge from evict well beyond the floor."""
    failures = []
    runs = {}
    for mode in ("evict", "recompute"):
        worker = build(NARROW_WINDOW, mode)
        runs[mode] = run_probs(worker, images)
        evicted = worker._n_evicted
        release(worker)
        if not evicted:
            failures.append(f"[{mode}] the window never evicted, so nothing was tested; "
                            f"raise N_STEPS or lower NARROW_WINDOW")

    deltas = [float(np.abs(a - b).max()) for a, b in zip(runs["evict"], runs["recompute"])]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_evict - p_recompute| = {delta:.5f}")
    signal = max(deltas)
    print(f"  signal={signal:.5f} vs numerical floor={floor:.5f} ({signal / max(floor, 1e-9):.1f}x)")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"recompute differs from evict by only {signal:.5f}, within "
                        f"{DIVERGENCE_MARGIN}x of the {floor:.5f} arithmetic floor -- the ablation "
                        f"would be measuring rounding rather than leaked context")
    return failures


def main():
    failures = []

    print(f"=== token parity: context_window={PARITY_WINDOW}, {len(PARITY_TURNS)} turns (no model) ===")
    failures += check_token_parity()
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    images = make_images(N_STEPS)
    print(f"=== fidelity: context_window={WIDE_WINDOW} (no eviction), forced rebuild, "
          f"{N_STEPS} steps ===")
    fidelity_failures, floor = check_fidelity(images)
    failures += fidelity_failures

    print(f"=== divergence: context_window={NARROW_WINDOW}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"\nPASS: rebuilding the same tokens leaves the decision put (within {floor:.5f}), and "
          f"the rebuild changes it once the window starts evicting.")


if __name__ == "__main__":
    main()
