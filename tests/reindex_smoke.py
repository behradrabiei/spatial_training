"""Smoke test for context_window_mode='reindex' (StreamingLLM-style pre-rotation cache).

Stock eviction slices K/V out of the cache but leaves the survivors' mRoPE positions
absolute, so a positional hole opens between the pinned prefix and the window. 'reindex'
caches keys pre-rotation and applies RoPE at attention time from a per-slot position
table (longnav.utils.pre_rope), which lets _apply_context_window renumber the survivors
to sit flush against the prefix.

Three things have to hold:

  1. Schedule parity + contiguity -- reindex must retire exactly the same tokens on
     exactly the same steps as evict (it is the same eviction, positions aside), and
     after every eviction the position table must be hole-free: prefix..window one dense
     run, with worker.offset continuing where the table ends so the next turn stays
     contiguous. Checked on synthetic records, CPU only, no model.

  2. Fidelity -- with a window wider than the run nothing is ever evicted and nothing is
     renumbered, so caching keys unrotated and rotating at read time must reproduce the
     stock forward: rotation commutes with concatenation, and cos/sin are recomputed
     bit-identically every step (fp32 rotary_emb over integer positions -- no
     accumulation). Measured behaviourally on action probabilities, same tolerance as
     ctx_recompute_smoke: the single bf16 rounding moves from rotate-then-store to
     rotate-at-read, which is arithmetic noise, not drift.

  3. Non-vacuity -- with a window that does evict, the renumbered positions must move the
     decision well beyond that floor. If they did not, the re-indexing would be inert and
     the eval would return a null result for the wrong reason.

Check 1 runs anywhere; checks 2 and 3 need the GPU and the HF model (no habitat, ray or
dataset). Run inside longnav_vlm:
    python tests/reindex_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.pre_rope import ReindexState
from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
NARROW_WINDOW = 2     # evicts from step 2 onwards
WIDE_WINDOW = 10_000  # never evicts, so no renumbering ever happens
FIDELITY_TOL = 2e-2      # how far bf16 reduction order alone may move an action probability
DIVERGENCE_MARGIN = 3.0  # renumbering must beat that floor by this factor to be a real signal

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


# --- check 1: schedule parity + position contiguity, CPU only ------------------------
# Synthetic pure-text turns. Every token is its own global id AND its own mRoPE position
# (text tokens have t == h == w), so both the cache slices and the position table can be
# compared as plain lists of integers.
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
    worker._reindex_state = ReindexState() if mode == "reindex" else None
    worker.offset = 0
    return worker


def check_parity_and_contiguity():
    workers = {mode: parity_worker(mode) for mode in ("evict", "reindex")}
    live = {mode: [] for mode in workers}
    retained = {mode: [] for mode in workers}
    failures = []

    next_id = 0
    for length in PARITY_TURNS:
        ids = torch.arange(next_id, next_id + length)
        next_id += length
        for mode, worker in workers.items():
            live[mode] += ids.tolist()
            worker.past_key_values = FakeCache(torch.tensor(live[mode]))
            if mode == "reindex":
                # Pure text turns: _pos_id_fast assigns arange(offset, offset+length) on
                # all three mRoPE components (deltas=0), then advances offset. offset was
                # rebased by the previous eviction, so positions diverge from token ids
                # once the window starts evicting -- exactly the property under test.
                pos = torch.arange(worker.offset, worker.offset + length)
                worker._reindex_state.append(pos.reshape(1, 1, -1).expand(3, 1, -1).clone())
                worker.offset += length
            worker._apply_context_window()
            live[mode] = worker.past_key_values.layers[0].ids()
            retained[mode].append(list(live[mode]))

        st = workers["reindex"]._reindex_state
        kv_len = workers["reindex"].past_key_values.get_seq_length()
        if st.pos_table.shape[-1] != kv_len:
            failures.append(f"position table has {st.pos_table.shape[-1]} slots for a "
                            f"{kv_len}-slot cache")
        positions = st.pos_table[0, 0].tolist()
        if positions != list(range(kv_len)):
            failures.append(f"positions are not hole-free after this turn: {positions}")
        if workers["reindex"].offset != kv_len:
            failures.append(f"offset {workers['reindex'].offset} does not continue the "
                            f"table (len {kv_len}); the next turn would open a hole")

    if not workers["evict"]._n_evicted:
        failures.append("the window never evicted, so nothing was tested; "
                        "lower PARITY_WINDOW or add turns")
    for idx, (a, b) in enumerate(zip(retained["evict"], retained["reindex"])):
        print(f"  step {idx}: {len(a)} tokens retained ({'match' if a == b else 'MISMATCH'})")
        if a != b:
            failures.append(f"step {idx}: evict retains {a} but reindex retains {b}")
    for field in ("_dropped", "_n_evicted"):
        got = {mode: getattr(w, field) for mode, w in workers.items()}
        if got["evict"] != got["reindex"]:
            failures.append(f"{field} diverged: {got}")
    return failures


def check_fidelity(images):
    """With nothing evicted, pre-rotation caching must not move the decision."""
    plain = build(None, "evict")  # stock full-context path, rotated keys
    baseline = run_probs(plain, images)
    release(plain)

    worker = build(WIDE_WINDOW, "reindex")
    rotated_at_read = run_probs(worker, images)
    evicted = worker._n_evicted
    table_len = worker._reindex_state.pos_table.shape[-1]
    kv_len = worker.past_key_values.get_seq_length()
    release(worker)

    failures = []
    if evicted:
        failures.append(f"the window evicted after {N_STEPS} steps, so this compares two "
                        f"different token sets; raise WIDE_WINDOW")
    if table_len != kv_len:
        failures.append(f"position table ({table_len}) desynced from cache ({kv_len})")
    deltas = [float(np.abs(a - b).max()) for a, b in zip(baseline, rotated_at_read)]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_stock - p_reindex| = {delta:.5f}")
    floor = max(deltas)
    if floor > FIDELITY_TOL:
        failures.append(f"with no eviction the pre-rotation cache moved the action "
                        f"distribution by {floor:.5f}, more than arithmetic noise should "
                        f"account for ({FIDELITY_TOL}); the read-time rotation is not "
                        f"reproducing the stock forward")
    return failures, floor


def check_divergence(images, floor):
    """With a window that evicts, the renumbered positions must actually change decisions."""
    failures = []
    runs = {}
    for mode in ("evict", "reindex"):
        worker = build(NARROW_WINDOW, mode)
        runs[mode] = run_probs(worker, images)
        evicted = worker._n_evicted
        release(worker)
        if not evicted:
            failures.append(f"[{mode}] the window never evicted, so nothing was tested; "
                            f"raise N_STEPS or lower NARROW_WINDOW")

    deltas = [float(np.abs(a - b).max()) for a, b in zip(runs["evict"], runs["reindex"])]
    for idx, delta in enumerate(deltas):
        print(f"  step {idx}: max |p_evict - p_reindex| = {delta:.5f}")
    signal = max(deltas)
    print(f"  signal={signal:.5f} vs numerical floor={floor:.5f} ({signal / max(floor, 1e-9):.1f}x)")
    if signal < DIVERGENCE_MARGIN * max(floor, 1e-6):
        failures.append(f"reindex differs from evict by only {signal:.5f}, within "
                        f"{DIVERGENCE_MARGIN}x of the {floor:.5f} arithmetic floor -- the "
                        f"renumbering is inert and the eval would measure rounding")
    return failures


def main():
    failures = []

    print(f"=== parity + contiguity: context_window={PARITY_WINDOW}, "
          f"{len(PARITY_TURNS)} turns (no model) ===")
    failures += check_parity_and_contiguity()
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    images = make_images(N_STEPS)
    print(f"=== fidelity: context_window={WIDE_WINDOW} (no eviction), {N_STEPS} steps ===")
    fidelity_failures, floor = check_fidelity(images)
    failures += fidelity_failures

    print(f"=== divergence: context_window={NARROW_WINDOW}, {N_STEPS} steps ===")
    failures += check_divergence(images, floor)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"\nPASS: read-time rotation reproduces the stock forward (within {floor:.5f}), "
          f"the eviction schedule matches evict exactly, and the renumbered positions "
          f"change the decision once the window starts evicting.")


if __name__ == "__main__":
    main()
