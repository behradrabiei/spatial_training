"""Smoke test for the attention visualizations under a sliding context window.

`_apply_context_window` evicts whole turns from the KV cache and shifts its frame->key
bookkeeping down by the number of dropped tokens. The captured attention rows are
indexed by those same keys, so they have to be cropped in lockstep -- otherwise every
surviving frame's heat map reads the wrong key on any step where eviction fires, which
in steady state is every step.

The invariant checked here: an eviction must not change what the maps say about the
frames that survive it. Each step's maps are captured once from inside infer_step just
before the cut (row and indices both in pre-eviction space, correct by construction)
and again after infer_step returns (both in post-eviction space); the two must agree.

Run inside the VLM conda env (longnav_vlm), no habitat or dataset needed:
    python tests/attn_evict_smoke.py
"""
import numpy as np
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
CONTEXT_WINDOW = 2
N_STEPS = 5  # enough to evict on several consecutive steps past the window
LAYER = -1

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
    """Fixed pseudo-random frames, so every weighting sees byte-identical input."""
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def run(weighting, images):
    """Step a windowed worker, capturing each step's maps either side of the eviction.

    Returns one (pre_maps, post_maps, n_evicted) per step, where the maps are
    {layer: [per-frame float16 bytes]} and n_evicted is how many leading frames the
    window had dropped by the end of that step.
    """
    worker = VLMWorker(
        model_id=MODEL_ID,
        attn_impl="eager",
        dtype="bfloat16",
        use_sparse=True,  # _record_frame_keys needs the sparse model's patch masks
        visualize_attention_3d=True,
        attn3d_layers=[LAYER],
        attn_weighting=weighting,
        context_window=CONTEXT_WINDOW,
    )
    # The rollout reads the viz only after infer_step returns, so the sole place to
    # observe the pre-eviction maps is from inside infer_step, just before the cut.
    evict = worker._apply_context_window
    pre = []

    def capture_then_evict():
        pre.append(worker.get_attention_3d_visualization())
        evict()

    worker._apply_context_window = capture_then_evict

    steps = []
    for idx, image in enumerate(images):
        worker.infer_step(START if idx == 0 else TURN, [image])
        post = worker.get_attention_3d_visualization()
        assert pre[-1] is not None, f"[{weighting}] step {idx}: no maps before eviction"
        assert post is not None, f"[{weighting}] step {idx}: no maps after eviction"
        steps.append((pre[-1][0], post[0], worker._n_evicted))

    del worker
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return steps


def check(weighting, steps):
    failures = []
    if not any(n_evicted for _, _, n_evicted in steps):
        failures.append(f"[{weighting}] the window never evicted, so nothing was tested; "
                        f"raise N_STEPS or lower CONTEXT_WINDOW")

    for idx, (pre_maps, post_maps, n_evicted) in enumerate(steps):
        for layer, post in post_maps.items():
            pre = pre_maps[layer]
            # Evicted frames are blank by design; only the survivors carry a claim that
            # has to hold on both sides of the cut.
            survivors = range(n_evicted, len(post))
            moved = [i for i in survivors if pre[i] != post[i]]
            if moved:
                failures.append(f"[{weighting}] step {idx} layer {layer}: maps for surviving "
                                f"frames {moved} changed across the eviction")
            if not any(np.frombuffer(post[i], dtype=np.float16).any() for i in survivors):
                failures.append(f"[{weighting}] step {idx} layer {layer}: every surviving map "
                                f"is zero, so the comparison is vacuous")
    return failures


def main():
    images = make_images(N_STEPS)
    failures = []

    for weighting in ("raw", "grad"):
        print(f"=== attn_weighting={weighting!r}, context_window={CONTEXT_WINDOW}, "
              f"{N_STEPS} steps ===")
        steps = run(weighting, images)
        for idx, (_, post_maps, n_evicted) in enumerate(steps):
            frames = len(next(iter(post_maps.values())))
            print(f"  step {idx}: frames={frames} evicted={n_evicted}")
        failures += check(weighting, steps)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: eviction leaves every surviving frame's heat map unchanged.")


if __name__ == "__main__":
    main()
