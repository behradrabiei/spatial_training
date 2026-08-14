"""Smoke test for attn_weighting="grad".

The gradient heat maps come from replaying the action-decision token with autograd
enabled, which temporarily crops and re-appends a KV entry. This runs the same
synthetic episode under "wo_norm" and under "grad" and checks that the replay is
invisible to the rollout: identical cache lengths and identical action logprobs. A
drift here would mean the visualization is quietly changing the decision it claims
to explain.

Also checks the maps themselves are populated and are not a relabelled copy of the
attention maps.

Run inside the VLM conda env (longnav_vlm), no habitat or dataset needed:
    python tests/grad_attn_smoke.py
"""
import numpy as np
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 3
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
    """Fixed pseudo-random frames, so both runs see byte-identical input."""
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def run(weighting, images, freeze=False):
    worker = VLMWorker(
        model_id=MODEL_ID,
        attn_impl="eager",
        dtype="bfloat16",
        use_sparse=True,  # _record_frame_keys needs the sparse model's patch masks
        visualize_attention_3d=True,
        attn3d_layers=[LAYER],
        attn_weighting=weighting,
    )
    if freeze:
        for p in worker.model.parameters():
            p.requires_grad_(False)
    logprobs, cache_lens, maps = [], [], []
    for step, image in enumerate(images):
        lp, _ = worker.infer_step(START if step == 0 else TURN, [image])
        logprobs.append(np.asarray(lp).reshape(-1))
        cache_lens.append(worker.past_key_values.get_seq_length())
        viz = worker.get_attention_3d_visualization()
        assert viz is not None, f"[{weighting}] step {step}: no 3D visualization produced"
        per_layer, _ = viz
        layer_id = worker.attn3d_layer_ids[0]
        assert layer_id in per_layer, f"[{weighting}] step {step}: layer {layer_id} missing"
        # Latest frame's map, back from the float16 wire format.
        maps.append(np.frombuffer(per_layer[layer_id][-1], dtype=np.float16).astype(np.float32))

    del worker
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return logprobs, cache_lens, maps


def main():
    images = make_images(N_STEPS)

    print(f"=== attn_weighting='wo_norm' ({N_STEPS} steps) ===")
    ref_lp, ref_lens, ref_maps = run("wo_norm", images)
    print(f"=== attn_weighting='grad' ({N_STEPS} steps) ===")
    grad_lp, grad_lens, grad_maps = run("grad", images)

    failures = []

    if ref_lens != grad_lens:
        failures.append(f"cache lengths diverged: wo_norm={ref_lens} grad={grad_lens}")
    print(f"cache lengths: {grad_lens}")

    # The replay restores the KV entry it borrowed, so the rollout should be bitwise
    # identical -- not merely close. Anything above zero means recomputed values are
    # leaking into the cache and compounding across steps.
    against = [float(np.abs(a - b).max()) for a, b in zip(ref_lp, grad_lp)]
    print("per-step max |logprob difference| vs wo_norm:")
    for step, diff in enumerate(against):
        print(f"  step {step}: {diff:.2e}")
    if max(against) > 0.0:
        failures.append(f"action logprobs drifted by {max(against):.2e}; the replay is "
                        f"leaking recomputed values into the cache")

    for step, (ref, got) in enumerate(zip(ref_maps, grad_maps)):
        nz_ref, nz_got = int((ref > 0).sum()), int((got > 0).sum())
        print(f"step {step}: patches={got.size} nonzero wo_norm={nz_ref} grad={nz_got} "
              f"peak={got.max():.3f}")
        if nz_got == 0:
            failures.append(f"step {step}: gradient map is entirely zero")
        elif np.allclose(ref, got, atol=1e-3):
            failures.append(f"step {step}: gradient map is identical to the attention map")

    print(f"=== attn_weighting='grad' frozen params (eval-like) ===")
    # Eval freezes the loaded checkpoint; without a requires_grad input the
    # attribution silently produces empty maps. This is the case that bit the
    # attn3d_4ep_grad_all run.
    frz_lp, frz_lens, frz_maps = run("grad", images, freeze=True)
    if frz_lens != ref_lens:
        failures.append(f"frozen grad cache lengths diverged: {frz_lens}")
    against_frz = [float(np.abs(a - b).max()) for a, b in zip(ref_lp, frz_lp)]
    print("per-step max |logprob difference| vs wo_norm (frozen):")
    for step, diff in enumerate(against_frz):
        print(f"  step {step}: {diff:.2e}")
    if max(against_frz) > 0.0:
        failures.append(f"frozen grad drifted logprobs by {max(against_frz):.2e}")
    for step, got in enumerate(frz_maps):
        nz = int((got > 0).sum())
        print(f"step {step}: frozen-grad nonzero={nz} peak={got.max():.3f}")
        if nz == 0:
            failures.append(f"step {step}: frozen-grad map is entirely zero")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: the attribution replay leaves the cache and the decision untouched.")


if __name__ == "__main__":
    main()
