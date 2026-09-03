"""GPU smoke for immutable pre-RoPE decision replay and all visual prune methods."""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
REPLAY_TOL = 2e-2
BUDGET = 280
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


def make_worker(method, budget, attn_impl="sdpa"):
    return VLMWorker(
        model_id=MODEL_ID, attn_impl=attn_impl, dtype="bfloat16", use_sparse=True,
        context_window=32, context_window_mode="prune", kv_budget=budget,
        kv_prune_candidate_scope="all", kv_prune_importance=method,
        kv_prune_seed=17, kv_prune_fisher_pool_factor=2.0,
    )


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def check_replay(image):
    worker = make_worker("fisher", 10**9)
    original, _, _ = worker.infer_probs(START, [image])
    kv_len = worker.past_key_values.get_seq_length()
    keys = [layer.keys for layer in worker.past_key_values.layers]
    values = [layer.values for layer in worker.past_key_values.layers]
    pos_table = worker._reindex_state.pos_table
    meta = (worker._kv_prune_state.turn_id.clone(),
            worker._kv_prune_state.is_visual.clone(),
            worker._kv_prune_state.importance.clone())
    embed_db = worker.past_image_embeds[0].clone()

    empty = torch.zeros(kv_len, dtype=torch.bool)
    base = worker._decision_replay(worker._replay_base_bias(empty))
    base_probs = torch.softmax(base.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    delta = float(np.abs(base_probs - original).max())
    assert delta <= REPLAY_TOL, f"unmasked replay delta {delta} exceeds {REPLAY_TOL}"

    bias = worker._replay_base_bias(empty)
    bias[..., worker._kv_prune_state.is_visual.to(bias.device)] = -torch.inf
    ablated = worker._decision_replay(bias)
    ablated_probs = torch.softmax(
        ablated.logits[0, -1, worker.vocab_ids].float(), -1).cpu().numpy()
    assert float(np.abs(ablated_probs - base_probs).max()) > 1e-5
    assert all(layer.keys is key and layer.values is value for layer, key, value in
               zip(worker.past_key_values.layers, keys, values))
    assert worker._reindex_state.pos_table is pos_table
    assert all(torch.equal(a, b) for a, b in zip(
        meta, (worker._kv_prune_state.turn_id, worker._kv_prune_state.is_visual,
               worker._kv_prune_state.importance)))
    assert torch.equal(embed_db, worker.past_image_embeds[0])

    before = worker._prune_replay_count
    fisher = worker._score_fisher_slots(empty)
    assert fisher.numel() == kv_len and bool(torch.isfinite(fisher).all())
    assert worker._prune_replay_count - before == 1
    print(f"  replay delta={delta:.5f}, Fisher range=[{float(fisher.min()):.3g}, "
          f"{float(fisher.max()):.3g}]")
    release(worker)


def check_integrated_methods(images):
    for method in ("random", "stratified", "diversity", "kl", "fisher",
                   "fisher_diversity"):
        worker = make_worker(method, BUDGET)
        replay_count = 0
        for step, image in enumerate(images):
            probs, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
            replay_count += worker._prune_replay_count
            state = worker._kv_prune_state
            kv_len = worker.past_key_values.get_seq_length()
            assert np.isfinite(probs).all() and kv_len <= BUDGET
            assert kv_len == state.turn_id.numel() == state.is_visual.numel()
            assert kv_len == state.importance.numel() == worker._reindex_state.pos_table.shape[-1]
            assert worker.past_image_embeds[0].shape[0] == int(state.is_visual.sum())
        assert worker._kv_prune_state.n_budget_dropped > 0
        if method in ("kl", "fisher", "fisher_diversity"):
            assert replay_count > 0
        print(f"  {method}: kv={kv_len}, visual={worker._prune_visual_slots}, "
              f"text={worker._prune_text_slots}, replays={replay_count}")
        release(worker)


def check_production_shape_fisher(image):
    """SDPA inference plus eager replay must survive the actual B4 KV shape."""
    worker = make_worker("fisher", 1187, attn_impl="sdpa")
    for step in range(40):
        probs, _, _ = worker.infer_probs(START if step == 0 else TURN, [image])
        assert np.isfinite(probs).all()
        assert worker.language_model.config._attn_implementation == "sdpa"
        if worker._kv_prune_state.n_budget_dropped:
            assert worker.past_key_values.get_seq_length() == 1187
            assert worker._prune_replay_count == 1
            print(f"  production Fisher: step={step + 1}, kv=1187, "
                  f"dropped={worker._kv_prune_state.n_budget_dropped}")
            release(worker)
            return
    release(worker)
    raise AssertionError("B4 pruning threshold was not reached")


def main():
    rng = np.random.RandomState(19)
    images = [Image.fromarray(rng.randint(0, 255, (240, 320, 3), dtype=np.uint8))
              for _ in range(6)]
    print("=== immutable decision replay ===")
    check_replay(images[0])
    print("=== integrated visual pruning methods ===")
    check_integrated_methods(images)
    print("=== production-shape Fisher replay ===")
    check_production_shape_fisher(images[0].resize((640, 480)))
    print("PASS: replay fidelity/state restoration/Fisher and all method alignments")


if __name__ == "__main__":
    main()
