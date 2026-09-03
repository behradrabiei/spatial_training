"""GPU smoke for vlm.kv_prune_log_influence (leave-one-frame-out KL rows every step).

Four things have to hold:
  1. The flag is refused outside prune mode (CPU, no model).
  2. With the flag off nothing is replayed and the row stays empty.
  3. With the flag on (stride 1), step t logs exactly t+1 finite non-negative KLs, costs
     exactly t+2 replays, leaves the cache/table/metadata untouched, and does not move the
     decision relative to the flag-off run (the replay is side-effect-free). reset() clears it.
  4. stride 2 scores only even steps.

Run inside longnav_vlm:  python tests/kl_influence_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 5
PROB_TOL = 1e-4
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


def make_worker(log, stride=1):
    return VLMWorker(
        model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
        context_window=None, context_window_mode="prune", kv_budget=10 ** 9,
        kv_prune_importance="random", kv_prune_log_influence=log,
        kv_prune_influence_stride=stride,
    )


def make_images(n, h=240, w=320):
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def run(worker, images):
    steps = []
    for t, image in enumerate(images):
        probs, _, _ = worker.infer_probs(START if t == 0 else TURN, [image])
        ps = worker._kv_prune_state
        steps.append({
            "probs": np.asarray(probs),
            "row": list(worker._kl_influence_row),
            "replays": worker._prune_replay_count,
            "ms_per_replay": 1e3 * worker._prune_score_latency / max(worker._prune_replay_count, 1),
            "kv_len": worker.past_key_values.get_seq_length(),
            "meta_len": ps.turn_id.numel(),
            "table_len": worker._reindex_state.pos_table.shape[-1],
            "visual_new": worker._prune_visual_new,
        })
    return steps


def main():
    failures = []

    print("=== 1. flag refused outside prune mode (no model) ===")
    try:
        VLMWorker(model_id=MODEL_ID, load_model=False, kv_prune_log_influence=True)
    except ValueError as exc:
        print(f"  refused: {exc}")
    else:
        failures.append("kv_prune_log_influence=True was accepted in evict mode")

    images = make_images(N_STEPS)

    print("=== 2. flag off ===")
    worker = make_worker(log=False)
    off = run(worker, images)
    release(worker)
    for t, s in enumerate(off):
        if s["row"] or s["replays"]:
            failures.append(f"flag off, step {t}: row={s['row']} replays={s['replays']}")

    print("=== 3. flag on, stride 1 ===")
    worker = make_worker(log=True)
    on = run(worker, images)
    worker.reset()
    if worker._kl_influence_row != []:
        failures.append("reset() did not clear the influence row")
    release(worker)
    for t, (a, b) in enumerate(zip(on, off)):
        row = np.asarray(a["row"], dtype=float)
        if row.size != t + 1:
            failures.append(f"step {t}: row has {row.size} entries, expected {t + 1}")
        if not (np.isfinite(row).all() and (row >= 0).all()):
            failures.append(f"step {t}: non-finite or negative KL in {row}")
        if row.size and row.max() <= 1e-6:
            failures.append(f"step {t}: every frame scored ~0 ({row}); the masking is inert")
        if a["replays"] != t + 2:
            failures.append(f"step {t}: {a['replays']} replays, expected {t + 2}")
        if not (a["kv_len"] == a["meta_len"] == a["table_len"] == b["kv_len"]):
            failures.append(f"step {t}: cache {a['kv_len']} / meta {a['meta_len']} / table "
                            f"{a['table_len']} vs flag-off cache {b['kv_len']}")
        delta = float(np.abs(a["probs"] - b["probs"]).max())
        if delta > PROB_TOL:
            failures.append(f"step {t}: logging moved the decision by {delta:.2e}")
        print(f"  step {t}: KL row {np.array2string(row, precision=4)}  visual_new={a['visual_new']}  "
              f"{a['replays']} replays @ {a['ms_per_replay']:.1f} ms  |dp|={delta:.1e}")

    print("=== 4. flag on, stride 2 ===")
    worker = make_worker(log=True, stride=2)
    strided = run(worker, images)
    release(worker)
    for t, s in enumerate(strided):
        want = 0 if t % 2 else t + 1
        if len(s["row"]) != want:
            failures.append(f"stride 2, step {t}: row has {len(s['row'])} entries, expected {want}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: influence rows have t+1 finite entries from t+2 replays, leave the cache and the "
          "decision untouched, clear on reset, and honour the stride.")


if __name__ == "__main__":
    main()
