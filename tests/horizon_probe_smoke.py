"""GPU smoke for vlm.horizon_probe_layers (paper/info_horizon: frame-level layer-wise KL).

What has to hold:
  1. The option is refused without token_info_probe (CPU).
  2. In both arms (evict = leak-free re-prefill, prune = leaky decision replay) the live decision
     is unchanged relative to a plain full-context worker: the probes only read.
  3. Every step logs one finite non-negative keep-one-in and leave-one-out value per probe
     layer; L = n_layers is exactly 0; the base log-probs normalise; the number of forwards
     is 1 + 3 * #{L < n_layers} per step.
  4. The signal is not inert at L = 0 (the current frame moves the decision).
  5. reset() clears the row and the counters.

Run inside longnav_vlm:  python tests/horizon_probe_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 5
LAYERS = (0, 8, 18, 28)
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


def make_worker(mode, layers=()):
    kw = dict(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True, context_window=None,
              context_window_mode=mode, token_info_probe=bool(layers), horizon_probe_layers=layers)
    if mode == "prune":
        kw |= dict(kv_budget=10 ** 9, kv_prune_importance="random", kv_prune_granularity="slot",
                   kv_prune_candidate_scope="all", kv_prune_recent_turns=2, kv_prune_seed=17)
    return VLMWorker(**kw)


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
        row = dict(worker._horizon_row) if getattr(worker, "_horizon_row", None) else {}
        steps.append({"probs": np.asarray(probs), "row": row,
                      "count": getattr(worker, "_horizon_probe_count", 0)})
    return steps


def main():
    failures = []
    print("=== 1. refused without token_info_probe (no model) ===")
    try:
        VLMWorker(model_id=MODEL_ID, load_model=False, use_sparse=True, horizon_probe_layers=(0, 4))
    except ValueError as exc:
        print(f"  refused: {exc}")
    else:
        failures.append("horizon_probe_layers accepted without token_info_probe")

    images = make_images(N_STEPS)
    print("=== 2. plain full-context worker ===")
    worker = make_worker("evict")
    plain = run(worker, images)
    release(worker)

    for mode in ("evict", "prune"):
        print(f"=== 3. {mode} arm ===")
        worker = make_worker(mode, LAYERS)
        n_layers = len(worker.language_model.layers)
        on = run(worker, images)
        worker.reset()
        if worker._horizon_row or worker._horizon_probe_count or worker._horizon_probe_latency:
            failures.append(f"{mode}: reset() did not clear the probe state")
        release(worker)
        prev = 0
        for t, (a, b) in enumerate(zip(plain, on)):
            dp = float(np.abs(a["probs"] - b["probs"]).max())
            if dp > PROB_TOL:
                failures.append(f"{mode} step {t}: probing moved the live decision by {dp:.2e}")
            row = b["row"]
            keep, leave = row.get("keep_one", []), row.get("leave_one", [])
            if row.get("layers") != list(LAYERS) or len(keep) != len(LAYERS) or len(leave) != len(LAYERS):
                failures.append(f"{mode} step {t}: malformed row {row}")
                continue
            for name, vals in (("keep_one", keep), ("leave_one", leave)):
                if not all(np.isfinite(v) and v >= 0 for v in vals):
                    failures.append(f"{mode} step {t}: {name} non-finite or negative: {vals}")
                if vals[-1] != 0.0:
                    failures.append(f"{mode} step {t}: {name} at L=n_layers is {vals[-1]}, expected 0")
            if abs(np.exp(row["base_logp"]).sum() - 1) > 1e-4:
                failures.append(f"{mode} step {t}: base log-probs do not normalise")
            expected = 1 + 3 * sum(1 for L in LAYERS if L < n_layers) if row["n_cur_tokens"] else 1
            got = b["count"] - prev
            if got != expected:
                failures.append(f"{mode} step {t}: {got} forwards, expected {expected}")
            prev = b["count"]
            print(f"  step {t}: n_cur={row['n_cur_tokens']} keep={['%.2e' % v for v in keep]} "
                  f"leave={['%.2e' % v for v in leave]} {row['seconds']:.2f}s")
        if not any(b["row"].get("leave_one", [0])[0] > 1e-3 or b["row"].get("keep_one", [0])[0] > 1e-3 for b in on):
            failures.append(f"{mode}: the current frame never moves the decision at L=0 (probe inert?)")

    print("=== result ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
