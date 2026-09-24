"""GPU smoke for vlm.token_info_probe + longnav.utils.token_info (paper/layer_token_info).

What has to hold:
  1. The flag is refused without use_sparse, with a context window, outside prune/evict mode,
     and off sdpa (CPU, no model).
  2. Evict arm: the flag itself changes nothing (live decisions match a plain worker); after the
     probe the episode continues unchanged; probe_recompute_arm returns [L, N] finite non-negative
     rows with raster == arange(N) at sparse_threshold=2.0; the rebuilt base matches the live
     decision to bf16 precision; every hook is removed; the forward count is 1 + L(2 + 3N).
  3. Hook semantics: a hook that starts past the last layer reproduces the base bit-for-bit
     (same mask shape, same arithmetic), a hook from layer 0 hiding the whole current frame moves
     the decision, and that hidden result equals physically deleting the tokens from the stored
     record (_strip_visual_rec) to the bf16 floor of two prefills of different lengths (~2e-2
     in probability, the same floor as the rebuilt base vs the live decision). Within an arm every
     probe is a same-length forward, so that floor never enters a measured KL.
  4. Prune arm (leaky): same shape/finiteness/raster checks, live decision unchanged by the probe,
     replay count 1 + L(2 + 3N); with the bf16 head the replayed base equals the live decision
     exactly and the per-slot machinery agrees with the frame-level _score_layer_frame_influence
     (KL(full || current frame hidden) at layer 0 coincides); with the fp32 head (default) the
     base keeps the live argmax and resolves at least as many distinct leave-one-out values.
  5. Prune vs evict decisions agree on the same frames (nothing evicted => same cache).
  6. Both arms, no-visual-history measures (``measures=``): only the requested arrays are returned,
     the forward count is n_forwards(L, N, measures), and on the FIRST frame of an episode (no
     history to hide) leave_nohist == leave and keep_all == keep_cur bit-for-bit.

Run inside longnav_vlm:  python tests/token_info_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker
from longnav.utils import token_info as ti

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 3  # probes run after the 3rd decision, then a 4th step checks nothing moved
LAYERS = (0, 14, 27)
PROB_TOL = 1e-4  # the flag / a probe must not move the live decision
BASE_TOL = 5e-2  # from-scratch bf16 re-prefill vs the live incremental forward
PARITY_TOL = 5e-2  # hook hiding vs physical deletion: two bf16 prefills of DIFFERENT lengths, the
#                    same floor as the rebuilt base vs the live decision (influence_recompute_smoke)
SUBSTRATE_TOL = 1e-2  # prune (read-time RoPE) vs stock decisions on identical frames
PIN_TOL = 1e-5  # per-slot vs per-frame replay with identical masks
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


def make_worker(mode, flag=True):
    kw = dict(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
              sparse_threshold=2.0, context_window=None, context_window_mode=mode,
              token_info_probe=flag)
    if mode == "prune":
        kw |= dict(kv_budget=10 ** 9, kv_prune_importance="random")
    return VLMWorker(**kw)


def make_images(n, h=240, w=320):
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def step(worker, t, image):
    probs, _, _ = worker.infer_probs(START if t == 0 else TURN, [image])
    return np.asarray(probs)


def check_arrays(failures, name, out, n_tokens):
    L = len(LAYERS)
    for key in ti.MEASURES:
        arr = out[key]
        if arr.shape != (L, n_tokens):
            failures.append(f"{name}: {key} has shape {arr.shape}, expected {(L, n_tokens)}")
        if not (np.isfinite(arr).all() and (arr >= 0).all()):
            failures.append(f"{name}: {key} has non-finite or negative entries")
    if not np.array_equal(out["raster"], np.arange(n_tokens)):
        failures.append(f"{name}: raster is not arange({n_tokens}) with the filter disabled")
    if abs(np.exp(out["base_logp"]).sum() - 1) > 1e-4:
        failures.append(f"{name}: base log-probs do not normalise")
    expected = 1 + L * (2 + 3 * n_tokens)
    if int(out["n_probes"]) != expected:
        failures.append(f"{name}: {int(out['n_probes'])} probes, expected {expected}")


def check_nohist(failures, name, worker, probe, image):
    """Check 6: measure selection on the live state, then the no-history identity on a fresh episode."""
    n_tokens = ti.current_frame_tokens(worker)[0].numel()
    out = probe(worker, (0,), measures=ti.NOHIST_MEASURES)
    absent = [m for m in ti.ALL_MEASURES if m not in ti.NOHIST_MEASURES and m in out]
    if absent:
        failures.append(f"{name}: unrequested measures returned: {absent}")
    for key in ti.NOHIST_MEASURES:
        if out[key].shape != (1, n_tokens) or not (np.isfinite(out[key]).all() and (out[key] >= 0).all()):
            failures.append(f"{name}: {key} malformed under measures={ti.NOHIST_MEASURES}")
    if int(out["n_probes"]) != ti.n_forwards(1, n_tokens, ti.NOHIST_MEASURES):
        failures.append(f"{name}: {int(out['n_probes'])} probes for the no-history measures, "
                        f"expected {ti.n_forwards(1, n_tokens, ti.NOHIST_MEASURES)}")
    print(f"  {name} no-history: mean keep_all {out['keep_all'].mean():.2e}, leave_nohist "
          f"{out['leave_nohist'].mean():.2e}, KL(full || history visuals hidden) {float(out['text_hist_kl'][0]):.2e}")
    worker.reset()
    step(worker, 0, image)
    first = probe(worker, (0,), measures=ti.ALL_MEASURES)
    for a, b in (("leave_nohist", "leave"), ("keep_all", "keep_cur")):
        if not np.array_equal(first[a], first[b]):
            failures.append(f"{name}: {a} != {b} on the first frame (no history to hide), "
                            f"max gap {np.abs(first[a] - first[b]).max():.2e}")
    if float(first["text_hist_kl"][0]) != 0.0:
        failures.append(f"{name}: hiding an empty history moved the decision")


def main():
    failures = []
    print("=== 1. refused configurations (no model) ===")
    for kwargs, why in ((dict(use_sparse=False), "use_sparse=False"),
                        (dict(use_sparse=True, context_window=4), "context_window=4"),
                        (dict(use_sparse=True, context_window_mode="reindex"), "reindex mode"),
                        (dict(use_sparse=True, attn_impl="eager"), "eager attention")):
        try:
            VLMWorker(model_id=MODEL_ID, load_model=False, token_info_probe=True, **kwargs)
        except (ValueError, NotImplementedError) as exc:
            print(f"  refused ({why}): {exc}")
        else:
            failures.append(f"token_info_probe accepted with {why}")

    images = make_images(N_STEPS + 1)

    print("=== 2. plain evict worker (reference decisions) ===")
    worker = make_worker("evict", flag=False)
    ref = [step(worker, t, im) for t, im in enumerate(images)]
    release(worker)

    print("=== 3. evict worker + flag, recompute arm ===")
    worker = make_worker("evict")
    live = [step(worker, t, im) for t, im in enumerate(images[:N_STEPS])]
    grid = ti.current_frame_grid(worker)
    n_tokens = (grid[1] // 2) * (grid[2] // 2)
    print(f"  frame grid {grid} -> {n_tokens} tokens")
    rec = ti.probe_recompute_arm(worker, LAYERS,
                                 progress=lambda li, l, s: print(f"  layer {l}: {s:.1f}s"))
    if any(layer.self_attn._forward_pre_hooks for layer in worker.language_model.layers):
        failures.append("recompute arm left forward pre-hooks installed")
    check_arrays(failures, "recompute", rec, n_tokens)
    base_gap = float(np.abs(np.exp(rec["base_logp"]) - live[-1]).max())
    print(f"  rebuilt base vs live decision: max |dp| = {base_gap:.2e}")
    if base_gap > BASE_TOL:
        failures.append(f"recompute base differs from the live decision by {base_gap:.2e}")
    # hook parity: hide all current-frame columns on every layer vs delete them from the record
    li0 = LAYERS.index(0)
    prefix, turns = worker._device_recs()
    stripped, n_removed = worker._strip_visual_rec(turns[-1])
    if n_removed != n_tokens:
        failures.append(f"_strip_visual_rec removed {n_removed} tokens, expected {n_tokens}")
    deleted = ti.forward_logp_masked(worker, [prefix] + list(turns[:-1]) + [stripped], fp32_head=True)  # same kernel path, same head
    parity = float(np.abs(np.exp(rec["text_cur_logp"][li0]) - np.exp(deleted.numpy())).max())
    print(f"  hook hiding vs deletion: max |dp| = {parity:.2e}")
    if parity > PARITY_TOL:
        failures.append(f"hook hiding differs from deletion by {parity:.2e}")
    recs_all = [prefix] + list(turns)
    T = sum(int(r["inputs_embeds"].shape[1]) for r in recs_all)
    cur_hide = torch.zeros(T, dtype=torch.bool, device=recs_all[0]["inputs_embeds"].device)
    cur_hide[ti.current_frame_tokens(worker)[0].to(cur_hide.device)] = True
    n_layers = len(worker.language_model.layers)
    noop = ti.forward_logp_masked(worker, recs_all, cur_hide, n_layers, fp32_head=True)
    if not torch.equal(noop, torch.from_numpy(rec["base_logp"]).to(noop.dtype)):
        failures.append("a hook starting past the last layer changed the base decision")
    moved = float(np.abs(np.exp(rec["text_cur_logp"][li0]) - np.exp(rec["base_logp"])).max())
    print(f"  hiding the whole current frame from layer 0 moves the decision by max |dp| = {moved:.2e}")
    if moved <= 1e-6:
        failures.append("hiding every current-frame column from layer 0 did not move the decision")
    live.append(step(worker, N_STEPS, images[N_STEPS]))
    for t, (a, b) in enumerate(zip(live, ref)):
        d = float(np.abs(a - b).max())
        if d > PROB_TOL:
            failures.append(f"evict step {t}: flag/probe moved the live decision by {d:.2e}")
    check_nohist(failures, "recompute", worker, ti.probe_recompute_arm, images[0])
    worker.reset()
    if worker._window_recs or worker._prefix_rec is not None or worker._influence_recs:
        failures.append("reset() did not clear the stored records")
    release(worker)
    print(f"  recompute arm: {int(rec['n_probes'])} prefills in {float(rec['seconds']):.1f}s; "
          f"mean leave {rec['leave'].mean():.2e}, keep_cur {rec['keep_cur'].mean():.2e}, "
          f"keep_all {rec['keep_all'].mean():.2e}")

    print("=== 4. prune worker + flag, mask arm ===")
    worker = make_worker("prune")
    live = [step(worker, t, im) for t, im in enumerate(images[:N_STEPS])]
    msk = ti.probe_mask_arm(worker, LAYERS,
                            progress=lambda li, l, s: print(f"  layer {l}: {s:.1f}s"))
    check_arrays(failures, "mask", msk, n_tokens)
    base_gap = float(np.abs(np.exp(msk["base_logp"]) - live[-1]).max())
    print(f"  replayed base (fp32 head) vs live decision: max |dp| = {base_gap:.2e}")
    if base_gap > BASE_TOL or int(np.argmax(msk["base_logp"])) != int(np.argmax(live[-1])):
        failures.append(f"mask base (fp32 head) differs from the live decision by {base_gap:.2e}")
    # the bf16 reading reproduces the live decision exactly and pins the per-slot mask machinery
    # against the frame-level diagnostic (identical masks -> identical KL)
    msk16 = ti.probe_mask_arm(worker, (0,), fp32_head=False)
    base_gap = float(np.abs(np.exp(msk16["base_logp"]) - live[-1]).max())
    print(f"  replayed base (bf16 head) vs live decision: max |dp| = {base_gap:.2e}")
    if base_gap > PROB_TOL:
        failures.append(f"mask base (bf16 head) differs from the live decision by {base_gap:.2e}")
    n_uniq16, n_uniq32 = len(np.unique(msk16["leave"][0])), len(np.unique(msk["leave"][li0]))
    print(f"  distinct leave-one-out values at layer 0: bf16 head {n_uniq16}, fp32 head {n_uniq32}")
    if n_uniq32 < n_uniq16:
        failures.append("the fp32 head resolves fewer distinct leave-one-out values than the bf16 head")
    # per-slot machinery vs the frame-level diagnostic with identical masks
    ps = worker._kv_prune_state
    saved_starts = worker.kv_prune_layer_influence_starts
    worker.kv_prune_layer_influence_starts = (0,)
    try:
        _, leave_rows = worker._score_layer_frame_influence(
            ps.is_visual.clone(), torch.zeros(worker._master_len(), dtype=torch.bool))
    finally:
        worker.kv_prune_layer_influence_starts = saved_starts
    frame_leave = float(leave_rows[0][ps.n_turns - 1])
    pin = abs(frame_leave - float(msk16["text_cur_kl"][0]))
    print(f"  frame-level leave-one {frame_leave:.3e} vs per-slot text_cur_kl (bf16 head) {float(msk16['text_cur_kl'][0]):.3e}")
    if pin > PIN_TOL:
        failures.append(f"per-slot and per-frame replays disagree by {pin:.2e}")
    live.append(step(worker, N_STEPS, images[N_STEPS]))
    for t, (a, b) in enumerate(zip(live, ref)):
        d = float(np.abs(a - b).max())
        if d > SUBSTRATE_TOL:  # different substrate (read-time RoPE), bf16-level differences expected
            failures.append(f"prune step {t}: decision differs from the evict reference by {d:.2e}")
        if int(np.argmax(a)) != int(np.argmax(b)):
            failures.append(f"prune step {t}: argmax differs from the evict reference")
    check_nohist(failures, "mask", worker, ti.probe_mask_arm, images[0])
    release(worker)
    print(f"  mask arm: {int(msk['n_probes'])} replays in {float(msk['seconds']):.1f}s; "
          f"mean leave {msk['leave'].mean():.2e}, keep_cur {msk['keep_cur'].mean():.2e}, "
          f"keep_all {msk['keep_all'].mean():.2e}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("token_info smoke PASS")


if __name__ == "__main__":
    main()
