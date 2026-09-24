"""GPU smoke for vlm.influence_recompute (leak-free leave-one-frame-out probes).

Five things have to hold:
  1. The flag is refused without use_sparse, with a context window, or outside evict mode (CPU).
  2. With the flag on, the live decision is unchanged relative to a plain full-context worker
     (the probes only read; nothing they build is kept).
  3. Step t logs t+1 finite non-negative KLs and t+1 log-prob rows that normalise and reproduce
     the KLs; the rebuilt base distribution matches the live decision to bf16 precision; joint
     probes and the horizon bands are well-formed (history_all == 0 at t=0, bands only for K <= t);
     the number of forwards is 1 + t+1 + 2 + #{K <= t} (minus frames with no visual tokens).
  4. The signal is not inert (some frame moves the decision by more than the arithmetic floor).
  5. stride 2 scores only even steps; reset() clears everything.
  6. kv_prune_influence_measures=[keep_one]: leave-one rows empty, t+1 keep-one KLs whose
     only-frame log-probs reproduce KL(only f || no visual) against the visual_all log-probs,
     1 + 1 + (t+1) + history_all + bands forwards, joint probes identical to the leave-one run.

Run inside longnav_vlm:  python tests/influence_recompute_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
HORIZONS = (2, 4)
PROB_TOL = 1e-4  # probing must not move the live decision
BASE_TOL = 5e-2  # from-scratch bf16 re-prefill vs the live incremental forward
KL_TOL = 1e-5
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
FIELDS = ("_kl_influence_row", "_kl_influence_logp", "_kl_influence_base_logp",
          "_kl_influence_history_all", "_kl_influence_visual_all", "_kl_influence_horizon")


def make_worker(recompute, stride=1, measures=("leave_one",)):
    return VLMWorker(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
                     context_window=None, context_window_mode="evict", influence_recompute=recompute,
                     kv_prune_influence_stride=stride, kv_prune_influence_horizons=HORIZONS,
                     kv_prune_influence_measures=measures)


def make_images(n, h=240, w=320):
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


def release(worker):
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def kl_from_logp(base, other):
    base, other = np.asarray(base, dtype=float), np.asarray(other, dtype=float)
    return float(np.sum(np.exp(base) * (base - other)))


def run(worker, images):
    steps = []
    for t, image in enumerate(images):
        probs, _, _ = worker.infer_probs(START if t == 0 else TURN, [image])
        steps.append({
            "probs": np.asarray(probs),
            "row": list(worker._kl_influence_row),
            "logp": [list(r) for r in worker._kl_influence_logp],
            "base_logp": list(worker._kl_influence_base_logp),
            "history_all": list(worker._kl_influence_history_all),
            "visual_all": list(worker._kl_influence_visual_all),
            "horizon": [list(h) for h in worker._kl_influence_horizon],
            "keep_one": list(worker._kl_influence_keep_one),
            "keep_one_logp": [list(r) for r in worker._kl_influence_keep_one_logp],
            "forwards": worker._influence_probe_count,
            "seconds": worker._influence_probe_latency,
            "visual_new": worker._prune_visual_new,
            "kv_len": worker.past_key_values.get_seq_length(),
        })
    return steps


def check_joint(failures, t, name, probe, n_vocab):
    if len(probe) != 2 or len(probe[1]) != n_vocab:
        failures.append(f"step {t}: {name} malformed: {probe}")
        return
    kl, logp = probe
    if not (np.isfinite(kl) and kl >= 0 and np.all(np.isfinite(logp))):
        failures.append(f"step {t}: {name} non-finite or negative: {probe}")
    if abs(np.exp(logp).sum() - 1) > 1e-4:
        failures.append(f"step {t}: {name} log-probs do not normalise: {logp}")


def main():
    failures = []
    print("=== 1. refused configurations (no model) ===")
    for kwargs, why in ((dict(use_sparse=False), "use_sparse=False"),
                        (dict(use_sparse=True, context_window=4), "context_window=4"),
                        (dict(use_sparse=True, context_window_mode="prune", kv_budget=10), "prune mode")):
        try:
            VLMWorker(model_id=MODEL_ID, load_model=False, influence_recompute=True, **kwargs)
        except (ValueError, NotImplementedError) as exc:
            print(f"  refused ({why}): {exc}")
        else:
            failures.append(f"influence_recompute accepted with {why}")

    images = make_images(N_STEPS)
    print("=== 2. flag off ===")
    worker = make_worker(False)
    off = run(worker, images)
    release(worker)

    print("=== 3. flag on ===")
    worker = make_worker(True)
    n_vocab = len(worker.vocab_ids)
    on = run(worker, images)
    worker.reset()
    if any(getattr(worker, f) != [] for f in FIELDS) or worker._influence_recs or worker._influence_probe_count:
        failures.append("reset() did not clear the influence state")
    release(worker)
    prev_forwards = 0
    for t, (a, b) in enumerate(zip(on, off)):
        row = np.asarray(a["row"], dtype=float)
        delta = float(np.abs(a["probs"] - b["probs"]).max())
        if delta > PROB_TOL:
            failures.append(f"step {t}: probing moved the live decision by {delta:.2e}")
        if a["kv_len"] != b["kv_len"]:
            failures.append(f"step {t}: cache length {a['kv_len']} vs {b['kv_len']} without probes")
        if row.size != t + 1 or len(a["logp"]) != t + 1:
            failures.append(f"step {t}: {row.size} KLs / {len(a['logp'])} logp rows, expected {t + 1}")
        if not (np.isfinite(row).all() and (row >= 0).all()):
            failures.append(f"step {t}: non-finite or negative KL in {row}")
        if len(a["base_logp"]) != n_vocab or abs(np.exp(a["base_logp"]).sum() - 1) > 1e-4:
            failures.append(f"step {t}: base log-probs malformed")
        for f, (kl, logp) in enumerate(zip(row, a["logp"])):
            if len(logp) != n_vocab or abs(np.exp(logp).sum() - 1) > 1e-4:
                failures.append(f"step {t}, frame {f}: log-probs malformed")
            elif abs(kl_from_logp(a["base_logp"], logp) - kl) > KL_TOL:
                failures.append(f"step {t}, frame {f}: KL {kl:.6f} != recomputed {kl_from_logp(a['base_logp'], logp):.6f}")
        base_gap = float(np.abs(np.exp(a["base_logp"]) - b["probs"]).max())
        if base_gap > BASE_TOL:
            failures.append(f"step {t}: rebuilt base is {base_gap:.2e} from the live decision")
        check_joint(failures, t, "history_all", a["history_all"], n_vocab)
        check_joint(failures, t, "visual_all", a["visual_all"], n_vocab)
        if t == 0 and a["history_all"] and a["history_all"][0] != 0.0:
            failures.append("step 0: history_all should be 0 with no history")
        want_k = [K for K in HORIZONS if K <= t]
        if [h[0] for h in a["horizon"]] != want_k:
            failures.append(f"step {t}: horizon bands {[h[0] for h in a['horizon']]}, expected {want_k}")
        for h in a["horizon"]:
            check_joint(failures, t, f"horizon K={h[0]}", h[1:], n_vocab)
        want_fw = 1 + (t + 1) + 1 + (1 if t >= 1 else 0) + len(want_k)  # base, frames, visual_all, history_all, bands
        got_fw = a["forwards"] - prev_forwards
        prev_forwards = a["forwards"]
        if got_fw != want_fw:
            failures.append(f"step {t}: {got_fw} forwards, expected {want_fw}")
        hist = a["history_all"][0] if a["history_all"] else float("nan")
        vis = a["visual_all"][0] if a["visual_all"] else float("nan")
        bands = " ".join(f"K{h[0]}={h[1]:.4f}" for h in a["horizon"])
        print(f"  step {t}: KL row {np.array2string(row, precision=4)}  hist={hist:.4f} vis={vis:.4f} {bands}  "
              f"visual_new={a['visual_new']}  {got_fw} forwards  |dp|={delta:.1e} |base-live|={base_gap:.1e}")
    if max(max(s["row"]) for s in on if s["row"]) <= 1e-4:
        failures.append("no frame ever moved the decision above 1e-4: the removal is inert")
    print(f"  total probe time {on[-1]['seconds']:.1f} s for {on[-1]['forwards']} forwards")

    print("=== 4. stride 2 ===")
    worker = make_worker(True, stride=2)
    strided = run(worker, images)
    release(worker)
    for t, s in enumerate(strided):
        want = 0 if t % 2 else t + 1
        if len(s["row"]) != want or len(s["logp"]) != want:
            failures.append(f"stride 2, step {t}: {len(s['row'])} KLs, expected {want}")
        if t % 2 and (s["horizon"] or s["history_all"] or s["visual_all"]):
            failures.append(f"stride 2, step {t}: joint probes logged on a skipped step")

    print("=== 5. keep_one (leave-one-in) ===")
    worker = make_worker(True, measures=("keep_one",))
    keep = run(worker, images)
    release(worker)
    prev_forwards = 0
    for t, (a, b, c) in enumerate(zip(keep, off, on)):
        if a["row"] or a["logp"]:
            failures.append(f"keep_one step {t}: leave-one rows logged")
        row = np.asarray(a["keep_one"], dtype=float)
        if row.size != t + 1 or len(a["keep_one_logp"]) != t + 1:
            failures.append(f"keep_one step {t}: {row.size} KLs, expected {t + 1}")
        if not (np.isfinite(row).all() and (row >= 0).all()):
            failures.append(f"keep_one step {t}: non-finite or negative KL in {row}")
        vis_logp = a["visual_all"][1] if len(a["visual_all"]) == 2 else []
        for f, (kl, logp) in enumerate(zip(row, a["keep_one_logp"])):
            if len(logp) != n_vocab or abs(np.exp(logp).sum() - 1) > 1e-4:
                failures.append(f"keep_one step {t}, frame {f}: log-probs malformed")
            elif vis_logp and abs(kl_from_logp(logp, vis_logp) - kl) > KL_TOL:
                failures.append(f"keep_one step {t}, frame {f}: KL {kl:.6f} != recomputed {kl_from_logp(logp, vis_logp):.6f}")
        want_k = [K for K in HORIZONS if K <= t]
        want_fw = 1 + 1 + (t + 1) + (1 if t >= 1 else 0) + len(want_k)  # base, visual_all, frames, history_all, bands
        got_fw = a["forwards"] - prev_forwards
        prev_forwards = a["forwards"]
        if got_fw != want_fw:
            failures.append(f"keep_one step {t}: {got_fw} forwards, expected {want_fw}")
        for name in ("base_logp", "history_all", "visual_all", "horizon"):
            if a[name] != c[name]:
                failures.append(f"keep_one step {t}: {name} differs from the leave-one run")
        delta = float(np.abs(a["probs"] - b["probs"]).max())
        if delta > PROB_TOL:
            failures.append(f"keep_one step {t}: probing moved the live decision by {delta:.2e}")
        print(f"  step {t}: keep-one row {np.array2string(row, precision=4)}  vis={a['visual_all'][0]:.4f}  {got_fw} forwards")
    if max(max(s["keep_one"]) for s in keep if s["keep_one"]) <= 1e-4:
        failures.append("keep_one: no frame alone ever moved the decision above 1e-4")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: leak-free probes leave the live decision and cache untouched, log t+1 consistent "
          "rows plus well-formed joint/horizon probes, rebuild the base to bf16 precision, honour "
          "the stride, and clear on reset.")


if __name__ == "__main__":
    main()
