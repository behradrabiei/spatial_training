"""GPU smoke for vlm.kv_prune_log_influence (leave-one-frame-out rows every step).

Five things have to hold:
  1. The flag is refused outside prune mode, and horizons < 2 are refused (CPU, no model).
  2. With the flag off nothing is replayed and every row stays empty.
  3. With the flag on (stride 1), step t logs t+1 finite non-negative KLs and t+1 hidden-frame
     log-prob rows that normalise and reproduce the logged KLs; the base log-probs match the
     live decision to bf16 replay precision; the joint probes (history_all, visual_all, and one
     age band per horizon K <= t) are finite and non-negative with history_all == 0 at t=0; the
     replay count is exactly what those probes cost; the cache/table/metadata are untouched and
     the decision does not move relative to the flag-off run. reset() clears everything.
  4. stride 2 scores only even steps.
  5. horizon K=t+1.. are absent (no frame is that old), and history_all equals the K=1 band by
     construction (asserted through the mask identity on the smallest configured horizon).
  6. kv_prune_influence_measures=[keep_one] (leave-one-in): the leave-one rows stay empty, step t
     logs t+1 finite non-negative keep-one KLs whose only-frame log-probs reproduce them against
     the visual_all log-probs (the reference), the replay count is 1 base + (t+1) + history_all
     + visual_all + bands, the joint probes are unchanged and the decision does not move; an
     empty or unknown measure list is refused.

Run inside longnav_vlm:  python tests/kl_influence_smoke.py
"""
import gc

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
HORIZONS = (2, 4)  # small so the smoke exercises populated bands within N_STEPS
PROB_TOL = 1e-4  # logging must not move the live decision
BASE_TOL = 5e-2  # bf16 single-token replay vs the live multi-token forward (a diagnostic only:
#                  every logged KL compares two replays, so this gap never enters the metric)
KL_TOL = 1e-5  # KL recomputed from the logged log-probs vs the logged KL (rounded to 6 dp)
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
NEW_FIELDS = ("_kl_influence_logp", "_kl_influence_base_logp", "_kl_influence_history_all",
              "_kl_influence_visual_all", "_kl_influence_horizon")


def make_worker(log, stride=1, measures=("leave_one",)):
    return VLMWorker(
        model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True,
        context_window=None, context_window_mode="prune", kv_budget=10 ** 9,
        kv_prune_importance="random", kv_prune_log_influence=log,
        kv_prune_influence_stride=stride, kv_prune_influence_horizons=HORIZONS,
        kv_prune_influence_measures=measures,
    )


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
        ps = worker._kv_prune_state
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
            "replays": worker._prune_replay_count,
            "ms_per_replay": 1e3 * worker._prune_score_latency / max(worker._prune_replay_count, 1),
            "kv_len": worker.past_key_values.get_seq_length(),
            "meta_len": ps.turn_id.numel(),
            "table_len": worker._reindex_state.pos_table.shape[-1],
            "visual_new": worker._prune_visual_new,
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

    print("=== 1. refused outside prune mode / bad horizons (no model) ===")
    try:
        VLMWorker(model_id=MODEL_ID, load_model=False, kv_prune_log_influence=True)
    except ValueError as exc:
        print(f"  refused: {exc}")
    else:
        failures.append("kv_prune_log_influence=True was accepted in evict mode")
    try:
        VLMWorker(model_id=MODEL_ID, load_model=False, context_window_mode="prune", kv_budget=10,
                  use_sparse=True, kv_prune_log_influence=True, kv_prune_influence_horizons=(1, 4))
    except ValueError as exc:
        print(f"  refused: {exc}")
    else:
        failures.append("kv_prune_influence_horizons=(1, 4) was accepted (K=1 is history_all)")

    images = make_images(N_STEPS)

    print("=== 2. flag off ===")
    worker = make_worker(log=False)
    off = run(worker, images)
    release(worker)
    for t, s in enumerate(off):
        if s["row"] or s["logp"] or s["horizon"] or s["history_all"] or s["replays"]:
            failures.append(f"flag off, step {t}: rows non-empty or replays={s['replays']}")

    print("=== 3. flag on, stride 1 ===")
    worker = make_worker(log=True)
    n_vocab = len(worker.vocab_ids)
    on = run(worker, images)
    worker.reset()
    if worker._kl_influence_row != [] or any(getattr(worker, f) != [] for f in NEW_FIELDS):
        failures.append("reset() did not clear every influence field")
    release(worker)
    for t, (a, b) in enumerate(zip(on, off)):
        row = np.asarray(a["row"], dtype=float)
        if row.size != t + 1 or len(a["logp"]) != t + 1:
            failures.append(f"step {t}: row has {row.size} KLs / {len(a['logp'])} logp rows, expected {t + 1}")
        if not (np.isfinite(row).all() and (row >= 0).all()):
            failures.append(f"step {t}: non-finite or negative KL in {row}")
        if row.size and row.max() <= 1e-6:
            failures.append(f"step {t}: every frame scored ~0 ({row}); the masking is inert")
        if len(a["base_logp"]) != n_vocab or abs(np.exp(a["base_logp"]).sum() - 1) > 1e-4:
            failures.append(f"step {t}: base log-probs malformed: {a['base_logp']}")
        for f, (kl, logp) in enumerate(zip(row, a["logp"])):
            if len(logp) != n_vocab or abs(np.exp(logp).sum() - 1) > 1e-4:
                failures.append(f"step {t}, frame {f}: hidden-frame log-probs malformed: {logp}")
                continue
            if abs(kl_from_logp(a["base_logp"], logp) - kl) > KL_TOL:
                failures.append(f"step {t}, frame {f}: KL {kl:.6f} != recomputed "
                                f"{kl_from_logp(a['base_logp'], logp):.6f} from the logged log-probs")
        base_gap = float(np.abs(np.exp(a["base_logp"]) - b["probs"]).max())
        if base_gap > BASE_TOL:
            failures.append(f"step {t}: replayed base distribution is {base_gap:.2e} from the live one")
        check_joint(failures, t, "history_all", a["history_all"], n_vocab)
        check_joint(failures, t, "visual_all", a["visual_all"], n_vocab)
        if t == 0 and a["history_all"] and a["history_all"][0] != 0.0:
            failures.append(f"step 0: history_all should be 0 with no history, got {a['history_all'][0]}")
        want_k = [K for K in HORIZONS if K <= t]
        got_k = [h[0] for h in a["horizon"]]
        if got_k != want_k:
            failures.append(f"step {t}: horizon bands {got_k}, expected {want_k}")
        for h in a["horizon"]:
            check_joint(failures, t, f"horizon K={h[0]}", h[1:], n_vocab)
        want_replays = (t + 1) + 1 + 1 + (1 if t >= 1 else 0) + len(want_k)
        if a["replays"] != want_replays:
            failures.append(f"step {t}: {a['replays']} replays, expected {want_replays}")
        if not (a["kv_len"] == a["meta_len"] == a["table_len"] == b["kv_len"]):
            failures.append(f"step {t}: cache {a['kv_len']} / meta {a['meta_len']} / table "
                            f"{a['table_len']} vs flag-off cache {b['kv_len']}")
        delta = float(np.abs(a["probs"] - b["probs"]).max())
        if delta > PROB_TOL:
            failures.append(f"step {t}: logging moved the decision by {delta:.2e}")
        hist = a["history_all"][0] if a["history_all"] else float("nan")
        vis = a["visual_all"][0] if a["visual_all"] else float("nan")
        bands = " ".join(f"K{h[0]}={h[1]:.4f}" for h in a["horizon"])
        print(f"  step {t}: KL row {np.array2string(row, precision=4)}  hist={hist:.4f} vis={vis:.4f} "
              f"{bands}  visual_new={a['visual_new']}  {a['replays']} replays @ {a['ms_per_replay']:.1f} ms  "
              f"|dp|={delta:.1e} |base-live|={base_gap:.1e}")

    print("=== 4. flag on, stride 2 ===")
    worker = make_worker(log=True, stride=2)
    strided = run(worker, images)
    release(worker)
    for t, s in enumerate(strided):
        want = 0 if t % 2 else t + 1
        if len(s["row"]) != want or len(s["logp"]) != want:
            failures.append(f"stride 2, step {t}: {len(s['row'])} KLs / {len(s['logp'])} logp rows, expected {want}")
        if t % 2 and (s["horizon"] or s["history_all"] or s["visual_all"]):
            failures.append(f"stride 2, step {t}: joint probes logged on a skipped step")

    print("=== 5. keep_one (leave-one-in) ===")
    for bad in ((), ("keep_all",)):
        try:
            VLMWorker(model_id=MODEL_ID, load_model=False, context_window_mode="prune", kv_budget=10,
                      use_sparse=True, kv_prune_log_influence=True, kv_prune_influence_measures=bad)
        except ValueError as exc:
            print(f"  refused measures={bad}: {exc}")
        else:
            failures.append(f"kv_prune_influence_measures={bad} was accepted")
    worker = make_worker(log=True, measures=("keep_one",))
    keep = run(worker, images)
    release(worker)
    for t, (a, b, c) in enumerate(zip(keep, off, on)):
        if a["row"] or a["logp"]:
            failures.append(f"keep_one step {t}: leave-one rows logged ({len(a['row'])})")
        row = np.asarray(a["keep_one"], dtype=float)
        if row.size != t + 1 or len(a["keep_one_logp"]) != t + 1:
            failures.append(f"keep_one step {t}: {row.size} KLs / {len(a['keep_one_logp'])} logp rows, expected {t + 1}")
        if not (np.isfinite(row).all() and (row >= 0).all()):
            failures.append(f"keep_one step {t}: non-finite or negative KL in {row}")
        if row.size and row.max() <= 1e-6:
            failures.append(f"keep_one step {t}: every frame scored ~0 ({row}); the masking is inert")
        check_joint(failures, t, "visual_all", a["visual_all"], n_vocab)
        vis_logp = a["visual_all"][1] if len(a["visual_all"]) == 2 else []
        for f, (kl, logp) in enumerate(zip(row, a["keep_one_logp"])):
            if len(logp) != n_vocab or abs(np.exp(logp).sum() - 1) > 1e-4:
                failures.append(f"keep_one step {t}, frame {f}: only-frame log-probs malformed: {logp}")
            elif vis_logp and abs(kl_from_logp(logp, vis_logp) - kl) > KL_TOL:
                failures.append(f"keep_one step {t}, frame {f}: KL {kl:.6f} != recomputed "
                                f"{kl_from_logp(logp, vis_logp):.6f} (KL(only f || no visual))")
        want_k = [K for K in HORIZONS if K <= t]
        want_replays = 1 + (t + 1) + 1 + (1 if t >= 1 else 0) + len(want_k)
        if a["replays"] != want_replays:
            failures.append(f"keep_one step {t}: {a['replays']} replays, expected {want_replays}")
        # joint probes and base are the same replays as in the leave-one run
        for name in ("base_logp", "history_all", "visual_all", "horizon"):
            if a[name] != c[name]:
                failures.append(f"keep_one step {t}: {name} differs from the leave-one run")
        delta = float(np.abs(a["probs"] - b["probs"]).max())
        if delta > PROB_TOL:
            failures.append(f"keep_one step {t}: logging moved the decision by {delta:.2e}")
        print(f"  step {t}: keep-one row {np.array2string(row, precision=4)}  vis={a['visual_all'][0]:.4f}  "
              f"{a['replays']} replays  |dp|={delta:.1e}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: influence rows carry t+1 KLs and log-prob rows that reproduce them, the base "
          "matches the live decision, joint/horizon probes are well-formed, the cache and the "
          "decision are untouched, reset clears everything, the stride is honoured, and the "
          "keep-one measure logs consistent leave-one-in rows against the no-visual reference.")


if __name__ == "__main__":
    main()
