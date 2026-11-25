"""GPU smoke for vlm.prune_kl_probe + longnav.utils.prune_kl_probe (paper/prune_kl).

What has to hold:
  1. (CPU, no model) The flag is refused in evict mode, with a window, candidate_scope='all',
     recent_turns=2, layer_start=2, a frame budget, a non-zero kv_prune_score_layer, an unknown or
     repeated method, or no budget; an accepted worker has kv_prune_score_layer == 0 and one
     shadow per (method, k).
  2. The probe changes nothing live: a plain prune worker (random importance, unreachable budget)
     and the probe worker produce identical probabilities on every step, and identical cache
     length, visual count, embed-DB rows and random-generator state.
  3. The shadows ARE the flat arms: with the sparse filter off (so the tokens of a frame do not
     depend on what was pruned), the surviving (frame, patch) set of every shadow equals the
     survivors of a LIVE flat worker of that (method, k) -- at every step for random and DivPrune,
     through the first binding step for FastV and DART (afterwards the live arm scores against
     its pruned cache; documented caveat), with equal survivor counts throughout.
  4. The logged numbers are the masked replay: for every shadow, the hidden set is the visual
     slots it dropped BEFORE this step (the frame decided on is never hidden), `hidden + kept`
     is the history's visual count, `kept` equals the previous selection's capacity, and the
     logged log-probs / KL equal a manual decision replay with those slots at -inf.
  5. A budget that never binds scores exactly 0 with the base log-probs and no replay; the
     replay count is 2 + (shadows that hide something).
  6. log_rows carries the sup/prune_kl_* keys; reset() clears the row, the counters and rebuilds
     empty shadows.

Run inside longnav_vlm:  python tests/prune_kl_smoke.py [--cpu-only]
"""
import gc
import sys

import numpy as np
import torch
from PIL import Image

from longnav.utils import prune_kl_probe as pk
from longnav.utils.vlm_worker import VLMWorker

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 6
BUDGETS = (1, 2, 50)  # frames' worth: k=1 binds from the 2nd frame on, k=2 from the 3rd, k=50 never
LIVE_BUDGETS = (1, 2)
PROB_TOL = 1e-6
LOGP_TOL = 2e-6  # the row rounds log-probs to 6 decimals
LABEL = "smoke_episode_3"
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
BASE = dict(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True, sparse_threshold=2.0,
            context_window=None, context_window_mode="prune", kv_prune_candidate_scope="visual",
            kv_prune_recent_turns=0, kv_prune_layer_start=0, kv_prune_seed=17, kv_prune_seed_per_step=True,
            kv_prune_ema_beta=0.0)
LIVE = {  # the flat arms of paper/token_prune (run_r1.sh, VARIANT=flat)
    "random": dict(kv_prune_importance="random"),
    "fastv": dict(kv_prune_importance="attn", kv_prune_score_layer=0),
    "divprune": dict(kv_prune_importance="divprune"),
    "dart": dict(kv_prune_importance="dart"),
}
EXACT = ("random", "divprune")  # selection independent of the cache's pruning history


def make_worker(kind, method=None, k=None, **extra):
    kw = dict(BASE)
    if kind == "probe":
        kw |= dict(kv_budget=10 ** 9, kv_prune_importance="random", prune_kl_probe=True,
                   prune_kl_methods=pk.METHODS, prune_kl_budgets=BUDGETS)
    elif kind == "reference":
        kw |= dict(kv_budget=10 ** 9, kv_prune_importance="random")
    elif kind == "live":
        kw |= dict(LIVE[method], kv_prune_budget_frames=float(k))
    else:
        raise ValueError(kind)
    return VLMWorker(**(kw | extra))


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


def survivors(ps, alive=None):
    """Surviving visual slots as a sorted list of (frame, patch)."""
    m = ps.is_visual if alive is None else ps.is_visual & alive
    return sorted(zip(ps.turn_id[m].tolist(), ps.raster[m].tolist()))


def cpu_checks(failures):
    print("=== 1. refused configurations (no model) ===")
    good = dict(BASE, load_model=False, kv_budget=10 ** 9, kv_prune_importance="random",
                prune_kl_probe=True, prune_kl_methods=pk.METHODS, prune_kl_budgets=BUDGETS)
    for kwargs, why in ((dict(context_window_mode="evict"), "evict mode"),
                        (dict(context_window=4), "a window"),
                        (dict(kv_prune_candidate_scope="all"), "candidate_scope=all"),
                        (dict(kv_prune_recent_turns=2), "recent_turns=2"),
                        (dict(kv_prune_layer_start=2), "layer_start=2"),
                        (dict(kv_budget=None, kv_prune_budget_frames=1), "a frame budget"),
                        (dict(kv_prune_score_layer=3), "kv_prune_score_layer=3"),
                        (dict(prune_kl_methods=("random", "bogus")), "an unknown method"),
                        (dict(prune_kl_methods=("random", "random")), "a repeated method"),
                        (dict(prune_kl_methods=()), "no method"),
                        (dict(prune_kl_budgets=()), "no budgets"),
                        (dict(prune_kl_budgets=(0, 4)), "a zero budget")):
        try:
            VLMWorker(**(good | kwargs))
        except (ValueError, NotImplementedError) as exc:
            print(f"  refused ({why}): {str(exc)[:90]}")
        else:
            failures.append(f"prune_kl_probe accepted with {why}")
    w = VLMWorker(**good)
    if w.kv_prune_score_layer != 0:
        failures.append(f"accepted probe worker has kv_prune_score_layer={w.kv_prune_score_layer}, expected 0")
    arms = [(s.method, s.k) for s in w._prune_kl_shadows]
    if arms != [(m, k) for m in pk.METHODS for k in BUDGETS]:
        failures.append(f"shadows {arms} are not the (method, k) grid")
    plain = VLMWorker(**dict(BASE, load_model=False, kv_budget=10 ** 9, kv_prune_importance="random"))
    if plain.prune_kl_probe or plain.prune_kl_methods or plain._prune_kl_shadows:
        failures.append("a worker without the flag carries prune_kl state")
    del w, plain


def check_row(failures, worker, t, row, prev_row, pre_alive, replays_before):
    """Checks 4 and 5 on the live state right after step t."""
    ps = worker._kv_prune_state
    kv_len = worker._master_len()
    cur = ps.n_turns - 1
    history = ps.is_visual & (ps.turn_id < cur)
    n_hist = int(history.sum())
    base_bias = worker._replay_base_bias(torch.zeros(kv_len, dtype=torch.bool))
    base = worker._replay_action_logp(base_bias)
    if not np.allclose(row["base_logp"], base.cpu().tolist(), atol=LOGP_TOL, rtol=0):
        failures.append(f"t={t}: logged base log-probs differ from a fresh base replay")
    novis = worker._replay_action_logp(base_bias, extra=worker._replay_ablation_term(ps.is_visual))
    if not np.isclose(row["visual_all"][0], worker._action_kl(base, novis), rtol=1e-5, atol=1e-9):
        failures.append(f"t={t}: visual_all KL differs from a manual no-visual replay")
    n_hiding = 0
    for i, s in enumerate(worker._prune_kl_shadows):
        tag = f"t={t} {s.method} k={s.k}"
        before = torch.cat([pre_alive[i], torch.ones(kv_len - pre_alive[i].numel(), dtype=torch.bool)])
        mask = ps.is_visual & ~before
        if bool((mask & ~history).any()):
            failures.append(f"{tag}: the probe hid a slot of the frame being decided on")
        if row["hidden"][i] != int(mask.sum()) or row["hidden"][i] + row["kept"][i] != n_hist:
            failures.append(f"{tag}: hidden {row['hidden'][i]} + kept {row['kept'][i]} != history {n_hist} "
                            f"(expected hidden {int(mask.sum())})")
        if prev_row is not None and prev_row["cap"][i] != pk.NONE and row["kept"][i] != prev_row["cap"][i]:
            failures.append(f"{tag}: {row['kept'][i]} history slots visible, the last selection kept "
                            f"{prev_row['cap'][i]}")
        if not bool(mask.any()):
            if row["kl"][i] != 0.0 or row["logp"][i] != row["base_logp"]:
                failures.append(f"{tag}: nothing hidden but KL {row['kl'][i]} / log-probs differ from base")
            continue
        n_hiding += 1
        manual = worker._replay_action_logp(base_bias, extra=worker._replay_ablation_term(mask))
        if not np.allclose(row["logp"][i], manual.cpu().tolist(), atol=LOGP_TOL, rtol=0):
            gap = float(np.abs(np.asarray(row["logp"][i]) - manual.cpu().numpy()).max())
            failures.append(f"{tag}: logged log-probs differ from the manual masked replay (max gap {gap:.2e})")
        kl = worker._action_kl(base, manual)
        if not np.isclose(row["kl"][i], kl, rtol=1e-5, atol=1e-9):
            failures.append(f"{tag}: logged KL {row['kl'][i]:.6g} != manual {kl:.6g}")
        # the selection after this decision drops only from survivors + this frame
        if bool((s.alive & ~before).any()):
            failures.append(f"{tag}: the selection revived a dropped slot")
    never = [i for i, s in enumerate(worker._prune_kl_shadows) if s.k == max(BUDGETS)]
    if any(row["hidden"][i] or row["cap"][i] != pk.NONE or row["kl"][i] != 0.0 for i in never):
        failures.append(f"t={t}: the k={max(BUDGETS)} shadows bind or score")
    if row["n_probes"] != 2 + n_hiding or worker._prune_kl_count - replays_before != row["n_probes"]:
        failures.append(f"t={t}: {row['n_probes']} replays, expected {2 + n_hiding}")
    by = {(m, k): v for m, k, v in zip(row["methods"], row["budgets"], row["kl"])}
    print(f"  t={t}: history visual {n_hist}, {row['n_probes']} replays in {row['seconds']:.2f}s; "
          f"no-visual KL {row['visual_all'][0]:.3e}; k=1 KL " +
          " ".join(f"{m} {by[(m, 1)]:.2e}" for m in pk.METHODS))


def main():
    failures = []
    cpu_checks(failures)
    if "--cpu-only" in sys.argv:
        finish(failures)
        return
    images = make_images(N_STEPS + 1)

    print("=== 2. reference prune worker (random importance, no probe) ===")
    worker = make_worker("reference")
    worker.begin_episode(LABEL)
    ref, ref_state = [], []
    for t, im in enumerate(images):
        ref.append(step(worker, t, im))
        ps = worker._kv_prune_state
        ref_state.append((worker._master_len(), int(ps.is_visual.sum()), int(worker.past_image_embeds[0].shape[0]),
                          ps.generator.get_state().clone()))
    release(worker)

    print("=== 3. live flat workers: survivors per step ===")
    live = {}
    for method in pk.METHODS:
        for k in LIVE_BUDGETS:
            worker = make_worker("live", method, k)
            worker.begin_episode(LABEL)
            rows = []
            for t, im in enumerate(images[:N_STEPS]):
                step(worker, t, im)
                rows.append(survivors(worker._kv_prune_state))
            live[(method, k)] = rows
            print(f"  {method} k={k}: visual survivors per step {[len(r) for r in rows]}")
            release(worker)

    print("=== 4/5. probe worker ===")
    worker = make_worker("probe")
    worker.begin_episode(LABEL)
    probs, prev_row = [], None
    agree = {arm: 0 for arm in live}
    for t, im in enumerate(images[:N_STEPS]):
        pre_alive = [s.alive.clone() for s in worker._prune_kl_shadows]
        replays_before = worker._prune_kl_count
        probs.append(step(worker, t, im))
        row = dict(worker._prune_kl_row)
        ps = worker._kv_prune_state
        state = (worker._master_len(), int(ps.is_visual.sum()), int(worker.past_image_embeds[0].shape[0]))
        if state != ref_state[t][:3] or not torch.equal(ps.generator.get_state(), ref_state[t][3]):
            failures.append(f"t={t}: the probe worker's live state {state} differs from the reference "
                            f"{ref_state[t][:3]} (or its random generator moved)")
        if worker._hidden_db is None or worker._hidden_db.shape[0] != state[1]:
            failures.append(f"t={t}: DART hidden DB out of step with the {state[1]} visual slots")
        check_row(failures, worker, t, row, prev_row, pre_alive, replays_before)
        for s in worker._prune_kl_shadows:
            arm = (s.method, s.k)
            if arm not in live:
                continue
            mine, theirs = survivors(ps, s.alive), live[arm][t]
            if len(mine) != len(theirs):
                failures.append(f"t={t} {arm}: shadow holds {len(mine)} visual slots, the live arm {len(theirs)}")
            same = mine == theirs
            agree[arm] += int(same)
            if not same and (s.method in EXACT or t <= s.k):
                failures.append(f"t={t} {arm}: shadow survivors differ from the live flat arm's "
                                f"({len(set(mine) & set(theirs))} of {len(theirs)} shared)")
        prev_row = row
    print("  steps on which the shadow equals the live arm: " +
          ", ".join(f"{m} k={k}: {n}/{N_STEPS}" for (m, k), n in agree.items()))
    probs.append(step(worker, N_STEPS, images[N_STEPS]))
    gaps = [float(np.abs(a - b).max()) for a, b in zip(probs, ref)]
    for t, d in enumerate(gaps):
        if d > PROB_TOL:
            failures.append(f"step {t}: the probe moved the live decision by {d:.2e}")
    print(f"  live vs reference decisions: max |dp| = {max(gaps):.2e}")
    row = worker._prune_kl_row
    if not any(k > 0 for k in row["kl"]):
        failures.append("no shadow moved the decision at the last step: the probe hides nothing")

    print("=== 6. log rows, reset ===")
    logs = pk.log_rows(worker)
    missing = [k for k in ("prune_kl_methods", "prune_kl_budgets", "prune_kl_kl", "prune_kl_logp",
                           "prune_kl_hidden", "prune_kl_kept", "prune_kl_cap", "prune_kl_base_logp",
                           "prune_kl_visual_all", "max/prune_kl_probe_count") if k not in logs]
    if missing or len(logs["prune_kl_kl"]) != len(pk.METHODS) * len(BUDGETS):
        failures.append(f"log_rows misses {missing} or has {len(logs.get('prune_kl_kl', []))} KLs")
    worker.reset()
    if worker._prune_kl_row or worker._prune_kl_count or worker._prune_kl_latency \
            or any(s.alive.numel() for s in worker._prune_kl_shadows) \
            or len(worker._prune_kl_shadows) != len(pk.METHODS) * len(BUDGETS):
        failures.append("reset() did not clear the prune_kl state")
    release(worker)
    finish(failures)


def finish(failures):
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("prune_kl smoke PASS")


if __name__ == "__main__":
    main()
