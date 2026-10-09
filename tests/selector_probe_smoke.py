"""GPU smoke for vlm.selector_probe + longnav.utils.selector_probe (paper/selector_kl).

What has to hold:
  1. (CPU, no model) The flag is refused without token_info_probe, in evict mode, with
     candidate_scope='all', recent_turns=2, layer_start=2, a frame budget, or a non-zero
     kv_prune_score_layer; an accepted worker has kv_prune_score_layer == 0. keep_rank /
     member_rank on synthetic inputs, including "no budget binds" -> every rank NEVER.
  2. The probe changes no decision: a plain prune worker (random importance, unreachable budget)
     and the probe worker produce identical probabilities on every step, including a step after
     the probes have run.
  3. FastV row: the probe's `attn` equals bit for bit the layer-0 mean decision row of a worker
     configured as the flat fastv arm (kv_prune_importance=attn, kv_prune_score_layer=0); exactly
     one decision row is folded.
  4. Ranks reproduce the selectors: at every binding budget the tokens with `*_pos < cap`
     (attn, divprune, random) / `dart_rank <= k` are exactly those divprune_select / the
     seeded random_select / VLMWorker._dart_select / the attention top-k keep; DART's sets are
     nested; the budgets equal frame_budget(ps, 0, k) and do not bind while n_frames <= k. The
     continuous criteria equal a direct computation: divprune_nn = 1 - max cosine similarity to
     any other pool token (embed DB), dart_dis = 1 - min cosine similarity to the top-8 key-norm
     pivots on the layer-0 output (pivots 2.0).
  5. KLs: the row's keep_cur / leave equal probe_mask_arm(worker, [0], measures) (to the row's
     6-significant-digit rounding); the per-step replay count is n_forwards(1, N, measures).
  6. reset() clears the row, the counters and the DART hidden DB.

Run inside longnav_vlm:  python tests/selector_probe_smoke.py [--cpu-only]
"""
import gc
import sys

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker
from longnav.utils import token_info as ti
from longnav.utils import selector_probe as sp
from longnav.utils.kv_prune import divprune_select, frame_budget, random_select, step_seed, visual_slot_embeddings

MODEL_ID = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
N_STEPS = 5
BUDGETS = (1, 2, 3)  # frames' worth: k=1 binds from the 2nd frame on, k=3 from the 4th
MEASURES = ("keep_cur", "leave", "keep_all")
PROB_TOL = 1e-6
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
BASE = dict(model_id=MODEL_ID, attn_impl="sdpa", dtype="bfloat16", use_sparse=True, sparse_threshold=0.95,
            context_window=None, context_window_mode="prune", kv_budget=10 ** 9,
            kv_prune_candidate_scope="visual", kv_prune_recent_turns=0, kv_prune_layer_start=0,
            kv_prune_seed=17, token_info_probe=True)


def make_worker(kind):
    kw = dict(BASE)
    if kind == "probe":
        kw |= dict(kv_prune_importance="random", selector_probe=True, selector_probe_budgets=BUDGETS,
                   selector_probe_measures=MEASURES)
    elif kind == "reference":
        kw |= dict(kv_prune_importance="random")
    elif kind == "attn":
        kw |= dict(kv_prune_importance="attn", kv_prune_score_layer=0)
    else:
        raise ValueError(kind)
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


def cpu_checks(failures):
    print("=== 1. refused configurations (no model) ===")
    good = dict(BASE, load_model=False, kv_prune_importance="random", selector_probe=True,
                selector_probe_budgets=BUDGETS, selector_probe_measures=MEASURES)
    for kwargs, why in ((dict(token_info_probe=False), "no token_info_probe"),
                        (dict(context_window_mode="evict"), "evict mode"),
                        (dict(kv_prune_candidate_scope="all"), "candidate_scope=all"),
                        (dict(kv_prune_recent_turns=2), "recent_turns=2"),
                        (dict(kv_prune_layer_start=2), "layer_start=2"),
                        (dict(kv_prune_budget_frames=1), "a frame budget"),
                        (dict(kv_prune_score_layer=3), "kv_prune_score_layer=3"),
                        (dict(selector_probe_measures=("bogus",)), "an unknown measure"),
                        (dict(selector_probe_budgets=()), "no budgets")):
        try:
            VLMWorker(**(good | kwargs))
        except (ValueError, NotImplementedError) as exc:
            print(f"  refused ({why}): {str(exc)[:90]}")
        else:
            failures.append(f"selector_probe accepted with {why}")
    w = VLMWorker(**good)
    if w.kv_prune_score_layer != 0:
        failures.append(f"accepted probe worker has kv_prune_score_layer={w.kv_prune_score_layer}, expected 0")
    if w.selector_probe_budgets != BUDGETS:
        failures.append(f"budgets {w.selector_probe_budgets} != {BUDGETS}")
    del w
    # keep_rank / member_rank
    pos = torch.tensor([0, 1, 2, -1, 5])
    got = sp.keep_rank(pos, {1: 1, 2: 2, 3: None}, BUDGETS).tolist()
    if got != [1, 2, sp.NEVER, sp.NEVER, sp.NEVER]:
        failures.append(f"keep_rank gave {got}")
    got = sp.keep_rank(pos, {1: None, 2: None, 3: None}, BUDGETS).tolist()
    if got != [sp.NEVER] * 5:
        failures.append(f"keep_rank with no binding budget gave {got}")
    got = sp.keep_rank(pos, {1: 1, 2: 4, 3: 6}, BUDGETS).tolist()
    if got != [1, 2, 2, sp.NEVER, 3]:
        failures.append(f"keep_rank (all binding) gave {got}")
    members = {2: torch.tensor([True, False, True]), 3: torch.tensor([True, True, True])}
    got = sp.member_rank(members, BUDGETS, 3).tolist()
    if got != [2, 3, 2]:
        failures.append(f"member_rank gave {got}")
    print("  keep_rank / member_rank ok")


def check_row(failures, worker, t, row, n_before):
    """Checks 4 and 5 on the live state right after step t (n_frames = t + 1)."""
    ps, st = worker._kv_prune_state, worker._reindex_state
    cols, raster, _ = ti.current_frame_tokens(worker)
    n = cols.numel()
    if row["n_cur"] != n or row["raster"] != raster.tolist() or row["n_frames"] != t + 1:
        failures.append(f"t={t}: row header disagrees with the live state")
    kv_len = worker._master_len()
    cand = torch.nonzero(ps.is_visual).flatten()  # scope=visual, recent_turns=0: every visual slot
    if row["n_cand"] != cand.numel():
        failures.append(f"t={t}: n_cand {row['n_cand']} != {cand.numel()} visual slots")
    caps = {}
    for k, c in zip(row["budget_frames"], row["budgets"]):
        exp = frame_budget(ps, 0, k)
        exp = sp.NEVER if exp is None else min(int(exp), cand.numel())
        if c != exp:
            failures.append(f"t={t}: budget k={k} is {c}, frame_budget says {exp}")
        if (t + 1 <= k) != (c == sp.NEVER):
            failures.append(f"t={t}: budget k={k} binding state wrong (n_frames {t + 1}, cap {c})")
        caps[k] = None if c == sp.NEVER else c
    binding = [k for k in BUDGETS if caps[k] is not None]
    col_of = {int(c): i for i, c in enumerate(cols.tolist())}

    def kept_by_row(key, k):
        if key == "dart_rank":
            return {int(cols[i]) for i in range(n) if 0 <= row[key][i] <= k}
        return {int(cols[i]) for i in range(n) if 0 <= row[key][i] < caps[k]}

    scores = st.mean_scores().cpu()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(step_seed(worker.kv_prune_seed, worker._episode_label, ps.n_turns - 1))
    rand_order = random_select(cand, cand.numel(), gen)
    db = visual_slot_embeddings(ps, worker.past_image_embeds)
    db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
    rows = db[db_rank[cand].to(db.device)].to(worker.device)
    prev_dart = None
    for k in binding:
        cap = caps[k]
        exp_attn = set(cand[torch.argsort(scores[cand], descending=True, stable=True)[:cap]].tolist()) & set(col_of)
        if kept_by_row("attn_pos", k) != exp_attn:
            failures.append(f"t={t} k={k}: attn keep set differs from the top-{cap} of the decision row")
        exp_rand = set(rand_order[:cap].tolist()) & set(col_of)
        if kept_by_row("random_pos", k) != exp_rand:
            failures.append(f"t={t} k={k}: random keep set differs from the seeded permutation prefix")
        exp_div = set(divprune_select(rows, cand, cap).tolist()) & set(col_of)
        if kept_by_row("divprune_pos", k) != exp_div:
            failures.append(f"t={t} k={k}: divprune keep set differs from divprune_select")
        dart = set(worker._dart_select(ps, st, cand, cap).tolist())
        if kept_by_row("dart_rank", k) != dart & set(col_of):
            failures.append(f"t={t} k={k}: dart keep set differs from _dart_select")
        if prev_dart is not None and not prev_dart <= dart:
            failures.append(f"t={t} k={k}: DART sets are not nested")
        prev_dart = dart
        # rank <-> pos consistency: smallest binding k
        for key_pos, key_rank in (("attn_pos", "attn_rank"), ("divprune_pos", "divprune_rank"),
                                  ("random_pos", "random_rank")):
            for i in range(n):
                p = row[key_pos][i]
                smallest = next((kk for kk in binding if 0 <= p < caps[kk]), sp.NEVER)
                if row[key_rank][i] != smallest:
                    failures.append(f"t={t}: {key_rank}[{i}]={row[key_rank][i]} but pos {p} -> {smallest}")
                    break
    if not binding:
        for key in ("attn_rank", "divprune_rank", "dart_rank", "random_rank"):
            if any(r != sp.NEVER for r in row[key]):
                failures.append(f"t={t}: {key} not NEVER while no budget binds")
        if any(p != sp.NEVER for p in row["divprune_pos"]):
            failures.append(f"t={t}: divprune positions defined while no budget binds")
    if not row["dart_nested"]:
        failures.append(f"t={t}: row reports DART sets not nested")
    # continuous criteria against a direct computation
    import torch.nn.functional as F
    x = F.normalize(rows.float(), dim=-1)
    sim = x @ x.T
    sim.fill_diagonal_(-float("inf"))
    nn_ref = (1.0 - sim.amax(dim=1)).cpu()
    cand_pos = {int(c): i for i, c in enumerate(cand.tolist())}
    nn_ref_cur = torch.tensor([float(nn_ref[cand_pos[int(c)]]) for c in cols.tolist()])
    if not torch.allclose(torch.tensor(row["divprune_nn"]), nn_ref_cur, rtol=6e-6, atol=1e-6):
        gap = float((torch.tensor(row["divprune_nn"]) - nn_ref_cur).abs().max())
        failures.append(f"t={t}: divprune_nn differs from the direct nearest-neighbour distance (max gap {gap:.2e})")
    from longnav.utils.pre_rope import _rotate
    keys = worker.past_key_values.layers[0].keys
    cos_, sin_ = st.layer_tables(0)
    rot = _rotate(keys, cos_, sin_)[0].permute(1, 0, 2).reshape(keys.shape[-2], -1)
    norms = rot.float().abs().sum(dim=-1).cpu()
    pivots = cand[torch.argsort(norms[cand], descending=True, stable=True)[:worker.kv_prune_dart_pivots]]
    h = F.normalize(worker._hidden_db.float(), dim=-1)
    hp = h[db_rank[pivots].to(h.device)]
    hc = h[db_rank[cols].to(h.device)]
    dis_ref = (1.0 - (hc @ hp.T).amin(dim=1)).cpu()
    dis_ref[torch.isin(cols, pivots)] = 2.0
    if not torch.allclose(torch.tensor(row["dart_dis"]), dis_ref, rtol=6e-6, atol=1e-6):
        gap = float((torch.tensor(row["dart_dis"]) - dis_ref).abs().max())
        failures.append(f"t={t}: dart_dis differs from the direct pivot dissimilarity (max gap {gap:.2e})")
    # a pivot inside the current frame scores 2.0; every other value lies in the cosine-distance range
    dd = torch.tensor(row["dart_dis"])
    if not bool(((dd >= 0) & (dd <= 2.0)).all()):
        failures.append(f"t={t}: dart_dis outside [0, 2]")
    # 5. KLs against probe_mask_arm
    ref = ti.probe_mask_arm(worker, [0], measures=MEASURES)
    for m in MEASURES:
        a, b = np.asarray(row[m], dtype=np.float64), ref[m][0].astype(np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=6e-6, atol=1e-12):
            gap = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
            failures.append(f"t={t}: {m} differs from probe_mask_arm (max gap {gap:.2e})")
    if not np.isclose(row["text_cur_kl"], float(ref["text_cur_kl"][0]), rtol=6e-6, atol=1e-12):
        failures.append(f"t={t}: text_cur_kl differs from probe_mask_arm")
    expected = ti.n_forwards(1, n, MEASURES)
    if row["n_probes"] != expected or worker._selector_probe_count - n_before != expected:
        failures.append(f"t={t}: {row['n_probes']} / {worker._selector_probe_count - n_before} replays, "
                        f"expected {expected}")
    print(f"  t={t}: N={n}, pool {cand.numel()}, budgets {row['budgets']}, {row['n_probes']} replays in "
          f"{row['seconds']:.1f}s; mean keep_cur {np.mean(row['keep_cur']):.2e}; attn max {max(row['attn']):.3e}; "
          f"divprune_nn mean {np.mean(row['divprune_nn']):.3f}; dart_dis mean {np.mean(row['dart_dis']):.3f}")


def main():
    failures = []
    cpu_checks(failures)
    if "--cpu-only" in sys.argv:
        finish(failures)
        return
    images = make_images(N_STEPS + 1)

    print("=== 2. reference prune worker (random importance, no probe) ===")
    worker = make_worker("reference")
    ref = [step(worker, t, im) for t, im in enumerate(images)]
    release(worker)

    print("=== 3. flat fastv worker: layer-0 decision rows ===")
    worker = make_worker("attn")
    attn_rows = []
    for t, im in enumerate(images[:N_STEPS]):
        step(worker, t, im)
        cols, _, _ = ti.current_frame_tokens(worker)
        attn_rows.append(worker._reindex_state.mean_scores().cpu()[cols].clone())
        if worker._reindex_state.score_layers != 1:
            failures.append(f"attn worker folded {worker._reindex_state.score_layers} rows at t={t}")
    release(worker)

    print("=== 4/5. probe worker ===")
    worker = make_worker("probe")
    live = []
    for t, im in enumerate(images[:N_STEPS]):
        n_before = worker._selector_probe_count
        live.append(step(worker, t, im))
        row = worker._selector_row
        if not torch.equal(torch.tensor(row["attn"], dtype=torch.float32), attn_rows[t].float()):
            gap = float((torch.tensor(row["attn"]) - attn_rows[t].float()).abs().max())
            failures.append(f"t={t}: probe attn row differs from the fastv worker's (max gap {gap:.2e})")
        check_row(failures, worker, t, row, n_before)
    live.append(step(worker, N_STEPS, images[N_STEPS]))
    for t, (a, b) in enumerate(zip(live, ref)):
        d = float(np.abs(a - b).max())
        if d > PROB_TOL:
            failures.append(f"step {t}: the probe moved the live decision by {d:.2e}")
    print(f"  live vs reference decisions: max |dp| = {max(float(np.abs(a - b).max()) for a, b in zip(live, ref)):.2e}")
    logs = sp.log_rows(worker)
    missing = [k for k in ("selector_keep_cur", "selector_attn", "selector_dart_rank", "selector_budgets",
                           "selector_n_cand", "max/selector_probe_count") if k not in logs]
    if missing:
        failures.append(f"log_rows misses {missing}")
    print("=== 6. reset ===")
    worker.reset()
    if worker._selector_row or worker._selector_probe_count or worker._selector_probe_latency \
            or worker._hidden_db is not None:
        failures.append("reset() did not clear the selector probe state")
    release(worker)
    finish(failures)


def finish(failures):
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("selector_probe smoke PASS")


if __name__ == "__main__":
    main()
