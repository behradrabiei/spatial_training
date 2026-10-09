"""Flat-variant selector criteria vs per-token KL of the CURRENT frame (paper/selector_kl).

After a decision, on the prune substrate with nothing pruned (``vlm.selector_probe``), for every
visual token of the frame just decided on:

* the layer-0 criterion of each selector of the flat token-pruning variant
  (paper/token_prune, "Flat variant"), applied ONE-SHOT to the whole unpruned pool -- every visual
  slot in the cache, the current frame included (``kv_prune_candidate_scope='visual'``,
  ``kv_prune_recent_turns=0``):

  - ``attn``      FastV: the layer-0 decision-row attention, mean over heads
                  (``ReindexState.mean_scores`` with ``kv_prune_score_layer=0``); its position in
                  the descending order;
  - ``divprune``  the position in DivPrune's greedy max-min order (``divprune_select`` on the
                  sparse embed DB), run to the largest binding budget; ``-1`` beyond it;
  - ``dart``      whether ``VLMWorker._dart_select`` (layer-0 keys + layer-0 output) keeps the
                  token at each budget; DART's sets are nested across capacity (pivots fixed,
                  quotas grow), and the row records whether they were (``dart_nested``);
  - ``random``    one permutation seeded like the flat random arm (``step_seed``), whose prefixes
                  are that arm's draws at every capacity; the live ``ps.generator`` is untouched.

  Each diversity method also gets a CONTINUOUS per-token criterion, the quantity its rule ranks
  by, defined for every token whatever the budget: ``divprune_nn`` = the token's nearest-neighbour
  cosine distance in the pool (``1 - max_j cos``; DivPrune's max-min objective, its own
  initialisation score) and ``dart_dis`` = ``1 - min_p cos`` to DART's pivots on the layer-0 output
  hidden states (a pivot claims the tokens least similar to it; pivots themselves score 2.0, the
  top of the cosine-distance range, since DART always keeps them).

  Budgets are frames' worth ``k`` (``frame_budget``: ``round(k * mean visual count per turn)``),
  binding only while the cache holds more than ``k`` frames. For every method the token's
  ``*_rank`` is the smallest binding ``k`` at which it is kept (``NEVER`` = -1 otherwise), a scale
  the four methods share.

* the leaky per-token KL of :mod:`longnav.utils.token_info` at layer 0 (``probe_mask_arm``:
  decision-token replay with key columns hidden, fp32 action head). The paper measure is
  ``keep_all[k] = KL(P_only k, every other visual token hidden || P_no visual)`` -- the
  token-level version of the information horizon's keep-one (no visual history; the past
  action text stays); ``vlm.selector_probe_measures`` selects it (``keep_cur`` keeps the
  history intact, ``leave`` is leave-one-out).

Everything is in raster order (``current_frame_tokens``), so a token maps to its image patch.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from longnav.utils import token_info as ti
import torch.nn.functional as F

from longnav.utils.kv_prune import (build_prune_partition, divprune_select, frame_budget,
                                    random_select, step_seed, visual_slot_embeddings)
from longnav.utils.pre_rope import _rotate

BUDGETS = (4, 8, 16, 32)
NEVER = -1  # keep rank of a token no binding budget keeps; also "outside the order" for positions
LIST_KEYS = ("raster", "grid_thw", "budget_frames", "budgets", "attn", "attn_pos", "divprune_pos",
             "random_pos", "divprune_nn", "dart_dis", "attn_rank", "divprune_rank", "dart_rank",
             "random_rank", "base_logp", "text_cur_logp")
SCALAR_KEYS = ("n_cand", "n_frames", "n_cur", "dart_nested", "text_cur_kl", "text_all_kl")


def _sig(v, digits=6):
    return float(f"{float(v):.{digits}g}")


def keep_rank(pos, caps, budgets):
    """Smallest binding budget at which a token in position ``pos`` of a method's order is kept.

    ``pos`` (N,) long, ``-1`` = outside the order; ``caps`` maps each budget to its capacity or
    ``None`` when it does not bind. Returns (N,) long with ``NEVER`` where no binding budget
    keeps the token."""
    pos = torch.as_tensor(pos, dtype=torch.long)
    out = torch.full_like(pos, NEVER)
    for k in sorted(budgets, reverse=True):  # descending: the smallest k writes last
        c = caps.get(k)
        if c is None:
            continue
        out[(pos >= 0) & (pos < int(c))] = int(k)
    return out


def member_rank(members, budgets, n):
    """Same as :func:`keep_rank` from per-budget membership masks ``members[k]`` ((n,) bool)."""
    out = torch.full((n,), NEVER, dtype=torch.long)
    for k in sorted(budgets, reverse=True):
        m = members.get(k)
        if m is None:
            continue
        out[m] = int(k)
    return out


def nn_cosine_distance(rows_cur, rows_pool, self_idx, block=4096):
    """(N,) nearest-neighbour cosine distance of each current-frame row within the pool
    (``1 - max_{j != i} cos``); ``self_idx`` (N,) = each row's index in ``rows_pool``."""
    x = F.normalize(rows_cur.float(), dim=-1)
    best = torch.full((x.shape[0],), -torch.inf, device=x.device)
    ar = torch.arange(x.shape[0], device=x.device)
    for start in range(0, rows_pool.shape[0], block):
        y = F.normalize(rows_pool[start:start + block].float(), dim=-1)
        sim = x @ y.T
        own = (self_idx.to(x.device) - start)
        hit = (own >= 0) & (own < y.shape[0])
        sim[ar[hit], own[hit]] = -torch.inf
        best = torch.maximum(best, sim.amax(dim=1))
    return (1.0 - best).cpu()


def dart_criterion(hidden_cur, hidden_pivots, is_pivot):
    """(N,) DART dissimilarity: ``1 - min_p cos`` to the pivots on the cut layer's hidden states;
    pivots score 2.0 (always kept)."""
    x = F.normalize(hidden_cur.float(), dim=-1)
    pv = F.normalize(hidden_pivots.float(), dim=-1).to(x.device)
    dis = 1.0 - (x @ pv.T).amin(dim=1)
    dis[is_pivot.to(dis.device)] = 2.0
    return dis.cpu()


def _dart_pivots(worker, st, cand, n_pivots):
    """The pool's DART pivots (the top-P layer-0 rotated-key L1 norms), as slot indices."""
    src = 0 if st.pruned_layers is None else max(st.pruned_layers.start - 1, 0)
    keys = worker.past_key_values.layers[src].keys
    cos, sin = st.layer_tables(src)
    rot = _rotate(keys, cos, sin)[0].permute(1, 0, 2).reshape(keys.shape[-2], -1)
    norms = rot.float().abs().sum(dim=-1).cpu()
    n_piv = max(1, min(int(n_pivots), cand.numel()))
    return cand[torch.argsort(norms[cand], descending=True, stable=True)[:n_piv]]


def _positions(order, kv_len):
    """(kv_len,) long: each slot's position in ``order`` (a 1-D tensor of slot indices), -1 if absent."""
    pos = torch.full((kv_len,), NEVER, dtype=torch.long)
    order = torch.as_tensor(order, dtype=torch.long).cpu()
    pos[order] = torch.arange(order.numel(), dtype=torch.long)
    return pos


def selector_ranks(worker, budgets=BUDGETS):
    """The four selectors' layer-0 criteria for the current frame's tokens (see module docstring).

    Reads the captured layer-0 decision row, the sparse embed DB and the DART hidden DB of the
    forward just run; changes nothing on the worker.
    """
    budgets = tuple(sorted({int(k) for k in budgets}))
    ps, st = worker._kv_prune_state, worker._reindex_state
    if st is None or st.cos is None:
        raise RuntimeError("selector_probe needs the prune substrate's rotation tables of the last forward")
    cols, raster, grid = ti.current_frame_tokens(worker)
    kv_len = worker._master_len()
    part = build_prune_partition(ps, prefix_len=worker._prefix_len, context_window=None,
                                 holdback_n=len(worker.prefix_ids), budget=None, recent_turns=0,
                                 candidate_scope="visual")
    if not bool(part.candidates[cols].all()):
        raise RuntimeError("the current frame's visual slots are not all candidates")
    cand = torch.nonzero(part.candidates).flatten()
    n_cand = int(cand.numel())
    caps = {}
    for k in budgets:
        c = frame_budget(ps, 0, k)
        caps[k] = None if c is None else min(int(c), n_cand)
    binding = [k for k in budgets if caps[k] is not None]
    cap_max = max(caps[k] for k in binding) if binding else None

    # FastV: the layer-0 decision row (the worker armed the capture for this forward)
    scores = st.mean_scores().detach().float().cpu()
    if scores.numel() != kv_len:
        raise RuntimeError(f"decision row covers {scores.numel()} slots, master holds {kv_len}")
    if st.score_layers != 1:
        raise RuntimeError(f"{st.score_layers} decision rows folded; the probe reads layer 0 alone")
    attn_pos = _positions(cand[torch.argsort(scores[cand], descending=True, stable=True)], kv_len)

    # random: the flat arm's permutation for this (episode, step)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(step_seed(worker.kv_prune_seed, worker._episode_label, ps.n_turns - 1))
    random_pos = _positions(random_select(cand, n_cand, gen), kv_len)

    # DivPrune: greedy order to the largest binding budget (capacity == n returns the pool
    # unordered, so run one short of it and give the leftover the last position)
    divprune_pos = torch.full((kv_len,), NEVER, dtype=torch.long)
    if cap_max is not None:
        cap_run = min(cap_max, n_cand - 1)
        if cap_run > 0:
            db = visual_slot_embeddings(ps, worker.past_image_embeds)
            db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
            rows = db[db_rank[cand].to(db.device)].to(getattr(worker, "device", db.device))
            order = divprune_select(rows, cand, cap_run)
            del rows
            divprune_pos = _positions(order, kv_len)
        if cap_max == n_cand:
            rest = cand[divprune_pos[cand] < 0]
            divprune_pos[rest] = torch.arange(n_cand - rest.numel(), n_cand, dtype=torch.long)

    # continuous criteria of the two diversity methods, for the current frame's tokens
    db = visual_slot_embeddings(ps, worker.past_image_embeds)
    db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
    dev = getattr(worker, "device", db.device)
    pool_rows = db[db_rank[cand].to(db.device)].to(dev)
    self_idx = torch.searchsorted(cand, cols)  # cols are candidates (checked above), cand is sorted
    divprune_nn = nn_cosine_distance(pool_rows[self_idx.to(dev)], pool_rows, self_idx)
    if worker._hidden_db is None or worker._hidden_db.shape[0] != int(ps.is_visual.sum()):
        raise RuntimeError("DART hidden DB out of step with the visual slots")
    pivots = _dart_pivots(worker, st, cand, worker.kv_prune_dart_pivots)
    hdb = worker._hidden_db
    dart_dis = dart_criterion(hdb[db_rank[cols].to(hdb.device)], hdb[db_rank[pivots].to(hdb.device)],
                              torch.isin(cols, pivots))

    # DART at every binding budget; nesting is recorded, not assumed
    members, nested = {}, True
    prev = None
    for k in binding:
        kept = worker._dart_select(ps, st, cand, caps[k])
        m = torch.zeros(kv_len, dtype=torch.bool)
        m[torch.as_tensor(kept, dtype=torch.long).cpu()] = True
        if prev is not None and bool((prev & ~m).any()):
            nested = False
        members[k] = m
        prev = m

    n_cur = int(cols.numel())
    return {
        "raster": raster.tolist(),
        "grid_thw": [int(x) for x in grid],
        "budget_frames": list(budgets),
        "budgets": [NEVER if caps[k] is None else int(caps[k]) for k in budgets],
        "n_cand": n_cand,
        "n_frames": int(ps.n_turns),
        "n_cur": n_cur,
        "attn": [float(x) for x in scores[cols].tolist()],
        "attn_pos": attn_pos[cols].tolist(),
        "divprune_pos": divprune_pos[cols].tolist(),
        "random_pos": random_pos[cols].tolist(),
        "divprune_nn": [_sig(v) for v in divprune_nn.tolist()],
        "dart_dis": [_sig(v) for v in dart_dis.tolist()],
        "attn_rank": keep_rank(attn_pos[cols], caps, budgets).tolist(),
        "divprune_rank": keep_rank(divprune_pos[cols], caps, budgets).tolist(),
        "random_rank": keep_rank(random_pos[cols], caps, budgets).tolist(),
        "dart_rank": member_rank({k: m[cols] for k, m in members.items()}, budgets, n_cur).tolist(),
        "dart_nested": int(nested),
    }


def selector_probe(worker, budgets=BUDGETS, measures=("keep_cur", "leave"), fp32_head=True):
    """Selector criteria plus the leaky layer-0 per-token KLs, as one flat row of plain lists.

    Cost: the selector work (a few matvecs over the pool) plus ``token_info.n_forwards(1, N,
    measures)`` decision replays.
    """
    t0 = time.perf_counter()
    ranks = selector_ranks(worker, budgets)
    out = ti.probe_mask_arm(worker, layers=[0], measures=tuple(measures), fp32_head=fp32_head)
    if not np.array_equal(out["raster"], np.asarray(ranks["raster"])):
        raise RuntimeError("the selector ranks and the KL probe disagree on the current frame's tokens")
    row = dict(ranks)
    for m in measures:
        row[m] = [_sig(v) for v in out[m][0].tolist()]
    row["text_cur_kl"] = _sig(out["text_cur_kl"][0])
    row["text_all_kl"] = _sig(out["text_all_kl"][0])
    row["base_logp"] = [_sig(v, 8) for v in out["base_logp"].tolist()]
    row["text_cur_logp"] = [_sig(v, 8) for v in out["text_cur_logp"][0].tolist()]
    row["n_probes"] = int(out["n_probes"])
    row["seconds"] = time.perf_counter() - t0
    return row


def log_rows(worker):
    """The step-log entries of the last probe (``sup/selector_*`` in sequence.json)."""
    row = getattr(worker, "_selector_row", {}) or {}
    logs = {}
    for key in LIST_KEYS + tuple(getattr(worker, "selector_probe_measures", ())):
        logs[f"selector_{key}"] = list(row.get(key, []))
    for key in SCALAR_KEYS:
        logs[f"selector_{key}"] = row.get(key, NEVER)
    logs["mean/kv_visual_new"] = row.get("n_cur", 0)
    logs["max/selector_probe_count"] = getattr(worker, "_selector_probe_count", 0)
    logs["max/selector_probe_latency"] = getattr(worker, "_selector_probe_latency", 0.0)
    return logs
