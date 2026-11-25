"""Pruning KL: KL(P full context || P pruned context) per pruning method and frame budget (paper/prune_kl).

After a decision, on the prune substrate with nothing pruned (``vlm.prune_kl_probe``), one SHADOW
pruner per (method, budget k) replays what the flat token-pruning arm of paper/token_prune would
hold at this step had it run along the full-context trajectory:

* a shadow is a bool mask over the master cache (``alive``): the visual slots its method has not
  dropped yet. Text is never a candidate (``kv_prune_candidate_scope='visual'``,
  ``kv_prune_recent_turns=0``), deletion is permanent, and the pool of a step is the shadow's
  survivors plus the frame just decided on -- the live arm's rule, with ``alive`` standing for the
  physical slice (``build_prune_partition(alive=...)``).
* timing as the live arm: the decision at step t saw the survivors of the selections made through
  step t-1 plus ALL of frame t, so the probe at t hides ``is_visual & ~alive`` first and the
  shadow selects on survivors + frame t afterwards (taking effect at t+1).
* the probe is the leaky decision replay of the recent-window measure
  (``DiagnosticsMixin._kl_joint_probe``): the cached decision token is replayed against the full
  cache with the hidden slots at -inf on every layer; reference = the replayed full decision, so a
  shadow that has dropped nothing scores exactly 0 without a replay. Action distribution = the 4
  action ids, bf16 head, T = 1.

Selection rules are the live arm's expressions (``PruningMixin._apply_kv_prune``): ``random`` =
``random_select`` under a private generator seeded ``step_seed(seed, episode label, step)`` (the
live ``ps.generator`` is never touched), ``fastv`` = top-capacity of the layer-0 decision row,
``divprune`` = ``divprune_select`` on the sparse embed DB, ``dart`` = ``VLMWorker._dart_select``
(layer-0 keys + layer-0 output). Capacity = ``frame_budget(ps, 0, k)``.

What a shadow on the full-context trajectory cannot reproduce of the live pruned arm (stated in
paper/prune_kl/README.md): the sparse filter saw the full embed DB, the FastV row is a softmax over
every slot rather than the survivors, and DART's layer-0 outputs attended to the full cache.
"""
from __future__ import annotations

import time

import torch

from longnav.utils.kv_prune import (build_prune_partition, build_selected_keep, divprune_select,
                                    frame_budget, random_select, step_seed, visual_slot_embeddings)

METHODS = ("random", "fastv", "divprune", "dart")
BUDGETS = (4, 8, 16, 32)
NONE = -1  # capacity of a step whose budget does not bind
LIST_KEYS = ("methods", "budgets", "kl", "logp", "hidden", "kept", "cap", "pool", "base_logp",
             "visual_all")


class ShadowPruner:
    """One (method, k) arm of the flat pruning study as a mask over the UNPRUNED master cache."""

    def __init__(self, method, k):
        if method not in METHODS:
            raise ValueError(f"prune_kl method must be one of {METHODS}, got {method!r}")
        self.method, self.k = method, int(k)
        self.alive = torch.zeros(0, dtype=torch.bool)
        self.cap = NONE  # capacity of the last selection (NONE while the budget does not bind)
        self.pool = 0  # candidates the last selection drew from

    def extend(self, kv_len):
        """Register the slots the cache gained since the last step (all alive)."""
        n = self.alive.numel()
        if kv_len < n:
            raise RuntimeError(f"master cache shrank from {n} to {kv_len} slots under a shadow pruner")
        if kv_len > n:
            self.alive = torch.cat([self.alive, torch.ones(kv_len - n, dtype=torch.bool)])

    def hidden(self, ps):
        """Master slots this arm no longer holds: visual slots dropped by earlier selections."""
        return ps.is_visual & ~self.alive

    def select(self, ps, partition_kwargs, select_fn):
        """One pruning step on survivors + the newest frame; returns the number of slots dropped."""
        cap = frame_budget(ps, 0, self.k)
        part = build_prune_partition(ps, budget=None, capacity=cap, alive=self.alive, **partition_kwargs)
        cand = torch.nonzero(part.candidates).flatten()
        self.cap = NONE if cap is None else int(part.capacity)
        self.pool = int(cand.numel())
        if part.capacity == cand.numel():
            return 0
        keep = build_selected_keep(part, select_fn(self.method, cand, part.capacity))
        if bool((keep & ~self.alive).any()):
            raise RuntimeError("a shadow selection revived a dropped slot")
        dropped = int((self.alive & ~keep).sum())
        self.alive = keep
        return dropped


def make_shadows(methods=METHODS, budgets=BUDGETS):
    """Shadows in (method, k) order, methods as given and budgets ascending."""
    return [ShadowPruner(m, k) for m in methods for k in sorted({int(b) for b in budgets})]


def partition_kwargs(worker):
    """The flat arm's partition: full context, every visual slot a candidate, text protected."""
    return dict(prefix_len=worker._prefix_len, context_window=None, holdback_n=len(worker.prefix_ids),
                recent_turns=0, candidate_scope="visual", frame_cut=False)


def make_select_fn(worker):
    """``select_fn(method, candidate_idx, capacity)`` with the live flat arm's selection rules,
    reading the forward just run (decision row, embed DB, DART hidden DB) and mutating nothing."""
    ps, st = worker._kv_prune_state, worker._reindex_state
    cache = {}

    def scores():
        if "scores" not in cache:
            row = st.mean_scores().detach().float().cpu()
            if row.numel() != ps.turn_id.numel():
                raise RuntimeError(f"decision row covers {row.numel()} slots, metadata {ps.turn_id.numel()}")
            if st.score_layers != 1:
                raise RuntimeError(f"{st.score_layers} decision rows folded; FastV reads layer 0 alone")
            cache["scores"] = row
        return cache["scores"]

    def select(method, cand, capacity):
        if method == "random":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(step_seed(worker.kv_prune_seed, worker._episode_label, ps.n_turns - 1))
            return random_select(cand, capacity, gen)
        if method == "fastv":
            return cand[torch.argsort(scores()[cand], descending=True, stable=True)[:capacity]]
        if method == "divprune":
            db = visual_slot_embeddings(ps, worker.past_image_embeds)
            db_rank = torch.cumsum(ps.is_visual.long(), 0) - 1
            rows = db[db_rank[cand].to(db.device)].to(getattr(worker, "device", db.device))
            return divprune_select(rows, cand, capacity)
        if method == "dart":
            return worker._dart_select(ps, st, cand, capacity)
        raise AssertionError(f"unhandled prune_kl method {method!r}")

    return select


def prune_kl_probe(worker):
    """Probe every shadow on the decision just made, then advance every shadow by one selection.

    Returns one flat row of plain lists (``LIST_KEYS`` plus counters). Cost: 2 decision replays
    (the full reference and the no-visual decision) + one per shadow that has dropped something,
    plus the selectors.
    """
    t0 = time.perf_counter()
    ps, st = worker._kv_prune_state, worker._reindex_state
    kv_len = worker._master_len()
    if ps.n_budget_dropped or worker._dropped or ps.turn_id.numel() != kv_len:
        raise RuntimeError("prune_kl_probe needs the unpruned cache: keep kv_budget unreachable and "
                           "set no window or frame budget")
    if st is None or st.cos is None:
        raise RuntimeError("prune_kl_probe needs the prune substrate's rotation tables of the last forward")
    shadows = worker._prune_kl_shadows
    cur = ps.n_turns - 1
    for s in shadows:
        s.extend(kv_len)
    replays = worker._prune_replay_count

    base_bias = worker._replay_base_bias(torch.zeros(kv_len, dtype=torch.bool))
    base_logp = worker._replay_action_logp(base_bias)
    visual_all = worker._kl_joint_probe(ps.is_visual, base_bias, base_logp)
    history = ps.is_visual & (ps.turn_id < cur)
    kls, logps, hidden, kept = [], [], [], []
    for s in shadows:
        mask = s.hidden(ps)
        if bool((mask & ~history).any()):
            raise RuntimeError("a shadow pruner hides a slot of the frame being decided on")
        kl, logp = worker._kl_joint_probe(mask, base_bias, base_logp)
        kls.append(kl)
        logps.append(logp)
        hidden.append(int(mask.sum()))
        kept.append(int((history & s.alive).sum()))
    n_probes = worker._prune_replay_count - replays

    # The selection that follows this decision (the live arm prunes after the action is read).
    select_fn = make_select_fn(worker)
    kwargs = partition_kwargs(worker)
    for s in shadows:
        s.select(ps, kwargs, select_fn)

    return {
        "methods": [s.method for s in shadows],
        "budgets": [s.k for s in shadows],
        "kl": kls,
        "logp": logps,
        "hidden": hidden,  # visual slots hidden in this probe
        "kept": kept,  # history visual slots visible in this probe (the frame decided on is extra)
        "cap": [s.cap for s in shadows],  # capacity of the selection made after this decision
        "pool": [s.pool for s in shadows],
        "base_logp": [round(float(v), 6) for v in base_logp.detach().cpu().float().tolist()],
        "visual_all": visual_all,  # [kl, logp] with every visual slot hidden (the no-visual decision)
        "n_cur": int((ps.is_visual & (ps.turn_id == cur)).sum()),
        "n_visual": int(ps.is_visual.sum()),
        "n_probes": int(n_probes),
        "seconds": time.perf_counter() - t0,
    }


def log_rows(worker):
    """The step-log entries of the last probe (``sup/prune_kl_*`` in sequence.json)."""
    row = getattr(worker, "_prune_kl_row", {}) or {}
    logs = {f"prune_kl_{key}": list(row.get(key, [])) for key in LIST_KEYS}
    logs["mean/kv_visual_new"] = row.get("n_cur", 0)
    logs["mean/kv_visual_total"] = row.get("n_visual", 0)
    logs["max/prune_kl_probe_count"] = getattr(worker, "_prune_kl_count", 0)
    logs["max/prune_kl_latency"] = getattr(worker, "_prune_kl_latency", 0.0)
    return logs
