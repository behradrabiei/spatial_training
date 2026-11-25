"""Run ``lightnav.cli.eval_habitat`` with the keep-one-in KL probe of paper/frame_influence.

The leave-one-in measure (paper/frame_influence/README.md, "Leave-one-in variant") scores every
history frame ``f`` of decision ``t`` by
``KeepOne_t(f) = KL(P(action | only f's visual tokens) || P(action | no visual tokens))``, T = 1,
the expectation under the only-``f`` distribution; age 0 is the current frame alone against the
same no-visual reference. LongNav-R1 replays the decision token against its KV cache with
columns masked, Uni-NaVid re-scores its prompt with every other visual column closed on every
layer. LightNav-0 re-prefills its whole prompt inside vLLM every step, so this wrapper, without
touching the upstream clone:

* navigates exactly as ``lightnav_ref`` (eager vLLM, ``eval_blind``'s exact masked SDPA kernel on
  every layer, nothing hidden for the real decision) and, after each decision, re-runs the same
  prompt with ``<|video_pad|>`` columns hidden on every layer: once with every visual column
  hidden (the reference, ``visual_all_mask``), once with only the current tubelet open (age 0),
  once per history **tubelet** -- LightNav's 2-frame visual unit -- of age ``t - f1 <= max_age``
  (frame = the newer frame ``f1``), plus one replay with nothing hidden (the kernel's noise floor,
  ``kl_live_vs_replay``). Timestamps, vision markers and the task text stay visible, so the
  reference is the existing no-visual semantics;
* reads the **action distribution** at the third generated position: LightNav emits
  ``<apos_K><opos_K><act_l0_a><act_l1_b><act_l2_c>``, the pointing tokens first, then the RVQ
  codes whose coarse level ``act_l0`` (256 codes) carries the motion. Every probe free-runs its
  own two pointing tokens under its mask (3-token greedy generate; nothing from the full-context
  decision enters a probe, as for the other two models) and the logits of the third token are
  restricted to the 256 ``<act_l0_*>`` ids and log-softmaxed (fp32, T = 1; ``p_mass`` = the
  full-vocabulary mass those ids hold). The logits come from a hook on the in-process model's
  ``compute_logits`` (one request per generate, so call ``i`` is generated position ``i``);
* also folds the 256 codes onto the six executed-action classes the velocity controller
  produces (stop, forward, forward-left/right, rotate-left/right: decode the coarse codeword alone,
  first non-zero waypoint, Habitat's velocity_control clipping) and reports the KL over those bins
  as ``kl_bin`` next to the 256-way ``kl``; the per-probe 256 log-probs are stored (float16,
  base64) so anything else can be derived offline;
* resets vLLM's prefix cache before every generate and asserts that each prefill recomputes the
  whole prompt, so K/V computed under one mask are never reused under another.

Per inference one record goes to ``<output_dir>/influence/episode_NNN.jsonl`` (keys follow
Uni-NaVid's ``influence/<label>.jsonl``: ``step, full, replay, visual_all_mask, keep_one_age0,
keep_one[], n_probes, probe_seconds``); ``convert_results.py`` renames them to
``influence/<label>.jsonl`` and ``meta.json`` carries the bin table and token ids.

``--measure recent_k`` runs the recent-window variant instead (README, "Recent-window variant"):
``Recent_t(K) = KL(P(action | full context) || P(action | only the K most recent frames))``, K
counting frames with the current one, the expectation under the full-context decision. LightNav
has two ways to drop a frame, the two arms of its context-window ablation, and both are probed
after every decision, on the same decision:

* **old blocks kept** (``recent_keep``): the full prompt with the ``<|video_pad|>`` columns of
  every older tubelet hidden on every layer; their ``<t s>`` timestamps and vision markers stay
  and positions are unchanged (the pruning study's ``recency`` layout). One probe per prefix of
  the newest tubelets whose frames are exactly the newest K (K = 2, 4, ... in steady state: the
  tubelet is the unit), plus K = 1 (the current tubelet rebuilt as ``(t, t)``, upstream's own
  step-0 layout, every history block hidden) and K = 0 (``visual_all_mask``);
* **window** (``recent_window``): the prompt rebuilt as ``eval_ctx.py`` builds it for
  ``CONTEXT_FRAMES=K`` -- the layout of a fresh episode that started K steps ago, old blocks
  gone, nothing hidden -- for every K = 1..``max_k``.

Only K <= t is probed (at least one frame is lost). The rebuilt prompts go through the engine's
own sample builder and ViT path with a per-step copy of the episode's tubelet cache, so the
navigation cache and the decision are untouched.

``--measure blind_layer`` runs the depth analogue (paper/info_horizon/README.md, "Blind-from-layer
KL"): ``Blind_t(L) = KL(P(action | full context) || P(action | every visual token hidden from
decoder layer L on))``, the expectation under the full-context decision. One probe per ``L`` in
``--layers`` (default: every second layer) after every decision: the same prompt with every
``<|video_pad|>`` column closed for every query on layers ``>= L`` and open below, i.e. exactly
the forward ``eval_blind.py --blind-layer L`` runs, on the unblinded trajectory. ``L = 0`` is the
no-visual reference of the other two measures (``visual_all_mask``); the replay with nothing
hidden is the noise floor and the ``L = n_layers`` point. Each record keeps every probe's 256
log-probs, its six-bin fold and its free-run pointing tokens.

``--measure prune_kl`` runs the pruning analogue (paper/prune_kl/README.md):
``Prune_t(M, k) = KL(P(action | full context) || P(action | the visual tokens pruning method M
holds at k frames' worth))``, the expectation under the full-context decision. One SHADOW pruner per
``--prune-kl-arms method:k`` (default random / fastv / divprune / dart x k = 4, 8, 16, 32) is
advanced along the full-context trajectory with ``eval_prune.py``'s own rule -- ledger of
permanently dropped ``(f0, f1, pool, cell)`` tokens, pool = history tokens not in the ledger, the
current tubelet exempt, budget ``k x 14`` tokens, selection BEFORE the decision of the same step,
FastV / DART scored in a one-token pre-pass with that shadow's ledger hidden -- and after every
decision the prompt is re-run with each shadow's hidden set closed on every layer. The tubelet
layout depends on the step alone, so a shadow's ledger is well defined on a trajectory its own
masks never steered; the random arm's masks are those of the live ``lightnav_random_b<k>`` run
when ``--prune-kl-seed-episodes`` gives each episode its index in that run. ``recency:k`` arms
are accepted as a check (they reproduce ``recent_keep`` at ``K = k + 2``).

Runs in the LightNav client runtime (``baselines/.envs/lightnav``) with
``PYTHONPATH=baselines/LightNav-0/src``; ``baselines/run_lightnav.sh`` uses it when
``INFLUENCE=1`` (``INFLUENCE_MEASURE=recent_k MAX_K=32`` for the recent-window variant):

    python baselines/lightnav_hm3d/eval_influence.py --max-age 33 <eval_habitat args...>
    python baselines/lightnav_hm3d/eval_influence.py --measure recent_k --max-k 32 <eval_habitat args...>
    python baselines/lightnav_hm3d/eval_influence.py --measure blind_layer --layers 0 2 4 <eval_habitat args...>
    python baselines/lightnav_hm3d/eval_influence.py --measure prune_kl <eval_habitat args...>
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from baselines.lightnav_hm3d.eval_blind import (
        _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )
    from baselines.lightnav_hm3d.eval_ctx import windowed_segments
    from baselines.lightnav_hm3d.eval_prune import (
        METHODS as PRUNE_METHODS, TOKENS_PER_FRAME, PruneState, Tubelet, apply_drop, decision_row, kept_per_tubelet,
        output_dir_from, pack_bits, plan_step, select_keep, single_current, split_history, tubelet_layout,
    )
except ImportError:  # run as a script: sys.path[0] is this directory
    from eval_blind import (  # type: ignore
        _LAYER_RE, attention_mask, gather_kv, layer_index, masked_attention, visual_positions,
    )
    from eval_ctx import windowed_segments  # type: ignore
    from eval_prune import (  # type: ignore
        METHODS as PRUNE_METHODS, TOKENS_PER_FRAME, PruneState, Tubelet, apply_drop, decision_row, kept_per_tubelet,
        output_dir_from, pack_bits, plan_step, select_keep, single_current, split_history, tubelet_layout,
    )

N_CODES = 256  # <act_l0_*> coarse RVQ codes
N_GENERATE = 6  # <apos><opos><act_l0><act_l1><act_l2><|im_end|>
ACTION_POS = 2  # generated position of <act_l0_*>
PROBE_TOKENS = ACTION_POS + 1
MAX_AGE = 33  # current tubelet + the whole fast tier (ages 2-33 at pool 2)
MEASURES = ("keep_one", "recent_k", "blind_layer", "prune_kl")
PRUNE_KL_METHODS = ("random", "fastv", "divprune", "dart")  # prune_kl: the flat study's selectors
PRUNE_KL_BUDGETS = (4, 8, 16, 32)  # frames' worth
PRUNE_KL_ARMS = tuple((m, k) for m in PRUNE_KL_METHODS for k in PRUNE_KL_BUDGETS)
BLIND_LAYERS = tuple(range(0, 36, 2))  # blind_layer: default probe layers (every second of the 36)
MAX_K = 32  # recent_k: largest number of most recent frames kept (within the dense tiers)
BINS = ("stop", "fwd", "fwd_L", "fwd_R", "rot_L", "rot_R")
# Habitat velocity_control of the ObjectNav server (habitat_server/lightnav_habitat/objectnav.py).
DT, LIN_RANGE, ANG_RANGE = 1.0, (0.0, 0.25), (-30.0, 30.0)
MIN_LIN, MIN_ANG = 0.025, 1.0  # below both: Habitat treats the command as a stop
FWD_STRAIGHT_DEG = 3.0  # |yaw rate| below this is "forward"; above it an arc


# --- pure helpers -------------------------------------------------------------------------
def velocity_bin(lin_raw: float, ang_raw: float) -> str:
    """Executed-action class of a normalised Habitat velocity command (raw in [-1, 1])."""
    v = LIN_RANGE[0] + (lin_raw + 1.0) / 2.0 * (LIN_RANGE[1] - LIN_RANGE[0])
    w = ANG_RANGE[0] + (ang_raw + 1.0) / 2.0 * (ANG_RANGE[1] - ANG_RANGE[0])
    if v < MIN_LIN and abs(w) < MIN_ANG:
        return "stop"
    if v < MIN_LIN:
        return "rot_L" if w > 0 else "rot_R"
    if abs(w) < FWD_STRAIGHT_DEG:
        return "fwd"
    return "fwd_L" if w > 0 else "fwd_R"


def waypoints_bin(waypoints: np.ndarray) -> str:
    """Bin of what the policy executes from decoded ``(H, 3)`` waypoints (LightNav's own mapping)."""
    from lightnav.habitat.policy import select_action_waypoint
    from lightnav.velocity import first_waypoint_to_velocity_cmd

    if np.all(np.abs(waypoints) <= 1e-6):
        return "stop"
    _, wp = select_action_waypoint(np.asarray(waypoints, dtype=np.float32))
    args = first_waypoint_to_velocity_cmd(wp, DT, LIN_RANGE, ANG_RANGE)["action_args"]
    return velocity_bin(float(args["linear_velocity"]), float(args["angular_velocity"]))


def code_bins(bundle) -> List[str]:
    """Bin of every coarse code decoded on its own (zero residual); the stop code is ``stop``."""
    from lightnav.traj_vocab import compose_to_abs

    out = []
    for k in range(N_CODES):
        if bundle.stop_l0 is not None and k == bundle.stop_l0:
            out.append("stop")
            continue
        feat = (bundle.codebooks[0][k] / bundle.jacobian_weights).reshape(bundle.horizon, 3)
        wp = compose_to_abs(feat[None])[0] if bundle.representation == "se2_diff" else feat
        out.append(waypoints_bin(wp))
    return out


def bin_matrix(bins: Sequence[str]) -> torch.Tensor:
    """``[N_CODES, len(BINS)]`` 0/1 matrix folding code probabilities onto bins."""
    m = torch.zeros(len(bins), len(BINS))
    for i, b in enumerate(bins):
        m[i, BINS.index(b)] = 1.0
    return m


def action_logp(logits: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """(log-probs over ``ids`` renormalised, fp32, T = 1; full-vocabulary mass of ``ids``)."""
    full = torch.log_softmax(logits.detach().float(), dim=-1)
    sel = logits.detach().float()[ids]
    return torch.log_softmax(sel, dim=-1), float(full[ids].exp().sum())


def kl(p_logp: Optional[torch.Tensor], q_logp: Optional[torch.Tensor]) -> Optional[float]:
    """KL(p || q) in nats, clamped at 0; None when either side is missing."""
    if p_logp is None or q_logp is None:
        return None
    return max(float(torch.sum(p_logp.exp() * (p_logp - q_logp))), 0.0)


def entropy(logp: torch.Tensor) -> float:
    return max(float(-torch.sum(logp.exp() * logp)), 0.0)


def fold_logp(logp: torch.Tensor, fold: torch.Tensor) -> torch.Tensor:
    """Log-probs over the bins: sum of the code probabilities in each bin."""
    return torch.log((logp.exp() @ fold).clamp_min(1e-30))


def pack_f16(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().cpu().to(torch.float16).numpy().tobytes()).decode("ascii")


def unpack_f16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def only_mask(visual: torch.Tensor, pad_pos: Sequence[int], tubelets: Sequence[Tubelet]) -> torch.Tensor:
    """Hide every visual column except the cells of ``tubelets``."""
    hide = visual.clone()
    for tub in tubelets:
        for c in range(tub.n_tokens):
            hide[pad_pos[tub.start + c]] = False
    return hide


def recent_keep_sets(hist: Sequence[Tubelet], cur: Sequence[Tubelet], t_now: int, max_k: int) -> List[Tuple[int, List[Tubelet]]]:
    """``(K, tubelets left open)`` for the old-blocks-kept recent-window probes of decision
    ``t_now``: the current tubelet(s) plus the newest history tubelets, one entry per prefix whose
    frames are exactly the newest K frames ``t_now - K + 1 .. t_now``, for ``K <= max_k`` and
    ``K <= t_now`` (at least one frame is hidden). The tubelet is the unit, so K runs over
    2, 4, ... in steady state (odd K only while a dense tier holds an odd frame count)."""
    kept = list(cur)
    frames = {f for tub in kept for f in (tub.f0, tub.f1)}
    out: List[Tuple[int, List[Tubelet]]] = []
    for tub in [None] + sorted(hist, key=lambda x: (x.f1, x.f0), reverse=True):
        if tub is not None:
            kept.append(tub)
            frames |= {tub.f0, tub.f1}
        k = len(frames)
        if k > max_k or k > t_now:
            break
        if frames == set(range(t_now - k + 1, t_now + 1)):
            out.append((k, list(kept)))
    return out


def parse_arms(specs: Optional[Sequence[str]]) -> List[Tuple[str, int]]:
    """``method:k`` strings -> ``[(method, k)]`` (prune_kl); None = the paper's 4 x 4 grid."""
    if not specs:
        return list(PRUNE_KL_ARMS)
    arms: List[Tuple[str, int]] = []
    for spec in specs:
        method, _, k = str(spec).partition(":")
        if method not in PRUNE_METHODS or not k.isdigit():
            raise ValueError(f"prune_kl arm must be method:k with method in {PRUNE_METHODS}, got {spec!r}")
        arms.append((method, int(k)))
    if len(set(arms)) != len(arms):
        raise ValueError(f"repeated prune_kl arm in {list(specs)}")
    return arms


def shadow_step(shadow: PruneState, items, tubelets: Sequence[Tubelet], t_now: int, pads: torch.Tensor,
                pad_pos: Sequence[int], prompt_len: int, run_prepass: Callable[[PruneState, torch.Tensor], int]):
    """One step of a shadow pruner on the prompt of decision ``t_now``: the selection
    ``eval_prune``'s wrapper makes before its generate (``plan_step`` -> ``select_keep`` ->
    ``apply_drop``, same order, same budget test), against the shadow's own ledger. Returns
    ``(plan, n_pool, n_dropped, prepass_forwards)``; ``plan.hide`` is the hidden set that pruned
    arm's decision at this step runs under. ``run_prepass(shadow, hide)`` runs the scoring
    pre-pass of fastv / dart with ``hide`` closed and the captures written into ``shadow``."""
    shadow.current_abs = int(t_now)
    shadow.prompt_len = int(prompt_len)
    plan = plan_step(tubelets, t_now, shadow.ledger, pad_pos, prompt_len)
    n_pool, n_dropped, forwards = len(plan.cand_ranks), 0, 0
    if n_pool > shadow.budget:
        if shadow.budget == 0:
            keep = torch.empty(0, dtype=torch.long)
        else:
            keep, forwards = select_keep(shadow, items, plan.cand_ranks, pads,
                                         lambda: run_prepass(shadow, plan.hide), plan.cand_keys)
        if int(keep.numel()) != shadow.budget:
            raise RuntimeError(f"{shadow.method} kept {int(keep.numel())} of {n_pool} candidates, budget {shadow.budget}")
        n_dropped = apply_drop(plan, keep, shadow.ledger, pad_pos)
    return plan, n_pool, n_dropped, forwards


# --- shared state -------------------------------------------------------------------------
class InfluenceState:
    def __init__(self, max_age: int = MAX_AGE, debug_steps: int = 0, measure: str = "keep_one", max_k: int = MAX_K,
                 layers: Sequence[int] = BLIND_LAYERS, arms: Sequence[Tuple[str, int]] = PRUNE_KL_ARMS,
                 seed_episodes: Optional[Sequence[int]] = None, tokens_per_frame: int = TOKENS_PER_FRAME):
        if max_age < 1:
            raise ValueError(f"max age must be >= 1, got {max_age}")
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
        if max_k < 1:
            raise ValueError(f"max K must be >= 1, got {max_k}")
        self.max_age = int(max_age)
        self.measure = measure
        self.max_k = int(max_k)
        # blind_layer: the probe layers (first decoder layer without visual tokens), ascending
        self.layers = tuple(sorted({int(l) for l in layers}))
        if measure == "blind_layer" and (not self.layers or self.layers[0] < 0):
            raise ValueError(f"blind_layer needs non-negative probe layers, got {tuple(layers)}")
        # prune_kl: one shadow pruner (eval_prune.PruneState: budget, seed, ledger, pre-pass captures)
        # per (method, k); seed_episodes[i] = the episode counter the random selector seeds on for
        # the i-th episode of this run (its index in the 100-episode pruning runs), default i
        self.arms = [(str(m), int(k)) for m, k in arms] if measure == "prune_kl" else []
        if measure == "prune_kl" and (not self.arms or len(set(self.arms)) != len(self.arms)):
            raise ValueError(f"prune_kl needs distinct (method, k) arms, got {list(arms)}")
        self.shadows = [PruneState(m, k, tokens_per_frame) for m, k in self.arms]
        self.seed_episodes = None if seed_episodes is None else [int(e) for e in seed_episodes]
        if self.seed_episodes is not None and (measure != "prune_kl" or min(self.seed_episodes, default=0) < 0):
            raise ValueError("seed episodes are non-negative episode counters for the prune_kl measure")
        self.capture_into: Optional[PruneState] = None  # the shadow whose scoring pre-pass is in flight
        # recent_k: the arguments of the decision in flight (to rebuild its prompt under another
        # frame layout) and the layout the sampler hook applies to the next sample it builds
        self.call: Optional[Tuple[Any, str, Any]] = None
        self.layout: Optional[Tuple[str, int]] = None  # ("window", K) | ("single", 1); None = the model's own
        self.probe_segments: Optional[List[Tuple[List[int], int]]] = None
        self.debug_steps = debug_steps
        self.n_layers: Optional[int] = None
        self.strict = False
        # token ids, known once the engine is built
        self.act_ids: Optional[torch.Tensor] = None  # [256] <act_l0_*>
        self.apos0 = self.opos0 = 0  # id of <apos_0> / <opos_0>
        self.eos_id: Optional[int] = None
        self.bins: Optional[List[str]] = None
        self.fold: Optional[torch.Tensor] = None
        self.bundle = None  # RVQ bundle (decodes the executed command)
        # per generate call
        self.hide: Optional[torch.Tensor] = None  # bool [prompt_len]; None = no generate in flight
        self._hide_dev: Optional[torch.Tensor] = None
        self.min_layer = 0  # the hidden set applies on decoder layers >= this (0 = every layer)
        self.calls_hidden = 0  # layer calls of this generate that applied the hidden set
        self.prompt_len = 0
        self.calls = 0
        self.logits: List[torch.Tensor] = []  # one [V] fp32 row per compute_logits call
        self.generates = 0
        # per episode
        self.episode = -1
        self.segments: Optional[List[Tuple[List[int], int]]] = None
        self.current_abs: Optional[int] = None
        self.log_dir: Optional[Path] = None
        self._log = None

    def hide_on(self, device: torch.device) -> torch.Tensor:
        if self._hide_dev is None or self._hide_dev.device != device:
            self._hide_dev = self.hide.to(device)
        return self._hide_dev

    def set_hide(self, hide: Optional[torch.Tensor], min_layer: int = 0) -> None:
        self.hide = hide
        self._hide_dev = None
        self.min_layer = int(min_layer)

    def seed_episode(self) -> int:
        """The episode counter the shadows' random selector seeds on for the current episode."""
        if self.seed_episodes is None:
            return self.episode
        if not 0 <= self.episode < len(self.seed_episodes):
            raise RuntimeError(f"episode {self.episode} has no entry in the {len(self.seed_episodes)} seed episodes")
        return self.seed_episodes[self.episode]

    def reset_episode(self) -> None:
        self.episode += 1
        self.segments = None
        self.current_abs = None
        for shadow in self.shadows:  # not PruneState.reset_episode: that advances its own counter
            shadow.ledger = set()
            shadow.episode = self.seed_episode()
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_record(self, rec: Dict[str, Any]) -> None:
        if self.log_dir is None:
            return
        if self._log is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log = (self.log_dir / f"episode_{self.episode:03d}.jsonl").open("a")
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()

    def describe(self) -> Dict[str, Any]:
        if self.measure == "prune_kl":
            return {"measure": "prune_kl", "arms": [[m, k] for m, k in self.arms],
                    "tokens_per_frame": self.shadows[0].tokens_per_frame, "budgets": [s.budget for s in self.shadows],
                    "seed": self.shadows[0].seed, "pivots": self.shadows[0].n_pivots, "seed_episodes": self.seed_episodes,
                    "prefix": "free_run (each probe emits its own <apos><opos>; logits read at generated position 2)",
                    "prune_kl_direction": "KL(full context || visual history pruned by the arm's shadow)", "temperature": 1.0,
                    "action_token": "<act_l0_*> (256 coarse RVQ codes, renormalised)",
                    "act_l0_ids": [int(self.act_ids[0]), int(self.act_ids[-1])] if self.act_ids is not None else None,
                    "apos0_id": self.apos0, "opos0_id": self.opos0, "bins": list(BINS), "code_bins": self.bins,
                    "bin_rule": {"dt": DT, "lin_vel_range": list(LIN_RANGE), "ang_vel_range": list(ANG_RANGE),
                                 "min_abs_lin_speed": MIN_LIN, "min_abs_ang_speed": MIN_ANG, "fwd_straight_deg": FWD_STRAIGHT_DEG,
                                 "decode": "coarse codeword alone (zero residual), first non-zero waypoint"},
                    "shadow": "eval_prune.py's rule on the full-context trajectory: ledger of permanently dropped "
                              "(f0, f1, pool, cell) tokens, pool = history tokens not in the ledger, current tubelet exempt, "
                              "selection before the decision of the same step, budget = k x tokens_per_frame; fastv / dart "
                              "scored in a one-token pre-pass with the shadow's ledger hidden",
                    "hiding": "the shadow's dropped <|video_pad|> columns closed for every query on every layer, prefill and "
                              "decode; timestamps, vision markers and text stay; positions unchanged",
                    "probes_per_step": "replay (nothing hidden) + no-visual + one per arm whose shadow hides something",
                    "bits": "keep mask over the history tokens in prompt order (numpy.packbits, base64), as eval_prune's log",
                    "flip": "code = the probe's argmax <act_l0> code differs from the decision's; bin = its executed-action "
                            "class (bin of the argmax code) differs from the decision's",
                    "prefix_cache": "reset before every generate; each prefill asserted to cover the whole prompt"}
        if self.measure == "blind_layer":
            return {"measure": "blind_layer", "layers": list(self.layers),
                    "prefix": "free_run (each probe emits its own <apos><opos>; logits read at generated position 2)",
                    "blind_kl_direction": "KL(full context || every visual token hidden from layer L on)", "temperature": 1.0,
                    "action_token": "<act_l0_*> (256 coarse RVQ codes, renormalised)",
                    "act_l0_ids": [int(self.act_ids[0]), int(self.act_ids[-1])] if self.act_ids is not None else None,
                    "apos0_id": self.apos0, "opos0_id": self.opos0, "bins": list(BINS), "code_bins": self.bins,
                    "bin_rule": {"dt": DT, "lin_vel_range": list(LIN_RANGE), "ang_vel_range": list(ANG_RANGE),
                                 "min_abs_lin_speed": MIN_LIN, "min_abs_ang_speed": MIN_ANG, "fwd_straight_deg": FWD_STRAIGHT_DEG,
                                 "decode": "coarse codeword alone (zero residual), first non-zero waypoint"},
                    "hiding": "every <|video_pad|> column (history and current tubelets) closed for every query on decoder "
                              "layers >= L, prefill and decode, and open on the layers below; timestamps, vision markers and "
                              "text stay; positions unchanged (= eval_blind.py --blind-layer L on the unblinded trajectory)",
                    "probes_per_step": "replay (nothing hidden; the L = n_layers point and the noise floor) + one per layer",
                    "flip": "code = the probe's argmax <act_l0> code differs from the decision's; bin = its executed-action "
                            "class (bin of the argmax code) differs from the decision's",
                    "prefix_cache": "reset before every generate; each prefill asserted to cover the whole prompt"}
        if self.measure == "recent_k":
            return {"measure": "recent_k", "max_k": self.max_k, "frame_unit": "frame; K = most recent frames kept, current included",
                    "prefix": "free_run (each probe emits its own <apos><opos>; logits read at generated position 2)",
                    "recent_kl_direction": "KL(full context || K most recent frames)", "temperature": 1.0,
                    "action_token": "<act_l0_*> (256 coarse RVQ codes, renormalised)",
                    "act_l0_ids": [int(self.act_ids[0]), int(self.act_ids[-1])] if self.act_ids is not None else None,
                    "apos0_id": self.apos0, "opos0_id": self.opos0, "bins": list(BINS), "code_bins": self.bins,
                    "bin_rule": {"dt": DT, "lin_vel_range": list(LIN_RANGE), "ang_vel_range": list(ANG_RANGE),
                                 "min_abs_lin_speed": MIN_LIN, "min_abs_ang_speed": MIN_ANG, "fwd_straight_deg": FWD_STRAIGHT_DEG,
                                 "decode": "coarse codeword alone (zero residual), first non-zero waypoint"},
                    "recent_keep": "old blocks kept: full prompt, every <|video_pad|> column of the tubelets older than the "
                                   "newest K frames closed for every query on every layer (timestamps, vision markers, text and "
                                   "positions stay); K = tubelet prefixes (even in steady state), K = 1 = current tubelet rebuilt "
                                   "as (t, t) with every history block hidden, K = 0 = visual_all_mask",
                    "recent_window": "prompt rebuilt as eval_ctx.py's CONTEXT_FRAMES=K (fresh K-frame episode layout shifted to "
                                     "the newest K frames, old blocks gone), nothing hidden, K = 1..max_k",
                    "probed": "K <= t only (at least one frame lost)",
                    "probes_per_step": "replay (nothing hidden) + no-visual + keep prefixes + keep K=1 + one window per K",
                    "vit": "rebuilt prompts use a per-step copy of the episode's tubelet cache; the navigation cache is not written",
                    "prefix_cache": "reset before every generate; each prefill asserted to cover the whole prompt"}
        return {"measure": "keep_one_mask", "max_age": self.max_age, "frame_unit": "tubelet (2 frames); frame = f1, age = t - f1",
                "prefix": "free_run (each probe emits its own <apos><opos>; logits read at generated position 2)",
                "keep_one_kl_direction": "KL(only tubelet || no visual)", "temperature": 1.0,
                "action_token": "<act_l0_*> (256 coarse RVQ codes, renormalised)",
                "act_l0_ids": [int(self.act_ids[0]), int(self.act_ids[-1])] if self.act_ids is not None else None,
                "apos0_id": self.apos0, "opos0_id": self.opos0, "bins": list(BINS), "code_bins": self.bins,
                "bin_rule": {"dt": DT, "lin_vel_range": list(LIN_RANGE), "ang_vel_range": list(ANG_RANGE),
                             "min_abs_lin_speed": MIN_LIN, "min_abs_ang_speed": MIN_ANG, "fwd_straight_deg": FWD_STRAIGHT_DEG,
                             "decode": "coarse codeword alone (zero residual), first non-zero waypoint"},
                "hiding": "every <|video_pad|> column but the probed tubelet's closed for every query on every layer; "
                          "timestamps, vision markers and text stay; positions unchanged",
                "probes_per_step": "replay (nothing hidden) + no-visual + only-current + one per history tubelet of age <= max_age",
                "prefix_cache": "reset before every generate; each prefill asserted to cover the whole prompt"}


# --- vLLM hooks ---------------------------------------------------------------------------
def make_influence_forward(state: InfluenceState, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """``FlashAttentionImpl.forward`` replacement: ``eval_blind``'s masked kernel with the hidden
    set ``state.hide`` on every layer ``>= state.min_layer`` (0 for every measure but
    ``blind_layer``; the layers below get the plain causal mask through the same kernel); a
    prefill must cover the whole prompt (no cached prefix)."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        attn_type = getattr(self, "attn_type", None)
        is_decoder = attn_type is None or str(getattr(attn_type, "name", attn_type)).split(".")[-1] == "DECODER"
        if attn_metadata is None or state.hide is None or not is_decoder:
            if state.strict and attn_metadata is not None and is_decoder:
                raise RuntimeError("decoder attention forward outside a hooked generate call")
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output=output,
                            output_scale=output_scale, output_block_scale=output_block_scale)
        if output is None:
            raise RuntimeError("vLLM must pass an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by the masked kernel")
        if str(getattr(self, "kv_cache_dtype", "auto")).startswith("fp8"):
            raise NotImplementedError("fp8 KV cache is not supported by the masked kernel")
        if tuple(getattr(self, "sliding_window", (-1, -1))) != (-1, -1) or getattr(self, "alibi_slopes", None) is not None \
                or getattr(self, "logits_soft_cap", None):
            raise NotImplementedError("sliding window / alibi / soft cap are not supported by the masked kernel")
        if not getattr(attn_metadata, "causal", True):
            raise RuntimeError("non-causal attention metadata")
        qsl = attn_metadata.query_start_loc
        if int(qsl.numel()) != 2:
            raise RuntimeError(f"expected one request per forward, got {int(qsl.numel()) - 1}")
        n_query = int(attn_metadata.num_actual_tokens)
        if int(qsl[1].item()) != n_query:
            raise RuntimeError(f"query_start_loc {int(qsl[1].item())} != num_actual_tokens {n_query}")
        seq_len = int(attn_metadata.seq_lens[0].item())
        if n_query < 1 or seq_len < n_query:
            raise RuntimeError(f"bad lengths: n_query={n_query} seq_len={seq_len}")
        if n_query > 1 and seq_len != n_query:
            raise RuntimeError(f"prefill of {n_query} tokens over {seq_len}: prefix-cached rows reused across masks")
        if seq_len > int(state.hide.numel()) + 4096:
            raise RuntimeError(f"seq_len {seq_len} far beyond the prompt ({int(state.hide.numel())} tokens)")

        hidden_here = state.min_layer <= 0 or layer_index(layer.layer_name) >= state.min_layer
        hide = state.hide_on(query.device) if hidden_here else None
        key_cache, value_cache = kv_cache.unbind(0)
        row = attn_metadata.block_table[0]
        k = gather_kv(key_cache, row, seq_len)
        v = gather_kv(value_cache, row, seq_len)
        mask = attention_mask(n_query, seq_len, hide, device=query.device)
        out = masked_attention(query[:n_query], k, v, float(self.scale), mask)
        output[:n_query].copy_(out.to(output.dtype))
        state.calls += 1
        state.calls_hidden += int(hidden_here)
        cap = state.capture_into
        if cap is not None and cap.capture and n_query > 1 and layer_index(layer.layer_name) == 0:
            # prune_kl scoring pre-pass (eval_prune.make_prune_forward): the last query's layer-0
            # attention over the columns this shadow still holds, and the rotated layer-0 keys
            if cap.row is not None:
                raise RuntimeError("layer-0 prefill captured twice in one pre-pass (chunked prefill?)")
            if seq_len != cap.prompt_len or not hidden_here:
                raise RuntimeError(f"pre-pass prefill seq_len {seq_len} != prompt length {cap.prompt_len} "
                                   "(or the shadow's hidden set is not applied on layer 0)")
            cap.row = decision_row(query[n_query - 1], k, float(self.scale), mask[-1]).cpu()
            cap.keys0 = key[:n_query].detach().float().cpu()
            cap.offset = seq_len - n_query
        return output

    forward._influence = True  # type: ignore[attr-defined]
    return forward


def make_capture_prehook(state: InfluenceState) -> Callable[..., None]:
    """Forward pre-hook on decoder layer 1 (eval_prune.make_layer1_prehook for the shadow whose
    pre-pass is in flight): the residual stream after layer 0 = ``hidden_states + residual``."""

    def hook(module, args):
        cap = state.capture_into
        if cap is None or not cap.capture:
            return
        hidden_states, residual = args[1], args[2]
        if hidden_states.shape[0] <= 1:
            return  # decode forward
        if residual is None:
            raise RuntimeError("layer 1 received no residual stream")
        if cap.h1 is not None:
            raise RuntimeError("layer-1 input captured twice in one pre-pass")
        cap.h1 = (hidden_states + residual).detach().float().cpu()

    return hook


def make_logits_hook(state: InfluenceState, original: Callable[..., Any]) -> Callable[..., Any]:
    """``compute_logits`` replacement: keeps a CPU fp32 copy of each call's row (one request per
    generate, so call ``i`` is the distribution of generated position ``i``)."""

    def compute_logits(hidden_states, *args, **kwargs):
        logits = original(hidden_states, *args, **kwargs)
        if state.hide is not None and logits is not None:
            if logits.shape[0] != 1:
                raise RuntimeError(f"expected one request's logits, got {tuple(logits.shape)}")
            state.logits.append(logits[0].detach().float().cpu())
        return logits

    return compute_logits


def check_generate_counts(state: InfluenceState, prompt_len: int, n_visual: int, n_tokens: int, min_layer: int = 0) -> int:
    """After one generate call: the masked kernel ran on every layer of every forward, the hidden
    set was applied on exactly the layers ``>= min_layer`` of each, and the logits of every
    generated position were captured. Returns the number of forwards."""
    n = state.n_layers
    if not n:
        raise RuntimeError("engine layer count unknown (engine hook did not run)")
    if state.calls == 0 or state.calls % n:
        raise RuntimeError(f"masked attention ran {state.calls} times, not a multiple of {n} layers "
                           "(compiled / CUDA-graph path?)")
    if n_visual == 0:
        raise RuntimeError(f"prompt of {prompt_len} tokens has no <|video_pad|> token")
    if len(state.logits) < n_tokens:
        raise RuntimeError(f"{len(state.logits)} logits rows captured for {n_tokens} generated tokens")
    forwards = state.calls // n
    expected_hidden = forwards * max(n - max(int(min_layer), 0), 0)
    if state.calls_hidden != expected_hidden:
        raise RuntimeError(f"hidden set applied on {state.calls_hidden} layer calls, expected {expected_hidden} "
                           f"({forwards} forwards x layers >= {min_layer} of {n})")
    return forwards


# --- one probe ------------------------------------------------------------------------------
def probe_fields(state: InfluenceState, logits: Sequence[torch.Tensor], tokens: Sequence[int]) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Record fields of one generate (the decision or a probe) from its captured logits rows:
    the free-run pointing tokens with their top-1 probabilities, the renormalised 256-way action
    log-probs at position 2 and their bin fold. Returns ``(fields, logp, logp_bin)``; the two
    distributions are None when the generate ended before the action position."""
    def top1(row):
        p = torch.softmax(row, dim=-1)
        i = int(p.argmax())
        return i, float(p[i])

    if len(tokens) <= ACTION_POS:
        return {"tokens": list(tokens), "code": None, "bin": None, "truncated": True}, None, None
    apos_id, p_apos = top1(logits[0])
    opos_id, p_opos = top1(logits[1])
    logp, p_mass = action_logp(logits[ACTION_POS], state.act_ids)
    logp_bin = fold_logp(logp, state.fold)
    code = int(logp.argmax())
    act_tok = int(tokens[ACTION_POS]) if len(tokens) > ACTION_POS else -1
    fields = {"apos": apos_id - state.apos0, "opos": opos_id - state.opos0, "p_apos": p_apos, "p_opos": p_opos,
              "code": code, "bin": state.bins[code], "bin_argmax": BINS[int(logp_bin.argmax())], "p_mass": p_mass,
              "entropy": entropy(logp), "entropy_bin": entropy(logp_bin), "in_set": bool(act_tok in state.act_ids.tolist()),
              "logp_bin": [round(float(x), 6) for x in logp_bin], "logp_f16": pack_f16(logp)}
    return fields, logp, logp_bin


def install_visual_influence(state: InfluenceState) -> None:
    """Patch vLLM's flash-attention kernel and logits head, LightNav's engine loader, sampler,
    generate and reset."""
    os.environ["VLN_VLLM_ENFORCE_EAGER"] = "1"
    import lightnav.inference.engine as engine_mod
    import lightnav.inference.model as model_mod
    import lightnav.inference.samples as samples_mod
    from lightnav.slowfast import slowfast_segments
    from vllm.v1.attention.backends import flash_attn as fa

    original_forward = fa.FlashAttentionImpl.forward
    if any(getattr(original_forward, a, None) is not None for a in ("_blind_layer", "_prune_method", "_influence")):
        raise RuntimeError("an attention hook is already installed")
    fa.FlashAttentionImpl.forward = make_influence_forward(state, original_forward)

    original_segments = samples_mod.slowfast_video_segments
    if getattr(original_segments, "_context_frames", None) is not None:
        raise RuntimeError("the context window wrapper is installed; INFLUENCE and CONTEXT_FRAMES are not combined")

    def recording_segments(episode_frames, current_abs, n_available, tiers):
        total = int(episode_frames.shape[0])
        if current_abs != total - 1 or n_available != total:
            raise AssertionError(f"unexpected SlowFast call convention: total={total} current_abs={current_abs} "
                                 f"n_available={n_available}")
        if state.layout is not None:  # a recent_k probe rebuilding the decision's prompt
            kind, k = state.layout
            if kind == "window":
                if total <= k:
                    raise RuntimeError(f"window K={k} probe asked with only {total} frames")
                segs = windowed_segments(episode_frames, current_abs, n_available, tiers, k, original_segments, slowfast_segments)
            elif kind == "single":
                segs = single_current(original_segments(episode_frames, current_abs, n_available, tiers), int(current_abs), episode_frames)
            else:
                raise RuntimeError(f"unknown probe layout {state.layout}")
            state.probe_segments = [([int(i) for i in s["frame_indices"]], int(s["pool_spatial"])) for s in segs]
            return segs
        segs = original_segments(episode_frames, current_abs, n_available, tiers)
        state.segments = [([int(i) for i in s["frame_indices"]], int(s["pool_spatial"])) for s in segs]
        state.current_abs = int(current_abs)
        return segs

    samples_mod.slowfast_video_segments = recording_segments

    original_from_frames = engine_mod.VLNInferenceEngine.generate_from_frames

    def generate_from_frames(self, video_tensor, instruction, *args, **kwargs):
        """Keep the decision's inputs while it is in flight (recent_k rebuilds its prompt)."""
        if state.measure == "recent_k":
            if args or kwargs.get("task_type") != "vlnce_traj":
                raise RuntimeError("recent_k expects the policy's keyword call with task_type='vlnce_traj'")
            state.call = (video_tensor, instruction, kwargs.get("frame_ids"))
        try:
            return original_from_frames(self, video_tensor, instruction, *args, **kwargs)
        finally:
            state.call = None

    engine_mod.VLNInferenceEngine.generate_from_frames = generate_from_frames

    original_load = model_mod.load_vllm_local

    def load_vllm_local(config, task_type="vlnce"):
        bundle, llm = original_load(config, task_type=task_type)
        cfg = llm.llm_engine.vllm_config
        if not cfg.model_config.enforce_eager:
            raise RuntimeError("vLLM is not in eager mode (VLN_VLLM_ENFORCE_EAGER=1 required)")
        from vllm.config.compilation import CUDAGraphMode
        if cfg.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(f"CUDA graphs enabled ({cfg.compilation_config.cudagraph_mode}); hooks would not run")
        if getattr(cfg.scheduler_config, "enable_chunked_prefill", False):
            raise RuntimeError("chunked prefill is on; a probe needs one prefill forward")
        names = [n for n in cfg.compilation_config.static_forward_context if _LAYER_RE.search(n)]
        indices = sorted(layer_index(n) for n in names)
        n_layers = len(indices)
        if indices != list(range(n_layers)) or n_layers == 0:
            raise RuntimeError(f"unexpected decoder attention layers: {names}")
        expected = int(cfg.model_config.hf_text_config.num_hidden_layers)
        if n_layers != expected:
            raise RuntimeError(f"{n_layers} attention layers registered, config says {expected}")
        backend_name = type(getattr(next(iter(cfg.compilation_config.static_forward_context.values())), "impl", None)).__name__
        if backend_name != "FlashAttentionImpl":
            raise RuntimeError(f"attention backend is {backend_name}, the hook patches FlashAttentionImpl")
        if float(os.environ.get("VLN_EVAL_TEMPERATURE", "0.0")) != 0.0 or os.environ.get("VLN_EVAL_TRAJ_TOP1", "0").lower() in ("1", "true", "yes"):
            raise RuntimeError("the probe assumes greedy, unconstrained decoding (VLN_EVAL_TEMPERATURE=0, no TRAJ_TOP1)")
        # the logits head of the in-process model (the runner calls ``self.model.compute_logits``)
        runner = _model_runner(llm)
        runner.model.compute_logits = make_logits_hook(state, runner.model.compute_logits)
        # token ids
        tok = bundle.tokenizer
        act = [tok.convert_tokens_to_ids(f"<act_l0_{k}>") for k in range(N_CODES)]
        if any(i is None or i == tok.unk_token_id for i in act) or act != list(range(act[0], act[0] + N_CODES)):
            raise RuntimeError(f"<act_l0_*> ids are not 256 contiguous rows: {act[:3]}..{act[-1]}")
        for lvl in (1, 2):
            if int(tok.convert_tokens_to_ids(f"<act_l{lvl}_0>")) != act[0] + lvl * N_CODES:
                raise RuntimeError(f"<act_l{lvl}_*> rows do not follow <act_l0_*> contiguously")
        state.act_ids = torch.as_tensor(act, dtype=torch.long)
        state.apos0 = int(tok.convert_tokens_to_ids("<apos_0>"))
        state.opos0 = int(tok.convert_tokens_to_ids("<opos_0>"))
        state.eos_id = int(tok.eos_token_id)
        if bundle.action_method != "rvq":
            raise RuntimeError(f"checkpoint action method {bundle.action_method!r}; the probe reads RVQ codes")
        state.n_layers = n_layers
        if state.measure == "prune_kl":
            # the DART capture is a forward pre-hook on decoder layer 1 (as eval_prune)
            from vllm.config.compilation import CompilationMode
            if cfg.compilation_config.mode != CompilationMode.NONE:
                raise RuntimeError(f"torch.compile enabled ({cfg.compilation_config.mode}); the layer-1 pre-hook would not run")
            if n_layers < 2:
                raise RuntimeError(f"{n_layers} decoder layer(s): no layer 1 to read the DART hidden states from")
            from lightnav.inference.vllm_utils import get_vllm_model
            get_vllm_model(llm).language_model.model.layers[1].register_forward_pre_hook(make_capture_prehook(state))
            for shadow in state.shadows:
                shadow.hidden_size = int(cfg.model_config.hf_text_config.hidden_size)
                shadow.n_layers = n_layers
        if state.measure == "blind_layer" and state.layers[-1] >= n_layers:
            raise RuntimeError(f"probe layers {state.layers} exceed the model's {n_layers} decoder layers "
                               f"(L = {n_layers} hides nothing: that point is the replay)")
        state.strict = True
        # one "Successfully reset prefix cache" INFO line per probe would swamp the client log
        logging.getLogger("vllm.v1.core.block_pool").setLevel(logging.WARNING)
        print(f"[influence] n_layers={n_layers} eager={cfg.model_config.enforce_eager} act_l0 ids {act[0]}..{act[-1]} "
              f"measure={state.measure} max_age={state.max_age} max_k={state.max_k} layers={list(state.layers) if state.measure == 'blind_layer' else None} bins={ {b: state.bins.count(b) for b in BINS} }", flush=True)
        return bundle, llm

    model_mod.load_vllm_local = load_vllm_local

    original_reset = engine_mod.VLNInferenceEngine.reset_episode_state

    def reset_episode_state(self):
        state.reset_episode()
        return original_reset(self)

    engine_mod.VLNInferenceEngine.reset_episode_state = reset_episode_state

    original_generate = engine_mod.VLNInferenceEngine.llm_generate_batch

    def run(self, items, hide: torch.Tensor, max_new_tokens: int, prompt_len: int, n_visual: int,
            min_layer: int = 0) -> Tuple[Any, List[int], List[torch.Tensor], int]:
        """One hooked generate: ``(texts, token ids, logits rows, forwards)``. The token ids are
        the greedy argmax of the captured rows (up to the first eos), checked against the text.
        ``hide`` applies on decoder layers ``>= min_layer``."""
        if not self.vllm_engine.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset its prefix cache (requests in flight?)")
        state.set_hide(hide, min_layer)
        state.calls, state.calls_hidden, state.logits = 0, 0, []
        try:
            out = original_generate(self, items, max_new_tokens)
        finally:
            state.set_hide(None)
        rows = state.logits
        tokens: List[int] = []
        for r in rows[:max_new_tokens]:
            tokens.append(int(r.argmax()))
            if tokens[-1] == state.eos_id:
                break
        n_forwards = check_generate_counts(state, prompt_len, n_visual, len(tokens), min_layer)
        tok = self.bundle.tokenizer
        text = tok.decode(tokens, skip_special_tokens=False)
        text_no_eos = tok.decode([t for t in tokens if t != state.eos_id], skip_special_tokens=False)
        if out[0] not in (text, text_no_eos):
            raise RuntimeError(f"captured logits do not reproduce the greedy output: {text!r} vs {out[0]!r}")
        return out, tokens, rows, n_forwards

    def rebuilt_item(self, probe_cache, layout: Tuple[str, int], t_now: int):
        """The decision in flight re-encoded under another frame layout: ``(VitResult, prompt
        ids, visual-column mask, <|video_pad|> positions, tubelets)``. Goes through the engine's
        own sample builder, processor and ViT path; ``probe_cache`` is this step's copy of the
        episode tubelet cache and staging slot 1 keeps the decision's embeds (slot 0) intact."""
        video_tensor, instruction, frame_ids = state.call
        state.layout, state.probe_segments = layout, None
        try:
            sample = engine_mod.build_vln_traj_sample(video_tensor, instruction, frame_ids, self.bundle)
        finally:
            state.layout = None
        if state.probe_segments is None:
            raise RuntimeError("the sample builder did not go through the SlowFast sampler")
        if self.enable_vit_cache and sample.get("_allow_vit_cache"):
            res = self._vit_forward_one_with_cache(sample, probe_cache, 1)
        else:
            res = self.vit_forward(sample)
        ids = list(res.prompt_ids)
        visual = visual_positions(ids)
        pad_pos = torch.nonzero(visual).flatten().tolist()
        if res.video_grid_thw is None:
            raise RuntimeError("rebuilt prompt carries no video grid")
        tubelets = tubelet_layout(state.probe_segments, [[int(x) for x in row] for row in res.video_grid_thw.tolist()])
        if sum(t.n_tokens for t in tubelets) != len(pad_pos):
            raise RuntimeError(f"rebuilt layout {layout} holds {sum(t.n_tokens for t in tubelets)} tokens but its prompt has "
                               f"{len(pad_pos)} <|video_pad|>")
        if not any(t_now in (t.f0, t.f1) for t in tubelets):
            raise RuntimeError(f"rebuilt layout {layout} lost the current frame {t_now}")
        return res, ids, visual, pad_pos, tubelets

    def recent_probes(self, items, t_now: int, visual, pad_pos, hist, cur, full_logp, full_bin):
        """The recent-window probes of one decision: ``(recent_keep, recent_window)`` record lists,
        each entry ``KL(full || probe)`` with the probe's own fields."""
        def scored(k: int, fields, lp, lb, **extra):
            return {"k": k, **extra, "kl": kl(full_logp, lp), "kl_bin": kl(full_bin, lb), **fields}

        def probe_on(its, hide, prompt_len: int, n_vis: int):
            _, toks, rws, _ = run(self, its, hide, PROBE_TOKENS, prompt_len, n_vis)
            return probe_fields(state, rws, toks)

        keep: List[Dict[str, Any]] = []
        for k, kept in recent_keep_sets(hist, cur, t_now, state.max_k):
            keep.append(scored(k, *probe_on(items, only_mask(visual, pad_pos, kept), len(visual), len(pad_pos)),
                               n_tubelets=len(kept), frames=[min(t.f0 for t in kept), t_now]))
        window: List[Dict[str, Any]] = []
        if t_now < 1:
            return keep, window
        if state.call is None:
            raise RuntimeError("recent_k probe outside generate_from_frames")
        # this step's copy of the episode tubelet cache (shared tensors): the rebuilt layouts add
        # (t, t) / odd-K padding tubelets the navigation path must never see
        probe_cache = self.new_vit_cache()
        probe_cache.max_entries = self._vit_cache.max_entries + 4 * state.max_k
        probe_cache._store.update(self._vit_cache._store)
        # K = 1 with the old blocks kept: current tubelet (t, t), every history block hidden
        res, ids1, vis1, pad1, tubs1 = rebuilt_item(self, probe_cache, ("single", 1), t_now)
        cur1 = [t for t in tubs1 if (t.f0, t.f1) == (t_now, t_now)]
        if len(cur1) != 1 or any(t_now in (t.f0, t.f1) for t in tubs1 if t is not cur1[0]):
            raise RuntimeError(f"K=1 layout: expected one ({t_now}, {t_now}) tubelet, got {[(t.f0, t.f1) for t in tubs1][-3:]}")
        keep.insert(0, scored(1, *probe_on([res], only_mask(vis1, pad1, cur1), len(ids1), len(pad1)),
                              n_tubelets=1, frames=[t_now, t_now], prompt_tokens=len(ids1)))
        for k in range(1, min(state.max_k, t_now) + 1):
            res, idsk, visk, padk, tubsk = rebuilt_item(self, probe_cache, ("window", k), t_now)
            frames = {f for t in tubsk for f in (t.f0, t.f1)}
            if frames != set(range(t_now - k + 1, t_now + 1)):
                raise RuntimeError(f"window K={k} at step {t_now} holds frames {sorted(frames)}")
            window.append(scored(k, *probe_on([res], torch.zeros(len(idsk), dtype=torch.bool), len(idsk), len(padk)),
                                 n_tubelets=len(tubsk), prompt_tokens=len(idsk), n_visual=len(padk)))
        return keep, window

    def llm_generate_batch(self, items, max_new_tokens):
        if len(items) != 1:
            raise RuntimeError(f"the influence wrapper handles one request per generate call, got {len(items)}")
        if state.segments is None or state.current_abs is None:
            raise RuntimeError("generate called before the SlowFast sampler recorded the segment layout")
        if state.episode < 0:
            raise RuntimeError("generate called before the first episode reset")
        t0 = time.monotonic()
        ids = list(items[0].prompt_ids)
        visual = visual_positions(ids)
        pad_pos = torch.nonzero(visual).flatten().tolist()
        grids = items[0].video_grid_thw
        if grids is None:
            raise RuntimeError("prompt carries no video grid")
        tubelets = tubelet_layout(state.segments, [[int(x) for x in row] for row in grids.tolist()])
        if sum(t.n_tokens for t in tubelets) != len(pad_pos):
            raise RuntimeError(f"tubelet layout holds {sum(t.n_tokens for t in tubelets)} tokens but the prompt has {len(pad_pos)} <|video_pad|>")
        t_now, n_visual = state.current_abs, len(pad_pos)
        state.segments = None  # one generate per sampled layout
        hist, cur = split_history(tubelets, t_now)
        probed = [tub for tub in hist if t_now - tub.f1 <= state.max_age]
        none_hidden = torch.zeros(len(ids), dtype=torch.bool)

        # the decision the agent acts on
        out, tokens, rows, n_forwards = run(self, items, none_hidden, max_new_tokens, len(ids), n_visual)
        full, full_logp, full_bin = probe_fields(state, rows, tokens)
        full["tokens"] = tokens
        codes = [int(tokens[ACTION_POS + lvl]) - int(state.act_ids[0]) - lvl * N_CODES for lvl in range(3)] if len(tokens) >= ACTION_POS + 3 else None
        exec_bin = None
        if codes is not None and all(0 <= c < N_CODES for c in codes):
            wp = np.zeros((state.bundle.horizon, 3), dtype=np.float32) if state.bundle.is_stop(codes) else state.bundle.decode_waypoints(codes)
            exec_bin = waypoints_bin(wp)
        full.update({"codes": codes, "exec_bin": exec_bin})

        def probe(hide, min_layer=0):
            _, toks, rws, _ = run(self, items, hide, PROBE_TOKENS, len(ids), n_visual, min_layer)
            return probe_fields(state, rws, toks)

        if state.measure == "blind_layer":
            replay, replay_logp, replay_bin = probe(none_hidden)
            replay.update({"kl": kl(full_logp, replay_logp), "kl_bin": kl(full_bin, replay_bin)})
            blind = []
            for L in state.layers:
                fields, lp, lb = probe(visual.clone(), L)
                blind.append({"layer": L, "kl": kl(full_logp, lp), "kl_bin": kl(full_bin, lb),
                              "flip_code": None if lp is None or full.get("code") is None else bool(fields["code"] != full["code"]),
                              "flip_bin": None if lp is None or full.get("bin") is None else bool(fields["bin"] != full["bin"]),
                              **fields})
            seconds = time.monotonic() - t0
            state.generates += 1
            state.log_record({
                "step": t_now, "episode_label": f"episode_{state.episode:03d}", "prompt_tokens": len(ids), "n_visual": n_visual,
                "n_tubelets": len(hist), "n_current": len(cur), "n_current_tokens": sum(c.n_tokens for c in cur),
                "n_layers": state.n_layers, "layers": list(state.layers),
                "full": full, "replay": replay, "blind_layer": blind,
                "n_probes": 1 + len(blind), "probe_seconds": seconds, "forwards_decision": n_forwards,
            })
            if state.generates <= state.debug_steps:
                def f4(x):
                    return "none" if x is None else format(x, ".4f")
                print(f"[influence] episode={state.episode} step={t_now} prompt_len={len(ids)} visual={n_visual} codes={codes} "
                      f"exec={exec_bin} blind={ {r['layer']: f4(r['kl']) for r in blind} } "
                      f"flip_bin={ {r['layer']: r['flip_bin'] for r in blind} } replay={f4(replay['kl'])} "
                      f"{seconds * 1000:.0f}ms", flush=True)
            return out

        if state.measure == "prune_kl":
            replay, replay_logp, replay_bin = probe(none_hidden)
            replay.update({"kl": kl(full_logp, replay_logp), "kl_bin": kl(full_bin, replay_bin)})
            ref, ref_logp, ref_bin = probe(visual.clone())
            ref.update({"kl": kl(full_logp, ref_logp), "kl_bin": kl(full_bin, ref_bin)})
            pads = torch.as_tensor(pad_pos, dtype=torch.long)

            def prepass(shadow, hide) -> int:
                state.capture_into = shadow
                try:
                    return run(self, items, hide, 1, len(ids), n_visual)[3]
                finally:
                    state.capture_into = None

            arms, n_probes, n_prepass = [], 2, 0
            for shadow in state.shadows:
                plan, n_pool, n_dropped, fwd = shadow_step(shadow, items, tubelets, t_now, pads, pad_pos, len(ids), prepass)
                n_prepass += fwd
                n_kept, bits = kept_per_tubelet(plan, pad_pos)
                if bool(plan.hide.any()):
                    fields, lp, lb = probe(plan.hide)
                    n_probes += 1
                else:  # this arm still holds every history token: its decision is the replay
                    fields, lp, lb = dict(replay), replay_logp, replay_bin
                    fields.pop("kl", None), fields.pop("kl_bin", None)
                arms.append({"method": shadow.method, "k": shadow.k, "budget": shadow.budget, "n_pool": n_pool,
                             "n_dropped": n_dropped, "ledger": len(shadow.ledger), "n_hidden": int(plan.hide.sum()),
                             "n_kept": int(sum(n_kept)), "bits": pack_bits(bits), "prepass_forwards": fwd,
                             "kl": kl(full_logp, lp), "kl_bin": kl(full_bin, lb),
                             "flip_code": None if lp is None or full.get("code") is None else bool(fields["code"] != full["code"]),
                             "flip_bin": None if lp is None or full.get("bin") is None else bool(fields["bin"] != full["bin"]),
                             **fields})
            seconds = time.monotonic() - t0
            state.generates += 1
            state.log_record({
                "step": t_now, "episode_label": f"episode_{state.episode:03d}", "seed_episode": state.seed_episode(),
                "prompt_tokens": len(ids), "n_visual": n_visual, "n_tubelets": len(hist), "n_current": len(cur),
                "frames": [t.f1 for t in hist], "n_tokens": [t.n_tokens for t in hist],
                "tubelets": [[t.f0, t.f1] for t in hist], "pools": [t.pool for t in hist],
                "full": full, "replay": replay, "visual_all_mask": ref, "prune_kl": arms,
                "n_probes": n_probes, "prepass_forwards": n_prepass, "probe_seconds": seconds, "forwards_decision": n_forwards,
            })
            if state.generates <= state.debug_steps:
                def f4(x):
                    return "none" if x is None else format(x, ".4f")
                print(f"[influence] episode={state.episode} step={t_now} prompt_len={len(ids)} visual={n_visual} codes={codes} "
                      f"exec={exec_bin} K0={f4(ref['kl'])} prune_kl={ {a['method'] + ':' + str(a['k']): f4(a['kl']) for a in arms} } "
                      f"kept={ {a['method'] + ':' + str(a['k']): a['n_kept'] for a in arms} } probes={n_probes} "
                      f"prepass={n_prepass} replay={f4(replay['kl'])} {seconds * 1000:.0f}ms", flush=True)
            return out

        if state.measure == "recent_k":
            _, replay_logp, replay_bin = probe(none_hidden)
            ref, ref_logp, ref_bin = probe(visual.clone())
            ref.update({"kl": kl(full_logp, ref_logp), "kl_bin": kl(full_bin, ref_bin)})
            keep, window = recent_probes(self, items, t_now, visual, pad_pos, hist, cur, full_logp, full_bin)
            seconds = time.monotonic() - t0
            state.generates += 1
            state.log_record({
                "step": t_now, "episode_label": f"episode_{state.episode:03d}", "prompt_tokens": len(ids), "n_visual": n_visual,
                "n_tubelets": len(hist), "n_current": len(cur),
                "full": full, "replay": {"kl": kl(full_logp, replay_logp), "kl_bin": kl(full_bin, replay_bin)},
                "visual_all_mask": ref, "recent_keep": keep, "recent_window": window,
                "n_probes": 2 + len(keep) + len(window), "probe_seconds": seconds, "forwards_decision": n_forwards,
            })
            if state.generates <= state.debug_steps:
                def f4(x):
                    return "none" if x is None else format(x, ".4f")
                print(f"[influence] episode={state.episode} step={t_now} prompt_len={len(ids)} visual={n_visual} codes={codes} "
                      f"exec={exec_bin} K0={f4(ref['kl'])} keep={ {r['k']: f4(r['kl']) for r in keep} } "
                      f"window={ {r['k']: f4(r['kl']) for r in window} } window_tokens={ {r['k']: r['prompt_tokens'] for r in window} } "
                      f"replay={f4(kl(full_logp, replay_logp))} {seconds * 1000:.0f}ms", flush=True)
            return out

        # the probes: replay floor, no-visual reference, age 0, one per recent history tubelet
        _, replay_logp, replay_bin = probe(none_hidden)
        ref, ref_logp, ref_bin = probe(visual.clone())
        cur_fields, cur_logp, cur_bin = probe(only_mask(visual, pad_pos, cur))
        keep_one = []
        for tub in probed:
            fields, lp, lb = probe(only_mask(visual, pad_pos, [tub]))
            keep_one.append({"frame": tub.f1, "f0": tub.f0, "pool": tub.pool, "n_tokens": tub.n_tokens, "age": t_now - tub.f1,
                             "kl": kl(lp, ref_logp), "kl_bin": kl(lb, ref_bin), **fields})
        seconds = time.monotonic() - t0
        state.generates += 1
        state.log_record({
            "step": t_now, "episode_label": f"episode_{state.episode:03d}", "prompt_tokens": len(ids), "n_visual": n_visual,
            "n_tubelets": len(hist), "n_tubelets_probed": len(probed), "n_current": len(cur),
            "frames_unprobed": [tub.f1 for tub in hist if tub not in probed],
            "full": full, "replay": {"kl": kl(full_logp, replay_logp), "kl_bin": kl(full_bin, replay_bin)},
            "visual_all_mask": ref,
            "keep_one_age0": {"kl": kl(cur_logp, ref_logp), "kl_bin": kl(cur_bin, ref_bin), "frames": [[c.f0, c.f1] for c in cur], **cur_fields},
            "keep_one": keep_one, "n_probes": 3 + len(probed), "probe_seconds": seconds, "forwards_decision": n_forwards,
        })
        if state.generates <= state.debug_steps:
            def fmt(x, spec):
                return "none" if x is None else format(x, spec)
            print(f"[influence] episode={state.episode} step={t_now} prompt_len={len(ids)} visual={n_visual} tubelets={len(hist)} "
                  f"probed={len(probed)} codes={codes} exec={exec_bin} p_apos={fmt(full.get('p_apos'), '.2f')} "
                  f"p_mass={fmt(full.get('p_mass'), '.3f')} age0 kl={fmt(kl(cur_logp, ref_logp), '.4f')}/"
                  f"{fmt(kl(cur_bin, ref_bin), '.4f')} replay={fmt(kl(full_logp, replay_logp), '.2e')} "
                  f"{seconds * 1000:.0f}ms", flush=True)
        return out

    engine_mod.VLNInferenceEngine.llm_generate_batch = llm_generate_batch


def _model_runner(llm):
    """The in-process vLLM model runner (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    for dotpath in ("llm_engine.model_executor.driver_worker.model_runner",
                    "llm_engine.model_executor.driver_worker.worker.model_runner"):
        obj = llm
        try:
            for attr in dotpath.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj.model, "compute_logits"):
            return obj
    raise RuntimeError("cannot reach the in-process model runner (is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")


# --- CLI ----------------------------------------------------------------------------------
def arg_from(rest: Sequence[str], name: str) -> Optional[str]:
    for i, a in enumerate(rest):
        if a == name and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_influence",
        description="lightnav.cli.eval_habitat with the keep-one-in KL probe after every decision "
        "(eager vLLM, masked SDPA on every layer). Every other argument is passed through.",
    )
    p.add_argument("--max-age", type=int, default=MAX_AGE, help="probe history tubelets of age t - f1 <= this (default: the fast tier)")
    p.add_argument("--measure", choices=MEASURES, default="keep_one",
                   help="keep_one (default: one probe per history tubelet against the no-visual reference), recent_k "
                        "(KL(full || K most recent frames), old blocks kept and windowed, K <= --max-k), blind_layer "
                        "(KL(full || every visual token hidden from layer L on), L in --layers) or prune_kl "
                        "(KL(full || the visual history a shadow pruner holds), one per --prune-kl-arms)")
    p.add_argument("--max-k", type=int, default=MAX_K, help="recent_k: largest number of most recent frames kept")
    p.add_argument("--layers", type=int, nargs="+", default=list(BLIND_LAYERS),
                   help="blind_layer: decoder layers L to probe (every visual token hidden on layers >= L); "
                        "default every second layer")
    p.add_argument("--prune-kl-arms", nargs="+", default=None, metavar="METHOD:K",
                   help="prune_kl: shadow pruners as method:k (k = frames' worth); default random/fastv/divprune/dart x 4/8/16/32")
    p.add_argument("--prune-kl-seed-episodes", type=int, nargs="+", default=None,
                   help="prune_kl: per episode of this run (in order), the episode counter the random selector seeds on "
                        "(its index in the 100-episode pruning runs); default: this run's own counter")
    p.add_argument("--prune-kl-tokens-per-frame", type=int, default=TOKENS_PER_FRAME,
                   help="prune_kl: history tokens one frame is worth (as eval_prune.py)")
    p.add_argument("--influence-log-dir", default=None, help="per-episode records (default <output_dir>/influence)")
    p.add_argument("--influence-debug-steps", type=int, default=0, help="Print the first N inferences' bookkeeping.")
    return p


def main(argv: List[str] | None = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    if args.measure != "prune_kl" and (args.prune_kl_arms or args.prune_kl_seed_episodes):
        raise SystemExit("--prune-kl-arms / --prune-kl-seed-episodes belong to --measure prune_kl")
    state = InfluenceState(args.max_age, args.influence_debug_steps, args.measure, args.max_k, args.layers,
                           arms=parse_arms(args.prune_kl_arms) if args.measure == "prune_kl" else (),
                           seed_episodes=args.prune_kl_seed_episodes, tokens_per_frame=args.prune_kl_tokens_per_frame)
    model_path = arg_from(rest, "--model_path")
    if not model_path:
        raise SystemExit("--model_path is required (the RVQ bundle is read from <model_path>/action_tokenizer)")
    from lightnav.traj_vocab import load_rvq_bundle

    state.bundle = load_rvq_bundle(Path(model_path) / "action_tokenizer", 10)
    state.bins = code_bins(state.bundle)
    state.fold = bin_matrix(state.bins)
    log_dir = args.influence_log_dir or (os.path.join(output_dir_from(rest), "influence") if output_dir_from(rest) else None)
    if log_dir:
        state.log_dir = Path(log_dir)
        state.log_dir.mkdir(parents=True, exist_ok=True)
        (state.log_dir / "meta.json").write_text(json.dumps(state.describe(), indent=2) + "\n")
    install_visual_influence(state)
    what = (f"recent-window KL(full || K most recent frames), old blocks kept + windowed, K <= {state.max_k}"
            if state.measure == "recent_k" else
            f"blind-from-layer KL(full || no visual from layer L on), L in {list(state.layers)}"
            if state.measure == "blind_layer" else
            f"pruning KL(full || shadow-pruned history), arms {[m + ':' + str(k) for m, k in state.arms]}"
            if state.measure == "prune_kl" else f"keep-one KL, history tubelets of age <= {state.max_age}")
    print(f"[influence] installed: {what} over <act_l0_*> (256-way + {len(BINS)} bins), free-run prefix, "
          f"eager vLLM, masked SDPA on every layer"
          + (f", records -> {state.log_dir}" if state.log_dir else ""), flush=True)
    from lightnav.cli.eval_habitat import main as upstream_main

    return upstream_main(rest)


if __name__ == "__main__":
    sys.exit(main())
