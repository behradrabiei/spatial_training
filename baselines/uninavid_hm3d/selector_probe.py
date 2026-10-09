"""Per-token selector criteria vs no-history keep-one-in KL for Uni-NaVid (paper/selector_kl).

After every inference (decisions untouched), for each visual token of the CURRENT frame -- its
4-token history group plus the 64 fine navigation tokens, the ``current`` columns of
``layer_blind.visual_columns`` -- the probe records:

* the layer-0 criterion of each flat-variant selector, read from ONE decoder-layer-0 forward on
  the captured prompt (the ``TokenPruner._prepass`` idiom) and applied to the whole visual pool
  (history groups, long-term slots and the current frame):
  - ``attn``        FastV: the last prompt token's layer-0 attention, mean over heads;
  - ``divprune_nn`` DivPrune's objective: the token's nearest-neighbour cosine distance among all
                    visual tokens on the LLM input embeddings (``1 - max_{j != i} cos``);
  - ``dart_dis``    DART: ``1 - min_p cos`` to the top-``n_pivots`` pivots by L1 norm of the
                    layer-0 keys, on the layer-0 output hidden states (pivots score 2.0);
  - ``random_pos``  a control: the token's position in one permutation of the pool seeded by
                    ``prune.selection_seed(seed, label, n_frames - 1)``;
* the leak-free per-token KL ``keep_all[k] = KL(P_only k || P_no_visual)``: the prompt is
  re-forwarded with every visual column except ``k`` hidden (additive ``finfo.min`` on the key
  columns for every query, ``layer_blind.hidden_mask``), the reference with every visual column
  hidden; past-action text stays visible. Uni-NaVid re-prefills each inference, so hiding a
  column removes it from the decision entirely (no leakage through cached states).

The action distribution is read through an fp32 projection of the last hidden state onto the four
action-word rows of ``lm_head`` (the fp16 head resolves ~3 significant digits per logit, the size
of a single token's effect; ``p_mass`` still comes from the fp16 head).
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from uninavid_hm3d.influence import ACTION_WORDS, action_logp, kl_divergence
from uninavid_hm3d.layer_blind import ColumnBlinder
from uninavid_hm3d.prune import selection_seed

N_PIVOTS = 8
DEFAULT_SEED = 17


# ------------------------------------------------------------------------------ criteria
def nn_cosine_distance(rows: torch.Tensor, cur_idx: torch.Tensor) -> torch.Tensor:
    """(N,) nearest-neighbour cosine distance of ``rows[cur_idx]`` among all ``rows``
    (``1 - max_{j != i} cos``)."""
    x = F.normalize(rows.float(), dim=-1)
    cur_idx = cur_idx.to(x.device)
    sim = x[cur_idx] @ x.T
    sim[torch.arange(cur_idx.numel(), device=x.device), cur_idx] = -float("inf")
    return 1.0 - sim.amax(dim=1)


def dart_dissimilarity(hidden: torch.Tensor, key_norms: torch.Tensor, cur_idx: torch.Tensor,
                       n_pivots: int = N_PIVOTS) -> torch.Tensor:
    """(N,) DART dissimilarity of ``hidden[cur_idx]`` to the pool's pivots (the ``n_pivots`` rows
    with the largest key norm): ``1 - min_p cos``; a pivot itself scores 2.0 (always kept)."""
    n_piv = max(1, min(int(n_pivots), int(key_norms.numel())))
    pivots = torch.topk(key_norms.float(), n_piv).indices.to(hidden.device)
    h = F.normalize(hidden.float(), dim=-1)
    cur_idx = cur_idx.to(hidden.device)
    dis = 1.0 - (h[cur_idx] @ h[pivots].T).amin(dim=1)
    dis[torch.isin(cur_idx, pivots)] = 2.0
    return dis


def random_positions(n_pool: int, cur_idx: torch.Tensor, seed: int, label: str, step: int) -> torch.Tensor:
    """(N,) positions of ``cur_idx`` (pool indices) in one seeded permutation of the pool."""
    gen = torch.Generator(device="cpu").manual_seed(selection_seed(seed, label, step))
    perm = torch.randperm(int(n_pool), generator=gen)
    pos = torch.empty(int(n_pool), dtype=torch.long)
    pos[perm] = torch.arange(int(n_pool))
    return pos[cur_idx.cpu()]


# --------------------------------------------------------------------------------- probe
class SelectorProbe:
    """Capture one prefill during ``generate`` and score the current frame's tokens afterwards."""

    def __init__(self, model: Any, tokenizer: Any, image_token_index: int, batch: int = 12,
                 seed: int = DEFAULT_SEED, n_pivots: int = N_PIVOTS) -> None:
        from uninavid_hm3d.influence import FrameInfluenceProbe

        self.model = model
        self.blinder = ColumnBlinder(image_token_index)
        self.batch = int(batch)
        self.seed = int(seed)
        self.n_pivots = int(n_pivots)
        if self.batch < 1:
            raise ValueError("batch must be positive")
        self.action_ids = FrameInfluenceProbe._resolve_action_ids(tokenizer)
        self.label = ""
        self._installed = False
        self._embeds: torch.Tensor | None = None
        self._input_ids: torch.Tensor | None = None
        self._head: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        return {"action_words": list(ACTION_WORDS), "action_token_ids": list(self.action_ids),
                "tokens_probed": "the current frame: its 4-token history group + the 64 fine tokens",
                "pool": "every visual column: history groups, long-term slots, the current frame",
                "criteria": {"attn": "layer-0 attention of the last prompt token, mean over heads",
                             "divprune_nn": "1 - max cosine similarity to any other pool token (LLM input embeddings)",
                             "dart_dis": f"1 - min cosine similarity to the top-{self.n_pivots} layer-0 key-norm "
                                         "pivots on the layer-0 output (pivots 2.0)",
                             "random_pos": f"position in a permutation of the pool seeded by ({self.seed}, label, step)"},
                "keep_all": "KL(P_only_k || P_no_visual), every other visual column hidden on every layer",
                "hiding": "additive finfo.min on the key columns for every query (leak-free: Uni-NaVid re-prefills)",
                "action_head": "fp32 projection of the last hidden state onto the four action rows",
                "kl_direction": "KL(a || b), expectation under a", "temperature": 1.0,
                "probe_batch": self.batch}

    def reset(self, label: str = "") -> None:
        self.label = str(label)
        self._embeds = self._input_ids = None
        self.blinder.finish()

    # -- capture -----------------------------------------------------------------------
    def install(self) -> None:
        if self._installed:
            return
        model = self.model
        self.blinder.install(model)
        orig_prepare = model.prepare_inputs_labels_for_multimodal
        probe = self

        def prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels,
                                                 images, prompts=None):
            out = orig_prepare(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)
            if input_ids is not None and input_ids.shape[1] > 1:  # the prefill, not a decode step
                probe._embeds = out[3][0].detach()
                probe._input_ids = input_ids[0].detach().clone()
            return out

        model.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
        self._installed = True

    def begin_inference(self, input_ids: torch.Tensor) -> None:
        """Before ``generate``: mark the prompt so the layout is captured; nothing is hidden."""
        self._embeds = self._input_ids = None
        self.blinder.begin_inference(input_ids)
        self.blinder.set_hide(None)

    # -- forwards ----------------------------------------------------------------------
    def _head_weight(self) -> torch.Tensor:
        if self._head is None:
            w = self.model.lm_head.weight
            self._head = w[torch.as_tensor(list(self.action_ids), device=w.device)].detach().float()
        return self._head

    def _logp4(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, 4) log action distribution from the last hidden state through the fp32 action rows."""
        return torch.log_softmax(hidden.float() @ self._head_weight().T, dim=-1)

    @torch.inference_mode()
    def _last_hidden(self, batch: torch.Tensor, hide: torch.Tensor | None) -> torch.Tensor:
        """(B, D) final hidden state of the last prompt position with ``hide`` (B, Q) columns closed."""
        self.blinder.set_hide(hide, 0)
        try:
            return self.model.model(inputs_embeds=batch, use_cache=False)[0][:, -1]
        finally:
            self.blinder.set_hide(None)

    @torch.inference_mode()
    def _layer0(self, embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decoder layer 0 alone on the prompt (unmasked): ``(hidden_out (Q, D), attention row of the
        last token (Q,), L1 norms of the layer-0 keys (Q,))``."""
        llama = self.model.get_model()
        q = int(embeds.shape[0])
        x = embeds[None]
        mask2d = torch.ones((1, q), dtype=torch.bool, device=embeds.device)
        mask = llama._prepare_decoder_attention_mask(mask2d, (1, q), x, 0)
        position_ids = torch.arange(q, dtype=torch.long, device=embeds.device).unsqueeze(0)
        out = llama.layers[0](x, attention_mask=mask, position_ids=position_ids, past_key_value=None,
                              output_attentions=True, use_cache=True)
        hidden, attn, present = out[0][0], out[1], out[2]
        row = attn[0, :, -1, :].float().mean(dim=0)
        keys = present[0][0].permute(1, 0, 2).reshape(q, -1)
        return hidden, row, keys.float().abs().sum(dim=-1)

    # -- scoring -----------------------------------------------------------------------
    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor) -> dict[str, Any]:
        """Score the prefill captured by the last ``generate`` call; clears the capture."""
        b = self.blinder
        if self._embeds is None or self._input_ids is None or b.result_list is None:
            raise RuntimeError("no prefill captured; was begin_inference() called before generate()?")
        started = time.perf_counter()
        try:
            embeds, ids = self._embeds, self._input_ids
            if not torch.equal(ids, input_ids.reshape(-1).to(ids.device)):
                raise RuntimeError("captured prompt ids differ from the generate() prompt")
            if int(embeds.shape[0]) != b.prompt_len:
                raise RuntimeError(f"assembled prompt has {embeds.shape[0]} embeddings, layout expects {b.prompt_len}")
            all_visual, current = b.all_visual, b.current
            n_frames = int(self.model.get_model().feat_cache.shape[0])
            n_long = sum(1 for g in b.result_list if g == 1)
            pool_cols = torch.nonzero(all_visual).flatten()  # prompt columns of the pool, ascending
            cur_cols = torch.nonzero(current).flatten()
            cur_idx = torch.searchsorted(pool_cols, cur_cols)  # the current tokens' indices in the pool
            n_cur = int(cur_cols.numel())
            nav_start = int(cur_cols[-1]) - 63  # the 64 fine tokens close the current column set
            is_nav = (cur_cols >= nav_start).tolist()

            # criteria from one layer-0 forward on the unmasked prompt
            hidden0, row, key_norms = self._layer0(embeds)
            attn = row[cur_cols.to(row.device)]
            divprune_nn = nn_cosine_distance(embeds[pool_cols.to(embeds.device)], cur_idx)
            dart_dis = dart_dissimilarity(hidden0[pool_cols.to(hidden0.device)],
                                          key_norms[pool_cols.to(key_norms.device)], cur_idx, self.n_pivots)
            random_pos = random_positions(int(pool_cols.numel()), cur_idx, self.seed, self.label, n_frames - 1)

            # KL: base, then the reference (no visual) and one keep-one-in mask per current token
            base_hidden = self._last_hidden(embeds[None], None)
            base_logp = self._logp4(base_hidden)[0]
            _, p_mass = action_logp(self.model.lm_head(base_hidden).float()[0], self.action_ids)
            hides = [all_visual.clone()]
            for c in cur_cols.tolist():
                h = all_visual.clone()
                h[c] = False
                hides.append(h)
            hides = torch.stack(hides, dim=0)  # (1 + N, Q)
            logps = []
            for start in range(0, hides.shape[0], self.batch):
                chunk = hides[start:start + self.batch]
                hidden = self._last_hidden(embeds[None].expand(chunk.shape[0], -1, -1), chunk)
                logps.append(self._logp4(hidden))
            logps = torch.cat(logps, dim=0).cpu()
            n_probes = 1 + int(hides.shape[0])
            no_visual = logps[0].tolist()
            keep_all = [kl_divergence(logps[1 + k].tolist(), no_visual) for k in range(n_cur)]
            base = base_logp.cpu().tolist()
            record = {
                "n_frames": n_frames,
                "n_long_slots": n_long,
                "prompt_tokens": int(embeds.shape[0]),
                "n_visual": int(pool_cols.numel()),
                "n_current": n_cur,
                "current_cols": cur_cols.tolist(),
                "is_nav": is_nav,
                "keep_all": keep_all,
                "attn": attn.float().cpu().tolist(),
                "divprune_nn": divprune_nn.cpu().tolist(),
                "dart_dis": dart_dis.cpu().tolist(),
                "random_pos": random_pos.tolist(),
                "logp_full": base,
                "logp_no_visual": no_visual,
                "kl_no_visual": kl_divergence(base, no_visual),
                "p_mass": p_mass,
                "argmax_action": ACTION_WORDS[int(torch.tensor(base).argmax())],
                "n_probes": n_probes,
                "probe_seconds": time.perf_counter() - started,
            }
            for key in ("keep_all", "attn", "divprune_nn", "dart_dis", "logp_full", "logp_no_visual"):
                if any(not math.isfinite(float(v)) for v in record[key]):
                    raise RuntimeError(f"non-finite values in {key}")
            return record
        finally:
            self._embeds = self._input_ids = None
            b.finish()
