"""Reusable probe for the action-decision token's attention over the KV cache.

Attaches a forward hook to the self-attention of any chosen decoder layers and
keeps, per layer, the last query row (the token whose logits pick the action).
Requires attn_impl='eager'; sdpa/flash kernels return None for attention weights.

The row can be weighted by how much each cached token actually contributes to the
residual stream, rather than by attention weight alone -- see `weighting` below.
"""

WEIGHTING_MODES = ("raw", "value_norm", "wo_norm", "grad")


def resolve_layer_ids(spec, num_layers):
    """Normalize a layer spec into sorted absolute indices.

    `None` means every layer; negative indices count from the end, so the
    default [-1] resolves to the last decoder layer.
    """
    if spec is None:
        return list(range(num_layers))
    ids = set()
    for layer in spec:
        layer = int(layer)
        if not -num_layers <= layer < num_layers:
            raise ValueError(f"layer index {layer} out of range for {num_layers} decoder layers")
        ids.add(layer % num_layers)
    return sorted(ids)


def wo_row_factors(attn_module, num_heads, head_dim):
    """Per-head R factors of W_O, such that ||W_O^(h) v|| == ||R_h v|| exactly.

    o_proj maps concat_h(head outputs) -> hidden, so its weight is
    (hidden, num_heads*head_dim) and head h owns the COLUMN slice
    [:, h*head_dim:(h+1)*head_dim]. R comes from a QR of that slice: Q has
    orthonormal columns, so it preserves the norm. Using R rather than the Gram
    matrix W^T W matters numerically -- the quadratic form v^T G v rounds negative
    for near-null-space v, and sqrt of that is NaN, which would propagate through
    the downstream peak-normalization and blank an entire video.

    Returns (num_heads, head_dim, head_dim) on the module's device.
    """
    import torch

    weight = attn_module.o_proj.weight.detach().to(torch.float32)
    factors = [
        torch.linalg.qr(weight[:, h * head_dim:(h + 1) * head_dim], mode="r").R
        for h in range(num_heads)
    ]
    return torch.stack(factors)


def chunk_value_norms(values_chunk, factors, num_kv_groups):
    """Norm of the residual-stream contribution of each cached token, per head.

    values_chunk is (n_kv_heads, m, head_dim). With `factors`, returns
    ||W_O^(h) v|| at (n_kv_heads*num_kv_groups, m) -- query-head resolution, since
    heads within a GQA group share v but own different W_O column slices, so their
    norms genuinely differ. Without `factors`, returns ||v|| at (n_kv_heads, m):
    identical within a group, so the caller broadcasts instead of materializing it.
    """
    import torch

    if factors is None:
        return values_chunk.norm(dim=-1)
    v = torch.repeat_interleave(values_chunk, num_kv_groups, dim=0)
    return torch.einsum("hij,hmj->hmi", factors, v).norm(dim=-1)


class AttentionProbe:
    """Captures per-layer attention rows of the last query position.

    rows[layer] is reduced over heads on the accelerator before the host copy: a
    full (heads, kv_len) row per layer would move an order of magnitude more data
    for no benefit to the 3D heat maps. Layers listed in `head_layers` additionally
    keep the unreduced (heads, kv_len) tensor, which the 2D per-head
    visualizations need.

    `weighting` selects what the row measures:
      "raw"        - attention weight alpha, reduced with amax over heads.
      "value_norm" - alpha * ||v||, summed over heads.
      "wo_norm"    - alpha * ||W_O^(h) v||, summed over heads.
      "grad"       - alpha * d(action score)/d(alpha), summed over heads.

    The weighted modes exist because alpha is only a routing coefficient: a head's
    output is sum_j alpha_j * W_O^(h) v_j, so a token with large alpha and a small
    value vector contributes nothing. Weighting also makes heads commensurable (they
    land in the same residual-stream units), which is why the weighted modes sum over
    heads where "raw" can only take a max.

    Caveat: sum_h alpha*||W_O^(h) v|| is the triangle-inequality bound on the true
    contribution norm ||sum_h alpha_h W_O^(h) v_h||, so it ignores inter-head
    cancellation. The exact form cannot be cached -- it depends on alpha, which
    changes every step -- whereas the per-token norms are static, which is what makes
    this cheap enough to run every step.

    "grad" closes the remaining gap: ||W_O v|| is a magnitude, blind to whether a
    contribution argues for or against the action being taken. The gradient replaces
    it with the projection onto the direction that raises the action score, so the row
    answers "which keys drove THIS decision" rather than "which keys were loud". It
    needs a backward pass, so it only populates on a grad-enabled forward -- the
    caller drives it through `backward_from` instead of reading `rows` straight out
    of the hook.
    """

    def __init__(self, num_layers, layers=None, head_layers=(), weighting="raw",
                 num_kv_groups=1):
        if weighting not in WEIGHTING_MODES:
            raise ValueError(f"weighting must be one of {WEIGHTING_MODES}, got {weighting!r}")
        self.num_layers = num_layers
        self.weighting = weighting
        self.num_kv_groups = num_kv_groups
        self.head_layer_ids = resolve_layer_ids(head_layers, num_layers) if head_layers else []
        self.layer_ids = sorted(set(resolve_layer_ids(layers, num_layers)) | set(self.head_layer_ids))
        self.rows = {}
        self.head_rows = {}
        self.grad_rows = {}  # layer -> live attention tensor, awaiting backward_from
        self.mass_rows = {}  # layer -> mean-over-heads row; true probability mass (raw mode only)
        self.kv_len = None
        # Only the cached generation forward should be captured; see `enabled`.
        self.enabled = False
        self.banks = {}    # layer -> (heads, kv_len) value-norm bank
        self.factors = {}  # layer -> (heads, head_dim, head_dim) W_O R factors
        self._handles = []

    def attach(self, decoder_layers):
        import torch

        def make_hook(idx):
            def hook_fn(module, args, kwargs, output):
                if not self.enabled:
                    return
                # The decoder layer forwards past_key_values as a keyword, and the
                # attention module has already called cache.update() by the time this
                # fires -- so the cache seen here includes the current chunk. Forwards
                # that carry no cache (the value head, training) must not clobber a
                # captured row, so bail without popping.
                cache = kwargs.get("past_key_values")
                if cache is None:
                    return
                weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
                if weights is None:
                    self.rows.pop(idx, None)
                    self.head_rows.pop(idx, None)
                    return
                self.kv_len = weights.shape[-1]

                if self.weighting == "grad":
                    # Keep the whole tensor, not the last-query slice: the slice is a
                    # dead-end branch of the graph (the score is computed from
                    # `weights` itself), so autograd would report it as unused.
                    # Only a grad-enabled forward has anything to differentiate; the
                    # cached forward runs under no_grad and is skipped.
                    if weights.requires_grad:
                        self.grad_rows[idx] = weights
                    return

                heads = weights[0, :, -1, :].detach()

                if self.weighting != "raw":
                    bank = self._update_bank(idx, module, cache)
                    if bank is None or bank.shape[-1] != heads.shape[-1]:
                        # Desynced bank: drop this capture rather than scatter a
                        # misaligned row onto the patch grid.
                        return
                    heads = self._apply_weights(heads, bank)

                if idx in self.head_layer_ids:
                    self.head_rows[idx] = heads.to("cpu", torch.float32)
                if self.weighting == "raw":
                    self.rows[idx] = heads.amax(0).to("cpu", torch.float32)
                    # Each head's softmax row sums to 1, so the mean over heads is a
                    # true probability distribution -- unlike the amax above, which
                    # exists for display. Mass statistics must read this row.
                    self.mass_rows[idx] = heads.mean(0).to("cpu", torch.float32)
                else:
                    row = heads.sum(0)
                    # Bound the range before the fp16 serialization downstream. Every
                    # consumer peak-normalizes, so rescaling here changes nothing.
                    self.rows[idx] = (row / row.amax().clamp_min(1e-12)).to("cpu", torch.float32)

            return hook_fn

        self.detach()
        for idx in self.layer_ids:
            handle = decoder_layers[idx].self_attn.register_forward_hook(
                make_hook(idx), with_kwargs=True
            )
            self._handles.append(handle)

    def backward_from(self, score):
        """Fill `rows` with alpha * d(score)/d(alpha) for every captured layer.

        Call once per grad-enabled forward, while its graph is still alive.

        Uses autograd.grad rather than score.backward() for two reasons: it leaves
        .grad on the model's parameters untouched (a backward would materialize a
        full gradient copy of the model every step, none of it ever read), and it
        prunes the graph to the paths that actually reach the attention tensors, so
        the weight gradients are never computed at all.

        Summing over heads before clamping keeps the honest total first-order effect
        of a key -- a head arguing against the action should cancel one arguing for
        it, which per-head clamping would hide. The clamp then drops keys whose net
        effect is negative, because the heat videos render a [0,1] colormap and would
        otherwise paint contrary evidence as support.
        """
        import torch

        # Rows are only ever produced here, so a step that captures nothing must
        # clear them rather than leave the previous step's map in place.
        self.rows = {}
        if not self.grad_rows:
            return
        layer_ids = sorted(self.grad_rows)
        weights = [self.grad_rows[idx] for idx in layer_ids]
        self.grad_rows = {}
        grads = torch.autograd.grad(score, weights, allow_unused=True)
        for idx, weight, grad in zip(layer_ids, weights, grads):
            if grad is None:
                continue
            row = (weight[0, :, -1, :].detach() * grad[0, :, -1, :]).sum(0).clamp_min(0)
            # Bound the range before the fp16 serialization downstream. Every
            # consumer peak-normalizes, so rescaling here changes nothing.
            self.rows[idx] = (row / row.amax().clamp_min(1e-12)).to("cpu", torch.float32)

    def _apply_weights(self, heads, bank):
        """Scale a (n_heads, kv_len) attention row by its per-token norms."""
        if bank.shape[0] == heads.shape[0]:
            return heads * bank
        # value_norm keeps the bank at kv-head resolution. repeat_kv is
        # repeat_interleave-style, so query heads form contiguous groups and the
        # reshape below pairs each group with its kv head.
        n_kv, kv_len = bank.shape
        return (heads.view(n_kv, -1, kv_len) * bank.unsqueeze(1)).view_as(heads)

    def _update_bank(self, idx, module, cache):
        """Extend this layer's value-norm bank to cover the whole cache.

        Value vectors are computed once at ingest and never change, so each token's
        norm is a static property -- only the newly appended chunk needs work.
        """
        import torch

        try:
            values = cache.layers[idx].values
        except (AttributeError, IndexError):
            return None
        if values is None:
            return None
        values = values[0].to(torch.float32)  # (n_kv_heads, seq, head_dim)
        total = values.shape[-2]
        bank = self.banks.get(idx)
        have = 0 if bank is None else bank.shape[-1]
        if have > total:
            # The cache shrank without evict() being called. The bank is a pure
            # function of the cached values, so a full rebuild is always available.
            bank, have = None, 0
        if have < total:
            factors = None
            if self.weighting == "wo_norm":
                factors = self.factors.get(idx)
                if factors is None:
                    # Built lazily: o_proj is a LoRA target, and merge_adapter() runs
                    # after the probe is attached, so factors taken at attach time
                    # would come from unmerged base weights.
                    factors = wo_row_factors(
                        module, values.shape[0] * self.num_kv_groups, values.shape[-1]
                    )
                    self.factors[idx] = factors
            new = chunk_value_norms(values[:, have:, :], factors, self.num_kv_groups)
            bank = new if bank is None else torch.cat([bank, new], dim=-1)
            self.banks[idx] = bank
        return bank

    def evict(self, prefix, drop_end):
        """Mirror a KV-cache eviction onto every per-key tensor.

        Banks and rows are both (..., seq), so their sequence axis is dim=-1 -- unlike
        the cache's keys/values, which are (batch, heads, seq, head_dim) and slice on
        dim=-2.

        The captured rows have to follow the cut, not just the banks. The caller shifts
        its frame->key bookkeeping into post-eviction index space immediately after this
        returns, and the 3D viz then reads row[abs_kv_idx] with those shifted indices --
        once infer_step has returned, so there is no window in which a pre-eviction row
        could be read with pre-eviction indices. Leaving the rows uncropped would point
        every surviving frame at the wrong key on any step where eviction fires.
        """
        import torch

        for store in (self.banks, self.rows, self.head_rows, self.mass_rows):
            for idx, tensor in store.items():
                store[idx] = torch.cat([tensor[..., :prefix], tensor[..., drop_end:]], dim=-1)

    def invalidate_factors(self):
        """Drop cached W_O factors. Call whenever o_proj weights change (LoRA merge)."""
        self.factors = {}

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def reset(self):
        self.rows = {}
        self.head_rows = {}
        self.grad_rows = {}
        self.mass_rows = {}
        self.kv_len = None
        self.banks = {}
