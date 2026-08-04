"""Reusable probe for the action-decision token's attention over the KV cache.

Attaches a forward hook to the self-attention of any chosen decoder layers and
keeps, per layer, the last query row (the token whose logits pick the action).
Requires attn_impl='eager'; sdpa/flash kernels return None for attention weights.
"""


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


class AttentionProbe:
    """Captures per-layer attention rows of the last query position.

    rows[layer] is the max over heads, reduced on the accelerator before the
    host copy: a full (heads, kv_len) row per layer would move an order of
    magnitude more data for no benefit to the 3D heat maps. Layers listed in
    `head_layers` additionally keep the unreduced (heads, kv_len) tensor, which
    the 2D per-head visualizations need.
    """

    def __init__(self, num_layers, layers=None, head_layers=()):
        self.num_layers = num_layers
        self.head_layer_ids = resolve_layer_ids(head_layers, num_layers) if head_layers else []
        self.layer_ids = sorted(set(resolve_layer_ids(layers, num_layers)) | set(self.head_layer_ids))
        self.rows = {}
        self.head_rows = {}
        self.kv_len = None
        self._handles = []

    def attach(self, decoder_layers):
        import torch

        def make_hook(idx):
            def hook_fn(module, args, output):
                weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
                if weights is None:
                    self.rows.pop(idx, None)
                    self.head_rows.pop(idx, None)
                    return
                heads = weights[0, :, -1, :].detach()
                self.kv_len = heads.shape[-1]
                if idx in self.head_layer_ids:
                    self.head_rows[idx] = heads.to("cpu", torch.float32)
                self.rows[idx] = heads.amax(0).to("cpu", torch.float32)

            return hook_fn

        self.detach()
        for idx in self.layer_ids:
            handle = decoder_layers[idx].self_attn.register_forward_hook(make_hook(idx))
            self._handles.append(handle)

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def reset(self):
        self.rows = {}
        self.head_rows = {}
        self.kv_len = None
