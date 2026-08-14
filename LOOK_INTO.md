# Things to look into

## Caveats on the 3D attention heat videos

Not bugs, but they shape how the videos should be read:

- 3D attn viz silently requires `use_sparse=True` (`visual_pos_masks` / `vis_keep_mask`); without sparse, `_frame_records` stays empty.
- The per-step normalization runs over **visual** keys only, per layer, so colour intensity is not comparable across layers, nor to the total attention mass sitting on text. `attn_norm_mode` picks how the range is taken (`attn3d.attention_range`) but does not change this.

## Fixed

### Context-window + 3D attention: same-step `abs_kv_idx` bug

Fixed by cropping the probe's captured rows in `AttentionProbe.evict`, alongside the value-norm banks it already cropped, so rows and the shifted `abs_kv_idx` land in the same index space. Regression test: `tests/attn_evict_smoke.py`, which compares each step's maps either side of the eviction and fails on the old behaviour at every evicting step.
