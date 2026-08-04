# Things to look into

## Context-window + 3D attention: same-step `abs_kv_idx` bug

**Status:** Real logic bug when `context_window` is set. Does **not** affect full-context runs (`context_window=None`, the default).

**Where:** `VLMWorker.infer_step` / `_apply_context_window` / `get_attention_3d_visualization` in `src/longnav/utils/vlm_worker.py`.

**What goes wrong:** On each step the order is:

1. Forward → attention row is captured in the **pre-eviction** KV index space
2. `_record_frame_keys()` stores `abs_kv_idx` for the new frame
3. `_apply_context_window()` may evict old turns and **subtract `n_drop` from surviving `abs_kv_idx`** (or set them to `None`)
4. `get_attention_3d_visualization()` indexes `row[abs_kv_idx]`

After eviction, frame-record indices have been shifted but the attention row has not. On any step where eviction fires, heatmaps for surviving frames can point at the wrong keys (or garbage). Later steps are fine again until the next eviction.

**Fix direction:** Either (a) call `get_attention_3d_visualization()` / snapshot the scattered maps **before** `_apply_context_window`, or (b) keep a pre-eviction copy of `abs_kv_idx` for the viz read, or (c) also crop/shift the attention row to match the post-eviction cache layout before indexing.

**Related notes (not the same bug):**

- 3D attn viz silently requires `use_sparse=True` (`visual_pos_masks` / `vis_keep_mask`); without sparse, `_frame_records` stays empty.
- Docstring in `attn3d.py` says “percentile + gamma”; code does peak-norm + gamma.
- Viz peak-normalizes **visual** keys only, per layer/step — color intensity is not comparable across layers or to total attention mass on text.
