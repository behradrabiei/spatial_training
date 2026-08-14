"""Accumulated 3D attention-heat visualization of the action decision.

At every timestep, the action token of a decoder layer attends over the
patches of ALL frames observed so far (they all live in the KV cache). This
module renders that as one video per probed layer: each frame's pixels are unprojected from depth
into a world-frame point cloud, accumulated over the trajectory, and RECOLORED
EVERY TIMESTEP by the current decision's score for the patch each pixel belongs
to. What that score measures depends on the rollout's attn_weighting -- attention,
a value-weighted variant, or a gradient attribution (see SIGNAL_LABELS). Hot
regions show what matters to the model right now, anywhere in the explored
environment; near-zero geometry stays as a faint grayscale ghost for spatial
context. A side panel shows the current RGB frame with the same (shared-scale)
overlay, plus step/action/goal text.

Attention is normalized per timestep (see `attention_range`) and gamma-corrected, so
colors answer "what is most informative for THIS decision", and the 3D cloud and the
RGB panel share the same scale so they are directly comparable.

Self-contained on purpose: runs inside the Habitat sim conda env ("vln"), which
has numpy/scipy/matplotlib/imageio but NOT einops/open3d/torch.
"""
import os

import numpy as np

from longnav.env.patch3d import (
    PATCH_PX,
    _cv_to_habitat,
    depth_to_pointcloud,
    pos_rots_to_matrix,
)

ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]


def _as_map(m):
    """Attention maps arrive as raw float16 bytes (numpy-version-agnostic across
    the mixed conda envs); accept arrays too for convenience."""
    if isinstance(m, (bytes, bytearray)):
        return np.frombuffer(m, dtype=np.float16).astype(np.float32)
    return np.asarray(m, dtype=np.float32)


def attention_range(values, mode="peak", floor_pct=5.0, ceil_pct=99.0):
    """Return the (lo, hi) that maps raw attention onto the [0,1] color range.

    "peak" is lo=0, hi=max: the historical behavior. Its failure mode on this signal is
    a single dominant patch setting the scale. Measured on the eval checkpoint, at the
    last decoder layer the median non-dropped patch sits at 0.0016 of the max and the
    99th percentile at 0.13 -- so after peak-norm + gamma only ~20% of patches clear
    heat_floor and the other 80% render as grey ghost, regardless of how much real
    structure they carry.

    "robust" clips BOTH tails to percentiles of the NONZERO entries: hi at `ceil_pct`
    (the important one -- the hottest patches simply saturate instead of dictating the
    scale, which lifts the same measurement to ~73% visible) and lo at `floor_pct`
    (guards the opposite pathology, a signal with a high floor). Zeros are excluded
    throughout because dropped patches are stored as 0 and would otherwise own the
    low percentiles.

    Shared by the 3D cloud and the 2D overlays so their colors stay comparable.
    """
    values = np.asarray(values)
    top = float(np.max(values)) if values.size else 0.0
    if top <= 0:
        return 0.0, 1e-8
    if mode != "robust":
        return 0.0, max(top, 1e-8)
    nz = values[values > 0]
    if not nz.size:
        return 0.0, max(top, 1e-8)
    lo = float(np.percentile(nz, floor_pct))
    hi = float(np.percentile(nz, ceil_pct))
    if hi - lo <= 1e-8:  # degenerate spread; fall back to peak scaling
        return 0.0, max(top, 1e-8)
    return lo, hi


def apply_attention_range(values, lo, hi, gamma=1.0):
    """Map values onto [0,1] using a range from `attention_range`, then gamma-correct."""
    return np.clip((np.asarray(values) - lo) / (hi - lo), 0.0, 1.0) ** gamma


def _attention_overlay(rgb, attn2d, cmap, alpha, floor):
    """Blend a per-patch attention map (already normalized to [0,1]) over an RGB
    frame, nearest-upsampled to pixel resolution. Blend strength scales with the
    attention value; patches below `floor` stay fully photographic."""
    H, W = rgb.shape[:2]
    gh, gw = attn2d.shape
    up = attn2d[(np.arange(H) * gh // H)][:, (np.arange(W) * gw // W)]
    heat = cmap(up)[..., :3]
    a = np.where(up >= floor, alpha * up, 0.0)[..., None]
    return (1 - a) * rgb.astype(np.float32) / 255.0 + a * heat


# What the per-patch values mean, keyed by VLMWorker.attn_weighting. A gradient map
# is alpha * d(action score)/d(alpha), not attention at all, so labelling it
# "attention" would invite exactly the wrong reading of the video.
SIGNAL_LABELS = {
    "raw": "attention",
    "value_norm": "attention x ||v||",
    "wo_norm": "attention x ||W_O v||",
    "grad": "action-logit attribution",
}


def signal_label(steps_data):
    """How to describe the per-patch values, from the mode the rollout recorded."""
    signals = steps_data.get("sup/attn_signal") or []
    mode = signals[0] if signals else "raw"
    return SIGNAL_LABELS.get(mode, mode)


def _draw_cloud(ax3d, pts, rgbs, nv, traj, cmap, elev, azim, heat_floor):
    """Draw the accumulated cloud, colored by normalized attention, plus the
    trajectory. Near-zero-attention geometry stays a faint grayscale ghost."""
    ax3d.clear()
    ax3d.set_axis_off()
    ax3d.set_facecolor("black")
    cold = nv < heat_floor
    if cold.any():
        cp = pts[cold]
        lum = rgbs[cold].mean(axis=1, keepdims=True) * np.ones((1, 3))
        ax3d.scatter(cp[:, 0], cp[:, 2], cp[:, 1], c=np.clip(lum, 0, 1), s=1.2,
                     marker=".", alpha=0.3, depthshade=False, linewidths=0)
    hot = ~cold
    if hot.any():
        order = np.argsort(nv[hot])  # hottest drawn last, i.e. on top
        hp, hv = pts[hot][order], nv[hot][order]
        ax3d.scatter(hp[:, 0], hp[:, 2], hp[:, 1], c=cmap(hv)[:, :3],
                     s=1.8 + 14.0 * hv, marker=".", alpha=0.95,
                     depthshade=False, linewidths=0)

    if len(traj) >= 2:
        tr = np.array(traj, dtype=np.float32)
        ax3d.plot(tr[:, 0], tr[:, 2], tr[:, 1], color="#00e5ff", linewidth=2.0)
    c = traj[-1]
    ax3d.scatter([c[0]], [c[2]], [c[1]], c="#00e5ff", s=90, marker="^",
                 depthshade=False, edgecolors="white", linewidths=0.8)

    ax3d.view_init(elev=elev, azim=azim)
    if len(pts):
        # Bounds must cover the trajectory too: the agent can walk beyond the
        # accumulated cloud (e.g. into not-yet-ingested space). Pad by 10%.
        bounds = np.concatenate([pts, np.asarray(traj, dtype=np.float32)], axis=0)
        lo, hi = bounds.min(0), bounds.max(0)
        ext = np.maximum(hi - lo, 1.0) * 1.1
        ctr = (lo + hi) / 2
        ax3d.set_xlim(ctr[0] - ext[0] / 2, ctr[0] + ext[0] / 2)
        ax3d.set_ylim(ctr[2] - ext[2] / 2, ctr[2] + ext[2] / 2)
        ax3d.set_zlim(ctr[1] - ext[1] / 2, ctr[1] + ext[1] / 2)
        ax3d.set_box_aspect((ext[0], ext[2], ext[1]), zoom=1.1)


def save_attention_cloud_videos(steps_data, output_dir, filename="video_attn3d", fps=4,
                                quality=4, resolution=0.02, stride=1, fov_deg=79.0,
                                min_depth=0.1, max_depth=4.9, elev=55, azim=-60,
                                max_render_points=400_000, heat_floor=0.08,
                                gamma=0.5, overlay_alpha=0.65,
                                depth_scale=5.0, cam_offset=(0.0, 0.88, 0.0),
                                norm_mode="peak"):
    """Render one growing 3D attention-heat video per probed decoder layer.

    Returns {layer_idx: mp4_path}, or {} if the required per-step data (depth,
    poses, attention history) is missing.

    steps_data must carry "sup/attn_hist" (per step: {layer_idx: list of per-frame
    flat patch-attention arrays covering every frame seen so far}) and
    "sup/attn_hist_grids" (matching [t,h,w] grids), as produced by
    VLMWorker.get_attention_3d_visualization().

    The cloud geometry and the camera are identical for every layer, so each step
    is ingested once and only the coloring is redone per layer. Frames stream
    straight into per-layer writers; buffering them would cost gigabytes.
    """
    if not steps_data or "obs" not in steps_data:
        return {}
    obs_list = steps_data["obs"]
    info_list = steps_data["info"]
    attn_hist = steps_data.get("sup/attn_hist")
    hist_grids = steps_data.get("sup/attn_hist_grids")
    if not attn_hist or not hist_grids:
        return {}
    if "depth" not in obs_list[0] or "pos_rots" not in info_list[0]:
        return {}
    layer_ids = sorted(attn_hist[0].keys())
    if not layer_ids:
        return {}

    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    cmap = matplotlib.colormaps["turbo"]
    start = np.array(info_list[0]["pos_rots"][:3], dtype=np.float64)
    cv2hab = _cv_to_habitat()
    goal = obs_list[0].get("instr_or_goal", "")
    actions = steps_data.get("action", [])

    # Global voxel store: voxel key -> row (latest observation wins). Each voxel
    # keeps its RGB plus provenance (flat patch id = frame offset + patch index)
    # so the whole cloud can be recolored from any step's attention vector.
    vox_index = {}
    coords_list, colors_list, pid_list = [], [], []
    frame_offsets = []  # flat-attention offset of each ingested frame
    total_patches = 0
    traj = []

    fig = plt.figure(figsize=(12.8, 5.44), dpi=100)  # 1280x544, divisible by 16 for ffmpeg
    fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.035, 1], wspace=0.05,
                          left=0.0, right=0.99, top=0.9, bottom=0.05)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    cax = fig.add_subplot(gs[1])
    ax_img = fig.add_subplot(gs[2])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    label = signal_label(steps_data)
    cb.set_label(f"{label} (relative, per step)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="gray", labelcolor="white", labelsize=7)
    cb.outline.set_edgecolor("gray")

    os.makedirs(output_dir, exist_ok=True)
    paths = {li: os.path.join(output_dir, f"{filename}_L{li}.mp4") for li in layer_ids}
    # Render under a temporary name and only publish on success, so a render that
    # raises (or an actor killed mid-flush) leaves no empty mp4 posing as output.
    tmp_paths = {li: os.path.join(output_dir, f".{filename}_L{li}.partial.mp4") for li in layer_ids}
    writers = {li: imageio.get_writer(tmp_paths[li], fps=fps, quality=quality) for li in layer_ids}
    n_written = 0

    n_steps = min(len(attn_hist), len(obs_list), len(info_list))
    try:
        for t in range(n_steps):
            info = info_list[t]
            traj.append(np.array(info["pos_rots"][:3], dtype=np.float64) - start)

            # --- Ingest frame t into the voxel store (shared by every layer) ---
            # habitat-lab normalizes depth to [0,1]; depth_scale (= sensor max_depth)
            # restores meters.
            depth = np.asarray(obs_list[t]["depth"], dtype=np.float32) * depth_scale
            if depth.ndim == 3:
                depth = depth[..., 0]
            H, W = depth.shape
            rgb = np.asarray(obs_list[t]["rgb"], dtype=np.uint8)
            grid = hist_grids[t][t] if t < len(hist_grids[t]) else hist_grids[t][-1]
            _, gh2, gw2 = (int(x) for x in grid)
            gh, gw = gh2 // 2, gw2 // 2  # merge_size = 2
            frame_offsets.append(total_patches)
            total_patches += gh * gw

            pts_cam = depth_to_pointcloud(depth, fov_deg)[::stride, ::stride]
            cols = rgb[::stride, ::stride].astype(np.float32) / 255.0
            d = depth[::stride, ::stride]
            rr = np.minimum(np.arange(0, H, stride) // PATCH_PX, gh - 1)
            cc = np.minimum(np.arange(0, W, stride) // PATCH_PX, gw - 1)
            pix_pid = frame_offsets[t] + (rr[:, None] * gw + cc[None, :])
            valid = (d > min_depth) & (d < max_depth)

            # pos_rots is the agent BASE (on the floor); the camera sits cam_offset
            # above it in the agent frame (sensor position in the habitat config).
            Tw = pos_rots_to_matrix(info["pos_rots"])
            Tw[:3, 3] += Tw[:3, :3] @ np.asarray(cam_offset, dtype=np.float64)
            Tw = Tw @ cv2hab
            world = (pts_cam[valid] @ Tw[:3, :3].T + Tw[:3, 3]) - start
            vox = np.round(world / resolution).astype(np.int32)
            for key, col, pid in zip(map(tuple, vox), cols[valid], pix_pid[valid]):
                row = vox_index.get(key)
                if row is None:
                    vox_index[key] = len(coords_list)
                    coords_list.append(key)
                    colors_list.append(col)
                    pid_list.append(pid)
                else:
                    colors_list[row] = col
                    pid_list[row] = pid

            pts = np.asarray(coords_list, dtype=np.float32) * resolution
            rgbs = np.asarray(colors_list, dtype=np.float32)
            pids = np.asarray(pid_list)
            # One subsample for all layers, so the videos are directly comparable.
            if len(pts) > max_render_points:
                sel = np.random.default_rng(0).choice(len(pts), max_render_points, replace=False)
                pts_r, rgbs_r, pids_r = pts[sel], rgbs[sel], pids[sel]
            else:
                pts_r, rgbs_r, pids_r = pts, rgbs, pids
            act = ACTION_NAMES[int(actions[t])] if t < len(actions) else "?"

            for li in layer_ids:
                layer_maps = attn_hist[t].get(li)
                if layer_maps is None:
                    continue

                # --- Attention of decision t over every frame seen so far ---
                flat_attn = np.zeros(total_patches, dtype=np.float32)
                for i, m in enumerate(layer_maps):
                    if i >= len(frame_offsets):
                        break
                    m = _as_map(m)
                    flat_attn[frame_offsets[i]:frame_offsets[i] + len(m)] = m

                # Shared per-step normalization (3D cloud + RGB panel) so the two are
                # directly comparable; gamma reveals the heavy-tailed secondary
                # structure. Each layer gets its own range, since attention mass
                # differs wildly across depth.
                lo, hi = attention_range(flat_attn, norm_mode)
                nv = apply_attention_range(flat_attn[pids_r], lo, hi, gamma)

                _draw_cloud(ax3d, pts_r, rgbs_r, nv, traj, cmap, elev, azim, heat_floor)
                ax3d.text2D(0.02, 0.98,
                            f"layer {li}: {label} of current action decision over ALL frames seen so far",
                            transform=ax3d.transAxes, color="white", fontsize=9, va="top")

                # --- Draw RGB panel with the current frame's attention (same scale) ---
                cur_map = _as_map(layer_maps[-1])[: gh * gw].reshape(gh, gw)
                attn2d = apply_attention_range(cur_map, lo, hi, gamma)
                ax_img.clear()
                ax_img.set_axis_off()
                ax_img.imshow(_attention_overlay(rgb, attn2d, cmap, overlay_alpha, heat_floor))
                ax_img.set_title(f"step {t}  |  layer {li}  |  action: {act}  |  goal: {goal}",
                                 color="white", fontsize=10)

                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                writers[li].append_data(buf[..., :3].copy())
                n_written += 1
    finally:
        plt.close(fig)
        for writer in writers.values():
            writer.close()

    if not n_written:
        for path in tmp_paths.values():
            if os.path.exists(path):
                os.remove(path)
        return {}
    for li in layer_ids:
        os.replace(tmp_paths[li], paths[li])
    return paths
