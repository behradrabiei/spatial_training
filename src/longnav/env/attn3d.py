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
import textwrap

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


class LiveAttentionCloudRenderer:
    """Render RGB, final-layer 3D attention, and Habitat's top-down map live.

    The renderer lives in the Habitat actor: that environment owns matplotlib and
    the unfiltered RGB-D observations.  Each call writes the exact frame appended
    to the video to ``current.png`` so an external viewer can follow the run.
    """

    def __init__(self, output_dir, fps=4, resolution=0.02, stride=1, fov_deg=79.0,
                 min_depth=0.1, max_depth=4.9, elev=55, azim=-60,
                 max_render_points=400_000, heat_floor=0.08, gamma=0.5,
                 depth_scale=5.0, cam_offset=(0.0, 0.88, 0.0), norm_mode="peak"):
        import imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from habitat.utils.visualizations import maps as habitat_maps

        self.output_dir = os.path.abspath(output_dir)
        self.resolution = resolution
        self.stride = stride
        self.fov_deg = fov_deg
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.elev = elev
        self.azim = azim
        self.max_render_points = max_render_points
        self.heat_floor = heat_floor
        self.gamma = gamma
        self.depth_scale = depth_scale
        self.cam_offset = np.asarray(cam_offset, dtype=np.float64)
        self.norm_mode = norm_mode
        self.cv2hab = _cv_to_habitat()
        self.start = None
        self.vox_index = {}
        self.coords = []
        self.colors = []
        self.patch_ids = []
        self.frame_offsets = []
        self.total_patches = 0
        self.trajectory = []
        self.current_frustum = None
        self.layer_id = None
        self.n_written = 0
        self.closed = False

        os.makedirs(self.output_dir, exist_ok=True)
        self.current_path = os.path.join(self.output_dir, "current.png")
        self.video_path = os.path.join(self.output_dir, "episode.mp4")
        self.partial_video_path = os.path.join(self.output_dir, ".episode.partial.mp4")
        for stale_path in (self.current_path, self.video_path, self.partial_video_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)
        self.writer = imageio.get_writer(self.partial_video_path, fps=fps, quality=4)

        self.plt = plt
        self.habitat_maps = habitat_maps
        self.cmap = matplotlib.colormaps["turbo"]
        self.fig = plt.figure(figsize=(19.2, 6.4), dpi=100)
        self.fig.patch.set_facecolor("black")
        grid = self.fig.add_gridspec(
            1, 4, width_ratios=[1.0, 1.2, 0.035, 1.0], wspace=0.04,
            left=0.01, right=0.99, top=0.94, bottom=0.04,
        )
        self.ax_img = self.fig.add_subplot(grid[0])
        self.ax3d = self.fig.add_subplot(grid[1], projection="3d")
        self.cax = self.fig.add_subplot(grid[2])
        self.ax_map = self.fig.add_subplot(grid[3])
        scalar_map = matplotlib.cm.ScalarMappable(
            cmap=self.cmap, norm=matplotlib.colors.Normalize(0, 1)
        )
        self.colorbar = self.fig.colorbar(scalar_map, cax=self.cax)
        self.colorbar.ax.yaxis.set_tick_params(color="gray", labelcolor="white", labelsize=7)
        self.colorbar.outline.set_edgecolor("gray")

    def _ingest_geometry(self, obs, info, grid):
        position = np.asarray(info["pos_rots"][:3], dtype=np.float64)
        if self.start is None:
            self.start = position.copy()
        self.trajectory.append(position - self.start)

        depth = np.asarray(obs["depth"], dtype=np.float32) * self.depth_scale
        if depth.ndim == 3:
            depth = depth[..., 0]
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        height, width = depth.shape
        _, grid_h2, grid_w2 = (int(x) for x in grid)
        grid_h, grid_w = grid_h2 // 2, grid_w2 // 2
        self.frame_offsets.append(self.total_patches)
        self.total_patches += grid_h * grid_w

        points_camera = depth_to_pointcloud(depth, self.fov_deg)[::self.stride, ::self.stride]
        colors = rgb[::self.stride, ::self.stride].astype(np.float32) / 255.0
        sampled_depth = depth[::self.stride, ::self.stride]
        rows = np.minimum(np.arange(0, height, self.stride) // PATCH_PX, grid_h - 1)
        cols = np.minimum(np.arange(0, width, self.stride) // PATCH_PX, grid_w - 1)
        pixel_patch_ids = self.frame_offsets[-1] + (rows[:, None] * grid_w + cols[None, :])
        valid = (sampled_depth > self.min_depth) & (sampled_depth < self.max_depth)

        transform = pos_rots_to_matrix(info["pos_rots"])
        transform[:3, 3] += transform[:3, :3] @ self.cam_offset
        transform = transform @ self.cv2hab
        self.current_frustum = _camera_frustum(transform, self.start, self.fov_deg,
                                               height, width)
        world = (
            points_camera[valid] @ transform[:3, :3].T + transform[:3, 3]
        ) - self.start
        voxels = np.round(world / self.resolution).astype(np.int32)
        for key, color, patch_id in zip(map(tuple, voxels), colors[valid], pixel_patch_ids[valid]):
            index = self.vox_index.get(key)
            if index is None:
                self.vox_index[key] = len(self.coords)
                self.coords.append(key)
                self.colors.append(color)
                self.patch_ids.append(patch_id)
            else:
                self.colors[index] = color
                self.patch_ids[index] = patch_id
        return rgb

    @staticmethod
    def _overlay_lines(info, prediction, mode, step, episode_index, action_names):
        action_id = int(prediction["model_action_id"])
        probabilities = prediction["action_probs"]
        prob_text = ", ".join(
            f"{name}={float(probabilities[i]):.3f}"
            for i, name in enumerate(action_names[:len(probabilities)])
        )
        oracle_id = int(info.get("oracle_action", -1))
        oracle = action_names[oracle_id] if 0 <= oracle_id < len(action_names) else "unknown"
        goal = str(prediction.get("goal", ""))
        goal_lines = textwrap.wrap(goal, width=44) or [""]
        return [
            f"episode[{episode_index}]: {info.get('episode_label', '?')}",
            f"step: {step}  mode: {mode}",
            f"model action: {action_names[action_id]}  P={float(probabilities[action_id]):.3f}",
            f"probs: {prob_text}",
            f"oracle: {oracle}",
            f"distance_to_goal: {info.get('distance_to_goal')}",
            f"distance_reward: {info.get('distance_to_goal_reward')}",
            f"spl: {info.get('spl')}  soft_spl: {info.get('soft_spl')}",
            "goal: " + goal_lines[0],
            *["      " + line for line in goal_lines[1:]],
        ]

    def append(self, obs, info, prediction, mode, step, episode_index, action_names):
        if self.closed:
            raise RuntimeError("Cannot append to a closed live attention renderer")
        logs = prediction.get("supplementary_logs", {})
        attention = logs.get("attn_hist")
        grids = logs.get("attn_hist_grids")
        if not attention or not grids:
            raise ValueError("Final-layer 3D attention was not produced for this timestep")
        layer_id = max(int(key) for key in attention)
        layer_maps = attention.get(layer_id, attention.get(str(layer_id)))
        if not layer_maps:
            raise ValueError(f"Missing attention maps for layer {layer_id}")
        if self.layer_id is None:
            self.layer_id = layer_id
        elif layer_id != self.layer_id:
            raise ValueError(f"Attention layer changed from {self.layer_id} to {layer_id}")

        current_grid = grids[-1]
        rgb = self._ingest_geometry(obs, info, current_grid)
        points = np.asarray(self.coords, dtype=np.float32) * self.resolution
        colors = np.asarray(self.colors, dtype=np.float32)
        patch_ids = np.asarray(self.patch_ids, dtype=np.int64)
        if len(points) > self.max_render_points:
            selected = np.random.default_rng(0).choice(
                len(points), self.max_render_points, replace=False
            )
            points, colors, patch_ids = points[selected], colors[selected], patch_ids[selected]

        flat_attention = np.zeros(self.total_patches, dtype=np.float32)
        for index, attention_map in enumerate(layer_maps):
            if index >= len(self.frame_offsets):
                break
            values = _as_map(attention_map)
            offset = self.frame_offsets[index]
            flat_attention[offset:offset + len(values)] = values
        low, high = attention_range(flat_attention, self.norm_mode)
        normalized = apply_attention_range(flat_attention[patch_ids], low, high, self.gamma)

        _draw_cloud(
            self.ax3d, points, colors, normalized, self.trajectory, self.cmap,
            self.elev, self.azim, self.heat_floor, frustum=self.current_frustum,
        )
        label = SIGNAL_LABELS.get(logs.get("attn_signal", "raw"), logs.get("attn_signal", "raw"))
        header = f"last layer {layer_id}: {label} over all frames seen"
        bits = []
        visual = logs.get("attn_visual_mass_frac")
        if visual is not None:
            bits.append(f"visual {visual:.0%}")
        instruction = logs.get("attn_instruction_mass_frac")
        if instruction is not None:
            bits.append(f"instruction {instruction:.0%}")
        if bits:
            header += "  |  " + ", ".join(bits) + " of the row"
        self.ax3d.text2D(
            0.02, 0.98, header,
            transform=self.ax3d.transAxes, color="white", fontsize=9, va="top",
        )
        self.colorbar.set_label(f"{label} (relative, per step)", color="white", fontsize=8)

        self.ax_img.clear()
        self.ax_img.set_axis_off()
        self.ax_img.imshow(rgb)
        lines = self._overlay_lines(info, prediction, mode, step, episode_index, action_names)
        self.ax_img.text(
            0.02, 0.98, "\n".join(lines), transform=self.ax_img.transAxes,
            color="white", fontsize=7.5, va="top", ha="left",
            bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 4},
        )

        if "top_down_map" not in info:
            raise ValueError("Habitat top_down_map is required for live rendering")
        top_down = self.habitat_maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], rgb.shape[0]
        )
        self.ax_map.clear()
        self.ax_map.set_axis_off()
        self.ax_map.imshow(top_down)
        self.ax_map.set_title(
            "Top-down map | goal instances + agent trajectory",
            color="white", fontsize=9, pad=4,
        )

        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        temporary_image = os.path.join(self.output_dir, ".current.tmp.png")
        from PIL import Image
        Image.fromarray(frame).save(temporary_image)
        os.replace(temporary_image, self.current_path)
        self.writer.append_data(frame)
        self.n_written += 1
        return self.current_path

    def close(self):
        if self.closed:
            return self.video_path if os.path.exists(self.video_path) else None
        self.closed = True
        self.plt.close(self.fig)
        self.writer.close()
        if self.n_written:
            os.replace(self.partial_video_path, self.video_path)
            return self.video_path
        if os.path.exists(self.partial_video_path):
            os.remove(self.partial_video_path)
        return None


def _camera_frustum(transform, start, fov_deg, height, width, depth=1.5):
    """Apex + 4 image-corner points of the camera frustum, world coords minus start.

    Makes the current field of view legible in the accumulated cloud: the wedge
    shows where the camera sits and which slice of the geometry the RGB panel
    covers. `transform` is the same world<-camera matrix used to unproject pixels
    (cam_offset and the CV->habitat flip already applied), so the wedge lands
    exactly on the points ingested from this frame.
    """
    tan_h = np.tan(np.deg2rad(fov_deg / 2))
    tan_v = tan_h * height / width
    x, y = depth * tan_h, depth * tan_v
    corners_cam = np.array([
        [0.0, 0.0, 0.0],
        [-x, -y, depth],
        [x, -y, depth],
        [x, y, depth],
        [-x, y, depth],
    ])
    world = corners_cam @ transform[:3, :3].T + transform[:3, 3]
    return world - start


def _draw_cloud(ax3d, pts, rgbs, nv, traj, cmap, elev, azim, heat_floor, frustum=None):
    """Draw the accumulated cloud, colored by normalized attention, plus the
    trajectory and (optionally) the current camera frustum. Near-zero-attention
    geometry stays a faint grayscale ghost.

    Habitat's frame is right-handed with -z forward; plotting (x, z, y) directly
    would swap two axes and render the scene as a mirror image. Negating z keeps
    the rendering right-handed and matches the top-down map panel's orientation
    (world x right, world z down).
    """
    flip = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    pts = pts * flip
    traj = np.asarray(traj, dtype=np.float32) * flip
    if frustum is not None:
        frustum = np.asarray(frustum, dtype=np.float32) * flip

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
        ax3d.plot(traj[:, 0], traj[:, 2], traj[:, 1], color="#00e5ff", linewidth=2.0)
    c = traj[-1]
    ax3d.scatter([c[0]], [c[2]], [c[1]], c="#00e5ff", s=90, marker="^",
                 depthshade=False, edgecolors="white", linewidths=0.8)

    if frustum is not None:
        apex, corners = frustum[0], frustum[1:]
        for corner in corners:
            ax3d.plot([apex[0], corner[0]], [apex[2], corner[2]], [apex[1], corner[1]],
                      color="#00e5ff", linewidth=1.0, alpha=0.7)
        ring = np.vstack([corners, corners[:1]])
        ax3d.plot(ring[:, 0], ring[:, 2], ring[:, 1], color="#00e5ff",
                  linewidth=1.2, alpha=0.9)

    ax3d.view_init(elev=elev, azim=azim)
    if len(pts):
        # Bounds must cover the trajectory too: the agent can walk beyond the
        # accumulated cloud (e.g. into not-yet-ingested space). Pad by 10%.
        parts = [pts, traj] + ([frustum] if frustum is not None else [])
        bounds = np.concatenate(parts, axis=0)
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
            frustum = _camera_frustum(Tw, start, fov_deg, H, W)
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

                _draw_cloud(ax3d, pts_r, rgbs_r, nv, traj, cmap, elev, azim, heat_floor,
                            frustum=frustum)
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
