"""Accumulated 3D rollout visualization of sparse-filtered image patches.

Mirrors the 2D `dim_filtered_patches` panel (see `habitat.py`) in 3D: each frame's
pixels are unprojected from depth into a world-frame RGB point cloud (origin set to
the agent's starting pose) and accumulated across the trajectory. Pixels belonging
to patches dropped by sparse filtering are rendered as a faint grayscale ghost,
while kept patches show in full color.

Self-contained on purpose: this runs inside the Habitat sim conda env ("vln"), which
has numpy/scipy/matplotlib/imageio but NOT einops/open3d. Do not import bev_utils here.
"""
import os

import numpy as np

PATCH_DIM_FACTOR = 0.35  # brightness multiplier for dropped patches (matches habitat.py)
PATCH_PX = 32  # pixel side of one LLM patch (Qwen 16px * merge_size 2)


def depth_to_pointcloud(depth, fov_deg=79.0):
    """Unproject an (H, W) depth map to (H, W, 3) camera-frame points (CV convention:
    X right, Y down, Z forward)."""
    H, W = depth.shape[-2:]
    f = (W / 2) / np.tan(np.deg2rad(fov_deg / 2))
    cx, cy = W / 2, H / 2
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (uu - cx) * z / f
    y = (vv - cy) * z / f
    return np.stack([x, y, z], axis=-1)


def pos_rots_to_matrix(pos_rots):
    """[x, y, z, qx, qy, qz, qw] -> 4x4 world<-agent transform."""
    from scipy.spatial.transform import Rotation as R

    pos_rots = np.asarray(pos_rots, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(pos_rots[3:]).as_matrix()
    T[:3, 3] = pos_rots[:3]
    return T


def _cv_to_habitat():
    """CV frame -> Habitat frame (rotate 180deg about X: flip Y and Z)."""
    c = np.eye(4)
    c[1, 1] = -1
    c[2, 2] = -1
    return c


def _render_frame(pts, cols, keep, traj, fig, ax, elev, azim, max_points):
    """Draw the accumulated RGB cloud + trajectory and return an RGB uint8 array.

    Kept-patch pixels are drawn in true color; filtered-out pixels as a faint
    grayscale ghost so the model's selection stands out while scene geometry
    remains readable.
    """
    # Negate z, matching attn3d._draw_cloud: habitat is right-handed with -z
    # forward, so plotting (x, z, y) directly would mirror the scene and disagree
    # with the top-down map orientation.
    flip = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    if len(pts):
        pts = pts * flip
    traj = [np.asarray(p, dtype=np.float32) * flip for p in traj]

    ax.clear()
    ax.set_axis_off()

    if len(pts):
        if len(pts) > max_points:
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts, cols, keep = pts[sel], cols[sel], keep[sel]
        drop = ~keep
        if drop.any():
            dp = pts[drop]
            lum = cols[drop].mean(axis=1, keepdims=True) * np.ones((1, 3))
            ax.scatter(dp[:, 0], dp[:, 2], dp[:, 1], c=np.clip(lum, 0, 1), s=1.5,
                       marker=".", alpha=0.25, depthshade=False, linewidths=0)
        if keep.any():
            kp = pts[keep]
            ax.scatter(kp[:, 0], kp[:, 2], kp[:, 1], c=np.clip(cols[keep], 0, 1),
                       s=4.0, marker=".", alpha=1.0, depthshade=False, linewidths=0)

    if len(traj) >= 2:
        t = np.array(traj, dtype=np.float32)
        ax.plot(t[:, 0], t[:, 2], t[:, 1], color="#00e5ff", linewidth=2.0)
    if traj:
        c = traj[-1]
        ax.scatter([c[0]], [c[2]], [c[1]], c="#00e5ff", s=90, marker="^",
                   depthshade=False, edgecolors="white", linewidths=0.8)

    ax.view_init(elev=elev, azim=azim)

    # Metric-equal aspect: per-axis limits hug the data, box aspect matches the
    # actual extents so the cloud fills the frame without distortion.
    if len(pts):
        lo, hi = pts.min(0), pts.max(0)
        ext = np.maximum(hi - lo, 1.0)
        ctr = (lo + hi) / 2
        ax.set_xlim(ctr[0] - ext[0] / 2, ctr[0] + ext[0] / 2)
        ax.set_ylim(ctr[2] - ext[2] / 2, ctr[2] + ext[2] / 2)
        ax.set_zlim(ctr[1] - ext[1] / 2, ctr[1] + ext[1] / 2)
        ax.set_box_aspect((ext[0], ext[2], ext[1]), zoom=1.25)
    else:
        ax.set_box_aspect((1, 1, 1))

    ax.text2D(0.02, 0.98, "color = kept patches   |   gray = filtered out",
              transform=ax.transAxes, color="white", fontsize=10, va="top")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def save_patch_cloud_video(steps_data, output_dir, filename="video_3d", fps=4,
                           quality=4, resolution=0.06, stride=4, fov_deg=79.0,
                           min_depth=0.1, max_depth=4.9, elev=55, azim=-60,
                           max_render_points=150_000):
    """Render an accumulated per-pixel RGB point-cloud video with sparse-filtered
    patches ghosted out. Returns the mp4 path, or None if the required per-step
    data (depth + keep masks) is unavailable.

    stride subsamples pixels (every stride-th row/col); resolution is the voxel
    size in meters used to dedupe accumulated points (latest observation wins).
    """
    if not steps_data or "obs" not in steps_data:
        return None
    obs_list = steps_data["obs"]
    info_list = steps_data["info"]
    masks = steps_data.get("sup/vis_keep_mask")
    grids = steps_data.get("sup/image_grid_thw")
    if not masks or not grids:
        return None
    if "depth" not in obs_list[0] or "pos_rots" not in info_list[0]:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
    from habitat.utils.visualizations import utils as vut

    start = np.array(info_list[0]["pos_rots"][:3], dtype=np.float64)
    cv2hab = _cv_to_habitat()

    # Global voxel store: voxel key -> row into the parallel lists (latest wins).
    vox_index = {}
    coords_list, colors_list, keeps_list = [], [], []
    traj = []
    images = []

    fig = plt.figure(figsize=(7.2, 5.4), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    for idx in range(len(obs_list)):
        info = info_list[idx]
        if "pos_rots" in info:
            traj.append(np.array(info["pos_rots"][:3], dtype=np.float64) - start)

        mask = masks[idx] if idx < len(masks) else None
        grid = grids[idx] if idx < len(grids) else None
        depth = obs_list[idx].get("depth")
        if mask is not None and grid is not None and depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            H, W = depth.shape
            rgb = np.asarray(obs_list[idx]["rgb"], dtype=np.float32) / 255.0

            t, h, w = (int(x) for x in grid)
            gh, gw = h // 2, w // 2  # merge_size = 2
            keep2d = np.asarray(mask, dtype=bool)[: gh * gw].reshape(gh, gw)

            # Subsampled per-pixel unprojection + per-pixel keep flag from the
            # patch the pixel belongs to.
            pts_cam = depth_to_pointcloud(depth, fov_deg)[::stride, ::stride]
            cols = rgb[::stride, ::stride]
            d = depth[::stride, ::stride]
            rr = np.minimum(np.arange(0, H, stride) // PATCH_PX, gh - 1)
            cc = np.minimum(np.arange(0, W, stride) // PATCH_PX, gw - 1)
            keep_pix = keep2d[rr[:, None], cc[None, :]]
            valid = (d > min_depth) & (d < max_depth)

            Tw = pos_rots_to_matrix(info["pos_rots"]) @ cv2hab
            flat = pts_cam[valid]
            world = (flat @ Tw[:3, :3].T + Tw[:3, 3]) - start
            vox = np.round(world / resolution).astype(np.int32)
            pix_cols = cols[valid]
            pix_keep = keep_pix[valid]

            for key, col, kf in zip(map(tuple, vox), pix_cols, pix_keep):
                row = vox_index.get(key)
                if row is None:
                    vox_index[key] = len(coords_list)
                    coords_list.append(key)
                    colors_list.append(col)
                    keeps_list.append(kf)
                else:
                    colors_list[row] = col
                    keeps_list[row] = kf

        pts = np.asarray(coords_list, dtype=np.float32) * resolution
        cols_arr = np.asarray(colors_list, dtype=np.float32)
        keep_arr = np.asarray(keeps_list, dtype=bool)
        images.append(_render_frame(pts, cols_arr, keep_arr, traj, fig, ax,
                                    elev, azim, max_render_points))

    plt.close(fig)
    if not images:
        return None

    vut.images_to_video(images=images, output_dir=output_dir, video_name=filename,
                        fps=fps, quality=quality, verbose=False)
    return os.path.join(output_dir, filename + ".mp4")
