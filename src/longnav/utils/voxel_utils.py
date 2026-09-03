"""Per-patch world voxel ids from depth + agent pose (numpy only).

Used by context_window_mode='prune' with kv_prune_importance='voxel_*': every visual cache
slot is tagged with the world voxel its 32x32-pixel patch looks at, so a selector can
dedupe or stratify by *geometry* -- which the appearance-based sparse filter cannot see:
the same wall from two viewpoints has two different embeddings but one voxel.

Deliberately numpy-only. It runs inside the habitat actor (`vln`, which has no einops) and
its output is consumed in the VLM actor (`longnav_vlm`, which has no scipy);
longnav.utils.bev_utils imports both and is therefore unusable on either side.
"""
import numpy as np

VOXEL_NONE = -(2 ** 31)  # sentinel: text slot, or a patch without usable depth


def quat_to_matrix(qx, qy, qz, qw):
    """Rotation matrix from a scalar-last unit quaternion (scipy's convention)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose_matrix(pos_rots):
    """4x4 world <- agent transform from [x, y, z, qx, qy, qz, qw] (habitat frame, y up)."""
    pos_rots = np.asarray(pos_rots, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*pos_rots[3:7])
    T[:3, 3] = pos_rots[:3]
    return T


def patch_voxels(pos_rots, depth_m, patch_size=32, resolution=0.15, fov_degrees=79,
                 valid_max_m=4.9, min_valid_frac=0.5):
    """(H/patch_size, W/patch_size, 3) int32 world voxel ids; VOXEL_NONE where depth is unusable.

    depth_m: (H, W) in METRES (habitat hands out normalised depth; see
    HabitatWorker._postprocess_step). Each patch takes the median of its valid pixels
    (0 < d < valid_max_m) -- a median rather than a mean, so a patch straddling a depth
    edge lands on one of its surfaces instead of in mid-air -- and unprojects its centre
    pixel at that depth. Points go CV (x right, y down, z forward) -> habitat camera
    (180 deg about x) -> world through the agent pose, then round to the voxel grid. A patch
    whose valid fraction is below min_valid_frac (clipped far range, holes) is VOXEL_NONE so
    no selector ever treats it as re-observed geometry.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    H, W = depth_m.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"patch_size={patch_size} must divide the depth image {H}x{W}"
    gh, gw = H // patch_size, W // patch_size
    blocks = (depth_m.reshape(gh, patch_size, gw, patch_size)
              .transpose(0, 2, 1, 3).reshape(gh, gw, -1))
    valid = (blocks > 0) & (blocks < valid_max_m)
    with np.errstate(all="ignore"):
        z = np.nanmedian(np.where(valid, blocks, np.nan), axis=-1)  # (gh, gw)
    ok = (valid.mean(axis=-1) >= min_valid_frac) & np.isfinite(z)
    z = np.where(ok, z, 1.0)

    f = (W / 2) / np.tan(np.deg2rad(fov_degrees / 2))
    cx, cy = W / 2, H / 2
    u = (np.arange(gw) + 0.5) * patch_size
    v = (np.arange(gh) + 0.5) * patch_size
    uu, vv = np.meshgrid(u, v)
    pts_cv = np.stack([(uu - cx) * z / f, (vv - cy) * z / f, z, np.ones_like(z)], axis=-1)

    correction = np.eye(4)
    correction[1, 1] = correction[2, 2] = -1.0  # CV -> habitat camera frame
    world = pts_cv @ (pose_matrix(pos_rots) @ correction).T
    vox = np.round(world[..., :3] / resolution).astype(np.int32)
    vox[~ok] = VOXEL_NONE
    return vox
