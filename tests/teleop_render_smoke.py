"""Synthetic live-render smoke test. Run in the Habitat (vln) environment."""

import os
import tempfile

import imageio
import numpy as np
from PIL import Image
from habitat.utils.visualizations import maps

from longnav.env.attn3d import LiveAttentionCloudRenderer


def prediction(history):
    return {
        "model_action_id": 1,
        "action_probs": [0.05, 0.8, 0.1, 0.05],
        "goal": "chair",
        "supplementary_logs": {
            "attn_hist": {7: history},
            "attn_hist_grids": [[1, 4, 4]] * len(history),
            "attn_signal": "raw",
        },
    }


with tempfile.TemporaryDirectory() as directory:
    renderer = LiveAttentionCloudRenderer(directory, fps=4, stride=2)
    info = {
        "pos_rots": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "episode_label": "scene_0",
        "distance_to_goal": 2.0,
        "oracle_action": 1,
        "spl": 0.0,
        "top_down_map": {
            "map": np.ones((64, 64), dtype=np.uint8),
            "fog_of_war_mask": np.ones((64, 64), dtype=np.uint8),
            "agent_map_coord": [(32, 32)],
            "agent_angle": [0.0],
        },
    }
    info["top_down_map"]["map"][10:14, 48:52] = maps.MAP_TARGET_POINT_INDICATOR
    first_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    first_rgb[..., 0] = 220
    obs = {"rgb": first_rgb, "depth": np.full((64, 64, 1), 0.4, dtype=np.float32)}
    map_a = np.array([0.1, 0.3, 0.6, 0.2], dtype=np.float16).tobytes()
    current = renderer.append(
        obs, info, prediction([map_a]), "teleop", 0, 0,
        ["stop", "forward", "left", "right"],
    )
    assert os.path.exists(current)
    assert Image.open(current).size == (1920, 640)

    second_info = dict(info)
    second_info["pos_rots"] = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    second_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    second_rgb[..., 1] = 220
    second_obs = {"rgb": second_rgb, "depth": np.full((64, 64, 1), 0.45, dtype=np.float32)}
    map_b = np.array([0.7, 0.1, 0.2, 0.4], dtype=np.float16).tobytes()
    renderer.append(
        second_obs, second_info, prediction([map_a, map_b]), "model", 1, 0,
        ["stop", "forward", "left", "right"],
    )
    video = renderer.close()
    assert video and os.path.exists(video)
    assert not os.path.exists(os.path.join(directory, ".current.tmp.png"))
    assert imageio.get_reader(video).count_frames() == 2

print("teleop render smoke passed")
