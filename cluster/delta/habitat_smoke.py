#!/usr/bin/env python3
"""Load, reset, and step one HM3D v2 ObjectNav episode."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EPISODE_LABEL = "4ok3usBNeis_0"


def main() -> int:
    repo_root = Path(os.environ["LONGNAV_REPO_ROOT"]).resolve()
    data_root = Path(os.environ["HABITAT_DATA_ROOT"]).resolve()
    dataset_path = Path(
        os.environ.get(
            "LONGNAV_HM3D_EPISODES",
            data_root
            / "evaluation_episodes/HM3D/objectnav_hm3d_v2/val/val.json.gz",
        )
    )
    scenes_dir = Path(
        os.environ.get(
            "LONGNAV_HM3D_SCENES", data_root / "scenes/HM3D/v2"
        )
    )

    from longnav.env.habitat import HabitatWorker

    worker = HabitatWorker(
        assigned_episode_labels=[EPISODE_LABEL],
        workspace=str(repo_root),
        config_path=str(repo_root / "habitat_configs/objectnav_hm3d_v2.yaml"),
        dataset_path=str(dataset_path),
        scenes_dir=str(scenes_dir),
        split="val",
        enable_caching=True,
        ep_seed=17,
        add_top_down_map=False,
        output_schema={
            "obs": {"rgb": True, "instr_or_goal": True},
            "info": {"episode_label": True, "distance_to_goal": True},
            "done": True,
            "reward": True,
        },
    )
    try:
        reset = worker.reset()
        if reset["info"]["episode_label"] != EPISODE_LABEL:
            raise RuntimeError(f"Loaded the wrong episode: {reset['info']}")
        stepped = worker.step(1)
        rgb = reset["obs"]["rgb"]
        print(
            f"Habitat reset/step: OK; episode={EPISODE_LABEL}; "
            f"rgb_shape={rgb.shape}; reward={stepped['reward']}"
        )
        return 0
    finally:
        worker.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Habitat smoke test failed: {error}", file=sys.stderr)
        raise
