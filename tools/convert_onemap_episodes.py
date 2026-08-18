"""Convert OneMap's multi-object navigation episodes into a habitat ObjectNav-v1 dataset.

OneMap ("One Map to Find Them All", ICRA 2025) ships 236 episodes on 36 HM3D v0.2
val scenes, each with a sequence of 3 object goals. This script rewrites them as a
single combined val.json.gz loadable by ObjectNavDatasetV1, with:
  - object_category = first goal (legs 2-3 live in episode.info["object_goals"],
    consumed by MultiObjectHabitatWorker at runtime)
  - goals_by_category copied wholesale from the existing objectnav_hm3d_v2 val
    content files, so every leg's category resolves to real goals + view_points
  - episodes emitted in OneMap's global order; ObjectNavDatasetV1.from_json
    overwrites episode_id with the positional index, so habitat ids == OneMap ids

Also emits an episode-label JSON ("<scene>_<id>") for task.episode_json.

Pure json+gzip — runs in any env, no habitat import.
"""

import argparse
import gzip
import json
import os

HM3D_CATEGORIES = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")


def load_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def onemap_files(onemap_dir):
    """Enumerate episode files exactly like OneMap's hm3d_multi_dataset.py so the
    global episode order (and therefore episode ids) match their benchmark."""
    for entry in sorted(os.listdir(onemap_dir), key=str.casefold):
        full = os.path.join(onemap_dir, entry)
        if not os.path.isdir(full):
            continue
        scene = entry.split("-")[1]
        yield scene, os.path.join(full, f"{scene}_episodes.json.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onemap-dir", default="/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/onemap_source/multiobject_episodes")
    parser.add_argument("--source-dir", default="/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/objectnav_hm3d_v2/val")
    parser.add_argument("--out", default="/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/onemap_multi/val/val.json.gz")
    parser.add_argument("--labels-out", default="src/longnav/conf/episode_jsons/onemap_multi.json")
    args = parser.parse_args()

    source_top = load_gz_json(os.path.join(args.source_dir, "val.json.gz"))

    episodes, labels = [], []
    goals_by_category = {}
    missing = []  # (scene, category) pairs a leg needs but the source goals lack
    category_leg_counts = {c: 0 for c in HM3D_CATEGORIES}
    leg1_dists = []
    global_id = 0

    for scene, path in onemap_files(args.onemap_dir):
        onemap_eps = load_gz_json(path)["episodes"]
        source = load_gz_json(os.path.join(args.source_dir, "content", f"{scene}.json.gz"))
        template = source["episodes"][0]
        scene_key = os.path.basename(template["scene_id"])  # "<scene>.basis.glb"
        goals_by_category.update(source["goals_by_category"])

        for om_ep in onemap_eps:
            assert om_ep["scene_id"] == template["scene_id"], (om_ep["scene_id"], template["scene_id"])
            for cat in om_ep["object_goals"]:
                category_leg_counts[cat] += 1
                if not source["goals_by_category"].get(f"{scene_key}_{cat}", []):
                    missing.append((scene, cat))
            episodes.append({
                "episode_id": str(global_id),
                "scene_id": template["scene_id"],
                "scene_dataset_config": template["scene_dataset_config"],
                "additional_obj_config_paths": template["additional_obj_config_paths"],
                "start_position": om_ep["start_position"],
                "start_rotation": om_ep["start_rotation"],
                "object_category": om_ep["object_goals"][0],
                "goals": [],
                "start_room": None,
                "shortest_paths": None,
                "info": {
                    "geodesic_distance": om_ep["best_seq_dists"][0][0],
                    "object_goals": om_ep["object_goals"],
                    "best_seq_dists": om_ep["best_seq_dists"],
                    "floor": om_ep["floor"],
                    "onemap_episode_id": global_id,
                },
            })
            labels.append(f"{scene}_{global_id}")
            leg1_dists.append((f"{scene}_{global_id}", om_ep["best_seq_dists"][0][0]))
            global_id += 1

    if missing:
        print("FATAL: legs reference categories with no goals/view_points in the source dataset:")
        for scene, cat in missing:
            print(f"  {scene}: {cat}")
        raise SystemExit(1)

    out = {
        "episodes": episodes,
        "goals_by_category": goals_by_category,
        "category_to_task_category_id": source_top["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": source_top["category_to_scene_annotation_category_id"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(out, f)
    with open(args.labels_out, "w") as f:
        json.dump(labels, f, indent=0)

    scenes = {e["scene_id"] for e in episodes}
    print(f"wrote {len(episodes)} episodes / {len(scenes)} scenes -> {args.out}")
    print(f"wrote {len(labels)} labels -> {args.labels_out}")
    print("legs per category:", category_leg_counts)
    print("shortest leg-1 episodes (good smoke candidates):")
    for label, d in sorted(leg1_dists, key=lambda x: x[1])[:5]:
        print(f"  {label}: {d:.2f} m")


if __name__ == "__main__":
    main()
