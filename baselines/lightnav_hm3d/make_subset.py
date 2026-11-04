"""Write a filtered copy of the HM3D ObjectNav v2 val dataset holding only the labelled episodes.

LightNav's Habitat env server has no episode filter: it loads ``<root>/val/val.json.gz`` (the
stub with the category maps) and then every ``<root>/val/content/<scene>.json.gz`` shard. The
held-out evaluation set is a list of ``"<scene>_<episode_id>"`` labels, so the subset is a
directory in the same layout whose shards contain exactly those episodes. The stub is copied
byte-for-byte; the shards keep the category maps verbatim, the episodes in their original
order, and only the ``goals_by_category`` entries the kept episodes reference. Each kept
episode's ``episode_id`` is rewritten to its index in the source shard (the id habitat-lab
assigns on load, which is what the labels refer to); the loader then exposes it as
``raw_episode_id`` in every result record.

Standard library only; run it with any Python 3.8+.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_LABELS = Path("dump/hm3d_v2_100_labels.json")
DEFAULT_SOURCE = Path("/home/brabiei/vault/habitat_data/evaluation_episodes/HM3D/objectnav_hm3d_v2")
DEFAULT_SCENES = Path("/home/brabiei/vault/habitat_data/scenes/HM3D/v2")


def scene_label(scene_id: str) -> str:
    return Path(scene_id).name.split(".")[0]


def episode_label(episode: Dict[str, Any], index: int) -> str:
    """Label of the ``index``-th episode of its shard.

    habitat-lab's ObjectNav loader (and LightNav's tolerant copy of it) renumbers
    ``episode_id`` to ``str(index)`` within each content shard, so the repo's labels
    ``"<scene>_<k>"`` name the k-th episode of the shard, not the raw id stored in the
    file (raw ids repeat per category).
    """
    return f"{scene_label(episode['scene_id'])}_{index}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_labels(path: Path, expected_count: int) -> List[str]:
    labels = json.loads(path.read_text())
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"{path}: expected a JSON list of strings")
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(f"{path}: duplicate labels {dup}")
    if expected_count and len(labels) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} labels, found {len(labels)}")
    return labels


def _load_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _dump_gz(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_subset(
    labels_path: Path,
    source_root: Path,
    dest_root: Path,
    scenes_dir: Path,
    expected_count: int = 100,
    split: str = "val",
    lightnav_repo: Path | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Create ``dest_root/<split>/{<split>.json.gz, content/*.json.gz}``; return the manifest."""
    labels = read_labels(labels_path, expected_count)
    labels_sha = sha256_file(labels_path)
    manifest_path = dest_root / "manifest.json"
    if manifest_path.is_file() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("labels_sha256") == labels_sha and old.get("source_root") == str(source_root):
            return old

    src_split = source_root / split
    stub = src_split / f"{split}.json.gz"
    content = src_split / "content"
    if not stub.is_file() or not content.is_dir():
        raise FileNotFoundError(f"{src_split}: need {stub.name} and content/")

    wanted = set(labels)
    dst_split = dest_root / split
    if dst_split.exists():
        shutil.rmtree(dst_split)
    dst_split.mkdir(parents=True)
    shutil.copyfile(stub, dst_split / stub.name)

    found: Dict[str, str] = {}
    per_scene: Dict[str, int] = {}
    for shard in sorted(content.glob("*.json.gz")):
        data = _load_gz(shard)
        kept = []
        for index, ep in enumerate(data.get("episodes", [])):
            label = episode_label(ep, index)
            if label not in wanted:
                continue
            if label in found:
                raise ValueError(f"label {label} matched twice ({found[label]} and {shard.name})")
            found[label] = shard.name
            ep = dict(ep)
            # The loader keeps the file's episode_id as info["raw_episode_id"] before
            # renumbering, so store the source index there: results map back to labels.
            info = dict(ep.get("info") or {})
            info["source_episode_id"] = ep["episode_id"]
            ep["info"] = info
            ep["episode_id"] = str(index)
            kept.append(ep)
        if not kept:
            continue
        goals_by_category = data.get("goals_by_category", {})
        keys = []
        for ep in kept:
            key = f"{Path(ep['scene_id']).name}_{ep['object_category']}"
            if key not in goals_by_category:
                raise KeyError(f"{shard.name}: goals_by_category lacks {key}")
            if key not in keys:
                keys.append(key)
            scene_path = scenes_dir / ep["scene_id"]
            navmesh = scene_path.with_name(scene_path.name[: -len(".glb")] + ".navmesh")
            if not scene_path.is_file():
                raise FileNotFoundError(f"scene missing: {scene_path}")
            if not navmesh.is_file():
                raise FileNotFoundError(f"navmesh missing: {navmesh}")
        out = dict(data)
        out["episodes"] = kept
        out["goals_by_category"] = {k: goals_by_category[k] for k in keys}
        _dump_gz(dst_split / "content" / shard.name, out)
        per_scene[shard.name[: -len(".json.gz")]] = len(kept)

    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{len(missing)} labels not found in {content}: {missing[:10]}")

    manifest = {
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha,
        "source_root": str(source_root),
        "scenes_dir": str(scenes_dir),
        "split": split,
        "n_episodes": len(found),
        "n_scenes": len(per_scene),
        "per_scene": per_scene,
        "max_episodes_per_scene": max(per_scene.values()) if per_scene else 0,
        "lightnav_sha": _git_sha(lightnav_repo) if lightnav_repo else "",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    dest_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="dataset root (parent of val/)")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    p.add_argument("--expected-count", type=int, default=100)
    p.add_argument("--split", default="val")
    p.add_argument("--lightnav-repo", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    m = build_subset(
        a.labels.resolve(), a.source.resolve(), a.dest.resolve(), a.scenes_dir.resolve(),
        a.expected_count, a.split, a.lightnav_repo, a.force,
    )
    print(json.dumps({k: m[k] for k in ("n_episodes", "n_scenes", "max_episodes_per_scene", "labels_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
