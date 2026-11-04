"""Unit tests for the LightNav-0 baseline glue (no torch, no Habitat, no GPU).

Run from the repository root:
    python3 -m unittest baselines.tests.test_lightnav
"""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from baselines.lightnav_hm3d import convert_results, make_subset


def _gz_dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _gz_load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


CATS = {"chair": 0, "bed": 1}


def _episode(scene: str, raw_id: str, category: str) -> dict:
    return {
        "episode_id": raw_id,
        "scene_id": f"hm3d_v0.2/val/00001-{scene}/{scene}.basis.glb",
        "scene_dataset_config": "./data/x.json",
        "start_position": [0, 0, 0],
        "start_rotation": [0, 0, 0, 1],
        "info": {"geodesic_distance": 1.0},
        "goals": [],
        "object_category": category,
    }


def _shard(scene: str, episodes: list[dict]) -> dict:
    cats = sorted({e["object_category"] for e in episodes})
    return {
        "goals_by_category": {f"{scene}.basis.glb_{c}": [{"position": [1, 2, 3]}] for c in cats},
        "episodes": episodes,
        "category_to_task_category_id": CATS,
        "category_to_scene_annotation_category_id": CATS,
    }


class SubsetSource:
    """A synthetic stub + two shards and fake scene files inside a temp dir."""

    def __init__(self, root: Path) -> None:
        self.source = root / "src"
        self.scenes = root / "scenes"
        self.labels = root / "labels.json"
        stub = {"episodes": [], "category_to_task_category_id": CATS,
                "category_to_scene_annotation_category_id": CATS}
        _gz_dump(self.source / "val" / "val.json.gz", stub)
        # Raw ids repeat per category on purpose, as in the real HM3D v2 shards.
        _gz_dump(self.source / "val" / "content" / "A.json.gz", _shard("A", [
            _episode("A", "0", "chair"), _episode("A", "0", "bed"), _episode("A", "1", "chair")]))
        _gz_dump(self.source / "val" / "content" / "B.json.gz", _shard("B", [
            _episode("B", "0", "bed"), _episode("B", "1", "bed")]))
        for scene in ("A", "B"):
            d = self.scenes / "hm3d_v0.2" / "val" / f"00001-{scene}"
            d.mkdir(parents=True)
            (d / f"{scene}.basis.glb").write_bytes(b"glb")
            (d / f"{scene}.basis.navmesh").write_bytes(b"nav")
        self.labels.write_text(json.dumps(["A_0", "A_2", "B_1"]))


class MakeSubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = SubsetSource(self.root)
        self.dest = self.root / "dest"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, **kw):
        return make_subset.build_subset(self.src.labels, self.src.source, self.dest, self.src.scenes,
                                        expected_count=3, **kw)

    def test_filters_by_shard_index_and_rewrites_ids(self) -> None:
        manifest = self.build()
        self.assertEqual(manifest["n_episodes"], 3)
        self.assertEqual(manifest["per_scene"], {"A": 2, "B": 1})
        self.assertEqual(manifest["max_episodes_per_scene"], 2)
        stub_src = (self.src.source / "val" / "val.json.gz").read_bytes()
        self.assertEqual((self.dest / "val" / "val.json.gz").read_bytes(), stub_src)
        a = _gz_load(self.dest / "val" / "content" / "A.json.gz")
        self.assertEqual([e["episode_id"] for e in a["episodes"]], ["0", "2"])
        self.assertEqual([e["info"]["source_episode_id"] for e in a["episodes"]], ["0", "1"])
        self.assertEqual([e["object_category"] for e in a["episodes"]], ["chair", "chair"])
        self.assertEqual(list(a["goals_by_category"]), ["A.basis.glb_chair"])
        self.assertEqual(a["category_to_task_category_id"], CATS)
        b = _gz_load(self.dest / "val" / "content" / "B.json.gz")
        self.assertEqual([e["episode_id"] for e in b["episodes"]], ["1"])
        self.assertEqual(manifest["labels_sha256"], make_subset.sha256_file(self.src.labels))

    def test_idempotent_unless_forced(self) -> None:
        first = self.build()
        second = self.build()
        self.assertEqual(first["generated_at"], second["generated_at"])
        third = self.build(force=True)
        self.assertEqual(third["n_episodes"], 3)

    def test_missing_label_raises(self) -> None:
        self.src.labels.write_text(json.dumps(["A_0", "A_9", "B_1"]))
        with self.assertRaisesRegex(ValueError, "A_9"):
            self.build()

    def test_duplicate_or_wrong_count_raises(self) -> None:
        self.src.labels.write_text(json.dumps(["A_0", "A_0", "B_1"]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build()
        self.src.labels.write_text(json.dumps(["A_0", "B_1"]))
        with self.assertRaisesRegex(ValueError, "expected 3"):
            self.build()

    def test_missing_scene_file_raises(self) -> None:
        (self.src.scenes / "hm3d_v0.2" / "val" / "00001-B" / "B.basis.navmesh").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "navmesh"):
            self.build()


def _result(scene: str, raw_id: str, idx: int, **kw) -> dict:
    res = {
        "episode_id": f"episode_{idx:03d}", "habitat_episode_id": str(idx), "raw_episode_id": raw_id,
        "scene_id": f"/scenes/hm3d_v0.2/val/00001-{scene}/{scene}.basis.glb", "rollout_idx": 0,
        "success": 1.0, "oracle_success": True, "spl": 0.5, "ndtw": 0.0, "soft_spl": 0.6,
        "object_category": "chair", "steps": 10, "final_distance": 0.4, "min_distance": 0.3,
        "instruction": "Find the chair.", "termination_reason": "agent_stop", "forced_stop": False,
        "termination_details": {},
    }
    res.update(kw)
    return res


class ConvertResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps(["A_0", "A_2", "B_1"]))
        self.run = self.root / "run"
        self.raw = self.run / "raw"
        self.raw.mkdir(parents=True)
        self.jsonl = self.raw / "results.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, results: list[dict]) -> None:
        self.jsonl.write_text("".join(json.dumps(r) + "\n" for r in results))

    def test_records_and_summary(self) -> None:
        self.write([
            _result("A", "0", 0),
            _result("A", "2", 1, success=0.0, spl=0.0, steps=500, forced_stop=True,
                    termination_reason="agent_stop", final_distance=float("inf")),
            _result("B", "1", 2, success=1.0, spl=1.0, steps=20),
        ])
        (self.raw / "summary.json").write_text(json.dumps({"metrics": {"SR_success_rate_pct": 66.67}}))
        summary = convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3)
        rec = json.loads((self.run / "episodes" / "A_2.json").read_text())
        self.assertEqual(rec["episode_label"], "A_2")
        self.assertEqual(rec["episode_id"], "2")
        self.assertEqual(rec["goal"], "chair")
        self.assertEqual(rec["steps"], 500)
        self.assertEqual(rec["inference_calls"], 499)
        self.assertTrue(rec["forced_stop"])
        self.assertNotIn("distance_to_goal", rec["metrics"])
        self.assertIsInstance(rec["metrics"]["success"], float)
        rec_b = json.loads((self.run / "episodes" / "B_1.json").read_text())
        self.assertEqual(rec_b["inference_calls"], 20)
        self.assertAlmostEqual(rec_b["metrics"]["distance_to_goal"], 0.4)
        self.assertEqual(summary["completed_episodes"], 3)
        self.assertEqual(summary["missing_episodes"], [])
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["spl"], 0.5)
        self.assertAlmostEqual(summary["average_final_distance"], 0.4)
        self.assertEqual(summary["termination_reasons"], {"agent_stop": 3})
        self.assertEqual(summary["forced_stops"], 1)
        # The row access used by longnav.analysis.results_io.load_uninavid_run.
        for path in (self.run / "episodes").glob("*.json"):
            r = json.loads(path.read_text())
            m = r["metrics"]
            float(m["success"]); float(m["spl"]); float(m.get("soft_spl", math.nan))
            float(m.get("distance_to_goal", math.nan)); int(r["steps"])
            float(r.get("elapsed_seconds", math.nan)); r.get("inference_calls"); r.get("peak_cuda_bytes")

    def test_missing_episodes(self) -> None:
        self.write([_result("A", "0", 0)])
        with self.assertRaises(SystemExit):
            convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3)
        summary = json.loads((self.run / "summary.json").read_text())
        self.assertEqual(summary["missing_episodes"], ["A_2", "B_1"])
        summary = convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3,
                                          allow_missing=True)
        self.assertEqual(summary["completed_episodes"], 1)

    def test_unknown_or_duplicate_label_raises(self) -> None:
        self.write([_result("A", "0", 0), _result("A", "0", 1)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3)
        self.write([_result("C", "0", 0)])
        with self.assertRaisesRegex(ValueError, "not in"):
            convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3)

    def test_sr_mismatch_with_lightnav_summary_raises(self) -> None:
        self.write([_result("A", "0", 0), _result("A", "2", 1), _result("B", "1", 2)])
        (self.raw / "summary.json").write_text(json.dumps({"metrics": {"SR_success_rate_pct": 50.0}}))
        with self.assertRaisesRegex(ValueError, "SR mismatch"):
            convert_results.convert(self.run, self.labels, [self.jsonl], expected_count=3)


if __name__ == "__main__":
    unittest.main()


class PruneLogMoveTest(unittest.TestCase):
    """convert_results moves the pruning wrapper's raw/prune/episode_NNN.jsonl to prune/<label>.jsonl."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps(["A_0", "A_2", "B_1"]))
        self.run = self.root / "run"
        self.raw = self.run / "raw"
        (self.raw / "prune").mkdir(parents=True)
        rows = [_result("A", "0", 0), _result("A", "2", 1), _result("B", "1", 2)]
        (self.raw / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for i in range(3):
            (self.raw / "prune" / f"episode_{i:03d}.jsonl").write_text(json.dumps({"step": 5, "frames": [i]}) + "\n")
        (self.raw / "prune" / "meta.json").write_text(json.dumps({"method": "random", "k": 4}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_logs_are_renamed_by_label(self):
        convert_results.convert(self.run, self.labels, [self.raw / "results.jsonl"], expected_count=3)
        self.assertEqual(sorted(p.name for p in (self.run / "prune").iterdir()),
                         ["A_0.jsonl", "A_2.jsonl", "B_1.jsonl", "meta.json"])
        self.assertEqual(json.loads((self.run / "prune" / "B_1.jsonl").read_text())["frames"], [2])
        self.assertFalse((self.raw / "prune").exists())
        # a second conversion (no raw/prune left) is a no-op for the logs
        convert_results.convert(self.run, self.labels, [self.raw / "results.jsonl"], expected_count=3)
        self.assertTrue((self.run / "prune" / "A_0.jsonl").is_file())

    def test_log_without_results_row(self):
        (self.raw / "prune" / "episode_007.jsonl").write_text("{}\n")
        with self.assertRaises(ValueError):
            convert_results.convert(self.run, self.labels, [self.raw / "results.jsonl"], expected_count=3)
        # with --allow-missing it stays under raw/prune, the others move
        convert_results.convert(self.run, self.labels, [self.raw / "results.jsonl"], expected_count=3, allow_missing=True)
        self.assertTrue((self.raw / "prune" / "episode_007.jsonl").is_file())
        self.assertTrue((self.run / "prune" / "A_2.jsonl").is_file())


class LoadLightnavRunTest(unittest.TestCase):
    """longnav.analysis.results_io.load_lightnav_run: forced stops at the step cap are failures."""

    def setUp(self) -> None:
        try:
            from longnav.analysis import results_io  # the paper aggregators' loader (src/ on PYTHONPATH)
        except ImportError:  # pragma: no cover - test runs from the lightnav client env
            self.skipTest("longnav.analysis.results_io not importable (add src/ to PYTHONPATH)")
        self.results_io = results_io
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        (self.run / "raw").mkdir(parents=True)
        labels = Path(self.tmp.name) / "labels.json"
        labels.write_text(json.dumps(["A_0", "A_2", "B_1", "B_3"]))
        jsonl = self.run / "raw" / "results.jsonl"
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in [
            _result("A", "0", 0, success=1.0, spl=0.8, steps=40),                       # agent stop, success
            _result("A", "2", 1, success=1.0, spl=1.0, steps=500, forced_stop=True,      # parked in radius at the cap
                    termination_reason="agent_stop", final_distance=0.9),
            _result("B", "1", 2, success=0.0, spl=0.0, steps=500, forced_stop=True,      # far away at the cap
                    termination_reason="agent_stop", final_distance=6.0),
            _result("B", "3", 3, success=0.0, spl=0.0, steps=70),                        # agent stop, miss
        ]))
        convert_results.convert(self.run, labels, [jsonl], expected_count=4)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_forced_stops_are_failures_and_raw_scores_kept(self) -> None:
        raw = self.results_io.load_uninavid_run(self.run)
        rows = self.results_io.load_lightnav_run(self.run)
        self.assertEqual(sorted(rows), ["A_0", "A_2", "B_1", "B_3"])
        self.assertEqual(raw["A_2"]["success"], 1.0)  # the harness's own verdict
        self.assertEqual((rows["A_2"]["success"], rows["A_2"]["spl"]), (0.0, 0.0))
        self.assertEqual((rows["A_2"]["success_raw"], rows["A_2"]["spl_raw"]), (1.0, 1.0))
        self.assertTrue(rows["A_2"]["rescored"])
        for label in ("A_0", "B_1", "B_3"):  # untouched: not forced, or forced but already a failure
            self.assertEqual(rows[label]["success"], raw[label]["success"])
            self.assertEqual(rows[label]["spl"], raw[label]["spl"])
            self.assertFalse(rows[label]["rescored"])
        self.assertEqual([rows[l]["forced_stop"] for l in ("A_0", "A_2", "B_1", "B_3")], [False, True, True, False])
        self.assertEqual(rows["A_2"]["n_steps"], 500)  # steps / other fields unchanged

    def test_refuses_runs_without_the_flag(self) -> None:
        for path in (self.run / "episodes").glob("*.json"):
            rec = json.loads(path.read_text()); rec.pop("forced_stop"); path.write_text(json.dumps(rec))
        with self.assertRaises(ValueError):
            self.results_io.load_lightnav_run(self.run)
