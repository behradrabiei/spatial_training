"""Unit tests for the LongNav-R1 frame-influence loader (src/longnav/analysis/influence.py) on a
synthetic keep-one run: the ``keep_one`` rows keep their meaning and the derived ``keep_one_full``
measure scores every lone frame against the replayed full decision; and on synthetic
joint-probes-only runs for the ``recent_k`` measure (KL(full || only the K most recent frames)).

    conda run -n longnav_vlm python -m unittest tests.influence_loaders_unit
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from longnav.analysis import influence as inf

LABEL = "ep_a"


def _logp(*p):
    p = np.asarray(p, dtype=float)
    return [float(x) for x in np.log(p / p.sum())]


BASE = [_logp(.1, .7, .1, .1), _logp(.1, .1, .7, .1), _logp(.7, .1, .1, .1)]  # the replayed full decision per step
NOVIS = [_logp(.25, .25, .25, .25)] * 3  # the no-visual reference
# only-frame log-probs per step t, frame f = 0..t (index t = the current frame)
ONLY = [[_logp(.2, .4, .2, .2)],
        [_logp(.1, .6, .2, .1), _logp(.1, .1, .6, .2)],
        [_logp(.4, .2, .2, .2), _logp(.1, .1, .6, .2), _logp(.6, .2, .1, .1)]]
KEEP = [[inf.kl_from_logp(ONLY[t][f], NOVIS[t]) for f in range(t + 1)] for t in range(3)]  # the logged keep-one KL
VIS_NEW = [3, 0, 2]  # frame 1 has no surviving visual slots


def write_fake_run(root: Path) -> None:
    seq_dir = root / "rollout" / "scene" / f"{LABEL}.0@0"
    seq_dir.mkdir(parents=True)
    seq_dir.joinpath("sequence.json").write_text(json.dumps({
        "sup/kl_influence_keep_one": KEEP,
        "sup/kl_influence_keep_one_logp": ONLY,
        "sup/kl_influence_base_logp": BASE,
        "sup/kl_influence_visual_all": [[inf.kl_from_logp(BASE[t], NOVIS[t]), NOVIS[t]] for t in range(3)],
        "sup/kl_influence_history_all": [[0.0, BASE[0]], [0.02, BASE[1]], [0.03, BASE[2]]],
        "sup/kl_influence_horizon": [[], [], []],
        "kv_visual_new": VIS_NEW,
        "sup/action_probs": [list(np.exp(b)) for b in BASE],
        "action_history": ["forward", "left", "stop"],
        "distance_to_goal": [3.0, 2.0, 1.0],
    }))
    (root / "rollout" / "results_0.jsonl").write_text(json.dumps({"episode_label": LABEL, "success": 1, "n_steps": 3}) + "\n")


class KlHelperTest(unittest.TestCase):
    def test_kl_from_logp_matches_kl_from_probs(self):
        a, b = _logp(.5, .3, .1, .1), _logp(.1, .2, .3, .4)
        self.assertAlmostEqual(inf.kl_from_logp(a, b), inf.kl_from_probs(np.exp(a), np.exp(b)), places=9)
        self.assertEqual(inf.kl_from_logp(a, a), 0.0)
        self.assertGreater(inf.kl_from_logp(a, b), 0.0)


class R1LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_fake_run(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _rows_by(self, rows):
        return {(r["step"], r["frame"]): r for r in rows}

    def test_keep_one_rows_unchanged(self):
        rows, steps = inf.load_r1_influence(self.root, measure="keep_one")
        by = self._rows_by(rows)
        self.assertEqual(sorted(by), [(1, 0), (2, 0), (2, 1)])
        self.assertEqual(by[(2, 0)]["infl"], KEEP[2][0])
        self.assertEqual(by[(2, 0)]["logp_full"], NOVIS[2])  # keep_one is scored against the no-visual reference
        self.assertEqual(by[(2, 0)]["logp_removed"], ONLY[2][0])
        self.assertTrue(math.isnan(by[(2, 1)]["infl"]))  # frame 1 has no visual slots
        self.assertEqual([s["age0"] for s in steps][::2], [KEEP[0][0], KEEP[2][2]])
        self.assertTrue(math.isnan(steps[1]["age0"]))
        self.assertAlmostEqual(steps[2]["sum_loo"], KEEP[2][0])
        self.assertEqual(steps[2]["n_scored"], 1)

    def test_keep_one_full_scores_against_the_full_decision(self):
        rows, steps = inf.load_r1_influence(self.root, measure="keep_one_full")
        by = self._rows_by(rows)
        self.assertEqual(sorted(by), [(1, 0), (2, 0), (2, 1)])
        r = by[(2, 0)]
        self.assertAlmostEqual(r["infl"], inf.kl_from_logp(BASE[2], ONLY[2][0]), places=12)
        self.assertEqual(r["logp_full"], BASE[2])  # the reference is the replayed full decision
        self.assertEqual(r["logp_removed"], ONLY[2][0])
        self.assertEqual((r["argmax_action"], r["argmax_removed"], r["flip"]), ("stop", "stop", 0))
        r10 = by[(1, 0)]
        self.assertAlmostEqual(r10["infl"], inf.kl_from_logp(BASE[1], ONLY[1][0]), places=12)
        self.assertEqual((r10["argmax_action"], r10["argmax_removed"], r10["flip"]), ("left", "forward", 1))
        self.assertTrue(math.isnan(by[(2, 1)]["infl"]))  # frame without slots stays unscored
        # age 0 = the current frame alone vs the full decision; NaN where the current frame has no slots
        self.assertAlmostEqual(steps[0]["age0"], inf.kl_from_logp(BASE[0], ONLY[0][0]), places=12)
        self.assertTrue(math.isnan(steps[1]["age0"]))
        self.assertAlmostEqual(steps[2]["age0"], inf.kl_from_logp(BASE[2], ONLY[2][2]), places=12)
        # the joint probes keep their full-context meaning: visual_all = KL(full || no visual) is the ceiling
        self.assertAlmostEqual(steps[2]["visual_all"], inf.kl_from_logp(BASE[2], NOVIS[2]), places=12)
        self.assertAlmostEqual(steps[2]["history_all"], 0.03)
        self.assertAlmostEqual(steps[2]["sum_loo"], r["infl"])
        self.assertEqual(steps[2]["n_scored"], 1)
        # the two measures differ on every scored row
        keep_rows = self._rows_by(inf.load_r1_influence(self.root, measure="keep_one")[0])
        for key, kr in keep_rows.items():
            if math.isfinite(kr["infl"]):
                self.assertNotAlmostEqual(kr["infl"], by[key]["infl"])
        self.assertTrue(set(inf.ROW_FIELDS) <= set(r))

    def test_unknown_measure_rejected(self):
        with self.assertRaises(ValueError):
            inf.load_r1_influence(self.root, measure="nope")


class RecentWindowTest(unittest.TestCase):
    """The recent_k measure, KL(full || only the K most recent frames): read from the joint probes
    alone (K 0 = visual_all, K 1 = history_all, K >= 2 = the logged age bands)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_r1(self):
        """A joint-probes-only run (vlm.kv_prune_influence_measures=[]): empty per-frame rows."""
        seq_dir = self.root / "rollout" / "scene" / f"{LABEL}.0@0"
        seq_dir.mkdir(parents=True)
        band = _logp(.6, .2, .1, .1)  # step 2, K = 2: argmax moves from 'stop' ... to 'stop' (no flip)
        seq_dir.joinpath("sequence.json").write_text(json.dumps({
            "sup/kl_influence": [[], [], []], "sup/kl_influence_logp": [[], [], []],
            "sup/kl_influence_keep_one": [[], [], []],
            "sup/kl_influence_base_logp": BASE,
            "sup/kl_influence_visual_all": [[inf.kl_from_logp(BASE[t], NOVIS[t]), NOVIS[t]] for t in range(3)],
            "sup/kl_influence_history_all": [[0.0, BASE[0]], [0.02, ONLY[1][0]], [0.03, BASE[2]]],
            "sup/kl_influence_horizon": [[], [], [[2, 0.004, band]]],
            "kv_visual_new": VIS_NEW,
            "sup/action_probs": [list(np.exp(b)) for b in BASE],
            "action_history": ["forward", "left", "stop"],
            "distance_to_goal": [3.0, 2.0, 1.0],
        }))
        (self.root / "rollout" / "results_0.jsonl").write_text(
            json.dumps({"episode_label": LABEL, "success": 1, "n_steps": 3}) + "\n")

    def test_r1_steps_from_joint_probes(self):
        self._write_r1()
        self.assertEqual(inf.load_r1_influence(self.root, measure="leave_one"), ([], []))  # nothing per-frame to read
        rows, steps = inf.load_r1_influence(self.root, measure="recent_k")
        self.assertEqual(rows, [])
        self.assertEqual([s["step"] for s in steps], [0, 1, 2])
        self.assertEqual([sorted(s["horizon"]) for s in steps], [[0, 1], [0, 1], [0, 1, 2]])
        self.assertAlmostEqual(steps[2]["horizon"][0], inf.kl_from_logp(BASE[2], NOVIS[2]), places=12)
        self.assertEqual((steps[1]["horizon"][1], steps[2]["horizon"][1], steps[2]["horizon"][2]), (0.02, 0.03, 0.004))
        self.assertEqual(steps[1]["horizon_flip"], {0: 1, 1: 1})  # full 'left'; no-visual and current-only say otherwise
        self.assertEqual(steps[2]["horizon_flip"], {0: 0, 1: 0, 2: 0})
        self.assertEqual((steps[2]["argmax_action"], steps[2]["chosen_action"], steps[2]["n_history"]), ("stop", "stop", 2))
        self.assertAlmostEqual(steps[0]["kl_live_vs_replay"], 0.0, places=9)

    def test_uninavid_steps_from_joint_probes(self):
        (self.root / "influence").mkdir(parents=True)
        (self.root / "episodes").mkdir()
        (self.root / "episodes" / f"{LABEL}.json").write_text(json.dumps({
            "episode_label": LABEL, "metrics": {"success": 1.0, "spl": 0.6}, "steps": 4}))
        novis, cur = _logp(.3, .3, .2, .2), _logp(.1, .2, .6, .1)
        rec = {"step": 3, "episode_label": LABEL, "action": "stop", "argmax_action": "stop", "logp_full": BASE[2],
               "p_mass": 0.999, "frames": [], "keep_one": [], "merged_control": [], "keep_one_age0": None,
               "age0": {"kl": 0.2, "logp": novis}, "history_all": {"kl": 0.11, "logp": cur},
               "visual_all": {"kl": 0.3, "logp": novis},
               "horizon": [{"K": 2, "kl": 0.07, "logp": BASE[2]}, {"K": 3, "kl": 0.05, "logp": BASE[2]}],
               "distance_to_goal": 2.0, "probe_seconds": 0.1}
        (self.root / "influence" / f"{LABEL}.jsonl").write_text(json.dumps(rec) + "\n")
        rows, steps = inf.load_uninavid_influence(self.root, measure="recent_k")
        self.assertEqual(rows, [])
        (s,) = steps
        self.assertEqual(s["horizon"], {0: 0.3, 1: 0.11, 2: 0.07, 3: 0.05})
        self.assertEqual(s["horizon_flip"], {0: 0, 1: 1, 2: 0, 3: 0})  # full 'stop'; the current frame alone says 'left'
        self.assertEqual((s["n_history"], s["visual_all"], s["history_all"]), (3, 0.3, 0.11))
        # the other measures do not gain a K = 0 band
        self.assertNotIn(0, inf.load_uninavid_influence(self.root, measure="leave_one")[1][0]["horizon"])

    def test_curve_uses_decisions_with_k_history_and_a_fixed_set(self):
        def step(ep, t, values):
            return {"episode": ep, "step": t, "n_history": t, "horizon": dict(values),
                    "horizon_flip": {k: int(v > 0.25) for k, v in values.items()}}
        steps = [step("a", 0, {0: 0.4, 1: 0.0}),            # t = 0: counts at K = 0 only
                 step("a", 1, {0: 0.6, 1: 0.2}),
                 step("a", 3, {0: 0.8, 1: 0.4, 2: 0.1}),
                 step("b", 2, {0: 0.2, 1: 0.1, 2: 0.05}),
                 step("b", 3, {0: 0.3, 1: 0.3, 2: 0.15, 3: float("nan")})]
        curve = {r["K"]: r for r in inf.recent_window_curve(steps, ks=(0, 1, 2, 3), fixed_min=3)}
        self.assertEqual([curve[k]["n_steps"] for k in (0, 1, 2, 3)], [5, 4, 3, 0])
        self.assertEqual([curve[k]["n_episodes"] for k in (0, 1, 2)], [2, 2, 2])
        self.assertAlmostEqual(curve[0]["mean"], (0.4 + 0.6 + 0.8 + 0.2 + 0.3) / 5)
        self.assertAlmostEqual(curve[0]["balanced_mean"], ((0.4 + 0.6 + 0.8) / 3 + (0.2 + 0.3) / 2) / 2)
        self.assertAlmostEqual(curve[1]["mean"], (0.2 + 0.4 + 0.1 + 0.3) / 4)  # the t = 0 decision is left out
        self.assertAlmostEqual(curve[1]["balanced_mean"], (0.3 + 0.2) / 2)
        self.assertAlmostEqual(curve[1]["balanced_sem"], np.std([0.3, 0.2], ddof=1) / math.sqrt(2))
        self.assertAlmostEqual(curve[2]["balanced_mean"], (0.1 + (0.05 + 0.15) / 2) / 2)
        self.assertAlmostEqual(curve[1]["flip_rate"], 2 / 4)
        self.assertTrue(math.isnan(curve[3]["mean"]) and math.isnan(curve[3]["balanced_mean"]))  # the NaN band is skipped
        # fixed set = the two decisions with >= 3 history frames, at every K
        self.assertEqual([curve[k]["fixed_n_steps"] for k in (0, 1, 2)], [2, 2, 2])
        self.assertAlmostEqual(curve[0]["fixed_mean"], (0.8 + 0.3) / 2)
        self.assertAlmostEqual(curve[2]["fixed_balanced_mean"], (0.1 + 0.15) / 2)
        # normalised: per episode, KL at K over the no-frame KL of the same fixed-set decisions
        self.assertAlmostEqual(curve[0]["norm_balanced_mean"], 1.0)
        self.assertAlmostEqual(curve[2]["norm_balanced_mean"], (0.1 / 0.8 + 0.15 / 0.3) / 2)
        self.assertAlmostEqual(curve[2]["norm_balanced_sem"], np.std([0.1 / 0.8, 0.15 / 0.3], ddof=1) / math.sqrt(2))


def write_fake_uninavid_run(root: Path) -> None:
    """A 2-decision Uni-NaVid masked keep-one run (influence/<label>.jsonl + episodes/<label>.json)."""
    (root / "influence").mkdir(parents=True)
    (root / "episodes").mkdir()
    (root / "episodes" / f"{LABEL}.json").write_text(json.dumps({
        "episode_label": LABEL, "metrics": {"success": 1.0, "spl": 0.6}, "steps": 4}))
    novis = _logp(.3, .3, .2, .2)
    recs = []
    for t in (0, 2):
        base = BASE[t]
        probes = [{"frame": f, "tier": "short", "weight": 1, "logp": ONLY[2][f], "argmax_action": inf._argmax_action(ONLY[2][f]),
                   "kl": inf.kl_from_logp(ONLY[2][f], novis)} for f in range(t)]
        cur = ONLY[t][t]
        recs.append({"step": t, "episode_label": LABEL, "action": "forward", "argmax_action": inf._argmax_action(base),
                     "logp_full": base, "p_mass": 0.999, "frames": [], "horizon": [], "merged_control": [],
                     "history_all": {"kl": 0.01, "logp": base}, "visual_all": {"kl": 0.05, "logp": novis},
                     "visual_all_mask": {"kl": 0.06, "logp": novis}, "keep_one_mode": "mask",
                     "keep_one": probes,
                     "keep_one_age0": {"frame": t, "tier": "short", "logp": cur, "argmax_action": inf._argmax_action(cur),
                                       "kl": inf.kl_from_logp(cur, novis)},
                     "distance_to_goal": 2.0, "probe_seconds": 0.1})
    (root / "influence" / f"{LABEL}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


class UniNavidLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_fake_uninavid_run(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_keep_one_full_rescored_against_logp_full(self):
        novis = _logp(.3, .3, .2, .2)
        krows, ksteps = inf.load_uninavid_influence(self.root, measure="keep_one")
        rows, steps = inf.load_uninavid_influence(self.root, measure="keep_one_full")
        self.assertEqual([(r["step"], r["frame"]) for r in rows], [(2, 0), (2, 1)])
        for kr, r in zip(krows, rows):
            self.assertAlmostEqual(kr["infl"], inf.kl_from_logp(ONLY[2][r["frame"]], novis), places=12)  # unchanged
            self.assertAlmostEqual(r["infl"], inf.kl_from_logp(BASE[2], ONLY[2][r["frame"]]), places=12)
            self.assertEqual(r["logp_full"], BASE[2])
            self.assertEqual(kr["logp_full"], novis)
            self.assertEqual(r["flip"], int(r["argmax_removed"] != inf._argmax_action(BASE[2])))
        # age 0 against the full decision; the ceiling is the masked no-visual prompt rescored the same way
        self.assertAlmostEqual(steps[0]["age0"], inf.kl_from_logp(BASE[0], ONLY[0][0]), places=12)
        self.assertAlmostEqual(steps[1]["age0"], inf.kl_from_logp(BASE[2], ONLY[2][2]), places=12)
        self.assertAlmostEqual(steps[1]["visual_all"], inf.kl_from_logp(BASE[2], novis), places=12)
        self.assertEqual((ksteps[1]["visual_all"], ksteps[1]["age0"]), (0.05, inf.kl_from_logp(ONLY[2][2], novis)))
        self.assertAlmostEqual(steps[1]["sum_loo"], sum(r["infl"] for r in rows), places=12)
        self.assertAlmostEqual(steps[1]["history_all"], 0.01)  # joint probes untouched


if __name__ == "__main__":
    unittest.main()
