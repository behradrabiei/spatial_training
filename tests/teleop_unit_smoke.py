"""CPU-only checks for teleop episode selection, controls, and prediction API."""

import json
import tempfile
from pathlib import Path

import numpy as np

from longnav.scripts.teleop_eval import (
    TeleopEvalConfig,
    interpret_key,
    resolve_episode_labels,
    select_episode_label,
)
from longnav.utils.rollout_core import EpisodeRolloutMixin, substitute_convo_template


def expect_raises(exception_type, function, *args):
    try:
        function(*args)
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}")


cfg = TeleopEvalConfig()
cfg.task.subset_label = "sample20"
labels = resolve_episode_labels(cfg)
assert select_episode_label(labels, 0) == labels[0]
expect_raises(ValueError, select_episode_label, labels, -1)
expect_raises(IndexError, select_episode_label, labels, len(labels))

with tempfile.TemporaryDirectory() as directory:
    episode_file = Path(directory) / "episodes.json"
    episode_file.write_text(json.dumps(["scene_0", "scene_1"]))
    cfg.task.subset_label = ""
    cfg.task.episode_json = str(episode_file)
    assert resolve_episode_labels(cfg) == ["scene_0", "scene_1"]

cfg.task.episode_json = ""
assert resolve_episode_labels(cfg, ["dataset_0"]) == ["dataset_0"]

actions = ["stop", "forward", "left", "right"]
assert interpret_key("w", actions) == ("action", 1)
assert interpret_key("A", actions) == ("action", 2)
assert interpret_key(" ", actions) == ("handoff", None)
assert interpret_key("q", actions) == ("abort", None)
assert interpret_key("r", actions) == ("invalid", None)

turn = [{"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]}]
messages = substitute_convo_template(turn, {"action": "left"})
assert messages[0]["content"][0]["text"] == "**left**"


class FakePredictor(EpisodeRolloutMixin):
    rollout_config = {
        "temperature": 1.0,
        "deterministic": True,
        "stop_prob_threshold": None,
        "visualize_token_filtering": False,
        "visualize_attention": False,
        "visualize_attention_heads": False,
        "visualize_attention_3d": False,
    }

    def infer_probs(self, **kwargs):
        return np.array([0.1, 0.6, 0.2, 0.1]), np.log([0.1, 0.6, 0.2, 0.1]), object()


prediction = FakePredictor().predict_rollout_step(
    np.zeros((8, 8, 3), dtype=np.uint8), messages=[]
)
assert prediction["model_action_id"] == 1
assert "outputs" not in prediction
assert prediction["supplementary_logs"]["action_probs"] == prediction["action_probs"]

print("teleop unit smoke passed")
