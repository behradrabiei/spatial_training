"""Bounded Ray smoke: real pretrained inference, two finite dummy simulators, log flushes."""
import json
from pathlib import Path
import tempfile

import numpy as np
import ray

from longnav.config_schema import InferenceConfig
from longnav.utils.factories import ExpBootstrapper
from longnav.utils.rollout_core import collect_rollouts


class FiniteEnv:
    def __init__(self, output):
        self.output = Path(output)
        self.labels = []

    def assign_shard(self, labels):
        self.labels = list(labels)

    def is_exhausted(self):
        return not self.labels

    def observation(self, done=False):
        return np.zeros((240,320,3), dtype=np.uint8), {
            'obs': {'instr_or_goal': 'chair'}, 'reward': 0., 'done': done,
            'is_exhausted': self.is_exhausted(),
            'info': {'episode_label': self.label}}

    def reset(self):
        self.label = self.labels.pop(0)
        self.steps = 0
        return self.observation()

    def step(self, action, supplementary_logs=None):
        self.steps += 1
        return self.observation(done=self.steps == 3 or action == 0)

    def flush_logs_to_disk(self):
        path = self.output / (self.label + '.json')
        path.write_text(json.dumps({'episode_label': self.label, 'steps': self.steps}))
        return str(path)


def main():
    cfg = InferenceConfig()
    cfg.resources.osm_gb = 1
    cfg.resources.vlm_conda_env = None
    cfg.adapter.path = 'Aasdfip/hm3d_rpp_ke_standard-checkpoint_231'
    cfg.rollout.max_steps = 3
    cfg.rollout.deterministic = True
    bootstrapper = ExpBootstrapper(cfg)
    workers, sims = [], []
    with tempfile.TemporaryDirectory(prefix='longnav-smoke-') as output:
        try:
            bootstrapper.setup_cluster()
            workers = bootstrapper.bootstrap_vlms_infer()
            sims = [ray.remote(FiniteEnv).remote(output) for _ in range(2)]
            rows, results, logs = collect_rollouts(sims, workers, iter([['a','b'], ['c']]))
            paths = ray.get(logs)
            assert len(rows) == len(results) == len(paths) == 3
            assert sorted(r['episode_label'] for r in results) == ['a','b','c']
            assert sorted(json.loads(Path(p).read_text())['episode_label'] for p in paths) == ['a','b','c']
            print('PASS: Ray inference, finite scheduling, episode reset, and log flushing')
        finally:
            for actor in workers + sims:
                ray.kill(actor)
            ray.shutdown()


if __name__ == '__main__':
    main()
