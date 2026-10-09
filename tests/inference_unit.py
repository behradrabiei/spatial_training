"""Inference configuration, API, and scheduling regressions; no GPU or model download."""
import importlib.abc
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from longnav.config_schema import InferenceConfig
from longnav.utils.checkpoints import normalize_inference_config


class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('verl', 'trl'):
            raise AssertionError('Inference imported training dependency: ' + fullname)


class ConfigTests(unittest.TestCase):
    def test_removed_options_are_rejected(self):
        from longnav.utils.vlm_worker import VLMWorker
        for method in ('stratified', 'diversity', 'grid', 'voxel', 'voxel_stratified',
                       'kl', 'keep_one', 'fisher', 'fisher_diversity'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'kv_prune_importance'):
                VLMWorker(load_model=False, kv_prune_importance=method)
        for key, value in [('kv_prune_merge', True), ('kv_prune_granularity', 'turn'),
                           ('kv_prune_fisher_pool_factor', 3.0), ('save_outputs', True)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'removed'):
                normalize_inference_config(OmegaConf.create({'vlm': {key: value}}))

    def test_in_process_bootstrap_resolves_prompt(self):
        from longnav.utils.factories import ExpBootstrapper
        boot = ExpBootstrapper(InferenceConfig())
        self.assertNotIn('${', boot.resolved_dict['rollout']['system_prompt'])
        self.assertEqual(boot.resolved_dict['rollout']['convo_start_template'][0]['content'][0]['text'],
                         boot.resolved_dict['rollout']['system_prompt'])

    def test_legacy_checkpoint(self):
        cfg = OmegaConf.structured(InferenceConfig)
        cfg.training.checkpoint = '/tmp/adapter'
        normalized = normalize_inference_config(cfg)
        self.assertEqual(normalized.adapter.path, '/tmp/adapter')

    def test_conflicting_checkpoints(self):
        cfg = OmegaConf.structured(InferenceConfig)
        cfg.training.checkpoint = 'old'
        cfg.adapter.path = 'new'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            normalize_inference_config(cfg)

    def test_saved_paper_config(self):
        saved = OmegaConf.load('paper_results/hybrid_prune/runs/r1_hybrid_mp16/config.yaml')
        normalized = normalize_inference_config(saved)
        self.assertEqual(normalized.adapter.path, saved.training.checkpoint)
        self.assertEqual(list(normalized.vlm.kv_prune_stages), [0, 7, 14, 20])

    def test_experiment_presets(self):
        from hydra import compose, initialize_config_dir
        from longnav.conf.register_configs import register_configs
        register_configs()
        with initialize_config_dir(config_dir=str(Path('src/longnav/config').resolve()), version_base=None):
            for p in Path('src/longnav/config/experiment').glob('*.yaml'):
                cfg = compose(config_name='inference_config', overrides=['+checkpoint=longnav', '+dataset=hm3d_v2_val', '+experiment='+p.stem, '+resources=single'])
                normalize_inference_config(cfg)


class FakeWorker:
    def __init__(self):
        from dataclasses import asdict
        from longnav.config_schema import RolloutConfig
        self.rollout_config = asdict(RolloutConfig())
        self.rollout_config['deterministic'] = True
        self.rollout_config['convo_start_template'][0]['content'][0]['text'] = 'Find $instr_or_goal'
        self.reset_count = 0
        self.turns = []

    def reset(self):
        self.reset_count += 1

    def infer_probs(self, **kwargs):
        self.turns.append(kwargs['messages'])
        return np.array([.01,.9,.05,.04]), None, None


class ServerTests(unittest.TestCase):
    def test_startup_requests_and_reset(self):
        from fastapi.testclient import TestClient
        from longnav.scripts.serve import create_app
        worker = FakeWorker()
        builds = []
        def factory():
            builds.append(True)
            return worker
        app = create_app(factory)
        self.assertEqual(builds, [])
        image = io.BytesIO()
        Image.new('RGB', (10,10)).save(image, format='PNG')
        with TestClient(app) as client:
            files = {'file': ('frame.png', image.getvalue(), 'image/png')}
            self.assertEqual(client.post('/act', files=files).status_code, 400)
            response = client.post('/act', files=files, data={'instr_or_goal':'chair'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['action_id'], 1)
            client.post('/act', files=files)
            self.assertEqual(worker.reset_count, 1)
            client.post('/act', files=files, data={'instr_or_goal':'table'})
            self.assertEqual(worker.reset_count, 2)
            self.assertIn('table', str(worker.turns[-1]))
        self.assertEqual(len(builds), 1)


class Ref:
    def __init__(self, value): self.value = value


class RemoteMethod:
    def __init__(self, fn): self.fn = fn
    def remote(self, *args, **kwargs): return Ref(self.fn(*[a.value if isinstance(a,Ref) else a for a in args], **kwargs))


class FakeRay:
    @staticmethod
    def get(ref): return [r.value for r in ref] if isinstance(ref,list) else ref.value
    @staticmethod
    def wait(refs, **kwargs): return refs[:1], refs[1:]


class Actor:
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return RemoteMethod(value) if name in ('assign_shard','is_exhausted','reset','run_episode','postprocess_episode','flush_logs_to_disk') else value


class Sim(Actor):
    def __init__(self): self.labels=[]; self.flushed=[]
    def assign_shard(self, labels): self.labels=list(labels)
    def is_exhausted(self): return not self.labels
    def reset(self):
        self.current=self.labels.pop(0)
        return None, {'info':{'episode_label':self.current}}
    def flush_logs_to_disk(self):
        self.flushed.append(self.current)
        return self.current


class Worker(Actor):
    def run_episode(self, sim, state):
        self.label=state[-1]['info']['episode_label']
        return not sim.labels, {'episode_label':self.label}
    def postprocess_episode(self, **kwargs): return self.label, None, None


class SchedulingTests(unittest.TestCase):
    def test_exactly_once_and_flush(self):
        from longnav.utils import rollout_core
        sims = [Sim(),Sim()]
        with patch.object(rollout_core,'ray',FakeRay):
            rows, results, logs = rollout_core.collect_rollouts(sims,[Worker(),Worker()],iter([['a','b'],['c'],['d','e']]))
        self.assertEqual(len(rows),5)
        self.assertEqual(sorted(r['episode_label'] for r in results),list('abcde'))
        self.assertEqual(sorted(x for s in sims for x in s.flushed),list('abcde'))
        self.assertEqual([r['episode_label'] for r in results],FakeRay.get(logs))

    def test_empty_split(self):
        from longnav.utils import rollout_core
        with patch.object(rollout_core,'ray',FakeRay):
            self.assertEqual(rollout_core.collect_rollouts([Sim()],[Worker()],iter([])),([],[],[]))


if __name__ == '__main__':
    sys.meta_path.insert(0, NoTrainingImports())
    unittest.main()
