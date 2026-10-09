"""Run one canonical HM3D episode through Ray and compare actions and Habitat metrics.

The reference can be recorded with PYTHONPATH pointing at bf4e65ff's src tree and
its archived verl dependency. Uses the existing dataset and paper checkpoint.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import ray

from longnav.utils import factories
from longnav.utils.rollout_core import collect_rollouts


def compare(old, record):
    assert old['result'] == record['result'], (old['result'], record['result'])
    assert old['trajectory'].keys() == record['trajectory'].keys()
    for key, value in old['trajectory'].items():
        if np.asarray(value).dtype.kind in 'OUS':
            np.testing.assert_array_equal(value, record['trajectory'][key], err_msg=key)
        else:
            np.testing.assert_allclose(value, record['trajectory'][key], rtol=0, atol=1e-7, err_msg=key)
    print('HABITAT PARITY PASS', record['result'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--compare-only', action='store_true')
    args = ap.parse_args()
    if args.compare_only:
        compare(json.loads(args.reference.read_text()), json.loads(args.output.read_text()))
        return
    from longnav.conf.register_configs import register_configs
    register_configs()
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / 'paper_results/ctx_ablation/runs/r1_full/config.yaml')
    cfg.resources.num_vlms = cfg.resources.num_sims = 1
    cfg.resources.osm_gb = 2
    cfg.resources.vlm_conda_env = 'longnav_vlm'
    cfg.resources.object_spilling_directory = str(args.output.parent.resolve() / 'ray-spill')
    cfg.task.wandb_project = None
    cfg.task.output_dir = str(args.output.parent.resolve())
    cfg.task.run_name = args.output.stem + '-logs'
    cfg.rollout.visualize_attention_3d = False
    cfg.rollout.visualize_token_filtering = False
    cfg.sim.visualize_attn3d = False
    cfg.sim.add_top_down_map = False
    factories.thread_cap_env['env_vars']['PYTHONPATH'] = os.environ.get('PYTHONPATH', str(root / 'src'))
    factories.thread_cap_env['env_vars']['HF_HUB_OFFLINE'] = '1'
    boot = factories.ExpBootstrapper(cfg)
    workers, sims = [], []
    try:
        boot.setup_cluster()
        workers = (boot.bootstrap_vlms_infer() if not hasattr(boot, 'bootstrap_vlms_rl')
                   else boot.bootstrap_vlms_rl(training=False))
        sims = boot.bootstrap_sims()
        trajectories, results, logs = collect_rollouts(
            sims, workers, iter([['4ok3usBNeis_0']]),
            postprocess_kwargs={'return_inputs': False, 'eval': True})
        ray.get(logs)
        assert len(trajectories) == len(results) == len(logs) == 1
        trajectory = trajectories[0][0]
        record = {'result': results[0], 'trajectory': {k: v.tolist() for k,v in trajectory.items()}}
        args.output.write_text(json.dumps(record, indent=2))
        if args.reference:
            compare(json.loads(args.reference.read_text()), record)
        else:
            print('HABITAT REFERENCE', record['result'], flush=True)
    finally:
        for actor in workers + sims:
            ray.kill(actor)
        ray.shutdown()


if __name__ == '__main__':
    main()
