"""Record or compare the same inference inputs across the pre-cleanup and current code.

Point PYTHONPATH at the baseline src/ tree when recording the reference. Uses the real
paper adapter, fixed image bytes, two independent repeats, and all retained pruning rules.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from longnav.config_schema import RolloutConfig, VLMConfig
from longnav.utils import rollout_core

CHECKPOINT = 'Aasdfip/hm3d_rpp_ke_standard-checkpoint_231'
START = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Find the chair.'}]},
         {'role': 'user', 'content': [{'type': 'image'}]},
         {'role': 'assistant', 'content': [{'type': 'text', 'text': '**forward**'}]}]
TURN = [{'role': 'assistant', 'content': [{'type': 'text', 'text': '**forward**'}]}, *START[1:]]
PRUNE = dict(context_window_mode='prune', context_cut='frame', context_frames=3,
             kv_prune_candidate_scope='visual', kv_prune_recent_turns=0,
             kv_prune_budget_frames=1., kv_prune_ema_beta=0., kv_prune_seed_per_step=True)
ARMS = {
    'full': {}, 'evict': dict(context_frames=3, context_cut='frame'),
    'recompute': dict(context_frames=3, context_cut='frame', context_window_mode='recompute'),
    'reindex': dict(context_frames=3, context_cut='frame', context_window_mode='reindex'),
    'random': dict(PRUNE, kv_prune_importance='random'),
    'fastv': dict(PRUNE, kv_prune_importance='attn', kv_prune_layer_start=2, kv_prune_score_layers='boundary'),
    'divprune': dict(PRUNE, kv_prune_importance='divprune'),
    'dart': dict(PRUNE, kv_prune_importance='dart', kv_prune_layer_start=2),
    'pyramiddrop': dict(PRUNE, context_frames=None, context_cut='header', kv_prune_stages=[0,7,14,21], kv_prune_budget_scope='model', kv_prune_schedule_anchor='pool'),
    'hybrid': dict(PRUNE, kv_prune_stages=[0,7,14,20], kv_prune_stage_importance=['attn','attn','attn','random'], kv_prune_tail_budget_frames=1., kv_prune_budget_scope='model', kv_prune_schedule_anchor='pool'),
    'uniform': dict(PRUNE, kv_prune_stages=[0,18], kv_prune_stage_importance=['random','random'], kv_prune_visual_blind=True, kv_prune_budget_scope='model', kv_prune_schedule_anchor='pool'),
    'blind': dict(context_window_mode='prune', kv_prune_visual_blind=True, kv_prune_layer_start=18),
}


def build(overrides):
    from longnav.utils.factories import resolve_checkpoint_path, get_base_model
    path = resolve_checkpoint_path(CHECKPOINT)
    cfg = asdict(VLMConfig()) | overrides
    cfg['model_id'] = get_base_model(path)
    if hasattr(rollout_core, 'InferenceWorker'):
        worker = rollout_core.InferenceWorker(asdict(RolloutConfig()), **cfg)
        worker.load_adapter(path)
    else:
        from longnav.config_schema import VLMTrainingConfig
        worker = rollout_core.RLWorker(asdict(RolloutConfig()), **cfg)
        worker._setup_peft(VLMTrainingConfig())
        worker.load_checkpoint(path, False, False)
    return worker


def mask_digest(worker):
    state = getattr(worker, '_reindex_state', None)
    ps = getattr(worker, '_kv_prune_state', None)
    tensors = [getattr(state, k, None) for k in ('alive', 'pos_table')]
    tensors += [getattr(ps, k, None) for k in ('turn_id', 'is_visual', 'raster')]
    return [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() if isinstance(t, torch.Tensor) else None for t in tensors]


def record():
    rng = np.random.RandomState(0)
    images = [Image.fromarray(rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in range(8)]
    results = {}
    for name, config in ARMS.items():
        results[name] = []
        for repeat in range(2):
            torch.manual_seed(17)
            np.random.seed(17)
            worker = build(config)
            worker.begin_episode('cleanup_parity_0')
            steps = []
            for i, image in enumerate(images):
                probs, _, _ = worker.infer_probs(START if i == 0 else TURN, [image])
                steps.append({'probs': probs.tolist(), 'action': int(np.argmax(probs)),
                              'cache_lengths': [int(l.keys.shape[-2]) for l in worker.past_key_values.layers],
                              'masks': mask_digest(worker)})
            worker.reset()
            assert worker.past_key_values is None
            results[name].append(steps)
            del worker
            gc.collect()
            torch.cuda.empty_cache()
        print('recorded', name, flush=True)
    return results


def compare(reference, current):
    worst = 0.
    for arm, repeats in reference.items():
        a = np.array([s['probs'] for s in repeats[0]])
        b = np.array([s['probs'] for s in repeats[1]])
        tolerance = max(float(np.abs(a-b).max()), 1e-7)
        for old, new in zip(repeats, current[arm]):
            for x, y in zip(old, new):
                assert {k:v for k,v in x.items() if k != 'probs'} == {k:v for k,v in y.items() if k != 'probs'}, arm
                delta = float(np.max(np.abs(np.array(x['probs'])-y['probs'])))
                assert delta <= tolerance, (arm, delta, tolerance)
                worst = max(worst, delta)
    print('PARITY PASS; maximum probability difference:', worst)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()
    results = record()
    args.output.write_text(json.dumps(results, indent=2))
    if args.reference:
        compare(json.loads(args.reference.read_text()), results)
