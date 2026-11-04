"""Stateful /act API for pretrained navigation inference.

One server process owns one conversation (one episode at a time); a client such as the NavVerse
LongNav agent posts one frame per step and names the goal on the first request of an episode.

    python -m longnav.scripts.serve --port 8000 --experiment paper_ctx \
        --set vlm.context_frames=8 --set vlm.context_window_mode=evict \
        --config-out run/server_config.json --step-log run/server_steps.jsonl

``--experiment`` applies the ``vlm:`` and ``rollout:`` sections of
``src/longnav/config/experiment/<name>.yaml`` (the sim/task sections are Habitat-only) on top of
the dataclass defaults; ``--set section.key=value`` overrides are applied after it and typed
from the dataclass field (``null`` -> None, ``true``/``false`` -> bool). The effective
configuration is served at ``GET /health`` and, with ``--config-out``, written as JSON.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
import io
import json
from pathlib import Path
from threading import Lock
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

from longnav.config_schema import RolloutConfig, VLMConfig
from longnav.utils.checkpoints import get_base_model, resolve_checkpoint_path
from longnav.utils.rollout_core import InferenceWorker, substitute_convo_template, select_action

CHECKPOINT = 'Aasdfip/hm3d_rpp_ke_standard-checkpoint_231'
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PACKAGE_ROOT / 'conf/prompts/objectnav_prompt.txt'
EXPERIMENT_DIR = PACKAGE_ROOT / 'config/experiment'
SECTIONS = {'vlm': VLMConfig, 'rollout': RolloutConfig}


def _coerce(section, key, raw):
    """Type a command-line value from the dataclass field it targets."""
    field = {f.name: f for f in fields(SECTIONS[section])}.get(key)
    if field is None:
        raise ValueError(f'Unknown option {section}.{key}')
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.lower() in ('null', 'none'):
        return None
    if text.lower() in ('true', 'false'):
        return text.lower() == 'true'
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    try:
        return json.loads(text)  # lists / dicts
    except ValueError:
        return text


def apply_overrides(vlm, rollout, experiment=None, sets=()):
    """Return ``(vlm, rollout)`` dicts with an experiment preset and ``section.key=value`` sets applied."""
    vlm, rollout = dict(vlm), dict(rollout)
    targets = {'vlm': vlm, 'rollout': rollout}
    if experiment:
        import yaml
        path = EXPERIMENT_DIR / f'{experiment}.yaml'
        if not path.is_file():
            raise FileNotFoundError(f'No experiment preset {path}')
        preset = yaml.safe_load(path.read_text()) or {}
        for section in SECTIONS:
            for key, value in (preset.get(section) or {}).items():
                targets[section][key] = _coerce(section, key, value)
    for item in sets:
        if '=' not in item or '.' not in item.split('=', 1)[0]:
            raise ValueError(f'Expected section.key=value, got {item!r}')
        dotted, value = item.split('=', 1)
        section, key = dotted.split('.', 1)
        if section not in SECTIONS:
            raise ValueError(f'Unknown section {section!r} in {item!r} (expected vlm or rollout)')
        targets[section][key] = _coerce(section, key, value)
    return vlm, rollout


def resolve_config(checkpoint=CHECKPOINT, experiment=None, sets=()):
    """Effective ``{checkpoint, vlm, rollout}`` for ``build_worker``; the prompt is substituted here."""
    checkpoint = resolve_checkpoint_path(checkpoint)
    vlm, rollout = apply_overrides(asdict(VLMConfig()), asdict(RolloutConfig()), experiment, sets)
    vlm['model_id'] = get_base_model(checkpoint) or vlm['model_id']
    # Strip like the Hydra path's read_text resolver does (src/longnav/conf/register_configs.py);
    # the un-stripped read fed the model one extra newline token in the 2026-10-20/22 NavVerse sweep.
    prompt = PROMPT_PATH.read_text().strip()
    rollout['system_prompt'] = prompt
    rollout['convo_start_template'][0]['content'][0]['text'] = prompt
    return {'checkpoint': checkpoint, 'vlm': vlm, 'rollout': rollout}


def build_worker(checkpoint=CHECKPOINT, experiment=None, sets=(), config=None):
    config = config or resolve_config(checkpoint, experiment, sets)
    worker = InferenceWorker(config['rollout'], **config['vlm'])
    worker.load_adapter(config['checkpoint'])
    return worker


def infer_step(worker, rgb_pil, instr_or_goal=None):
    if instr_or_goal is not None:
        worker.reset()
        worker.messages = substitute_convo_template(
            worker.rollout_config['convo_start_template'],
            worker.rollout_config | {'instr_or_goal': instr_or_goal})
    if not hasattr(worker, 'messages'):
        raise ValueError('The first request must include instr_or_goal')
    probs, _, _ = worker.infer_probs(
        images=[rgb_pil], messages=worker.messages,
        temperature=worker.rollout_config['temperature'], pos_id_kwargs=None)
    action, _ = select_action(probs, deterministic=worker.rollout_config.get('deterministic', False),
                             stop_prob_threshold=worker.rollout_config.get('stop_prob_threshold'))
    worker.messages = substitute_convo_template(worker.rollout_config['convo_turn_template'],
                                                {'action': worker.rollout_config['action_space'][action]})
    return action, probs


def _peak_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return None


def create_app(worker_factory=build_worker, info=None, step_log=None):
    """``info`` is served at /health; ``step_log`` (path) receives one JSON line per /act call."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.worker = worker_factory()
        app.state.lock = Lock()
        app.state.episode = 0  # increments on every goal (= episode start)
        app.state.step = 0
        app.state.goal = None
        app.state.step_log = open(step_log, 'a') if step_log else None
        yield
        if app.state.step_log:
            app.state.step_log.close()
        del app.state.worker

    application = FastAPI(lifespan=lifespan)

    @application.get('/health')
    def health():
        return {'ok': True, **(info or {})}

    @application.post('/reset')
    def reset():
        with application.state.lock:
            worker = application.state.worker
            worker.reset()
            if hasattr(worker, 'messages'):
                del worker.messages
            application.state.goal = None
            application.state.step = 0
        return {'ok': True}

    @application.post('/act')
    def act(file: UploadFile = File(...), instr_or_goal: Optional[str] = Form(None)):
        from fastapi import HTTPException
        rgb = Image.open(io.BytesIO(file.file.read())).convert('RGB')
        # A single worker owns one conversation/cache; concurrent requests must serialize.
        with application.state.lock:
            worker = application.state.worker
            if instr_or_goal is None and not hasattr(worker, 'messages'):
                raise HTTPException(400, 'The first request must include instr_or_goal')
            state = application.state
            if instr_or_goal is not None:
                state.episode += 1
                state.step = 0
                state.goal = instr_or_goal
            t0 = time.time()
            action, probs = infer_step(worker, rgb, instr_or_goal)
            latency_ms = (time.time() - t0) * 1000.0
            state.step += 1
            if state.step_log:
                kv = worker.kv_stats() if hasattr(worker, 'kv_stats') else None
                record = {'episode': state.episode, 'goal': state.goal, 'step': state.step,
                          'latency_ms': round(latency_ms, 2), 'action_id': int(action),
                          'action_probs': [round(float(p), 6) for p in probs],
                          'kv_len': None if kv is None else kv[0],
                          'kv_len_layer_max': None if kv is None else kv[1],
                          'peak_mem_GB': _peak_mem_gb(), 'time': time.time()}
                state.step_log.write(json.dumps(record) + '\n')
                state.step_log.flush()
        return {'action_id': int(action), 'action_probs': probs.tolist()}

    return application


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--experiment', default=None, help='preset name under src/longnav/config/experiment/')
    ap.add_argument('--set', dest='sets', action='append', default=[], metavar='SECTION.KEY=VALUE',
                    help='override a vlm.* or rollout.* option (repeatable)')
    ap.add_argument('--config-out', type=Path, default=None, help='write the effective config as JSON')
    ap.add_argument('--step-log', type=Path, default=None, help='append one JSON line per /act call')
    return ap.parse_args(argv)


def main(argv=None):
    import uvicorn
    args = _parse_args(argv)
    config = resolve_config(args.checkpoint, args.experiment, args.sets)
    info = {'host': args.host, 'port': args.port, 'checkpoint': config['checkpoint'],
            'experiment': args.experiment, 'sets': list(args.sets),
            'model_id': config['vlm']['model_id'],
            'context_frames': config['vlm']['context_frames'],
            'context_window': config['vlm']['context_window'],
            'context_window_mode': config['vlm']['context_window_mode'],
            'context_cut': config['vlm']['context_cut'],
            'use_sparse': config['vlm']['use_sparse'],
            'deterministic': config['rollout']['deterministic'],
            'stop_prob_threshold': config['rollout']['stop_prob_threshold']}
    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps({'info': info, **config}, indent=2, default=str) + '\n')
    if args.step_log:
        args.step_log.parent.mkdir(parents=True, exist_ok=True)
    app = create_app(lambda: build_worker(config=config), info=info, step_log=args.step_log)
    uvicorn.run(app, host=args.host, port=args.port)


app = create_app()

if __name__ == '__main__':
    main()
