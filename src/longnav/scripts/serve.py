"""Stateful /act API for pretrained navigation inference."""
from contextlib import asynccontextmanager
from dataclasses import asdict
import io
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

from longnav.config_schema import RolloutConfig, VLMConfig
from longnav.utils.checkpoints import get_base_model, resolve_checkpoint_path
from longnav.utils.rollout_core import InferenceWorker, substitute_convo_template, select_action

CHECKPOINT = 'Aasdfip/hm3d_rpp_ke_standard-checkpoint_231'


def build_worker():
    checkpoint = resolve_checkpoint_path(CHECKPOINT)
    vlm = asdict(VLMConfig())
    vlm['model_id'] = get_base_model(checkpoint) or vlm['model_id']
    rollout = asdict(RolloutConfig())
    prompt = (Path(__file__).resolve().parents[1] / 'conf/prompts/objectnav_prompt.txt').read_text()
    rollout['system_prompt'] = prompt
    rollout['convo_start_template'][0]['content'][0]['text'] = prompt
    worker = InferenceWorker(rollout, **vlm)
    worker.load_adapter(checkpoint)
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


def create_app(worker_factory=build_worker):
    @asynccontextmanager
    async def lifespan(app):
        app.state.worker = worker_factory()
        app.state.lock = Lock()
        yield
        del app.state.worker

    application = FastAPI(lifespan=lifespan)

    @application.post('/act')
    def act(file: UploadFile = File(...), instr_or_goal: Optional[str] = Form(None)):
        from fastapi import HTTPException
        rgb = Image.open(io.BytesIO(file.file.read())).convert('RGB')
        # A single worker owns one conversation/cache; concurrent requests must serialize.
        with application.state.lock:
            worker = application.state.worker
            if instr_or_goal is None and not hasattr(worker, 'messages'):
                raise HTTPException(400, 'The first request must include instr_or_goal')
            action, probs = infer_step(worker, rgb, instr_or_goal)
        return {'action_id': int(action), 'action_probs': probs.tolist()}

    return application


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
