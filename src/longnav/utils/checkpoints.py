"""Inference-only checkpoint resolution and adapter loading."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


def get_base_model(checkpoint):
    path = Path(checkpoint) / 'adapter_config.json'
    return json.loads(path.read_text()).get('base_model_name_or_path') if path.exists() else None


def resolve_checkpoint_path(path_or_id):
    if Path(path_or_id).exists():
        return str(path_or_id)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=path_or_id,
                             allow_patterns=['*.json', '*.bin', '*.safetensors', '*.yaml'],
                             ignore_patterns=['optimizer*', 'scheduler*', '*.msgpack', '*.h5'])


def normalize_inference_config(config):
    """Translate historical checkpoint fields without importing training support.

    Serialized historical optimizer/loss settings are discarded, never executed. Removed
    pruning knobs are accepted only at their inert historical values.
    """
    from omegaconf import OmegaConf
    from longnav.config_schema import InferenceConfig
    cfg = OmegaConf.create(asdict(config)) if is_dataclass(config) else OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    legacy = cfg.pop('training', None)
    adapter = OmegaConf.to_container(cfg.adapter, resolve=True) if 'adapter' in cfg else {}
    if legacy:
        for old, new in [('checkpoint', 'path'), ('peft_config', 'peft_config')]:
            value = legacy.get(old)
            if value is None:
                continue
            value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            existing = adapter.get(new)
            defaults = asdict(InferenceConfig().adapter).get(new)
            if existing is not None and existing != defaults and existing != value:
                raise ValueError(f'Conflicting training.{old} and adapter.{new}')
            adapter[new] = value
    cfg['adapter'] = adapter
    for key in ('master_addr', 'master_port'):
        cfg.get('resources', {}).pop(key, None)
    defaults = {'save_outputs': False, 'kv_prune_merge': False, 'kv_prune_granularity': 'slot',
                'kv_prune_fisher_pool_factor': 2.0, 'kv_prune_voxel_cap': 1,
                'kv_prune_voxel_scale': 2, 'kv_prune_voxel_2d': True}
    for key, default in defaults.items():
        if key in cfg.get('vlm', {}):
            value = cfg.vlm.pop(key)
            if value != default:
                raise ValueError(f'{key}={value!r} belongs to a removed exploratory method')
    return OmegaConf.merge(OmegaConf.structured(InferenceConfig), cfg)


def load_adapter(worker, path, peft_config=None):
    """Apply the paper adapter with the same PEFT wrapping and merge timing as before."""
    from peft import LoraConfig, get_peft_model
    from peft.utils import load_peft_weights, set_peft_model_state_dict
    from longnav.config_schema import HydraLoraConfig
    from omegaconf import OmegaConf
    path = resolve_checkpoint_path(path)
    config = peft_config if peft_config is not None else HydraLoraConfig()
    kwargs = asdict(config) if is_dataclass(config) else (
        OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config))
    # Historical paper adapters were wrapped with this modules_to_save entry even
    # when no value head existed. Preserve the wrapper exactly; no head is created.
    kwargs['modules_to_save'] = list(kwargs.get('modules_to_save') or [])
    if 'value_head' not in kwargs['modules_to_save']:
        kwargs['modules_to_save'].append('value_head')
    for key in ('target_modules', 'modules_to_save'):
        if kwargs.get(key) is not None:
            kwargs[key] = list(kwargs[key])
    if not any((Path(path) / name).is_file() for name in ('adapter_model.bin', 'adapter_model.safetensors')):
        raise FileNotFoundError(f'No adapter weights in {path}')
    worker.model = get_peft_model(worker.model, LoraConfig(**kwargs))
    set_peft_model_state_dict(worker.model, load_peft_weights(path))
    worker.model.eval()
    worker._is_lora = worker._is_merged = None
    if worker.attn_probe is not None:
        worker.attn_probe.invalidate_factors()
