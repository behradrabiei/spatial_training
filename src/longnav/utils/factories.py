import ray
import os
from omegaconf import OmegaConf
from longnav.config_schema import *
from longnav.utils.checkpoints import get_base_model, resolve_checkpoint_path, normalize_inference_config

# Use these imports for type hinting
from typing import List, Dict, Any, Iterator, Optional,Union
import logging
import json
thread_cap_env = {
    "env_vars": {
        "OMP_NUM_THREADS": "1", 
        "MKL_NUM_THREADS": "1", 
        "OPENBLAS_NUM_THREADS": "1", 
        "VECLIB_MAXIMUM_THREADS": "1", 
        "NUMEXPR_NUM_THREADS": "1",
        # Optional: Reduce Habitat/Magnum logging spam while we're at it
        "HABITAT_SIM_LOG": "quiet",
        "MAGNUM_LOG": "quiet",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }
}

def save_hydra_config(config, save_dir: str, filename: str = "config.yaml"):
    """
    Saves a Hydra/OmegaConf object to a YAML file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    OmegaConf.save(config=config, f=save_path)
    print(f"📄 Config saved to: {save_path}")

def load_hydra_config(load_dir: str, filename: str = "config.yaml"):
    """
    Loads a Hydra/OmegaConf object from a YAML file.
    Returns None if file is missing.
    """
    load_path = os.path.join(load_dir, filename)
    if not os.path.exists(load_path):
        return None
        
    try:
        conf = OmegaConf.load(load_path)
        return conf
    except Exception as e:
        print(f"⚠️ Failed to load config from {load_path}: {e}")
        return None


    
class InferenceWorkerFactory:
    @staticmethod
    def create(vlm_dict: dict, rollout_dict: dict, res_cfg: ResourceConfig):
        # res_cfg is fine to keep as object for resource logic
        from longnav.utils.rollout_core import InferenceWorker
        
        # We use the dicts directly to avoid pickling issues
        RemoteInferenceWorker = ray.remote(InferenceWorker).options(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env=({"conda": res_cfg.vlm_conda_env} if res_cfg.vlm_conda_env else {}) | thread_cap_env,
            max_restarts=0,        # <--- CRITICAL: Do not restart on crash.
            max_task_retries=-1,
        )

        return [
            RemoteInferenceWorker.remote(
                rollout_config=rollout_dict, 
                **vlm_dict
            ) for _ in range(res_cfg.num_vlms)
        ]
    

class SimWorkerFactory:
    @staticmethod
    def create(sim_dict: dict, res_cfg: ResourceConfig, task_cfg: RunConfig, logger_actor=None,config_overrides=None):
        from longnav.env.habitat import HabitatEnvActor
        env_dict = {}
        if res_cfg.habitat_conda_env is not None:
            env_dict = {"conda": res_cfg.habitat_conda_env}
        # habitat-sim (EGL) can fail to match the CUDA device when the system
        # enumerates both NVIDIA and Mesa EGL vendors ("unable to find CUDA
        # device N among M EGL devices"). Force the NVIDIA EGL vendor and clear
        # DISPLAY so the sim renders headlessly on the GPU.
        sim_env_vars = dict(thread_cap_env.get("env_vars", {}))
        nvidia_egl = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if os.path.exists(nvidia_egl):
            sim_env_vars["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_egl
        sim_env_vars["DISPLAY"] = ""
        sim_runtime_env = env_dict | {"env_vars": sim_env_vars}
        RemoteSim = ray.remote(HabitatEnvActor).options(
            resources={res_cfg.sim_resource_tag: 1},
            num_cpus=res_cfg.sim_cpus,
            num_gpus=res_cfg.sim_gpu_fraction,
            runtime_env=sim_runtime_env,
            max_restarts=0,        # <--- CRITICAL: Do not restart on crash.
            max_task_retries=-1,
        )

        handles = []
        for i in range(res_cfg.num_sims):
            if config_overrides is not None:
                print(f"overriding sim {i} config with: {config_overrides[i]}")
                sim_dict['config_path'] = config_overrides[i]
            # Calculate dynamic per-worker arguments
            log_dir = os.path.join(task_cfg.output_dir, task_cfg.run_name,"rollout")# f'worker_{i}')
            
            # We merge the static sim_dict with our dynamic arguments
            h = RemoteSim.remote(
                **sim_dict,
                logging_output_dir=log_dir,
                logger_actor=logger_actor,
                # Ensure these match your HabitatRayWorker __init__
            )
            handles.append(h)
        return handles

class WandbFactory:
    @staticmethod
    def create(run_cfg: RunConfig, res_cfg: ResourceConfig, full_dict_cfg: dict):
        '''
        Returns a Ray Actor for logging to WandB, and set of episode labels to skip.
        Checks the episode_label column of the existing run (if any) to determine which episodes have already been logged, and returns that as a set to skip.
        '''
        if not run_cfg.wandb_project: 
            return None, set()
        
        from longnav.utils.logging_workers import WandbLoggerActor
        
        RemoteLogger = ray.remote(WandbLoggerActor).options(
            num_cpus=0, 
            runtime_env={"conda": res_cfg.vlm_conda_env}
        )
        import wandb
        api = wandb.Api()
        # fetch latest run id matching name
        id = None
        try:
            runs = api.runs(run_cfg.wandb_project,filters={"displayName":run_cfg.run_name})
            print(f"Found {len(runs)} existing runs with name '{run_cfg.run_name}' in project '{run_cfg.wandb_project}'.")
        except:
            print(f"Could not fetch runs for project '{run_cfg.wandb_project}'. Check your WandB connection and project name.")
            runs = []
        episodes_to_skip = set()
        if len(runs) > 0:
            # Sort runs by creation time descending      
            latest_run = runs[-1] #defualt wandb behavior oldest to newest. we resume from newest
            id = latest_run.id
            print(f"Resuming WandB run '{run_cfg.run_name}' with ID: {id}")
            history = latest_run.history(samples=50000)
            try:
                episodes_to_skip = set(history['episode_label'])
                print(f"Skipping {len(episodes_to_skip)} episodes already logged in WandB.")
            except Exception as e:
                print(f"Could not determine episodes to skip from WandB history: {e}")
        return RemoteLogger.remote(
            wandb_init_kwargs={
                "project": run_cfg.wandb_project,
                "name": run_cfg.run_name,
                "job_type": "eval", # Hardcode or add to RunConfig schema,
                "id": id,
                "resume": "allow",
            },
            run_config=full_dict_cfg
        ),episodes_to_skip
        

class ExpBootstrapper:
    def __init__(self, cfg: InferenceConfig):
        cfg = normalize_inference_config(cfg)
        # Resolve all interpolations (Stage 1)
        # This turns ${read_text:...} into actual file content
        try:
            self.resolved_dict = OmegaConf.to_container(cfg, resolve=True)
        except Exception as e:
            print(f"⚠️ Failed to resolve config interpolations: {e}")
            try:
                from dataclasses import asdict
                self.resolved_dict = asdict(cfg)
            except Exception as e:
                print(f"⚠️ Failed to convert config to dict: {e}")
                self.resolved_dict = {}
        self.typed_cfg = cfg 

    def setup_cluster(self):
        res = self.typed_cfg.resources
        system_config = {
            # "automatic_object_spilling_enabled": False,
        }
        if res.ray_address == "local":
            # Passing a custom `resources` dict can suppress Ray's GPU
            # auto-detection (it may fall back to a phantom TPU), so detect and
            # pass num_gpus explicitly to guarantee GPU actors can schedule.
            try:
                import torch
                num_gpus = torch.cuda.device_count()
            except Exception:
                num_gpus = None
            ray.init(
                resources={
                    res.vlm_resource_tag: res.num_vlms, 
                    res.sim_resource_tag: res.num_sims,
                },
                num_gpus=num_gpus,
                ignore_reinit_error=True,
                object_store_memory = res.osm_gb * 1024 * 1024 * 1024,
                object_spilling_directory = res.object_spilling_directory, _system_config=system_config,
            )
        else:
            ray.init(address=res.ray_address, ignore_reinit_error=True,_system_config=system_config)#,object_store_memory=res.osm_gb * 1024 * 1024 * 1024)
    def bootstrap_logger(self):
        save_hydra_config(self.typed_cfg,os.path.join(self.typed_cfg.task.output_dir,self.typed_cfg.task.run_name))
        return WandbFactory.create(
            self.typed_cfg.task, 
            self.typed_cfg.resources, 
            self.resolved_dict
        )
    
    def bootstrap_vlms_infer(self):
        adapter = self.typed_cfg.adapter
        if adapter.path is not None:
            adapter.path = resolve_checkpoint_path(adapter.path)
            base = get_base_model(adapter.path)
            if base:
                self.resolved_dict['vlm']['model_id'] = base
                self.typed_cfg.vlm.model_id = base
        workers = InferenceWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], rollout_dict=self.resolved_dict['rollout'],
            res_cfg=self.typed_cfg.resources)
        if adapter.path is not None:
            for worker in workers:
                ray.get(worker.load_adapter.remote(adapter.path, adapter.peft_config))
        return workers

    def bootstrap_sims(self,logger=None):
        return SimWorkerFactory.create(
            sim_dict=self.resolved_dict['sim'], 
            res_cfg=self.typed_cfg.resources, 
            task_cfg=self.typed_cfg.task, 
            logger_actor=logger,
            config_overrides = self.typed_cfg.hab_config_list
        )
    
    def bootstrap_eval(self):
        self.setup_cluster()
        
        # 1. Spawn Logger (Pass the FULL resolved dict for WandB hyperparams)
        logger, _ = self.bootstrap_logger()
        
        # 2. Spawn Inference Workers
        # We pass the resolved dictionaries from our resolved_dict
        vlms = self.bootstrap_vlms_infer()
        
        # 3. Spawn Sim Workers
        sims = self.bootstrap_sims(logger)
        
        return vlms, sims, logger
    
def trivial_shard_iterator(n=256) -> Iterator[None]:
    """Yields the trivial shard (None) once. Habitat handles dataset loading."""
    for i in range(n):
        yield None

def chunk_list(all_episodes: List[str], shard_size: int) -> Iterator[List[str]]:
    """Yields chunks of episodes of a specific size."""
    for i in range(0, len(all_episodes), shard_size):
        yield all_episodes[i : i + shard_size]
        
def get_console_logger(name = "DriverMain"):
    """Sets up a central logger and directory structure."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def get_shard_iterator(
    shard_size: int, 
    subset_label: str = "", 
    episode_json: str = "", logger: Optional[logging.Logger] = None,
    excluded_episodes = None
) -> Iterator[Optional[List[str]]]:
    """
    Orchestrates shard creation based on config.
    Reproduces the original branching logic for trivial vs. explicit shards.
    """
    # Case A: Trivial Shard (Let Habitat handle loading via its own config)
    if shard_size <= 0:
        if logger is not None: logger.info("Using trivial shard (full dataset via Habitat config).")
        return trivial_shard_iterator()

    # Case B: Explicit Sharding (We must load the list first)
    all_episodes = []

    if subset_label:
        # Import inside function to avoid circular dependencies or heavy startup
        from longnav.constants import episode_labels_table
        if subset_label in episode_labels_table:
            all_episodes = episode_labels_table[subset_label]
            if logger is not None: logger.info(f"Loaded {len(all_episodes)} episodes from subset: {subset_label}")
        else:
            raise ValueError(f"Subset label '{subset_label}' not found in constants.")

    elif episode_json:
        with open(episode_json, 'r') as f:
            all_episodes = json.load(f)
        if logger is not None: logger.info(f"Loaded {len(all_episodes)} episodes from JSON: {episode_json}")

    else:
        raise ValueError("Shard size > 0 but no episode source (subset_label or episode_json) provided.")

    if not all_episodes:
        raise ValueError("The resolved episode list is empty.")
    if excluded_episodes is not None:
        excluded_episodes = set(excluded_episodes)
        all_episodes = [episode for episode in all_episodes if episode not in excluded_episodes]
        if logger is not None: logger.info(f"After exclusion, {len(all_episodes)} episodes remain for sharding.")
        
    return chunk_list(all_episodes, shard_size)