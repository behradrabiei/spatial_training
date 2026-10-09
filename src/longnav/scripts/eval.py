"""Evaluate a pretrained navigation model; no training runtime is imported."""
import hydra

from longnav.conf.register_configs import register_configs
from longnav.config_schema import InferenceConfig

register_configs()


@hydra.main(version_base=None, config_name='inference_config', config_path='../config')
def main(cfg: InferenceConfig):
    import ray
    from longnav.utils.factories import ExpBootstrapper, get_shard_iterator
    from longnav.utils.rollout_core import collect_rollouts

    bootstrapper = ExpBootstrapper(cfg)
    workers, sims, logger = [], [], None
    try:
        bootstrapper.setup_cluster()
        workers = bootstrapper.bootstrap_vlms_infer()
        logger, excluded = bootstrapper.bootstrap_logger()
        sims = bootstrapper.bootstrap_sims(logger)
        shards = get_shard_iterator(
            subset_label=cfg.task.subset_label, episode_json=cfg.task.episode_json,
            shard_size=cfg.task.shard_size, excluded_episodes=excluded)
        # One collection owns the reset/prefetch queues until the split is exhausted.
        # Recreating those queues between batches can discard prefetched episodes.
        _, _, log_refs = collect_rollouts(sims, workers, shards)
        ray.get(log_refs)
    finally:
        for actor in workers + sims:
            ray.kill(actor)
        if logger is not None:
            ray.get(logger.close.remote())
            ray.kill(logger)
        ray.shutdown()


if __name__ == '__main__':
    main()
