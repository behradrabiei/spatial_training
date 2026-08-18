"""Smoke test for the multi-object conversation path: goal switches mid-episode
must be delivered through rollout.convo_goal_template (text + image turn) without
breaking the incremental KV pipeline (tokenize -> sandwich crop -> pos ids ->
sparse filtering). No habitat needed."""
from longnav.config_schema import *
from longnav.utils.factories import ExpBootstrapper,get_shard_iterator
from longnav.env.env_base import MultiGoalDummyEnvActor
from longnav.utils.rollout_core import collect_rollouts
import ray
import sys

FLUSH = "--flush" in sys.argv  # exercise flush_on_goal_switch (mid-episode VLM reset) instead of the goal turn

cfg = RLConfig()
cfg.resources.osm_gb=28
cfg.resources.vlm_conda_env=None
cfg.vlm.attn_impl = "sdpa"
# outside hydra the ${read_text:...}/${rollout...} interpolations don't resolve,
# so use literal template texts (same convention as eval_smoke.py)
cfg.rollout.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": "Find \"$instr_or_goal\". Each step, output an action inside double asterisks."}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]
cfg.rollout.convo_goal_template=[
        {"role": "user", "content": [{"type": "text", "text": "You have found the previous target. Your new target is \"$instr_or_goal\". The same rules apply."}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]

cfg.rollout.flush_on_goal_switch = FLUSH

bootstrapper = ExpBootstrapper(cfg)
bootstrapper.setup_cluster()

vlms = bootstrapper.bootstrap_vlms_rl(training=False)
sims = [ray.remote(MultiGoalDummyEnvActor).remote() for _ in range(2)]

rollout_list,result_list,log_list = collect_rollouts(sims,vlms,get_shard_iterator(0),8,{"return_inputs":False,"eval":True})

assert all(r is not None for r in result_list), f"episodes crashed: {result_list}"
assert all(r['goal_idx'] == 2 for r in result_list), f"third goal never reached: {result_list}"
assert all(r['final_instr_or_goal'] == "blue chair" for r in result_list), result_list
assert all(r['instr_or_goal'] == "red cube" for r in result_list), result_list
mode = "flush restarts" if FLUSH else "goal-switch turns"
print(f"\nOK: {len(result_list)} episodes, each delivered 2 {mode} and reached goal 3")
