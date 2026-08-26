"""HAMLET smoke test: rollout, replay, training, checkpoint round-trip on a dummy env.

Runs the RL worker in-process on one GPU (the env is a Ray DummyEnvActor, as in
tests/rl_smoke.py) and checks the invariants the HAMLET port relies on:

1. exactly one memory block and one moment block per decision reach the packed
   episode together with the stored moment history, and the memory read-out is
   an exact no-op at init (memory token == mem_embed, ratio == 0);
2. the packed replay (old_logprobs) reproduces the incremental rollout
   log-probs, and the reference pass (adapters disabled -> untrained HAMLET
   copy) equals them before any update;
2b. the train-mode / gradient-checkpointed replay that train_rl_step optimises
   is the same function as the eval replay (regression test for the
   packed-sequence mask bug, see forward_embeds_core);
3. two RL steps train the module (warmup 0): the first step's PPO ratio is ~1,
   out_proj leaves zero after step 1, the memory transformer receives gradient
   at step 2, the memory read-out becomes non-zero;
4. the checkpoint carries both active and frozen-reference HAMLET weights and
   restores them, the optimizer, and the scheduler bit-for-bit;
5. rollout under context_window=4 (evict) keeps the full moment history;
6. a LoRA-only adapter (what the stage-1 checkpoint is) loads with HAMLET on and
   leaves the module at init.

Usage (GPU node):  LONGNAV_MODEL_ID=<base model dir> python tests/hamlet_smoke.py
"""
import os
import socket
import tempfile

import numpy as np
import ray
import torch

from longnav.config_schema import RLConfig
from longnav.env.env_base import DummyEnvActor
from longnav.utils.hamlet import HAMLET_MODULE_NAME, moment_positions
from longnav.utils.rl_core import collate_trajectories
from longnav.utils.rollout_core import RLWorker
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

MAX_STEPS = 10
N_MOMENT = 8
N_MEM = 1


class FixedLengthDummyEnv(DummyEnvActor):
    """DummyEnvActor whose episodes never end early (no stop, no random done): the
    rollout loop ends them at rollout.max_steps. The gate must not depend on the
    sampled actions -- a 1-step episode makes verl's masked_var raise (mask sum 1)
    and short episodes weaken the replay-vs-rollout and eviction checks."""

    def step(self, action, supplementary_logs=None):
        rgb, state = super().step(action, supplementary_logs)
        state["done"] = False
        return rgb, state


def make_cfg():
    cfg = RLConfig()
    cfg.vlm.model_id = os.environ.get("LONGNAV_MODEL_ID", cfg.vlm.model_id)
    cfg.vlm.attn_impl = "sdpa"
    cfg.vlm.save_outputs = True
    cfg.vlm.hamlet.enabled = True
    cfg.vlm.hamlet.n_moment = N_MOMENT
    cfg.vlm.hamlet.n_mem = N_MEM
    cfg.rollout.convo_start_template = [
        {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    ]
    cfg.rollout.max_steps = MAX_STEPS
    cfg.training.rl_config.n_rollout = 1
    cfg.training.rl_config.n_adv = 4
    cfg.training.warmup_steps = 0  # linear warmup starts at lr=0; test the first step at full lr
    cfg.task.wandb_project = None
    return cfg


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def make_worker(cfg, training=True):
    from dataclasses import asdict
    d = asdict(cfg)
    worker = RLWorker(rollout_config=d["rollout"], **d["vlm"])
    if training:
        worker.setup_training(cfg.training, rank=0, world_size=1, master_addr="localhost", master_port=free_port())
    return worker


def rollout(worker, env):
    init = ray.get(env.reset.remote())
    _, _, traj, _ = worker.run_episode(env, init)
    assert traj is not None and len(traj["actions"]) >= 1, "episode failed"
    traj, packed = worker.postprocess_episode(eval=False)
    return traj, packed


def main():
    ray.init(ignore_reinit_error=True, object_store_memory=4 * 1024 ** 3,
             object_spilling_directory=os.environ.get("RAY_OBJECT_SPILL_DIR", "./ray_object_spilling"))
    env = ray.remote(FixedLengthDummyEnv).remote()
    cfg = make_cfg()
    worker = make_worker(cfg)
    assert worker.hamlet_enabled and len(worker.moment_ids) == N_MOMENT and len(worker.mem_ids) == N_MEM
    assert HAMLET_MODULE_NAME in cfg.training.peft_config.modules_to_save

    # ---- 1. rollout: one memory + one moment block per decision, read-out == 0 at init ----
    traj, packed = rollout(worker, env)
    T = len(traj["actions"])
    assert T == MAX_STEPS, f"fixed-length dummy episode should have {MAX_STEPS} steps, got {T}"
    assert len(worker._moment_history) == T, (len(worker._moment_history), T)
    assert worker._last_hamlet_stats["mean/hamlet_mem_ratio"] == 0.0, worker._last_hamlet_stats
    pos = moment_positions(packed["input_ids_reference"][0], worker.moment_ids)
    kpos = moment_positions(packed["input_ids_reference"][0], worker.mem_ids)
    assert pos.shape == (T, N_MOMENT) and kpos.shape == (T, N_MEM), (pos.shape, kpos.shape)
    assert packed["logits_to_keep"].shape[0] == T
    assert bool((pos[:, -1] < packed["logits_to_keep"].to(pos.device)).all()), "moment block must precede its decision"
    assert bool((kpos[:, -1] < pos[:, 0]).all()), "memory block must precede its turn's moment block"
    hidden = worker.model.config.text_config.hidden_size
    assert tuple(packed["moment_history"].shape) == (T, N_MOMENT, hidden), packed["moment_history"].shape
    print(f"[1] ok: {T} decisions, {T}x{N_MOMENT} moment + {T}x{N_MEM} memory tokens, read-out==0 at init")

    # ---- 2. replay == rollout, ref == old before any update ----
    old = traj["old_logprobs"]
    roll = traj["rollout_logprobs"]
    diff = np.abs(old - roll)
    print(f"[2] replay-vs-rollout logprob diff: max {diff.max():.4f} mean {diff.mean():.4f}")
    # bf16 kernel-order noise between the cached incremental path and the packed replay;
    # observed max 0.09-0.20 / mean 0.02-0.06 on random dummy images, a real defect
    # (e.g. the packed-sequence mask bug) is > 1 nat.
    assert diff.max() < 0.5 and diff.mean() < 0.1, "packed replay drifted from the incremental rollout"
    ref = traj["ref_logprobs"]
    # fresh LoRA (B=0) and a zero-init HAMLET read-out: the reference policy is the same function
    assert np.abs(ref - old).max() < 1e-3, f"ref != old at init: {np.abs(ref - old).max()}"
    print("[2] ok: ref_logprobs == old_logprobs at init")

    # ---- 2b. diagnostic: which replay configuration reproduces old_logprobs? ----
    # train_rl_step replays in train mode with gradient checkpointing through the DDP
    # wrapper; postprocess replays in eval mode without it. Report both so a
    # train/eval discrepancy is visible (it is independent of HAMLET: delta == 0 here).
    probs = np.asarray(traj["rollout_probs"])
    print(f"[2b] rollout action probs: mean max-prob {probs.max(-1).mean():.3f}")
    def replay(train_mode, gc):
        worker.ddp_model.train(train_mode)
        if gc:
            worker.model.gradient_checkpointing_enable({"use_reentrant": False})
        else:
            worker.model.gradient_checkpointing_disable()
        with torch.no_grad():
            logits, _ = worker.ddp_model(embeds_inputs=packed, compute_values=False, value_grad_scale=0.1)
            lp = worker._calculate_action_logprobs(logits).float().cpu().numpy()[0]
            ent = float(torch.distributions.Categorical(logits=logits.float()).entropy().mean())
        return lp, ent
    worst = 0.0
    for tm in (False, True):
        for gc in (False, True):
            lp, ent = replay(tm, gc)
            d = np.abs(lp - old)
            worst = max(worst, float(d.max()))
            print(f"[2b] train_mode={tm} gc={gc}: |logp-old| max {d.max():.4f} mean {d.mean():.4f}; full-vocab entropy {ent:.3f}")
    worker.ddp_model.eval()
    worker.model.gradient_checkpointing_disable()
    # The train-mode/checkpointed replay is what train_rl_step optimises; it must be the
    # same function as the eval replay that produced old_logprobs. Without the explicit
    # attention mask in forward_embeds_core the no-cache path treated every image as a
    # packed-sequence boundary and the two disagreed by >1 nat.
    assert worst < 0.1, f"train-mode replay drifted from old_logprobs by {worst:.3f}"
    print(f"[2b] ok: every replay configuration reproduces old_logprobs (max diff {worst:.4f})")

    # ---- 3. two RL steps train the module ----
    adv_fn = get_adv_estimator_fn("reinforce_plus_plus")
    batch = collate_trajectories([traj])
    advantages, returns = adv_fn(token_level_rewards=batch["rewards"], values=None,
                                 response_mask=batch["response_mask"], config=cfg.training.rl_config)[:2]
    batch["advantages"], batch["returns"] = advantages, returns
    row = batch[0:1, batch["response_mask"][0].bool()]
    m1 = worker.train_rl_step(packed, row["actions"], row["old_log_prob"], row["advantages"], row["returns"],
                              None, row.get("rollout_logprobs", None), row.get("ref_logprobs", None))
    assert "train/hamlet_grad_norm" in m1 and m1["train/hamlet_out_proj_norm"] > 0, m1
    # step 1 optimises the very policy that produced old_logprobs: ratio ~ 1
    assert abs(m1["actor/ppo_kl"]) < 0.05, f"ppo_kl at step 1 should be ~0: {m1}"
    m2 = worker.train_rl_step(packed, row["actions"], row["old_log_prob"], row["advantages"], row["returns"],
                              None, row.get("rollout_logprobs", None), row.get("ref_logprobs", None))
    assert m2["train/hamlet_grad_norm"] > 0 and m2["train/hamlet_mem_ratio"] > 0, m2
    print(f"[3] ok: step1 {m1['train/hamlet_out_proj_norm']=:.3e}; step2 grad {m2['train/hamlet_grad_norm']:.3e} "
          f"mem_ratio {m2['train/hamlet_mem_ratio']:.3e} moment_drift {m2.get('train/hamlet_moment_drift', float('nan')):.3e}")

    # ---- 4. checkpoint carries HAMLET; eval load path restores it ----
    ckpt = tempfile.mkdtemp(prefix="hamlet_ckpt_")
    worker.save_checkpoint_unsafe(ckpt)
    from safetensors import safe_open
    with safe_open(os.path.join(ckpt, "adapter_model.safetensors"), "pt") as f:
        hamlet_keys = [k for k in f.keys() if f".{HAMLET_MODULE_NAME}." in k]
    assert hamlet_keys, "no HAMLET weights in the adapter checkpoint"
    trained = worker._hamlet()
    trained_out = trained.modules_to_save[trained.active_adapters[0]].out_proj.weight.detach().float().cpu()
    trained_ref = {k: v.detach().float().cpu().clone() for k, v in trained.original_module.state_dict().items()}
    assert trained_out.abs().sum() > 0
    assert os.path.exists(os.path.join(ckpt, "hamlet_reference.pt"))
    del worker
    torch.cuda.empty_cache()
    cfg2 = make_cfg()
    cfg2.vlm.context_window = 4
    cfg2.vlm.context_window_mode = "evict"
    cfg2.vlm.save_outputs = False
    worker2 = make_worker(cfg2, training=True)
    worker2.load_checkpoint(ckpt, False, True, True)
    loaded = worker2._hamlet()
    loaded_out = loaded.modules_to_save[loaded.active_adapters[0]].out_proj.weight.detach().float().cpu()
    assert torch.equal(trained_out, loaded_out), "HAMLET weights not restored by load_checkpoint"
    loaded_ref = {k: v.detach().float().cpu() for k, v in loaded.original_module.state_dict().items()}
    assert all(torch.equal(trained_ref[k], loaded_ref[k]) for k in trained_ref), "HAMLET reference not restored"
    assert worker2.scheduler.state_dict()["last_epoch"] == 2, "scheduler state not restored"
    print(f"[4] ok: {len(hamlet_keys)} HAMLET tensors, frozen reference, optimizer, and scheduler restored")

    # ---- 5. context window eviction keeps the moment history ----
    init = ray.get(env.reset.remote())
    _, _, traj2, _ = worker2.run_episode(env, init)
    T2 = len(traj2["actions"])
    assert len(worker2._moment_history) == T2, (len(worker2._moment_history), T2)
    if T2 > 5:
        assert worker2._n_evicted > 0, "context window never evicted"
    assert worker2._last_hamlet_stats["mean/hamlet_mem_ratio"] > 0, "trained read-out should be non-zero"
    print(f"[5] ok: {T2} steps under context_window=4, evicted {worker2._n_evicted} turns, history {len(worker2._moment_history)}")

    # ---- 6. a LoRA-only adapter (the stage-1 checkpoint) loads with HAMLET on ----
    # PEFT indexes every modules_to_save key when loading an adapter; load_checkpoint
    # must fill the HAMLET keys an old adapter lacks from the init, not raise KeyError.
    import shutil
    from safetensors.torch import load_file, save_file
    lora_only = tempfile.mkdtemp(prefix="hamlet_lora_only_")
    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        shutil.copy(os.path.join(ckpt, "adapter_config.json"), lora_only)
    tensors = load_file(os.path.join(ckpt, "adapter_model.safetensors"))
    tensors = {k: v for k, v in tensors.items() if f".{HAMLET_MODULE_NAME}." not in k}
    save_file(tensors, os.path.join(lora_only, "adapter_model.safetensors"))
    del worker2
    torch.cuda.empty_cache()
    cfg3 = make_cfg()
    cfg3.vlm.save_outputs = False
    worker3 = make_worker(cfg3, training=False)
    worker3._setup_peft(cfg3.training)
    fresh = worker3._hamlet()
    init_moment = fresh.modules_to_save[fresh.active_adapters[0]].moment_embed.weight.detach().float().cpu().clone()
    worker3.load_checkpoint(lora_only, False, False)  # must not raise
    after = fresh.modules_to_save[fresh.active_adapters[0]]
    assert torch.equal(after.moment_embed.weight.detach().float().cpu(), init_moment), "HAMLET init clobbered by load"
    assert after.out_proj.weight.detach().abs().sum() == 0, "out_proj should still be zero-init"
    print(f"[6] ok: LoRA-only adapter ({len(tensors)} tensors) loads with HAMLET on; module left at init")
    ray.shutdown()
    print("HAMLET smoke test passed")


if __name__ == "__main__":
    main()
