To run eval on full hm3d v2 run:
cd /home/brabiei/Projects/World-Modelling/spatial_training
conda activate longnav_vlm

python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources=single \
  task.run_name=hm3d_v2_val_full \
  task.subset_label="" \
  task.episode_json=$PWD/dump/hm3d_v2_val_labels.json \
  task.shard_size=6

To run full-val HM3D v1 / MP3D with full context vs window-32 (evict), viz off, argmax:
cd /home/brabiei/Projects/World-Modelling/spatial_training
conda activate longnav_vlm

VIZ_OFF=(
  rollout.visualize_token_filtering=false
  rollout.visualize_attention=false
  rollout.visualize_attention_heads=false
  rollout.visualize_attention_3d=false
  sim.add_top_down_map=false
  sim.visualize_3d=false
  sim.visualize_attn3d=false
)
DET=(rollout.deterministic=true sim.ep_seed=17)

# 1. HM3D v1 full context (2000 eps)
python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_val +experiment=eval +resources=single \
  task.run_name=hm3d_v1_val_full \
  task.shard_size=6 \
  "${DET[@]}" "${VIZ_OFF[@]}"

# 2. HM3D v1 window 32 evict
python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_val +experiment=eval +resources=single \
  task.run_name=hm3d_v1_val_win32 \
  task.shard_size=6 \
  vlm.context_window=32 vlm.context_window_mode=evict \
  "${DET[@]}" "${VIZ_OFF[@]}"

# 3. MP3D full context (2195 eps)
python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=mp3d_val +experiment=eval +resources=single \
  task.run_name=mp3d_val_full \
  task.subset_label="" \
  task.episode_json=$PWD/src/longnav/conf/episode_jsons/mp3d_val.json \
  task.shard_size=6 \
  "${DET[@]}" "${VIZ_OFF[@]}"

# 4. MP3D window 32 evict
python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=mp3d_val +experiment=eval +resources=single \
  task.run_name=mp3d_val_win32 \
  task.subset_label="" \
  task.episode_json=$PWD/src/longnav/conf/episode_jsons/mp3d_val.json \
  task.shard_size=6 \
  vlm.context_window=32 vlm.context_window_mode=evict \
  "${DET[@]}" "${VIZ_OFF[@]}"

To run eval on 20 episode hm3d v2 smoke test:
cd /home/brabiei/Projects/World-Modelling/spatial_training
conda activate longnav_vlm

python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources=single \
  task.run_name=hm3d_v2_val_smoke \
  task.subset_label="" \
  task.episode_json=$PWD/dump/hm3d_v2_smoke_labels.json \
  task.shard_size=5
## Delta: HAMLET v2 (memory token in context, n_moment=8) RL from the stage-1 adapter — 2026-08-26
# CPU unit checks (login node):
PYTHONPATH=src /work/nvme/bgon/brabiei/longnav_runtime/envs/longnav_vlm/bin/python tests/hamlet_unit.py
# 1-GPU pre-flight on full-length episodes (smoke gate + 2 real episodes at 350 steps, peak VRAM):
PRE=$(sbatch --parsable --export=ALL,MINI_STEPS=350,MINI_ROLLOUT=2,MINI_OPT_STEPS=2 cluster/delta/test_hamlet.sbatch)
# 12 h 4xA100 run, gated on the pre-flight (run_rl_train.sh pins stage-1 reward shaping + lr 1.25e-6):
MAIN=$(sbatch --parsable --time=12:00:00 --dependency=afterok:$PRE \
  --export=ALL,RUN_NAME=hamlet_mem_20260826,MAX_WALLCLOCK_HOURS=11.4 cluster/delta/train_hamlet.sbatch)
# Checkpoint watchdog: 36-episode argmax evals of checkpoints 3,7,11,15,31,47,...,final; scancel on success <= 0.10
nohup cluster/delta/watch_train.sh $MAIN hamlet_mem_20260826 \
  > /work/nvme/bgon/brabiei/longnav_runtime/logs/watch_hamlet_mem_20260826.out 2>&1 &
# results: /work/nvme/bgon/brabiei/longnav_runtime/runs/hamlet_mem_20260826/{progress.json,eval36.jsonl,checkpoints/}
# Tele-op a checkpoint (needs a TTY on a GPU node): allocate, attach, run
salloc --no-shell --account=bgon-delta-gpu --partition=gpuA100x4-interactive -N1 -n1 -c16 --gpus-per-node=1 --mem=64G --time=01:00:00 -J teleop
srun --jobid=<alloc id> --overlap --pty bash --login
TELEOP_CHECKPOINT=/work/nvme/bgon/brabiei/longnav_runtime/runs/hamlet_mem_20260826/checkpoints/checkpoint_111 EPISODE_INDEX=0 cluster/delta/run_teleop.sh
# EPISODE_INDEX indexes the full HM3D-v2 val set (1000 episodes; scenes alphabetical, 28 per scene); EVAL_EPISODES=<json> restricts to a label list.
# Live frame: $LONGNAV_OUTPUT_ROOT/<RUN_NAME>/teleop/<EPISODE_INDEX>_<label>/current.png (printed as 'Live image:' at start)
