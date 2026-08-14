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