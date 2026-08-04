To run eval on full hm3d v2 run:
cd /home/brabiei/Projects/World-Modelling/spatial_training
conda activate longnav_vlm

python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources=single \
  task.run_name=hm3d_v2_val_full \
  task.subset_label="" \
  task.episode_json=$PWD/dump/hm3d_v2_val_labels.json \
  task.shard_size=6

To run eval on 20 episode hm3d v2 smoke test:
cd /home/brabiei/Projects/World-Modelling/spatial_training
conda activate longnav_vlm

python -m longnav.scripts.eval \
  +checkpoint=longnav +dataset=hm3d_v2_val +experiment=eval +resources=single \
  task.run_name=hm3d_v2_val_smoke \
  task.subset_label="" \
  task.episode_json=$PWD/dump/hm3d_v2_smoke_labels.json \
  task.shard_size=5