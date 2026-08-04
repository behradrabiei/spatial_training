# Setup Notes

Local setup record for this machine (RTX 5090 / Blackwell `sm_120`). Two conda envs, both Python 3.10.16.

## Environments

| Env | Purpose | Key packages |
|-----|---------|--------------|
| `longnav_vlm` | VLM / RL trainer, driver, serving | torch 2.8.0+cu128, ray 2.53.0, transformers, trl, verl (editable), longnav (editable) |
| `vln` | Habitat sim workers | habitat-sim 0.3.3, habitat-lab 0.3.3, ray 2.53.0, longnav (editable, `--no-deps`) |

## What we did

```bash
# 1. Submodules (verl + ovon)
git submodule update --init --recursive

# 2. longnav_vlm env
conda create -n longnav_vlm python=3.10.16 -y
conda activate longnav_vlm
pip install -e .
pip install trl                       # needed by SFT scripts, not in pyproject
cd verl && pip install --no-dependencies -e . && cd ..

# 3. vln env (Habitat). Python 3.10 -> habitat-sim only on aihabitat-nightly
conda create -n vln python=3.10.16 cmake=3.14.0 -y
conda install -n vln habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly -y
git clone --branch stable --depth 1 https://github.com/facebookresearch/habitat-lab.git .habitat-lab
conda run -n vln pip install -e .habitat-lab/habitat-lab
conda run -n vln pip install "pillow==10.4.0" regex   # pillow pinned by habitat-sim
conda run -n vln pip install "ray[default]==2.53.0"  # required for Habitat Ray workers (match longnav_vlm)
conda run -n vln pip install --no-dependencies -e .    # register longnav for Ray
```

## Validated

- `longnav_vlm`: `tests/eval_smoke.py` and `tests/rl_smoke.py` pass (real LoRA training step on GPU); SDPA runs on the 5090.
- `vln`: `import habitat, habitat_sim, longnav` all clean.

## Skipped / optional

- **flash-attn** — no prebuilt wheel for `sm_120`; FA3/FA4 can't run on desktop Blackwell. Repo defaults to `attn_impl=sdpa`, which works natively. Build recent FA2 from source with `TORCH_CUDA_ARCH_LIST="12.0"` only if throughput matters.
- **wandb login** — DONE (entity `brabiei-university-of-michigan`, verified end-to-end). Enable per-run with `task.wandb_project=<name>`; same `run_name`+`project` resumes.
- **Hugging Face login** — public model download already works; only needed for private/push.
- **Habitat HM3D dataset** — required for real sim runs (`train_rl`/`eval` with `dataset=hm3d_*`). Status:
  - Scene meshes PRESENT at `/home/brabiei/vault/habitat_data/habitat_scenes/{HM3D,MP3D}` (HM3D: 904 scenes, `.basis.glb`+`.basis.navmesh`, flat layout; MP3D has `_semantic.ply`).
  - STILL MISSING: ObjectNav episode datasets (`objectnav/hm3d/v1/{split}/{split}.json.gz`), HM3D semantic annotations (`*.semantic.glb/.txt` — config uses a semantic sensor + `sim.semantic_scene`), and `hm3d_*.scene_dataset_config.json`.
  - Wiring (once complete): `sim.workspace=<root containing data/>`, `sim.scenes_dir=<scene_datasets>`, `sim.dataset_path=<{split}.json.gz>` (schema lines 152-156). Expected layout: `data/scene_datasets/hm3d/{train,val}/...` + `data/datasets/objectnav/hm3d/v1/{train,val}/...json.gz`.
- **verl extras** (`liger-kernel`, `math_verify`, ...) — only for those specific verl features.

## Notes

- All CUDA toolchain (`cuda-nvcc`/`cuda-toolkit`/`cuda-compiler` 12.8) was installed **into the `longnav_vlm` env only** — system drivers and base env untouched.
- `verl` pins `numpy<2.0.0` but env has numpy 2.x (from longnav deps); installed per README with `--no-dependencies`. Smoke tests pass; pin down only if a concrete verl error appears.
- habitat-lab lives at `./.habitat-lab` (gitignored clone), installed editable into `vln`.
