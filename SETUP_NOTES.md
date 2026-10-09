# Local inference setup

This records the existing RTX 5090 setup. Cleanup preserved these environments and package versions.

| Environment | Purpose | Existing stack |
|---|---|---|
| `longnav_vlm` | Model inference, Ray driver, serving | Python 3.10, torch 2.8.0+cu128, Ray 2.53.0, Transformers, PEFT |
| `vln` | Habitat actors and paper plotting | Python 3.10, Habitat Sim/Lab 0.3.3, Ray 2.53.0 |
| `baselines/.envs/uninavid-hm3d` | Uni-NaVid runtime | Separate baseline stack; see baselines/README.md |

LongNav is installed editable in both conda environments. For an existing Habitat environment,
use `pip install --no-deps -e .` and install matching Ray explicitly so Habitat pins stay intact.
PEFT and its runtime dependencies are needed to load the retained pretrained adapters. `verl` and
TRL are no longer inference requirements. Existing environment packages were not uninstalled.

SDPA is the default and is used for cleanup parity. The local `longnav_vlm` environment also has
FlashAttention 2.8.3 with Blackwell kernels; select it explicitly per run if needed. CUDA 12.8 tools
live in that environment. Cleanup did not change drivers, attention defaults, tokenizer settings,
or checkpoint weights.

Dataset presets in `src/longnav/config/dataset/` specify local HM3D v2 / MP3D dataset and scene paths
under `/home/brabiei/vault/habitat_data/`. Evaluation needs the original datasets, semantics, and
scene assets. The old HM3D v1 setup notes are superseded by these presets.

`task.wandb_project=null` runs without remote logging. Cached public checkpoints can be used with
`HF_HUB_OFFLINE=1`. All acceptance runs write under the ignored cleanup archive.

The default multi-goal assets `src/longnav/conf/episode_jsons/onemap_multi.json` and
`habitat_configs/objectnav_hm3d_multi.yaml` are absent in this checkout. Multi-goal unit tests cover
transitions, but a real benchmark integration run requires those original assets.

See CLEANUP.md for current validation results, rather than historical training smoke-test claims.
