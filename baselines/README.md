# Uni-NaVid HM3D-v2 baseline

This directory contains a standalone, inference-only Uni-NaVid evaluation on
the 100 labels in `../dump/hm3d_v2_100_labels.json`. It does not import LongNav
or install anything into the existing `vln` or `longnav_vlm` environments.

## Layout and isolation

- `.envs/uninavid-hm3d`: relocated local Conda prefix.
- `.cache`: Conda, pip, Hugging Face, Torch, XDG, and temporary caches.
- `Uni-NaVid`: pinned editable upstream clone.
- `habitat-lab`: pinned editable Habitat-Lab clone.
- `uninavid_hm3d`: HM3D ObjectNav adapter.
- `results`: resumable per-episode outputs and aggregate summaries.

The large/generated paths are ignored by `baselines/.gitignore`. HM3D scenes
are referenced by symlink and are only read by the evaluator.

## Commands

```bash
bash baselines/setup.sh
bash baselines/run.sh download
bash baselines/run.sh validate
bash baselines/run.sh model-check
bash baselines/run.sh smoke
bash baselines/run.sh full
```

The full command uses `--resume`; completed episode JSON files are skipped.
Pass evaluator overrides after the command, for example:

```bash
bash baselines/run.sh smoke --temperature 0.1 --actions-per-inference 1
```

Run the strict context-window ablation (the window counts previous frames; the
current frame is additional) and generate its SR/SPL plot with:

```bash
bash baselines/run_context_window_ablation.sh
```

Run the current-frame-only (`--context-window 0`) and no-visual-token sanity
checks with:

```bash
bash baselines/run_extreme_ablations.sh
```

The extreme launcher also runs `--current-frame-only-64`, which removes the
current frame's duplicate four-token history representation while retaining its
64-token detailed image representation.

Finite arms are written to `results/hm3d_v2_100_win{N}` and can be resumed by
rerunning the command. The completed native full-history evaluation in
`results/hm3d_v2_100` is reused as the control.

## Modifying inference

Uni-NaVid is installed editable, so changes under `baselines/Uni-NaVid/uninavid`
take effect immediately. The observation-history cache and online token merging
are implemented in `uninavid/model/uninavid_arch.py`; decoding and action
buffering are controlled by `uninavid_hm3d/eval.py`. No training packages are
installed or required.

Each run records the source revisions, dependency versions, label checksum,
effective inference settings, raw model outputs, chosen actions, timing, and
Habitat metrics in its result directory.
