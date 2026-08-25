# LongNav on NCSA Delta

Run installation and model downloads from the repository root on a login node:

```bash
cluster/delta/setup.sh
cluster/delta/stage_models.sh
```

Both commands are safe to rerun. By default they use
`/work/nvme/bgon/brabiei/longnav_runtime`; set `LONGNAV_RUNTIME_ROOT` before
running them to use another shared location. The Habitat data defaults to
`/work/nvme/bgon/brabiei/habitat_data`.

Delta currently has glibc 2.34, while the retained Habitat-Sim nightlies require
glibc 2.35. Setup therefore builds the exact 0.3.3 nightly source commit in
headless/Bullet mode and caches its wheel below the runtime root. A first setup
takes longer; reruns install the cached wheel.

Request a one-hour interactive A40 session using one of the GPU projects shown
by Delta's `accounts` command:

```bash
export DELTA_ACCOUNT=your-delta-gpu-account
cluster/delta/request_interactive.sh
```

To request another supported GPU type, set `DELTA_PARTITION` to
`gpuA100x4-interactive` or `gpuH200x8-interactive` first.

Inside the compute-node shell, activate the shared environment and run checks
individually so failures can be debugged without leaving the allocation:

```bash
source cluster/delta/activate_session.sh
cluster/delta/debug.sh gpu
cluster/delta/debug.sh imports
cluster/delta/debug.sh models
cluster/delta/debug.sh habitat
cluster/delta/debug.sh eval-smoke
cluster/delta/debug.sh rl-smoke
cluster/delta/debug.sh hm3d
```

The compute-node session enables Hugging Face, Transformers, and Datasets
offline modes. The final command evaluates `4ok3usBNeis_0` through
`4ok3usBNeis_4`, with a 50-step cap, and writes its resolved configuration and
results below `$LONGNAV_RUNTIME_ROOT/runs`.
