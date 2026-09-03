#!/usr/bin/env python3
"""Compare eval runs episode-by-episode: metrics table + paired sign-flip bootstrap.

All runs must share the episode split (e.g. dump/hm3d_v2_100_labels.json); rows are
paired on episode_label. Significance is a paired sign-flip permutation test on the
per-episode delta vs --baseline (two-sided): p = fraction of sign-flipped resamples whose
|mean| >= |observed mean|.

    python tools/compare_runs.py hm3d_v2_100_prune_w32_b1000 hm3d_v2_100_reindexv2win4 \
        --baseline hm3d_v2_100_reindexv2win32

Runs in longnav_vlm (numpy only, no matplotlib).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_COLS = [
    ("SR", "success", "mean"),
    ("SPL", "spl", "mean"),
    ("steps", "n_steps", "mean"),
    ("kv_len", "sup/mean_kv_len", "mean"),
    ("kv_max", "sup/max_kv_len", "max"),
    ("kv_lmax", "sup/max_kv_len_layer_max", "max"),
    ("kv_master", "sup/mean_kv_len_master", "mean"),
    ("mem_GB", "sup/max_vlm_mem_GB", "max"),
    ("lat_ms", "sup/mean_vlm_latency", "mean"),
    ("total_s", "sup/sum_vlm_latency", "mean"),
    ("score_ms", "sup/mean_prune_score_latency", "mean"),
    ("score_s", "sup/sum_prune_score_latency", "mean"),
    ("replays", "sup/sum_prune_replay_count", "mean"),
    ("kv_vis", "sup/mean_kv_visual_slots", "mean"),
    ("kv_text", "sup/mean_kv_text_slots", "mean"),
]


def load_run(run_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((run_dir / "rollout").glob("results_*")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = row["episode_label"]
                if label in rows:
                    print(f"[warn] {run_dir.name}: duplicate episode {label}, keeping first")
                    continue
                rows[label] = row
    return rows


def col(rows: dict[str, dict], labels: list[str], key: str) -> np.ndarray:
    return np.array([float(rows[l].get(key, np.nan)) for l in labels])


def sign_flip_p(deltas: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    observed = abs(deltas.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_boot, deltas.size))
    resampled = np.abs((signs * deltas).mean(axis=1))
    return float((resampled >= observed - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run names under --eval-root")
    ap.add_argument("--baseline", help="run name to pair every other run against")
    ap.add_argument("--eval-root", type=Path, default=Path("./dump/longnav_eval"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = list(args.runs)
    if args.baseline and args.baseline not in names:
        names.insert(0, args.baseline)
    runs = {}
    for name in names:
        rows = load_run(args.eval_root / name)
        if not rows:
            print(f"[skip] no results for {name}")
            continue
        runs[name] = rows

    header = f"{'run':<40} {'n':>4}" + "".join(f" {h:>8}" for h, _, _ in METRIC_COLS)
    print(header)
    print("-" * len(header))
    for name, rows in runs.items():
        labels = list(rows)
        cells = []
        for _, key, red in METRIC_COLS:
            v = col(rows, labels, key)
            v = v[~np.isnan(v)]
            if v.size == 0:
                cells.append(f" {'-':>8}")
                continue
            x = v.max() if red == "max" else v.mean()
            if key in ("sup/mean_vlm_latency", "sup/mean_prune_score_latency"):
                x *= 1000.0
            fmt = ".0f" if key.endswith("kv_len") or key == "n_steps" else ".3f"
            cells.append(f" {x:>8{fmt}}")
        print(f"{name:<40} {len(labels):>4}" + "".join(cells))

    if not args.baseline:
        return
    base = runs[args.baseline]
    rng = np.random.default_rng(args.seed)
    print(f"\npaired sign-flip bootstrap vs {args.baseline} "
          f"({args.n_boot} resamples, two-sided):")
    for name, rows in runs.items():
        if name == args.baseline:
            continue
        labels = sorted(set(rows) & set(base))
        if len(labels) < len(base) or len(labels) < len(rows):
            print(f"  [warn] {name}: paired on {len(labels)} episodes "
                  f"(baseline {len(base)}, run {len(rows)})")
        for pretty, key in (("dSR", "success"), ("dSPL", "spl")):
            d = col(rows, labels, key) - col(base, labels, key)
            p = sign_flip_p(d, args.n_boot, rng)
            print(f"  {name:<40} {pretty}={d.mean():+.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
