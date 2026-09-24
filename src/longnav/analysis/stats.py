"""Episode-level statistics for comparing eval arms.

Every arm is scored on the same episodes, so comparisons are paired on ``episode_label``.
Uncertainty on a single arm's mean is a bootstrap over episodes; differences between arms
use a paired sign-flip permutation test on the per-episode delta (two-sided), the standard
already used in KV_PRUNE_RESULTS.tex / tools/compare_runs.py.
"""
from __future__ import annotations

import numpy as np

N_BOOT_CI = 2000       # bootstrap resamples for a mean's confidence interval
N_PERM = 10000         # sign-flip resamples for a paired p-value
SEED = 17


def bootstrap_mean_ci(values, n_boot: int = N_BOOT_CI, seed: int = SEED, alpha: float = 0.05):
    """(mean, lo, hi): percentile bootstrap CI of the mean over finite ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def sign_flip_p(deltas, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Two-sided paired sign-flip test: P(|mean of randomly re-signed deltas| >= |observed|)."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    resampled = np.abs((signs * d).mean(axis=1))
    return float((resampled >= observed - 1e-12).mean())


def paired_values(rows_a: dict, rows_b: dict, key: str, labels=None):
    """Per-episode (a, b) arrays over the labels both arms have (or the given list)."""
    if labels is None:
        labels = sorted(set(rows_a) & set(rows_b))
    a = np.array([float(rows_a[l].get(key, np.nan)) for l in labels])
    b = np.array([float(rows_b[l].get(key, np.nan)) for l in labels])
    return a, b


def paired_delta(rows_a: dict, rows_b: dict, key: str, labels=None, n_boot: int = N_BOOT_CI,
                 n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """a - b on paired episodes: mean delta, bootstrap CI of the delta, sign-flip p, n."""
    a, b = paired_values(rows_a, rows_b, key, labels)
    d = a - b
    mean, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    return {"delta": mean, "lo": lo, "hi": hi, "p": sign_flip_p(d, n_perm=n_perm, seed=seed),
            "n": int(np.isfinite(d).sum())}


def stars(p: float) -> str:
    """Significance marker for tables: *** <0.001, ** <0.01, * <0.05, else ''."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
