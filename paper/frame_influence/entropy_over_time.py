"""Does LightNav-0 grow more confident over an episode? Entropy of the third generated token.

LightNav-0 emits ``<apos_K><opos_K><act_l0_a><act_l1_b><act_l2_c>``; the third token, the coarse
RVQ code ``<act_l0_*>`` (256 codes), carries the motion. The keep-one influence run
(``paper_results/frame_influence_keepone/runs/lightnav_influence_keepone``) logged, for every
step, the FULL-context decision's distribution at that position (``full.entropy`` = entropy over
the 256 codes renormalised, T = 1; ``full.logp_f16`` = the 256 log-probs), so the question is
answered from those records without running the model.

Outputs (``--out``): a per-step CSV, a per-episode summary CSV (Spearman rho of entropy vs step,
first- vs last-quarter means, exact sign test across episodes) and a two-panel figure of entropy
and top-1 probability against episode progress (fraction of the episode), thin lines per episode
and the episode-balanced decile mean with SEM, plus ``lightnav_entropy_simple``: the entropy panel
alone, mean and SEM only.

    conda run -n vln python paper/frame_influence/entropy_over_time.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = ROOT / "paper_results/frame_influence_keepone/runs/lightnav_influence_keepone"
DEFAULT_OUT = ROOT / "outputs/validation/lightnav_entropy_over_time"
N_DECILES = 10
SMOOTH = 5  # centred rolling mean over this many steps for the per-episode lines
STYLE = dict(color="tab:green", marker="o", markersize=4, linestyle="-")


def unpack_f16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (average ranks for ties)."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        # average tied ranks
        vs = v[order]
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """(n negative, n non-zero, exact two-sided binomial p) for H0: median delta = 0."""
    nz = [d for d in deltas if d != 0]
    n, k = len(nz), sum(d < 0 for d in nz)
    if n == 0:
        return 0, 0, float("nan")
    tail = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n
    return k, n, min(1.0, 2 * p)


def rolling(v: np.ndarray, w: int) -> np.ndarray:
    h = w // 2
    return np.array([v[max(0, i - h):i + h + 1].mean() for i in range(len(v))])


def load(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "influence").glob("*.jsonl")):
        label = path.stem
        ep = json.loads((run_dir / "episodes" / f"{label}.json").read_text())
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        steps = [int(r["step"]) for r in recs]
        if steps != list(range(len(recs))):
            raise ValueError(f"{label}: steps not contiguous from 0")
        n = len(recs)
        for r in recs:
            f = r["full"]
            logp = unpack_f16(f["logp_f16"])
            h = float(-(np.exp(logp) * logp).sum())
            if abs(h - float(f["entropy"])) > 0.01:
                raise ValueError(f"{label} step {r['step']}: stored entropy {f['entropy']} != recomputed {h}")
            rows.append({"label": label, "step": int(r["step"]), "n_steps": n, "frac": r["step"] / max(n - 1, 1),
                         "entropy": float(f["entropy"]), "entropy_bin": float(f["entropy_bin"]),
                         "p_top1": float(np.exp(logp.max())), "exec_bin": f.get("exec_bin") or "",
                         "p_apos": float(f["p_apos"]), "p_opos": float(f["p_opos"]), "p_mass": float(f["p_mass"]),
                         "success": int(ep["metrics"]["success"] > 0 and not ep["forced_stop"]),
                         "forced_stop": int(ep["forced_stop"])})
    return rows


def per_episode(rows: list[dict]) -> list[dict]:
    out = []
    for label in sorted({r["label"] for r in rows}):
        ep = [r for r in rows if r["label"] == label]
        step = np.array([r["step"] for r in ep], dtype=float)
        q = max(1, len(ep) // 4)
        rec = {"label": label, "n_steps": len(ep), "success": ep[0]["success"], "forced_stop": ep[0]["forced_stop"]}
        for key in ("entropy", "p_top1"):
            v = np.array([r[key] for r in ep])
            rec[f"{key}_rho"] = spearman(step, v)
            rec[f"{key}_q1"] = float(v[:q].mean())
            rec[f"{key}_q4"] = float(v[-q:].mean())
            rec[f"{key}_delta"] = rec[f"{key}_q4"] - rec[f"{key}_q1"]
        v = np.array([r["entropy"] for r in ep])
        rec["entropy_median"] = float(np.median(v))
        rec["entropy_at_stop"] = float(v[-1]) if not ep[0]["forced_stop"] else float("nan")
        out.append(rec)
    return out


def decile_means(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Episode-balanced mean per progress decile: mean within each episode first, then mean and
    SEM over episodes."""
    labels = sorted({r["label"] for r in rows})
    edges = np.linspace(0, 1, N_DECILES + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    per = np.full((len(labels), N_DECILES), np.nan)
    for i, label in enumerate(labels):
        ep = [r for r in rows if r["label"] == label]
        frac = np.array([r["frac"] for r in ep])
        v = np.array([r[key] for r in ep])
        d = np.minimum((frac * N_DECILES).astype(int), N_DECILES - 1)
        for j in range(N_DECILES):
            if np.any(d == j):
                per[i, j] = v[d == j].mean()
    mean = np.nanmean(per, axis=0)
    n = np.sum(~np.isnan(per), axis=0)
    sem = np.nanstd(per, axis=0, ddof=1) / np.sqrt(n)
    return centres, mean, sem


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def figure(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    labels = sorted({r["label"] for r in rows})
    for ax, key, ylabel in zip(axes, ("entropy", "p_top1"), ("Entropy (nats)", "Top-1 probability")):
        for label in labels:
            ep = [r for r in rows if r["label"] == label]
            frac = np.array([r["frac"] for r in ep])
            v = rolling(np.array([r[key] for r in ep]), SMOOTH)
            ax.plot(frac, v, color="0.6", linewidth=0.7, linestyle="--" if ep[0]["forced_stop"] else "-", alpha=0.8)
        c, m, s = decile_means(rows, key)
        ax.errorbar(c, m, yerr=s, fmt="none", ecolor=STYLE["color"], elinewidth=1, capsize=0)
        ax.plot(c, m, label=f"mean ({len(labels)} eps)", linewidth=1.8, **STYLE)
        ax.set_xlabel("Episode progress")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def figure_simple(rows: list[dict], out: Path) -> None:
    """One panel: mean action entropy against episode progress, SEM bars over episodes, nothing else."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    c, m, s = decile_means(rows, "entropy")
    ax.errorbar(c, m, yerr=s, fmt="none", ecolor=STYLE["color"], elinewidth=1, capsize=0)
    ax.plot(c, m, linewidth=1.8, **STYLE)
    ax.set_xlabel("Episode progress")
    ax.set_ylabel("Action entropy (nats)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = load(a.run_dir)
    eps = per_episode(rows)
    write_csv(a.out / "lightnav_entropy_over_time.csv", rows)
    write_csv(a.out / "lightnav_entropy_over_time_episodes.csv", eps)
    figure(rows, a.out / "lightnav_entropy_over_time")
    figure_simple(rows, a.out / "lightnav_entropy_simple")

    print(f"\n{len(rows)} decisions, {len(eps)} episodes, p_mass min {min(r['p_mass'] for r in rows):.4f}")
    print(f"{'episode':16s} {'steps':>5s} {'ok':>2s} {'rho_H':>6s} {'H_q1':>5s} {'H_q4':>5s} {'rho_p':>6s} {'p_q1':>5s} {'p_q4':>5s} {'H_stop':>6s} {'H_med':>5s}")
    for e in eps:
        print(f"{e['label']:16s} {e['n_steps']:5d} {e['success']:2d} {e['entropy_rho']:6.2f} {e['entropy_q1']:5.2f} {e['entropy_q4']:5.2f} "
              f"{e['p_top1_rho']:6.2f} {e['p_top1_q1']:5.2f} {e['p_top1_q4']:5.2f} {e['entropy_at_stop']:6.2f} {e['entropy_median']:5.2f}")
    for key, name in (("entropy", "entropy"), ("p_top1", "top-1 prob")):
        k, n, pv = sign_test([e[f"{key}_delta"] for e in eps])
        rhos = [e[f"{key}_rho"] for e in eps]
        print(f"{name}: last-quarter below first-quarter in {k}/{n} episodes (sign test p = {pv:.3f}); "
              f"median rho {np.median(rhos):+.2f}, rho < 0 in {sum(r < 0 for r in rhos)}/{len(rhos)}")
        c, m, s = decile_means(rows, key)
        print("  decile means: " + "  ".join(f"{mm:.2f}±{ss:.2f}" for mm, ss in zip(m, s)))


if __name__ == "__main__":
    main()
