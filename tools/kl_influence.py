#!/usr/bin/env python3
"""Does a frame's influence on the action decision persist into future steps?

Reads the per-step rows logged by `vlm.kv_prune_log_influence=true` -- for step t,
KL[t, f] = KL(P_t || P_t^{-f}) for every cached frame f <= t (P_t = action distribution at
the decision token; P_t^{-f} = same decision with frame f's visual slots masked out of
attention) -- from a run's sequence.json files, and reports:

  persistence  Spearman rho between KL[t, f] and KL[t+k, f] over the frames f <= t, per
               step (mean over steps) and pooled, plus top-quartile retention
               P(top 25% at t+k | top 25% at t), chance 0.25. Reported on RAW KL and on
               age-DETRENDED residuals KL[t, f] / mean_KL(age = t - f): the raw numbers are
               inflated by the shared decay curve (older frames stay older, so their ranks
               persist trivially); the residual numbers ask whether a frame is more or less
               influential than its age predicts, and whether THAT persists. Also split by
               age >= 2 (the current and previous frame dominate the mass).
  decay        mean / median KL and share of total KL mass by frame age (t - f).
  revival      frames whose KL stays below eps for >= `dormant` consecutive steps and later
               exceeds thr again.
  recency      share of each step's KL mass in the two most recent frames.

Frames whose visual tokens were all sparse-filtered (kv_visual_new == 0) are excluded, not
scored as zero-influence. Run with the vln python (matplotlib):
    /home/brabiei/miniconda3/envs/vln/bin/python tools/kl_influence.py dump/longnav_eval/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGE_BINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 9)]
MIN_FRAMES = 8  # per-step correlations need this many frames f <= t


def load_episodes(run_dir: Path):
    results = {}
    for path in run_dir.glob("rollout/results_*"):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                results.setdefault(r["episode_label"], r)
    episodes = []
    for seq in sorted(run_dir.glob("rollout/*/*/sequence.json")):
        d = json.load(seq.open())
        rows = d.get("sup/kl_influence")
        if not rows:
            continue
        T = len(rows)
        M = np.full((T, T), np.nan)
        bad = 0
        for t, row in enumerate(rows):
            if len(row) == t + 1:
                M[t, :t + 1] = row
            elif len(row):
                bad += 1
        vis_new = d.get("kv_visual_new") or []
        dead = [f for f, n in enumerate(vis_new[:T]) if n == 0]
        for f in dead:
            M[:, f] = np.nan
        label = seq.parent.name.split(".")[0]
        episodes.append({"label": label, "M": M, "T": T, "bad_rows": bad, "dead_frames": len(dead),
                         "success": results.get(label, {}).get("success"),
                         "entropy": d.get("entropy"), "actions": d.get("action_history")})
    return episodes


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=ranks)[inv] / counts[inv]


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rx, ry = _rank(x[m]), _rank(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def age_curve(episodes, max_age=256):
    """Mean KL per frame age, pooled over episodes (ages beyond max_age share one value)."""
    sums, counts = np.zeros(max_age + 1), np.zeros(max_age + 1)
    for ep in episodes:
        T = ep["T"]
        t_idx, f_idx = np.tril_indices(T)
        v = ep["M"][t_idx, f_idx]
        ok = np.isfinite(v)
        a = np.minimum(t_idx - f_idx, max_age)[ok]
        sums += np.bincount(a, weights=v[ok], minlength=max_age + 1)
        counts += np.bincount(a, minlength=max_age + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = sums / counts
    # fill empty ages from the nearest populated younger age
    for a in range(1, max_age + 1):
        if not np.isfinite(curve[a]):
            curve[a] = curve[a - 1]
    return np.maximum(curve, 1e-12)


def detrend(M, curve):
    T = M.shape[0]
    t_idx, f_idx = np.indices((T, T))
    age = np.minimum(np.maximum(t_idx - f_idx, 0), curve.size - 1)
    return M / curve[age]


def _lag_stats(mats, k):
    step_all, step_old, retained, pooled_x, pooled_y, per_ep = [], [], [], [], [], []
    for M in mats:
        T = M.shape[0]
        ep_rhos = []
        for t in range(T - k):
            x, y = M[t, :t + 1], M[t + k, :t + 1]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_FRAMES:
                continue
            rho = spearman(x, y)
            step_all.append(rho); ep_rhos.append(rho)
            pooled_x.append(x[ok]); pooled_y.append(y[ok])
            old = ok.copy(); old[max(t - 1, 0):] = False  # age >= 2
            if old.sum() >= MIN_FRAMES:
                step_old.append(spearman(x[old], y[old]))
            xo, yo = x[ok], y[ok]
            q = max(1, int(round(0.25 * xo.size)))
            top_x = set(np.argsort(-xo)[:q].tolist())
            top_y = set(np.argsort(-yo)[:q].tolist())
            retained.append(len(top_x & top_y) / q)
        per_ep.append(float(np.nanmean(ep_rhos)) if ep_rhos else np.nan)
    pooled = spearman(np.concatenate(pooled_x), np.concatenate(pooled_y)) if pooled_x else np.nan
    return {"rho_step_mean": float(np.nanmean(step_all)) if step_all else np.nan,
            "rho_step_mean_age2": float(np.nanmean(step_old)) if step_old else np.nan,
            "rho_pooled": pooled,
            "top_quartile_retention": float(np.mean(retained)) if retained else np.nan,
            "rho_per_episode": per_ep, "n_steps": len(step_all)}


def persistence(episodes, lags):
    curve = age_curve(episodes)
    raw = [ep["M"] for ep in episodes]
    resid = [detrend(ep["M"], curve) for ep in episodes]
    return {k: {"raw": _lag_stats(raw, k), "residual": _lag_stats(resid, k)} for k in lags}


def decay(episodes):
    ages, vals = [], []
    for ep in episodes:
        M, T = ep["M"], ep["T"]
        t_idx, f_idx = np.tril_indices(T)
        v = M[t_idx, f_idx]
        ok = np.isfinite(v)
        ages.append((t_idx - f_idx)[ok]); vals.append(v[ok])
    ages, vals = np.concatenate(ages), np.concatenate(vals)
    total = vals.sum()
    rows = []
    for lo, hi in AGE_BINS:
        m = (ages >= lo) & (ages <= hi)
        if not m.any():
            continue
        rows.append({"age": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"),
                     "n": int(m.sum()), "mean": float(vals[m].mean()),
                     "median": float(np.median(vals[m])), "mass_share": float(vals[m].sum() / total)})
    return rows


def revival(episodes, eps, dormant, thr):
    ever_dormant, revived, events = 0, 0, 0
    for ep in episodes:
        M, T = ep["M"], ep["T"]
        for f in range(T):
            s = M[f:, f]
            s = s[np.isfinite(s)]
            if s.size < dormant + 1:
                continue
            run, was_dormant, frame_revived = 0, False, False
            for v in s:
                if v < eps:
                    run += 1
                    if run >= dormant:
                        was_dormant = True
                else:
                    if was_dormant and v > thr:
                        events += 1
                        frame_revived = True
                    run = 0
            ever_dormant += was_dormant
            revived += frame_revived
    return {"ever_dormant": ever_dormant, "revived": revived, "events": events,
            "rate": revived / ever_dormant if ever_dormant else np.nan}


def recency(episodes):
    two, top1 = [], []
    for ep in episodes:
        M, T = ep["M"], ep["T"]
        for t in range(2, T):
            row = M[t, :t + 1]
            tot = np.nansum(row)
            if not np.isfinite(tot) or tot <= 0:
                continue
            two.append(np.nansum(row[t - 1:t + 1]) / tot)
            top1.append(np.nanmax(row) / tot)
    return {"two_newest_mean": float(np.mean(two)), "two_newest_median": float(np.median(two)),
            "top1_mean": float(np.mean(top1)), "n_steps": len(two)}


def plot(summary, episodes, out):
    lags = sorted(summary["persistence"])
    pers = summary["persistence"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    series = [("residual", "rho_step_mean", "rho, age-detrended", "#2a78d6", "o", "-"),
              ("residual", "top_quartile_retention", "top-25% kept, age-detrended", "#1baf7a", "^", "-"),
              ("raw", "rho_step_mean", "rho, raw (age-confounded)", "#2a78d6", "o", ":")]
    for kind, key, label, color, marker, ls in series:
        y = [pers[k][kind][key] for k in lags]
        ax.plot(lags, y, label=label, color=color, marker=marker, markersize=7, linewidth=2,
                linestyle=ls, alpha=0.9 if ls == "-" else 0.6)
        ax.annotate(label, (lags[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#444", va="center")
    ax.axhline(0.25, color="#888", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2); ax.set_xticks(lags); ax.set_xticklabels([str(k) for k in lags])
    ax.set_xlim(lags[0] / 1.3, lags[-1] * 3.2)
    ax.set_xlabel("lag k (steps)"); ax.set_ylabel("persistence of frame influence")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    dec = summary["decay"]
    x = np.arange(len(dec))
    ax.plot(x, [d["mean"] for d in dec], color="#2a78d6", marker="o", markersize=7, linewidth=2,
            label="mean KL")
    ax.plot(x, [d["median"] for d in dec], color="#eb6834", marker="s", markersize=6, linewidth=2,
            label="median KL")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([d["age"] for d in dec])
    ax.set_xlabel("frame age t - f (steps)"); ax.set_ylabel("KL(P_t || P_t without frame)")
    ax.legend(fontsize=8, frameon=False)
    rec = summary["recency"]
    ax.set_title(f"two newest frames carry {100 * rec['two_newest_mean']:.0f}% of KL mass "
                 f"(median {100 * rec['two_newest_median']:.0f}%)", fontsize=9)

    ax = axes[2]
    ep = max(episodes, key=lambda e: e["T"])
    with np.errstate(divide="ignore"):
        L = np.log10(np.where(ep["M"] > 0, ep["M"], np.nan))
    im = ax.imshow(L, cmap="Blues", origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("frame f"); ax.set_ylabel("decision step t")
    ax.set_title(f"log10 KL[t, f], episode {ep['label']} ({ep['T']} steps)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for ax in axes[:2]:
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Frame-influence persistence, {summary['n_episodes']} full-cache episodes "
                 f"({summary['n_pairs']} (t, f) pairs)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--lags", default="1,2,4,8,16")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--dormant", type=int, default=5)
    ap.add_argument("--thr", type=float, default=1e-2)
    args = ap.parse_args()
    lags = [int(k) for k in args.lags.split(",")]

    episodes = load_episodes(args.run_dir)
    if not episodes:
        raise SystemExit(f"no sup/kl_influence rows under {args.run_dir}")
    n_pairs = int(sum(np.isfinite(ep["M"]).sum() for ep in episodes))
    summary = {
        "run": str(args.run_dir), "n_episodes": len(episodes), "n_pairs": n_pairs,
        "episodes": [{k: ep[k] for k in ("label", "T", "bad_rows", "dead_frames", "success")}
                     for ep in episodes],
        "persistence": persistence(episodes, lags),
        "decay": decay(episodes),
        "revival": revival(episodes, args.eps, args.dormant, args.thr),
        "recency": recency(episodes),
    }
    out_json = args.run_dir / "kl_influence_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    plot(summary, episodes, args.run_dir / "kl_influence.png")

    print(f"{len(episodes)} episodes, {n_pairs} (t, f) pairs; steps: "
          + ", ".join(f"{ep['label']}={ep['T']}" for ep in episodes))
    print(f"{'lag':>4} | {'RESIDUAL (age-detrended)':^44} | {'RAW':^30}")
    print(f"{'':>4} | {'rho(step)':>10} {'age>=2':>8} {'pooled':>8} {'top25':>7} {'n':>6} | "
          f"{'rho(step)':>10} {'age>=2':>8} {'top25':>7}")
    for k in lags:
        r, w = summary["persistence"][k]["residual"], summary["persistence"][k]["raw"]
        print(f"{k:>4} | {r['rho_step_mean']:>10.3f} {r['rho_step_mean_age2']:>8.3f} "
              f"{r['rho_pooled']:>8.3f} {r['top_quartile_retention']:>7.3f} {r['n_steps']:>6} | "
              f"{w['rho_step_mean']:>10.3f} {w['rho_step_mean_age2']:>8.3f} {w['top_quartile_retention']:>7.3f}")
    for k in (lags[0], lags[len(lags) // 2]):
        per = summary["persistence"][k]["residual"]["rho_per_episode"]
        print(f"per-episode residual rho at lag {k}: "
              + ", ".join(f"{ep['label']}={v:.2f}" for ep, v in zip(episodes, per)))
    print(f"{'age':>6} {'n':>7} {'mean':>9} {'median':>9} {'mass':>6}")
    for d in summary["decay"]:
        print(f"{d['age']:>6} {d['n']:>7} {d['mean']:>9.4f} {d['median']:>9.4f} {d['mass_share']:>6.2f}")
    r, c = summary["revival"], summary["recency"]
    print(f"revival: {r['revived']}/{r['ever_dormant']} dormant frames revived "
          f"(rate {r['rate']:.2f}, {r['events']} events); two newest frames hold "
          f"{100 * c['two_newest_mean']:.0f}% of KL mass (median {100 * c['two_newest_median']:.0f}%), "
          f"top-1 frame {100 * c['top1_mean']:.0f}%")
    print(f"wrote {out_json} and {args.run_dir / 'kl_influence.png'}")


if __name__ == "__main__":
    main()
