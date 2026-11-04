"""Is LightNav-0's action distribution multi-modal in trajectory space? Pushforward Gaussian vs entropy.

LightNav-0 decodes one trajectory per step (greedy), but its action head is a categorical
distribution over the 256 coarse RVQ codes ``<act_l0_*>``, each a fixed 10-waypoint SE(2) chunk.
The 256-way entropy is metric-free: it cannot tell "spread over near-identical paths" from
"spread over left vs right". This script pushes the stored distribution through the action
decoder, exactly (probability-weighted moments of the 256 decoded codes, no sampling), and
fits a Gaussian in trajectory space. Per step, from the keep-one influence run's
``full.logp_f16`` (256 log-probs, T = 1):

* ``H_cat``   entropy of the 256-way distribution (nats; recomputed, checked against the log);
* ``H_gauss`` entropy of the Gaussian fitted to the chunk ENDPOINT POSE ``(x, y, heading)`` of
  every code (metres, metres, radians; heading = cumulative rotation, unwrapped), i.e.
  ``0.5 log det(2 pi e Sigma)`` with ``mu = sum_k p_k e_k`` and
  ``Sigma = sum_k p_k (e_k - mu)(e_k - mu)^T + ridge``;
* ``d_exec``  Mahalanobis distance of the executed (greedy) code's endpoint from ``mu``: ~0 when
  the Gaussian mean is the action itself, large when the mean sits between modes.

Each code is decoded ALONE (zero residual), as ``code_bins`` in the run's ``meta.json`` does; the
offline decode is checked against that table for all 256 codes.

Outputs (``--out``): ``action_spread.csv`` (one row per step), ``action_spread_episodes.csv``,
``summary.json`` and three figures: ``spread_scatter`` (H_gauss vs H_cat, coloured by the executed
motion class), ``spread_over_time`` (all three against episode progress: episode-balanced decile mean with SEM over episodes) and ``spread_showcase`` (four
steps: the 256 endpoints weighted by probability, the 1-sigma ellipse, the mean and the executed
endpoint).

    conda run -n vln python paper/frame_influence/action_spread.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = ROOT / "paper_results/frame_influence_keepone/runs/lightnav_influence_keepone"
DEFAULT_OUT = ROOT / "outputs/validation/lightnav_action_spread"
BUNDLE = ROOT / "baselines/LightNav-0/checkpoints/LightNav-0/action_tokenizer"
LIGHTNAV_SRC = ROOT / "baselines/LightNav-0/src"
N_CODES = 256
N_DECILES = 10
RIDGE = np.diag([1e-4, 1e-4, 1e-4])  # (1 cm)^2, (1 cm)^2, (0.57 deg)^2: keeps Sigma invertible on one-hot steps
D_MULTI = 1.5  # d_exec above this: the mean is not the action
STYLE = dict(color="tab:green", marker="o", markersize=4, linestyle="-")
BIN_COLORS = {"stop": "k", "fwd": "tab:green", "fwd_L": "tab:blue", "fwd_R": "tab:red",
              "rot_L": "tab:cyan", "rot_R": "tab:orange"}


def unpack_f16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (average ranks for ties)."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
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


# --- codebook -------------------------------------------------------------------------------
def velocity_bin(forward_m: float, yaw_rad: float, rule: dict) -> str:
    """Executed-action class of a waypoint under the server's velocity_control (meta.json bin_rule)."""
    dt = float(rule["dt"])
    lo, hi = rule["lin_vel_range"]
    alo, ahi = rule["ang_vel_range"]
    v = min(max(forward_m / dt, lo), hi)
    w = min(max(math.degrees(yaw_rad) / dt, alo), ahi)
    if v < rule["min_abs_lin_speed"] and abs(w) < rule["min_abs_ang_speed"]:
        return "stop"
    if v < rule["min_abs_lin_speed"]:
        return "rot_L" if w > 0 else "rot_R"
    if abs(w) < rule["fwd_straight_deg"]:
        return "fwd"
    return "fwd_L" if w > 0 else "fwd_R"


def decode_codebook(meta: dict) -> dict:
    """Decode every coarse code alone. Returns endpoints (256, 3) = (x, y, unwrapped heading),
    first executed waypoint (256, 2) = (forward, yaw) and the motion class per code, checked
    against meta.json's ``code_bins``."""
    sys.path.insert(0, str(LIGHTNAV_SRC))
    from lightnav.traj_vocab import compose_to_abs  # pure numpy

    man = json.loads((BUNDLE / "manifest.json").read_text())
    if man["representation"] != "se2_diff":
        raise ValueError(f"unexpected representation {man['representation']!r}")
    cb = np.load(BUNDLE / man["codebook_files"][0]).astype(np.float64)
    w = np.load(BUNDLE / man["jacobian_weights_file"]).astype(np.float64)
    horizon = int(man["horizon"])
    stop_l0 = int(man["stop"]["l0"])
    rule = meta["bin_rule"]
    endpoints = np.zeros((N_CODES, 3))
    first = np.zeros((N_CODES, 2))
    bins = []
    for k in range(N_CODES):
        feat = (cb[k] / w).reshape(horizon, 3)
        wp = compose_to_abs(feat[None])[0].astype(np.float64)
        if k == stop_l0:
            wp[:] = 0.0  # the explicit stop code decodes to exact zeros (policy.py)
            feat = np.zeros_like(feat)
        endpoints[k] = (wp[-1, 0], wp[-1, 1], feat[:, 2].sum())
        row = next((r for r in wp if abs(r[0]) > 1e-6 or abs(r[2]) > 1e-6), wp[0])
        first[k] = (row[0], row[2])
        bins.append("stop" if k == stop_l0 else velocity_bin(float(row[0]), float(row[2]), rule))
    bad = [k for k in range(N_CODES) if bins[k] != meta["code_bins"][k]]
    if bad:
        raise ValueError(f"offline decode disagrees with meta.json code_bins on codes {bad[:10]}")
    return {"endpoints": endpoints, "first": first, "bins": bins}


# --- per-step quantities ---------------------------------------------------------------------
def gaussian_fit(p: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = p @ e
    d = e - mu
    sigma = (d * p[:, None]).T @ d + RIDGE
    return mu, sigma


def load(run_dir: Path, book: dict) -> list[dict]:
    e = book["endpoints"]
    rows = []
    n_argmax_mismatch = 0
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
            p = np.exp(logp)
            p /= p.sum()
            h = float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
            if abs(h - float(f["entropy"])) > 0.01:
                raise ValueError(f"{label} step {r['step']}: stored entropy {f['entropy']} != recomputed {h}")
            code = int(f["code"])
            if code != int(p.argmax()):
                n_argmax_mismatch += 1
            mu, sigma = gaussian_fit(p, e)
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                raise ValueError(f"{label} step {r['step']}: covariance not positive definite")
            h_gauss = 0.5 * (3 * math.log(2 * math.pi * math.e) + logdet)
            dev = e[code] - mu
            d2 = float(dev @ np.linalg.solve(sigma, dev))
            logn = -0.5 * d2 - 0.5 * (3 * math.log(2 * math.pi) + logdet)
            rows.append({
                "label": label, "step": int(r["step"]), "n_steps": n, "frac": r["step"] / max(n - 1, 1),
                "code": code, "bin": book["bins"][code], "exec_bin": f.get("exec_bin") or "",
                "p_top1": float(p.max()), "H_cat": h, "H_gauss": h_gauss, "d_exec": math.sqrt(d2), "logN_exec": logn,
                "mu_x": float(mu[0]), "mu_y": float(mu[1]), "mu_heading_deg": math.degrees(mu[2]),
                "sigma_x": math.sqrt(sigma[0, 0]), "sigma_y": math.sqrt(sigma[1, 1]),
                "sigma_heading_deg": math.degrees(math.sqrt(sigma[2, 2])),
                "exec_x": float(e[code, 0]), "exec_y": float(e[code, 1]), "exec_heading_deg": math.degrees(e[code, 2]),
                "exec_first_forward": float(book["first"][code, 0]),
                "exec_first_yaw_deg": math.degrees(book["first"][code, 1]),
                "success": int(ep["metrics"]["success"] > 0 and not ep["forced_stop"]),
                "forced_stop": int(ep["forced_stop"]),
                "_p": p,
            })
    print(f"loaded {len(rows)} decisions; executed code != argmax of the stored log-probs on {n_argmax_mismatch} steps")
    return rows


def per_episode(rows: list[dict]) -> list[dict]:
    out = []
    for label in sorted({r["label"] for r in rows}):
        ep = [r for r in rows if r["label"] == label]
        step = np.array([r["step"] for r in ep], dtype=float)
        q = max(1, len(ep) // 4)
        rec = {"label": label, "n_steps": len(ep), "success": ep[0]["success"], "forced_stop": ep[0]["forced_stop"]}
        for key in ("H_cat", "H_gauss", "d_exec"):
            v = np.array([r[key] for r in ep])
            rec[f"{key}_median"] = float(np.median(v))
            rec[f"{key}_rho"] = spearman(step, v)
            rec[f"{key}_q1"] = float(v[:q].mean())
            rec[f"{key}_q4"] = float(v[-q:].mean())
            rec[f"{key}_delta"] = rec[f"{key}_q4"] - rec[f"{key}_q1"]
        rec["frac_multi"] = float(np.mean([r["d_exec"] > D_MULTI for r in ep]))
        rec["rho_cat_gauss"] = spearman(np.array([r["H_cat"] for r in ep]), np.array([r["H_gauss"] for r in ep]))
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
    keys = [k for k in rows[0] if not k.startswith("_")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def save(fig, out: Path) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


# --- figures ---------------------------------------------------------------------------------
def figure_scatter(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for b, c in BIN_COLORS.items():
        sel = [r for r in rows if r["exec_bin"] == b]
        if not sel:
            continue
        ax.scatter([r["H_cat"] for r in sel], [r["H_gauss"] for r in sel], s=7, color=c, alpha=0.5,
                   linewidths=0, label=f"{b} ({len(sel)})")
    rho = spearman(np.array([r["H_cat"] for r in rows]), np.array([r["H_gauss"] for r in rows]))
    ax.set_xlabel("Categorical entropy over 256 codes (nats)")
    ax.set_ylabel("Gaussian entropy of endpoint pose (nats)")
    ax.set_title(f"{len(rows)} decisions, Spearman rho = {rho:.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, title="executed action", title_fontsize=7, loc="lower right")
    save(fig, out)


def figure_over_time(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    labels = sorted({r["label"] for r in rows})
    panels = (("H_cat", "Categorical entropy (nats)"), ("H_gauss", "Gaussian entropy (nats)"),
              ("d_exec", "Executed action, dist. to mean ($\\sigma$)"))
    for ax, (key, ylabel) in zip(axes, panels):
        c, m, s = decile_means(rows, key)
        ax.errorbar(c, m, yerr=s, fmt="none", ecolor=STYLE["color"], elinewidth=1, capsize=0)
        ax.plot(c, m, label=f"mean ({len(labels)} eps)", linewidth=1.8, **STYLE)
        ax.set_xlabel("Episode progress")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[2].axhline(D_MULTI, color="k", linewidth=0.8, linestyle=":")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    save(fig, out)


def pick_showcase(rows: list[dict]) -> list[tuple[str, dict]]:
    hg = np.array([r["H_gauss"] for r in rows])
    de = np.array([r["d_exec"] for r in rows])
    order = np.argsort(hg)
    return [("lowest Gaussian entropy", rows[int(order[0])]),
            ("median Gaussian entropy", rows[int(order[len(order) // 2])]),
            ("highest Gaussian entropy", rows[int(order[-1])]),
            ("executed action farthest from the mean", rows[int(de.argmax())])]


def draw_step(ax, r: dict, book: dict, title: str) -> None:
    """Endpoint plane: horizontal = lateral (left positive, drawn to the left), vertical = forward."""
    e = book["endpoints"]
    p = r["_p"]
    hx, vy = -e[:, 1], e[:, 0]
    order = np.argsort(p)  # most probable drawn last
    ax.scatter(hx[order], vy[order], s=4 + 400 * p[order], c=[BIN_COLORS[b] for b in np.array(book["bins"])[order]],
               alpha=np.clip(0.15 + 2.5 * p[order], 0.15, 1.0), linewidths=0)
    # heading ticks for codes with p > 0.02
    for k in np.flatnonzero(p > 0.02):
        th = e[k, 2]
        ax.plot([hx[k], hx[k] - 0.12 * math.sin(th)], [vy[k], vy[k] + 0.12 * math.cos(th)], color="0.3", linewidth=0.6)
    mu = np.array([r["mu_x"], r["mu_y"]])
    cov = gaussian_fit(p, e)[1][:2, :2]
    # in plot coordinates (h, v) = (-y, x): swap rows and negate y
    m2 = np.array([[0.0, -1.0], [1.0, 0.0]])
    cov_plot = m2 @ cov @ m2.T
    vals, vecs = np.linalg.eigh(cov_plot)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse((-mu[1], mu[0]), 2 * math.sqrt(vals[1]), 2 * math.sqrt(vals[0]), angle=ang,
                         fill=False, edgecolor="k", linewidth=1.0))
    ax.plot(-mu[1], mu[0], marker="x", color="k", markersize=7, markeredgewidth=1.5, linestyle="none")
    ax.plot(-r["exec_y"], r["exec_x"], marker="*", color="gold", markeredgecolor="k", markersize=11, linestyle="none")
    ax.plot(0, 0, marker="^", color="0.4", markersize=6, linestyle="none")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.8)
    ax.set_xlabel("lateral (m), left is left")
    ax.set_ylabel("forward (m)")
    ax.grid(alpha=0.3)
    ax.set_title(f"{title}\n{r['label']} step {r['step']}/{r['n_steps'] - 1}, executed: {r['exec_bin']}\n"
                 f"H_cat {r['H_cat']:.2f}  H_gauss {r['H_gauss']:.2f}  d_exec {r['d_exec']:.2f}  top-1 {r['p_top1']:.2f}",
                 fontsize=7)


def figure_showcase(rows: list[dict], book: dict, out: Path) -> None:
    picks = pick_showcase(rows)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 4.0))
    for ax, (title, r) in zip(axes, picks):
        draw_step(ax, r, book, title)
    handles = [plt.Line2D([], [], marker="o", color=c, linestyle="none", markersize=5, label=b) for b, c in BIN_COLORS.items()]
    handles += [plt.Line2D([], [], marker="x", color="k", linestyle="none", markersize=7, label="pushforward mean + 1-sigma"),
                plt.Line2D([], [], marker="*", color="gold", markeredgecolor="k", linestyle="none", markersize=10, label="executed (greedy) code")]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    plt.close(fig)


# --- summary ---------------------------------------------------------------------------------
def pct(v: np.ndarray) -> dict:
    q = np.percentile(v, [10, 50, 90])
    return {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}


def summarise(rows: list[dict], eps: list[dict]) -> dict:
    hc = np.array([r["H_cat"] for r in rows])
    hg = np.array([r["H_gauss"] for r in rows])
    de = np.array([r["d_exec"] for r in rows])
    out = {"n_decisions": len(rows), "n_episodes": len(eps), "ridge_diag": np.diag(RIDGE).tolist(), "d_multi": D_MULTI,
           "H_cat": pct(hc), "H_gauss": pct(hg), "d_exec": pct(de), "p_top1": pct(np.array([r["p_top1"] for r in rows])),
           "sigma_x_m": pct(np.array([r["sigma_x"] for r in rows])), "sigma_y_m": pct(np.array([r["sigma_y"] for r in rows])),
           "sigma_heading_deg": pct(np.array([r["sigma_heading_deg"] for r in rows])),
           "frac_d_exec_above": float(np.mean(de > D_MULTI)), "frac_d_exec_above_1": float(np.mean(de > 1.0)),
           "rho_cat_gauss_all_steps": spearman(hc, hg),
           "rho_cat_gauss_per_episode_median": float(np.median([e["rho_cat_gauss"] for e in eps]))}
    for key in ("H_cat", "H_gauss", "d_exec"):
        k, n, pv = sign_test([e[f"{key}_delta"] for e in eps])
        rhos = [e[f"{key}_rho"] for e in eps]
        out[f"{key}_over_time"] = {"last_quarter_below_first": f"{k}/{n}", "sign_test_p": pv,
                                   "rho_vs_step_median": float(np.median(rhos)), "rho_negative": f"{sum(r < 0 for r in rhos)}/{len(rhos)}"}
    out["by_exec_bin"] = {}
    for b in BIN_COLORS:
        sel = [r for r in rows if r["exec_bin"] == b]
        if sel:
            out["by_exec_bin"][b] = {"n": len(sel), "H_cat_median": float(np.median([r["H_cat"] for r in sel])),
                                     "H_gauss_median": float(np.median([r["H_gauss"] for r in sel])),
                                     "d_exec_median": float(np.median([r["d_exec"] for r in sel]))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.run_dir / "influence" / "meta.json").read_text())
    book = decode_codebook(meta)
    print(f"codebook decoded: 256 coarse codes, motion classes match meta.json; "
          f"endpoint forward range [{book['endpoints'][:, 0].min():.2f}, {book['endpoints'][:, 0].max():.2f}] m, "
          f"heading range [{math.degrees(book['endpoints'][:, 2].min()):.0f}, {math.degrees(book['endpoints'][:, 2].max()):.0f}] deg")
    rows = load(a.run_dir, book)
    eps = per_episode(rows)
    write_csv(a.out / "action_spread.csv", rows)
    write_csv(a.out / "action_spread_episodes.csv", eps)
    figure_scatter(rows, a.out / "spread_scatter")
    figure_over_time(rows, a.out / "spread_over_time")
    figure_showcase(rows, book, a.out / "spread_showcase")
    summary = summarise(rows, eps)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", a.out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
