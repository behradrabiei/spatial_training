#!/usr/bin/env python3
"""Memory and inference-time benefit of the protocol at budget k = 16 vs the unpruned models.

Writes paper_results/hybrid_prune/figures/cost_k16.{pdf,png} (17 long episodes, below) or
cost_k16_seed17.{pdf,png} (--bench runs/cost_bench_seed17 --name cost_bench_seed17: the 10
seed-17 episodes of hm3d_v2_10_seed17.json, not selected by length): per-step curves over the episode,
mean over the episodes still running at that step (band = interquartile range across them):

  (a) LongNav-R1  KV-cache size, slots per decoder layer (layer mean; text and visual)
  (b) LongNav-R1  GPU memory of the VLM actor (GB; the model weights are the floor)
  (c) LongNav-R1  inference time per step (ms)
  (d) Uni-NaVid   prompt tokens per LLM call: the instruction plus the visual history the model
                  keeps (4 tokens per frame in its per-frame feature cache, 1-token long-term slots
                  beyond 64 frames), i.e. its memory of the episode
  (e) Uni-NaVid   peak GPU memory during the call (GB; the y-axis starts just under the 15.8 GB
                  fp16 weight floor, so the cache is what is visible)
  (f) Uni-NaVid   inference time per LLM call (ms; one call per two env steps)
Uni-NaVid rows are held over the two env steps each call decides.

Data: paper_results/hybrid_prune/runs/cost_bench/, a clean re-measurement of both arms of each
model, ONE ARM AT A TIME on the GPU (the sweep ran two arms concurrently, so its latencies are not
comparable), on the 17 HM3D-v2 val episodes that run >= 250 steps under either LongNav-R1 arm
(src/longnav/conf/episode_jsons/hm3d_v2_17_long.json; launcher paper/hybrid_prune/run_cost_bench.sh). LongNav-R1 is argmax and reproduces the sweep's trajectories; Uni-NaVid is
sampled with --sampling-seed 30. Unpruned = r1_full (whole episode in the cache) and Uni-NaVid's
native evaluator (its own per-frame cache with long-term compression); protocol = the hybrid arm
at k = 16 (paper/hybrid_prune/README.md). Numbers at the right edge are the unpruned / protocol
ratio at the last step at which both arms still have >= MIN_ALIVE episodes (memory: the
difference in GB, since the model weights are a floor shared by both arms).

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_cost.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "paper_results/hybrid_prune"
BENCH = ROOT / "runs/cost_bench"
ARMS = {  # label -> (LongNav-R1 run dir, Uni-NaVid run dir), style
    "Unpruned": ("cost_r1_full", "cost_uninavid_full", dict(color="0.35", linestyle="-", linewidth=1.4)),
    "Protocol (k=16)": ("cost_r1_hybrid_mp16", "cost_uninavid_hybrid_mp16", dict(color="tab:red", linestyle="-", linewidth=1.4)),
}
MAX_STEP = 500
MIN_ALIVE = 1  # a step is drawn while at least this many episodes are still running (1: out to the longest episode)


def r1_traces(run: Path) -> dict[str, list[np.ndarray]]:
    """Per-episode per-step arrays of the LongNav-R1 run (from each episode's sequence.json)."""
    out = {"kv": [], "mem": [], "lat": []}
    for f in glob.glob(str(run / "rollout/results_*")):
        for line in open(f):
            r = json.loads(line)
            s = json.load(open(r["raw/trace"]))
            out["kv"].append(np.asarray(s["kv_len"], dtype=float))
            out["mem"].append(np.asarray(s["vlm_mem_GB"], dtype=float))
            out["lat"].append(1000 * np.asarray(s["vlm_latency"], dtype=float))
    return out


UNI_FIELDS = {"tok": ("prompt_tokens", 1.0), "mem": ("cuda_peak_bytes", 1e-9), "lat": ("inference_seconds", 1000.0)}


def uninavid_traces(run: Path) -> dict[str, list[np.ndarray]]:
    """Per-episode per-call prompt tokens, peak CUDA memory (GB) and inference time (ms), indexed
    by env step. One call decides two env steps, so each value is held over both (the curve would
    otherwise break on every other step). Fields missing from a trace give an empty series."""
    out = {k: [] for k in UNI_FIELDS}
    for f in sorted(glob.glob(str(run / "traces/*.jsonl"))):
        calls = [e for e in map(json.loads, open(f)) if e.get("inference")]
        if not calls:
            continue
        T = max(int(e["step"]) for e in calls) + 1
        for key, (field, scale) in UNI_FIELDS.items():
            if any(e.get(field) is None for e in calls):
                continue
            a = np.full(T, np.nan)
            for e in calls:
                a[int(e["step"])] = scale * float(e[field])
            for t in range(1, T):  # hold the last call's value over the steps it decided
                if not np.isfinite(a[t]):
                    a[t] = a[t - 1]
            out[key].append(a)
    return out


def curve(arrs: list[np.ndarray]):
    """Mean and interquartile range over the episodes alive at each step, up to MAX_STEP."""
    T = min(MAX_STEP, max((len(a) for a in arrs), default=0))
    mean, lo, hi, n = (np.full(T, np.nan) for _ in range(4))
    for t in range(T):
        v = np.array([a[t] for a in arrs if len(a) > t and np.isfinite(a[t])])
        n[t] = len(v)
        if len(v) >= MIN_ALIVE:
            mean[t], lo[t], hi[t] = v.mean(), np.percentile(v, 25), np.percentile(v, 75)
    return mean, lo, hi, n


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average that ignores NaNs (latency is noisy step to step)."""
    if w <= 1:
        return y
    ok = np.isfinite(y).astype(float)
    k = np.ones(w)
    num = np.convolve(np.nan_to_num(y), k, mode="same")
    den = np.convolve(ok, k, mode="same")
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return np.where(np.isfinite(y), out, np.nan)


def draw(ax, series: dict[str, list[np.ndarray]], ylabel: str, title: str, w: int = 1, scale: float = 1.0,
         annot: str = "ratio", ylim=None):
    """One panel; returns the unpruned / protocol ratio at the last common step (or None). The
    label between the curves at that step is the ratio, or the difference when annot="diff"
    (memory: the model weights are a floor shared by both arms, so the ratio understates it)."""
    last = {}
    for label, arrs in series.items():
        if not arrs:
            continue
        mean, lo, hi, n = curve(arrs)
        mean, lo, hi = (smooth(v, w) * scale for v in (mean, lo, hi))
        x = np.arange(len(mean))
        style = ARMS[label][2]
        ax.plot(x, mean, label=label, **style)
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, linewidth=0)
        ok = np.where(np.isfinite(mean))[0]
        if len(ok):
            last[label] = (ok[-1], mean[ok[-1]])
    ax.set_xlim(0, MAX_STEP)
    if ylim is not None:  # e.g. Uni-NaVid memory: the 7B fp16 weights sit at ~15.8 GB, so start there
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("episode step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    if len(last) == 2:
        t = min(v[0] for v in last.values())
        vals = {}
        for label, arrs in series.items():
            mean = smooth(curve(arrs)[0], w) * scale
            vals[label] = mean[t]
        u, p = vals["Unpruned"], vals["Protocol (k=16)"]
        if np.isfinite(u) and np.isfinite(p) and p > 0:
            text = f"{u / p:.1f}×" if annot == "ratio" else f"−{u - p:.1f} GB"
            ax.annotate("", (t, p), (t, u), arrowprops=dict(arrowstyle="<->", color="0.2", linewidth=0.8, shrinkA=0, shrinkB=0))
            ax.annotate(text, (t, (u + p) / 2), fontsize=8, color="0.2", ha="right", va="center",
                        xytext=(-4, 0), textcoords="offset points")
            return u / p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=BENCH, help="benchmark run directory (runs/cost_bench*)")
    ap.add_argument("--name", default="cost_bench", help="figure name: cost_bench -> cost_k16, else cost_k16_<name suffix>")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--smooth", type=int, default=9, help="moving-average window (steps) for the latency panels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.name == "cost_bench" else "_" + args.name.replace("cost_bench_", "")

    def find(arm: str) -> Path | None:  # LongNav-R1 runs carry the benchmark's suffix in their run name
        for d in (args.bench / f"{arm}{suffix}", args.bench / arm):
            if d.is_dir():
                return d
        return None

    r1 = {label: r1_traces(d) if (d := find(dirs[0])) else {"kv": [], "mem": [], "lat": []}
          for label, dirs in ARMS.items()}
    uni = {label: uninavid_traces(d) if (d := find(dirs[1])) else {k: [] for k in UNI_FIELDS}
           for label, dirs in ARMS.items()}
    for label in ARMS:
        print(f"{label}: LongNav-R1 {len(r1[label]['kv'])} episodes, Uni-NaVid {len(uni[label]['lat'])} episodes")

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2))
    ratios = {}
    ratios["kv"] = draw(axes[0][0], {l: r1[l]["kv"] for l in ARMS}, "KV slots per layer (×10³)",
                        "LongNav-R1: KV cache", scale=1e-3)
    ratios["mem"] = draw(axes[0][1], {l: r1[l]["mem"] for l in ARMS}, "GPU memory (GB)", "LongNav-R1: memory", annot="diff")
    ratios["lat"] = draw(axes[0][2], {l: r1[l]["lat"] for l in ARMS}, "inference time (ms / step)",
                         "LongNav-R1: inference time", w=args.smooth)
    ratios["uni_tok"] = draw(axes[1][0], {l: uni[l]["tok"] for l in ARMS}, "prompt tokens per call",
                             "Uni-NaVid: visual history")
    ratios["uni_mem"] = draw(axes[1][1], {l: uni[l]["mem"] for l in ARMS}, "peak GPU memory (GB)",
                             "Uni-NaVid: memory", annot="diff", ylim=(15.5, 16.6))
    ratios["uni_lat"] = draw(axes[1][2], {l: uni[l]["lat"] for l in ARMS}, "inference time (ms / call)",
                             "Uni-NaVid: inference time", w=args.smooth)
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ax.set_title(f"({tag}) {ax.get_title()}", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out / f"cost_k16{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    print("ratios at the last common step:", {k: (round(v, 2) if v else None) for k, v in ratios.items()})
    plt.close(fig)


if __name__ == "__main__":
    main()
