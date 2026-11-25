#!/usr/bin/env python3
"""Memory and inference-time benefit of a protocol at budget k = 16 vs the unpruned models, with
PyramidDrop at the same budget on the inference-time panels.

Per-step curves over the episode, mean over the episodes still running at that step (band =
interquartile range across them):

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

Curves: Unpruned (gray), Protocol (k=16) (red; --protocol hybrid | uniform) and, on the two
inference-time panels only, PyramidDrop (k=16) (green; no-window PyramidDrop, Uni-NaVid 64 frames).
PyramidDrop meets the same visual budget, so only its time is compared; its total KV also carries
every past action's text.

The bracket compares Unpruned with the Protocol PAIRED: at step t* it averages both arms over the SAME
episodes -- those still running in both at t* -- and prints the ratio (memory: the GB difference, since
the weights are a floor shared by both); the episode count n is printed to the console. Its ends are those episodes' means,
so they need not sit on the curves, which average every episode still running in each arm (most
episodes end early, so late in the episode each curve rests on a few, mostly different, episodes; the
kinks in the unpruned curve are episodes ending). t* = the last step with >= 1 shared episode
(--edge last) or >= 5 (--edge min5). The PyramidDrop / Protocol inference-time ratio at its own t*
(same rule) is printed to the console only.

Data: paper_results/hybrid_prune/runs/<bench>/, clean re-measurements with ONE ARM AT A TIME on the GPU
(the sweep ran two arms concurrently, so its latencies are not comparable); launcher
paper/hybrid_prune/run_cost_bench.sh. LongNav-R1 is argmax and Uni-NaVid is sampled with
--sampling-seed 30; both reproduce the sweep's trajectories. Figure: cost_k16<bench suffix>
[_uniform][_min5].{pdf,png}; the console prints every paired and unpaired value.

Needs matplotlib, so run in the vln env:
    /home/brabiei/miniconda3/envs/vln/bin/python paper/hybrid_prune/plot_cost.py \\
        --bench paper_results/hybrid_prune/runs/cost_bench_100 --name cost_bench_100 --protocol uniform
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
UNPRUNED, PYRAMIDDROP, PROTOCOL = "Unpruned", "PyramidDrop (k=16)", "Protocol (k=16)"
RUN_DIRS = {  # arm -> (LongNav-R1 run dir, Uni-NaVid run dir)
    UNPRUNED: ("cost_r1_full", "cost_uninavid_full"),
    PYRAMIDDROP: ("cost_r1_pyramiddrop_mp16", "cost_uninavid_pyramiddrop_mp16"),
    "hybrid": ("cost_r1_hybrid_mp16", "cost_uninavid_hybrid_mp16"),
    "uniform": ("cost_r1_uniform_mp16", "cost_uninavid_uniform_mp16"),
}
STYLES = {
    UNPRUNED: dict(color="0.35", linestyle="-", linewidth=1.4),
    PYRAMIDDROP: dict(color="tab:green", linestyle="-", linewidth=1.4),
    PROTOCOL: dict(color="tab:red", linestyle="-", linewidth=1.4),
}
MAX_STEP = 500
MIN_ALIVE = 1  # a step is drawn while at least this many episodes are still running (1: out to the longest episode)
EDGE_MIN_SHARED = {"last": 1, "min5": 5}
DELTA_FONTSIZE = 10  # the bracket labels (ratio / GB difference)

Series = dict[str, np.ndarray]  # episode label -> per-step values


def r1_traces(run: Path) -> dict[str, Series]:
    """Per-episode per-step arrays of the LongNav-R1 run (from each episode's sequence.json)."""
    out: dict[str, Series] = {"kv": {}, "mem": {}, "lat": {}}
    for f in glob.glob(str(run / "rollout/results_*")):
        for line in open(f):
            r = json.loads(line)
            s = json.load(open(r["raw/trace"]))
            label = r["episode_label"]
            out["kv"][label] = np.asarray(s["kv_len"], dtype=float)
            out["mem"][label] = np.asarray(s["vlm_mem_GB"], dtype=float)
            out["lat"][label] = 1000 * np.asarray(s["vlm_latency"], dtype=float)
    return out


UNI_FIELDS = {"tok": ("prompt_tokens", 1.0), "mem": ("cuda_peak_bytes", 1e-9), "lat": ("inference_seconds", 1000.0)}


def uninavid_traces(run: Path) -> dict[str, Series]:
    """Per-episode per-call prompt tokens, peak CUDA memory (GB) and inference time (ms), indexed
    by env step. One call decides two env steps, so each value is held over both (the curve would
    otherwise break on every other step). Fields missing from a trace give no entry."""
    out: dict[str, Series] = {k: {} for k in UNI_FIELDS}
    for f in sorted(glob.glob(str(run / "traces/*.jsonl"))):
        calls = [e for e in map(json.loads, open(f)) if e.get("inference")]
        if not calls:
            continue
        label = calls[0].get("episode_label") or Path(f).stem
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
            out[key][label] = a
    return out


def curve(series: Series):
    """Mean and interquartile range over the episodes alive at each step, up to MAX_STEP."""
    arrs = list(series.values())
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


def alive(a: np.ndarray, t: int) -> bool:
    return len(a) > t and bool(np.isfinite(a[t]))


def paired(ref: Series, prot: Series, min_shared: int, w: int, scale: float):
    """(t*, n, ref mean, protocol mean) over the episodes running in BOTH arms at t*, the last step
    with >= min_shared such episodes; each episode's series is smoothed like the curves first."""
    for t in range(MAX_STEP - 1, -1, -1):
        eps = [e for e in ref if e in prot and alive(ref[e], t) and alive(prot[e], t)]
        if len(eps) >= min_shared:
            r = scale * float(np.mean([smooth(ref[e], w)[t] for e in eps]))
            p = scale * float(np.mean([smooth(prot[e], w)[t] for e in eps]))
            return t, len(eps), r, p
    return None


def draw(ax, series: dict[str, Series], ylabel: str, title: str, key: str, report: dict, min_shared: int,
         w: int = 1, scale: float = 1.0, annot: str = "ratio", ylim=None):
    """One panel. Curves for every arm with data; a paired Unpruned-vs-Protocol bracket at t*; on the
    time panels a PyramidDrop / Protocol ratio in the corner. Paired and unpaired values go to `report`."""
    means = {}
    for label, style in STYLES.items():
        s = series.get(label) or {}
        if not s:
            continue
        mean, lo, hi, _ = curve(s)
        mean, lo, hi = (smooth(v, w) * scale for v in (mean, lo, hi))
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label, **style)
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, linewidth=0)
        means[label] = mean
    ax.set_xlim(0, MAX_STEP)
    if ylim is not None:  # e.g. Uni-NaVid memory: the 7B fp16 weights sit at ~15.8 GB, so start there
        top = max([ylim[1]] + [float(np.nanmax(m)) * 1.02 for m in means.values() if np.isfinite(m).any()])
        ax.set_ylim(ylim[0], top)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("episode step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)

    prot = series.get(PROTOCOL) or {}
    for ref_label in (UNPRUNED, PYRAMIDDROP):
        ref = series.get(ref_label) or {}
        if not ref or not prot:
            continue
        res = paired(ref, prot, min_shared, w, scale)
        if res is None:
            continue
        t, n, r, p = res
        unpaired = (means[ref_label][t] if t < len(means[ref_label]) else np.nan,
                    means[PROTOCOL][t] if t < len(means[PROTOCOL]) else np.nan)
        value = r / p if annot == "ratio" else r - p
        report[(key, ref_label)] = dict(step=t, n=n, ref=r, protocol=p, value=value,
                                        unpaired=(unpaired[0] / unpaired[1] if annot == "ratio" else unpaired[0] - unpaired[1]))
        text = f"{value:.1f}×" if annot == "ratio" else f"−{value:.1f} GB"
        if ref_label == UNPRUNED:
            ax.annotate("", (t, p), (t, r), arrowprops=dict(arrowstyle="<->", color="0.2", linewidth=0.8,
                                                           shrinkA=0, shrinkB=0))
            y0, y1 = ax.get_ylim()
            if abs(r - p) < 0.3 * (y1 - y0):  # short bracket: label under its lower end
                ax.annotate(text, (t, min(r, p)), fontsize=DELTA_FONTSIZE, color="0.2", ha="right", va="top",
                            xytext=(-2, -10), textcoords="offset points")
            else:
                ax.annotate(text, (t, (r + p) / 2), fontsize=DELTA_FONTSIZE, color="0.2", ha="right", va="center",
                            xytext=(-4, 0), textcoords="offset points")
        # The PyramidDrop / Protocol comparison is printed to the console only (report), not drawn.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=BENCH, help="benchmark run directory (runs/cost_bench*)")
    ap.add_argument("--name", default="cost_bench", help="figure name: cost_bench -> cost_k16, else cost_k16_<name suffix>")
    ap.add_argument("--protocol", choices=("hybrid", "uniform"), default="hybrid",
                    help="which protocol is the red 'Protocol (k=16)' curve (uniform adds _uniform to the name)")
    ap.add_argument("--edge", choices=list(EDGE_MIN_SHARED), default="last",
                    help="bracket step: last step with >= 1 (last) or >= 5 (min5, adds _min5) shared episodes")
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

    dirs = {UNPRUNED: RUN_DIRS[UNPRUNED], PYRAMIDDROP: RUN_DIRS[PYRAMIDDROP], PROTOCOL: RUN_DIRS[args.protocol]}
    r1 = {label: r1_traces(d) if (d := find(pair[0])) else {"kv": {}, "mem": {}, "lat": {}} for label, pair in dirs.items()}
    uni = {label: uninavid_traces(d) if (d := find(pair[1])) else {k: {} for k in UNI_FIELDS}
           for label, pair in dirs.items()}
    for label in dirs:
        print(f"{label}: LongNav-R1 {len(r1[label]['kv'])} episodes, Uni-NaVid {len(uni[label]['lat'])} episodes")

    no_pd = (UNPRUNED, PROTOCOL)  # PyramidDrop is compared on inference time only
    ms = EDGE_MIN_SHARED[args.edge]
    report: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2))
    draw(axes[0][0], {l: r1[l]["kv"] for l in no_pd}, "KV slots per layer (×10³)", "LongNav-R1: KV cache",
         "r1_kv", report, ms, scale=1e-3)
    draw(axes[0][1], {l: r1[l]["mem"] for l in no_pd}, "GPU memory (GB)", "LongNav-R1: memory",
         "r1_mem", report, ms, annot="diff")
    draw(axes[0][2], {l: r1[l]["lat"] for l in dirs}, "inference time (ms / step)", "LongNav-R1: inference time",
         "r1_lat", report, ms, w=args.smooth)
    draw(axes[1][0], {l: uni[l]["tok"] for l in no_pd}, "prompt tokens per call", "Uni-NaVid: visual history",
         "uni_tok", report, ms)
    draw(axes[1][1], {l: uni[l]["mem"] for l in no_pd}, "peak GPU memory (GB)", "Uni-NaVid: memory",
         "uni_mem", report, ms, annot="diff", ylim=(15.5, 16.6))
    draw(axes[1][2], {l: uni[l]["lat"] for l in dirs}, "inference time (ms / call)", "Uni-NaVid: inference time",
         "uni_lat", report, ms, w=args.smooth)
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ax.set_title(f"({tag}) {ax.get_title()}", fontsize=9)
    handles, labels = axes[0][2].get_legend_handles_labels()  # the time panel carries all three arms
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"cost_k16{suffix}" + ("" if args.protocol == "hybrid" else f"_{args.protocol}") \
        + ("" if args.edge == "last" else f"_{args.edge}")
    out = args.out / name
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
    print(f"{'panel':<8} {'vs':<20} {'step':>5} {'n':>3} {'paired':>9} {'unpaired':>9}   (ratio; memory panels: GB difference)")
    for (key, ref), v in report.items():
        print(f"{key:<8} {ref:<20} {v['step']:>5} {v['n']:>3} {v['value']:>9.2f} {v['unpaired']:>9.2f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
