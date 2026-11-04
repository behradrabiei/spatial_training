"""Run ``lightnav.cli.eval_habitat`` with the visual history cut to the last K observed frames.

LightNav-0 keeps every RGB observation of an episode (frame ``i`` at buffer index ``i``) and
samples its history by age with the checkpoint's SlowFast tiers
(``lightnav.inference.samples._build_slowfast_eval_sample`` ->
``slowfast_video_segments(buffer, total - 1, total, tiers)``). ``--num_history_frames`` does
not shorten that buffer for SlowFast checkpoints, so the context window is applied here, in
the harness, without touching the upstream clone:

* when ``total <= K`` the original sampler runs unchanged;
* when ``total > K`` the segment layout is the one upstream computes at step ``K - 1`` of a
  fresh episode (``slowfast_segments(K - 1, K, tiers)``), every frame id is shifted by
  ``total - K`` and the full buffer is indexed with the shifted ids.

Shifting instead of slicing the buffer keeps the model input identical to an episode that
started ``K`` steps ago: the ``<t.t seconds>`` prefixes are relative to the newest frame
(``timestamp_relative`` in ``eval_config.json``), so they depend on age only, the prompt has
no step counter, and the ViT-tubelet cache (keyed by absolute frame id) keeps returning the
embeds of the right frames. Everything else -- tiers, pooling, prompt, decoding -- is the
model's own.

Runs in the LightNav client runtime (``baselines/.envs/lightnav``) with
``PYTHONPATH=baselines/LightNav-0/src``; ``baselines/run_lightnav.sh`` uses it when
``CONTEXT_FRAMES`` is set:

    python baselines/lightnav_hm3d/eval_ctx.py --context-frames 8 <eval_habitat args...>
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Sequence


def windowed_segments(
    episode_frames: Any,
    current_abs: int,
    n_available: int,
    tiers: Sequence[Dict[str, Any]],
    k: int,
    original: Callable[..., List[Dict[str, Any]]],
    slowfast_segments: Callable[..., List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """``slowfast_video_segments`` restricted to the newest ``k`` frames of ``episode_frames``.

    ``original`` is upstream's ``slowfast_video_segments`` and ``slowfast_segments`` upstream's
    layout function (passed in so this stays importable without ``lightnav``). The call
    convention of the SlowFast eval sample builder (``current_abs == total - 1`` and
    ``n_available == total``) is asserted so a change upstream fails loudly instead of
    windowing the wrong frames.
    """
    total = int(episode_frames.shape[0])
    if current_abs != total - 1 or n_available != total:
        raise AssertionError(
            "unexpected SlowFast call convention: "
            f"total={total} current_abs={current_abs} n_available={n_available}"
        )
    if k < 1:
        raise ValueError(f"context window must be >= 1, got {k}")
    if total <= k:
        return original(episode_frames, current_abs, n_available, tiers)
    offset = total - k
    out: List[Dict[str, Any]] = []
    for seg in slowfast_segments(k - 1, k, tiers):  # layout of a fresh k-frame episode
        ids = [int(i) + offset for i in seg["frame_ids"]]
        out.append(
            {
                "video": episode_frames[ids],
                "frame_indices": ids,
                "total_frames": total,
                "pool_spatial": seg["pool_spatial"],
                "pool_mode": seg["pool_mode"],
            }
        )
    return out


def install_context_window(k: int, debug_steps: int = 0) -> None:
    """Patch ``lightnav.inference.samples.slowfast_video_segments`` for this process.

    The sample builder looks the name up in its module globals at call time, so this is the
    one binding that takes effect (``lightnav.slowfast.slowfast_video_segments`` and
    ``build_vln_traj_sample`` are imported by name elsewhere and would not).
    """
    import lightnav.inference.samples as samples
    from lightnav.slowfast import slowfast_segments

    original = samples.slowfast_video_segments
    if getattr(original, "_context_frames", None) is not None:
        raise RuntimeError("context window already installed")
    calls = {"n": 0}

    def patched(episode_frames, current_abs, n_available, tiers):
        segs = windowed_segments(
            episode_frames, current_abs, n_available, tiers, k, original, slowfast_segments
        )
        if calls["n"] < debug_steps:
            calls["n"] += 1
            ids = [list(s["frame_indices"]) for s in segs]
            pools = [int(s["pool_spatial"]) for s in segs]
            print(
                f"[ctx] call={calls['n']} total={int(episode_frames.shape[0])} "
                f"window={k} frame_ids={ids} pool_spatial={pools}",
                flush=True,
            )
        return segs

    patched._context_frames = k  # type: ignore[attr-defined]
    samples.slowfast_video_segments = patched


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_ctx",
        description="lightnav.cli.eval_habitat with the visual history cut to the newest K "
        "frames (current frame included). Every other argument is passed through.",
    )
    p.add_argument(
        "--context-frames",
        type=int,
        required=True,
        help="Frames visible per inference including the current one (K >= 1).",
    )
    p.add_argument(
        "--ctx-debug-steps",
        type=int,
        default=0,
        help="Print the sampled frame ids of the first N inferences (smoke checks).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args, rest = build_parser().parse_known_args(argv)
    if args.context_frames < 1:
        raise SystemExit("--context-frames must be >= 1")
    install_context_window(args.context_frames, args.ctx_debug_steps)
    print(f"[ctx] context_frames={args.context_frames} (newest frames, current included)", flush=True)
    from lightnav.cli.eval_habitat import main as upstream_main

    return upstream_main(rest)


if __name__ == "__main__":
    sys.exit(main())
