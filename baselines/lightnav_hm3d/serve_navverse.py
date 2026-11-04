#!/usr/bin/env python
"""LightNav-0 behind NavVerse's LongNav client: a stateful ``/act`` server.

Speaks the contract of ``src/longnav/scripts/serve.py`` so NavVerse's LongNav agent
(``baselines/longnav_r1/longnav_agent.py`` in the NavVerse clone: one JPEG per step, the goal
word on the first step of an episode, no client-side history, its own STOP at the step cap)
drives LightNav-0 unchanged:

    POST /act   multipart ``file`` (JPEG) + form ``instr_or_goal`` (first step only)
                -> {"action_id": int, "action_probs": [4 floats]}   STOP FORWARD TURN_LEFT TURN_RIGHT
    GET  /health, POST /reset

LightNav-0 predicts a trajectory and executes its first waypoint as a 1 s ``velocity_control``
step on Habitat (forward in [0, 0.25] m, yaw in [-30, 30] deg, both at once;
``habitat_server/lightnav_habitat/objectnav.py``). NavVerse's client only takes discrete moves
(FORWARD 0.25 m, TURN of ``--turn-deg``: 30 for the client's TURN_LEFT/RIGHT, 15 for its
TURN_LEFT_SMALL/RIGHT_SMALL, the closest to LightNav's own 18 deg rotate step), so the step is
discretized here: STOP when Habitat's stop guard would fire (|lin| < 0.025 m/s and |ang| < 1 deg/s
-- the explicit stop code, a decode failure and an all-zero trajectory all produce that zero
command), else whichever of the normalized forward (lin*dt / 0.25 m) or turn (|ang|*dt / turn-deg)
components is larger, ties -> FORWARD. A combined move-and-turn step therefore becomes one or the
other. Action ids 2/3 are "left" / "right" whatever turn size the client maps them to.

The visual history is the model's own full-episode SlowFast prompt. ``--prune-method`` installs
``eval_prune.install_visual_prune`` (eager vLLM, masked attention on every layer) before the
engine is built, so e.g. ``--prune-method recency --prune-k 6 --prune-tokens-per-frame 15`` keeps
the image tokens of the 6 newest history frames and hides the rest while timestamps, vision
markers and the anchor pair stay in the prompt (paper/ctx_ablation, "old blocks kept").

    baselines/.envs/lightnav/bin/python baselines/lightnav_hm3d/serve_navverse.py --port 8000 \
        --aspect-mode keep --expect-video-size 288 384 \
        --prune-method recency --prune-k 6 --prune-tokens-per-frame 15 --prune-log-dir run/prune \
        --config-out run/server_config.json --step-log run/server_steps.jsonl

Run from the LightNav client runtime with ``PYTHONPATH=baselines/LightNav-0/src``.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
ROOT = BASELINES.parent
LIGHTNAV_ROOT = BASELINES / "LightNav-0"
DEFAULT_CKPT = LIGHTNAV_ROOT / "checkpoints" / "LightNav-0"

ACTIONS = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")  # NavVerse --longnav_action_map order
STOP, FORWARD, TURN_LEFT, TURN_RIGHT = range(4)
# Habitat velocity_control settings of LightNav's ObjectNav server (objectnav.py:121-131).
DT = 1.0
LIN_VEL_RANGE = (0.0, 0.25)
ANG_VEL_RANGE = (-30.0, 30.0)
MIN_ABS_LIN = 0.025  # m/s: below both guards Habitat treats the command as STOP
MIN_ABS_ANG = 1.0  # deg/s
# NavVerse's discrete moves (evaluation_worker._convert_action_to_waypoint).
FORWARD_M = 0.25
TURN_DEG = 30.0  # the client's TURN_LEFT/RIGHT; --turn-deg 15 when it maps ids 2/3 to TURN_*_SMALL
INSTRUCTION_TEMPLATE = "Find the {goal}."  # LightNav's HM3D ObjectNav server phrasing
VIDEO_PAD_ID = 151656  # <|video_pad|> in the LightNav-0 (Qwen3-VL) tokenizer
PRUNE_METHODS = ("random", "fastv", "divprune", "dart", "recency")


# --- pure helpers (no model) ---------------------------------------------------------------
def instruction_for(goal: str, template: str = INSTRUCTION_TEMPLATE) -> str:
    """NavVerse posts the bare object word (``tv_monitor``); HM3D's server says ``Find the tv monitor.``"""
    return template.format(goal=str(goal).strip().replace("_", " "))


def raw_to_velocity(raw: float, vmin: float, vmax: float) -> float:
    """Habitat's ``scaled = vmin + (raw + 1) / 2 * (vmax - vmin)`` with ``raw`` clipped to [-1, 1]."""
    raw = float(np.clip(float(raw), -1.0, 1.0))
    return float(vmin + (raw + 1.0) / 2.0 * (vmax - vmin))


def velocity_from_cmd(cmd: Dict[str, Any], lin_vel_range: Tuple[float, float] = LIN_VEL_RANGE,
                      ang_vel_range: Tuple[float, float] = ANG_VEL_RANGE) -> Tuple[float, float]:
    """``(lin m/s, ang deg/s)`` Habitat would execute for a ``velocity_control`` action dict."""
    if cmd.get("action") != "velocity_control":
        raise ValueError(f"expected a velocity_control action, got {cmd.get('action')!r}")
    args = cmd["action_args"]
    return (raw_to_velocity(args["linear_velocity"], *lin_vel_range),
            raw_to_velocity(args["angular_velocity"], *ang_vel_range))


def discretize(lin_mps: float, ang_dps: float, *, dt: float = DT, min_abs_lin: float = MIN_ABS_LIN,
               min_abs_ang: float = MIN_ABS_ANG, forward_m: float = FORWARD_M,
               turn_deg: float = TURN_DEG) -> Tuple[int, str]:
    """One NavVerse action id for a 1-step velocity command, and the rule that chose it."""
    if abs(lin_mps) < min_abs_lin and abs(ang_dps) < min_abs_ang:
        return STOP, "stop_guard"
    forward = abs(lin_mps) * dt / forward_m if lin_mps > 0 else 0.0
    turn = abs(ang_dps) * dt / turn_deg
    if forward >= turn and forward > 0.0:
        return FORWARD, "forward"
    return (TURN_LEFT, "turn_left") if ang_dps > 0 else (TURN_RIGHT, "turn_right")


def one_hot(action_id: int) -> list:
    probs = [0.0] * len(ACTIONS)
    probs[int(action_id)] = 1.0
    return probs


def discretization_constants(turn_deg: float = TURN_DEG) -> Dict[str, Any]:
    return {"dt_s": DT, "lin_vel_range_mps": list(LIN_VEL_RANGE), "ang_vel_range_dps": list(ANG_VEL_RANGE),
            "stop_guard": {"min_abs_lin_mps": MIN_ABS_LIN, "min_abs_ang_dps": MIN_ABS_ANG},
            "forward_m": FORWARD_M, "turn_deg": turn_deg,
            "rule": "STOP if both stop guards fire; else argmax(lin*dt/forward_m, |ang|*dt/turn_deg), ties -> FORWARD; "
                    "TURN_LEFT for ang > 0 (CCW)",
            "action_ids": dict(zip(ACTIONS, range(len(ACTIONS))))}


def waypoint_fields(info: Dict[str, Any]) -> Dict[str, Any]:
    """Step-log fields from ``policy.get_info()``: the executed waypoint and the decode status."""
    out: Dict[str, Any] = {"raw_text": info.get("raw_text"), "decode_ok": "predicted_traj" in info,
                           "waypoint_index": info.get("action_waypoint_index"),
                           "fwd_m": None, "lat_m": None, "yaw_deg": None,
                           "rvq_codes": info.get("rvq_codes")}
    wp = info.get("action_waypoint")
    if wp is not None:
        wp = np.asarray(wp, dtype=np.float64).reshape(-1)
        out.update(fwd_m=round(float(wp[0]), 5), lat_m=round(float(wp[1]), 5),
                   yaw_deg=round(float(np.degrees(wp[2])), 4))
    return out


def _peak_mem_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return None


def _git_sha(path: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in ("torch", "vllm", "transformers", "lightnav", "fastapi"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


# --- model ---------------------------------------------------------------------------------
def install_prompt_probe(engine) -> Dict[str, Any]:
    """Record the prompt length and ``<|video_pad|>`` count of every generate on the instance.

    Set on the instance so it wraps whatever ``llm_generate_batch`` the class holds (the pruning
    wrapper's replacement when installed). LightNav-0 rebuilds the whole prompt every step, so the
    prompt length is what the step log reports as ``kv_len``.
    """
    stats: Dict[str, Any] = {}
    original = engine.llm_generate_batch

    def probed(items, max_new_tokens):
        ids = [int(t) for t in items[0].prompt_ids]
        stats["prompt_tokens"] = len(ids)
        stats["n_visual"] = sum(1 for t in ids if t == VIDEO_PAD_ID)
        return original(items, max_new_tokens)

    engine.llm_generate_batch = probed
    return stats


def build_served(args: argparse.Namespace):
    """Engine + policy (+ pruning wrapper), the server's single inference owner."""
    prune_state = None
    last_prune: Dict[str, Any] = {}
    if args.prune_method:
        try:
            from baselines.lightnav_hm3d.eval_prune import PruneState, install_visual_prune
        except ImportError:  # run as a script: sys.path[0] is this directory
            from eval_prune import PruneState, install_visual_prune  # type: ignore
        prune_state = PruneState(args.prune_method, args.prune_k, args.prune_tokens_per_frame, args.prune_pivots,
                                 args.prune_seed, args.prune_debug_steps, args.prune_current_frames)
        if args.prune_log_dir:
            # One sub-directory per server launch: the wrapper numbers episodes from 0 per process and
            # appends to episode_NNN.jsonl, so a resumed arm must not share a directory with its first launch.
            prune_state.log_dir = prune_log_dir_for(Path(args.prune_log_dir))
            prune_state.log_dir.mkdir(parents=True, exist_ok=True)
            (prune_state.log_dir / "meta.json").write_text(json.dumps(prune_state.describe(), indent=2) + "\n")
        original_log = prune_state.log_record

        def log_record(rec):
            last_prune["rec"] = rec
            original_log(rec)

        prune_state.log_record = log_record
        install_visual_prune(prune_state)  # before anything imports vllm
        print(f"[prune] installed: {prune_state.method} k={prune_state.k} tokens_per_frame={prune_state.tokens_per_frame} "
              f"budget={prune_state.budget} current frames={prune_state.current_frames}"
              + (f", records -> {prune_state.log_dir}" if prune_state.log_dir else ""), flush=True)

    from lightnav.habitat.policy import TrajVocabVLNCEPolicy
    from lightnav.habitat.runner import HabitatEvalConfig, _apply_decode_budget, resolve_action_decoder
    from lightnav.inference import InferenceConfig, build_engine

    inf_cfg = InferenceConfig(model_path=str(args.model_path), backend=args.backend, max_new_tokens=args.max_new_tokens,
                              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                              aspect_mode=args.aspect_mode)
    engine, bundle = build_engine(inf_cfg, task_type="vlnce", max_new_tokens=args.max_new_tokens)
    eval_cfg = HabitatEvalConfig(model_path=str(args.model_path), backend=args.backend, aspect_mode=args.aspect_mode,
                                 gpu_memory_utilization=args.gpu_memory_utilization, max_new_tokens=args.max_new_tokens)
    centroids, rvq_bundle = resolve_action_decoder(eval_cfg, bundle)
    max_new_before = int(getattr(engine, "max_new_tokens", args.max_new_tokens))
    _apply_decode_budget(engine, bundle, rvq_bundle)
    policy = TrajVocabVLNCEPolicy(engine, centroids, dt=DT, lin_vel_range=LIN_VEL_RANGE, ang_vel_range=ANG_VEL_RANGE,
                                  num_history_frames=int(bundle.num_history_frames), rvq_bundle=rvq_bundle)
    prompt_stats = install_prompt_probe(engine)
    facts: Dict[str, Any] = {"num_history_frames": int(bundle.num_history_frames),
                             "base_video_size": [int(v) for v in bundle.video_size],
                             "action_method": getattr(bundle, "action_method", None),
                             "max_new_tokens_before": max_new_before,
                             "max_new_tokens": int(getattr(engine, "max_new_tokens", max_new_before))}
    try:
        llm = getattr(engine, "vllm_engine", None) or getattr(engine, "llm", None)
        cfg = llm.llm_engine.vllm_config
        facts.update(enforce_eager=bool(cfg.model_config.enforce_eager), max_model_len=int(cfg.model_config.max_model_len),
                     cudagraph_mode=str(getattr(cfg.compilation_config, "cudagraph_mode", None)))
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        facts["vllm_config_error"] = repr(exc)
    return SimpleNamespace(policy=policy, engine=engine, bundle=bundle, prune=prune_state, prompt_stats=prompt_stats,
                           last_prune=last_prune, facts=facts)


def prune_log_dir_for(base: Path, now: Optional[float] = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    path = base / stamp
    n = 1
    while path.exists():
        n += 1
        path = base / f"{stamp}_{n}"
    return path


# --- app -----------------------------------------------------------------------------------
def create_app(served_factory, info=None, step_log=None, instruction_template: str = INSTRUCTION_TEMPLATE,
               expect_video_size: Optional[Sequence[int]] = None, on_ready=None, turn_deg: float = TURN_DEG):
    """``served_factory()`` -> object with ``policy`` (reset/act/get_info/agent), optional ``prune``
    (PruneState), ``prompt_stats`` and ``last_prune`` dicts. ``info`` is served at /health,
    ``step_log`` (path) receives one JSON line per /act call, ``on_ready(served)`` runs after the build."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    @asynccontextmanager
    async def lifespan(app):
        app.state.served = served_factory()
        app.state.lock = Lock()
        app.state.episode = 0  # increments on every goal (= episode start)
        app.state.step = 0
        app.state.goal = None
        app.state.instruction = None
        app.state.step_log = open(step_log, "a") if step_log else None
        if on_ready is not None:
            on_ready(app.state.served)
        yield
        if app.state.step_log:
            app.state.step_log.close()
        del app.state.served

    application = FastAPI(lifespan=lifespan)

    @application.get("/health")
    def health():
        return {"ok": True, **(info or {})}

    @application.post("/reset")
    def reset():
        # Clears the conversation only: the policy is reset when the next goal arrives, so the
        # pruning wrapper's episode counter (bumped by policy.reset) stays equal to the step log's.
        with application.state.lock:
            application.state.goal = None
            application.state.instruction = None
            application.state.step = 0
        return {"ok": True}

    @application.post("/act")
    def act(file: UploadFile = File(...), instr_or_goal: Optional[str] = Form(None)):
        rgb = np.asarray(Image.open(io.BytesIO(file.file.read())).convert("RGB"), dtype=np.uint8)
        state = application.state
        with state.lock:  # one policy, one episode, one in-flight generate
            served = state.served
            if instr_or_goal is None and state.goal is None:
                raise HTTPException(400, "The first request must include instr_or_goal")
            if instr_or_goal is not None:
                state.episode += 1
                state.step = 0
                state.goal = instr_or_goal
                state.instruction = instruction_for(instr_or_goal, instruction_template)
                served.policy.reset({"instruction": {"text": state.instruction}})
            t0 = time.time()
            cmd = served.policy.act({"rgb": rgb})
            latency_ms = (time.time() - t0) * 1000.0
            state.step += 1
            video_size = getattr(getattr(served.policy, "agent", None), "video_size", None)
            if expect_video_size is not None and state.step == 1:
                got = None if video_size is None else [int(v) for v in video_size]
                if got != [int(v) for v in expect_video_size]:
                    raise RuntimeError(f"model frame size {got} != expected {list(expect_video_size)} "
                                       f"(frame {rgb.shape[1]}x{rgb.shape[0]}); the token budget assumes the expected grid")
            lin, ang = velocity_from_cmd(cmd, served.policy.lin_vel_range, served.policy.ang_vel_range) \
                if hasattr(served.policy, "lin_vel_range") else velocity_from_cmd(cmd)
            action, reason = discretize(lin, ang, turn_deg=turn_deg)
            if state.step_log:
                stats = getattr(served, "prompt_stats", None) or {}
                prune_rec = (getattr(served, "last_prune", None) or {}).get("rec") or {}
                record = {"episode": state.episode, "goal": state.goal, "step": state.step,
                          "latency_ms": round(latency_ms, 2), "action_id": int(action),
                          "action_probs": one_hot(action),
                          "kv_len": stats.get("prompt_tokens"), "kv_len_layer_max": stats.get("prompt_tokens"),
                          "peak_mem_GB": _peak_mem_gb(), "time": time.time(),
                          "instruction": state.instruction, "reason": reason,
                          "lin_mps": round(lin, 5), "ang_dps": round(ang, 4),
                          **waypoint_fields(served.policy.get_info()),
                          "prompt_tokens": stats.get("prompt_tokens"), "n_visual": stats.get("n_visual"),
                          "video_size": None if video_size is None else [int(v) for v in video_size],
                          "frame_hw": [int(rgb.shape[0]), int(rgb.shape[1])]}
                if served.prune is not None:
                    record.update(prune_episode=int(served.prune.episode),
                                  prune_step=prune_rec.get("step"), n_kept=sum(prune_rec.get("n_kept", []) or []),
                                  budget=prune_rec.get("budget"), n_pool=prune_rec.get("n_pool"),
                                  ledger=prune_rec.get("ledger"))
                state.step_log.write(json.dumps(record) + "\n")
                state.step_log.flush()
        return {"action_id": int(action), "action_probs": one_hot(action)}

    return application


# --- CLI -----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--arm", default=None, help="free-text arm name recorded in /health and the config")
    ap.add_argument("--model-path", default=str(DEFAULT_CKPT))
    ap.add_argument("--backend", default="vllm_local", choices=("vllm_local", "hf"))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--aspect-mode", default="keep", choices=("stretch", "keep"),
                    help="keep: the source aspect ratio at the checkpoint's pixel budget (640x480 -> 384x288); "
                         "stretch: upstream's default resize to 448x256")
    ap.add_argument("--expect-video-size", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="fail loudly if the model frame size differs (the pruning budget assumes its token grid)")
    ap.add_argument("--instruction-template", default=INSTRUCTION_TEMPLATE)
    ap.add_argument("--turn-deg", type=float, default=TURN_DEG,
                    help="the turn the client executes for action ids 2/3 (30: TURN_LEFT/RIGHT; 15: TURN_*_SMALL); "
                         "normalizes the turn component of the discretization rule")
    ap.add_argument("--prune-method", choices=PRUNE_METHODS, default=None)
    ap.add_argument("--prune-k", type=int, default=None, help="history frames' worth of visual tokens kept")
    ap.add_argument("--prune-tokens-per-frame", type=int, default=None,
                    help="tokens one history frame is worth under the served frame size (15 for 384x288: a pooled "
                         "2-frame pair holds 30; 14 for 448x256); required with --prune-method")
    ap.add_argument("--prune-current-frames", type=int, choices=(1, 2), default=2)
    ap.add_argument("--prune-pivots", type=int, default=8)
    ap.add_argument("--prune-seed", type=int, default=0)
    ap.add_argument("--prune-debug-steps", type=int, default=0)
    ap.add_argument("--prune-log-dir", type=Path, default=None, help="per-episode pruning records (a sub-directory per launch)")
    ap.add_argument("--config-out", type=Path, default=None, help="write the effective config as JSON")
    ap.add_argument("--step-log", type=Path, default=None, help="append one JSON line per /act call")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.prune_method:
        if args.prune_k is None or args.prune_tokens_per_frame is None:
            raise SystemExit("--prune-method needs --prune-k and --prune-tokens-per-frame")
        if args.prune_k < 0 or args.prune_tokens_per_frame < 1:
            raise SystemExit("--prune-k must be >= 0 and --prune-tokens-per-frame >= 1")
    elif args.prune_k is not None or args.prune_tokens_per_frame is not None:
        raise SystemExit("--prune-k / --prune-tokens-per-frame need --prune-method")
    return args


def server_info(args: argparse.Namespace) -> Dict[str, Any]:
    prune = None
    if args.prune_method:
        prune = {"method": args.prune_method, "k": args.prune_k, "tokens_per_frame": args.prune_tokens_per_frame,
                 "budget": args.prune_k * args.prune_tokens_per_frame, "current_frames": args.prune_current_frames,
                 "log_dir": None if args.prune_log_dir is None else str(args.prune_log_dir)}
    return {"host": args.host, "port": args.port, "arm": args.arm, "model": "LightNav-0", "model_path": str(args.model_path),
            "backend": args.backend, "aspect_mode": args.aspect_mode, "expect_video_size": args.expect_video_size,
            "gpu_memory_utilization": args.gpu_memory_utilization, "max_num_seqs": args.max_num_seqs,
            "max_new_tokens": args.max_new_tokens, "instruction_template": args.instruction_template,
            "prune": prune, "discretization": discretization_constants(args.turn_deg),
            "kv_len_meaning": "prompt tokens of the step (LightNav-0 re-prefills its whole prompt every step)"}


def write_config(path: Path, info: Dict[str, Any], served=None) -> None:
    doc = {"info": info, "argv": sys.argv, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "VLN_VLLM_ENFORCE_EAGER",
                                                   "VLLM_ENABLE_V1_MULTIPROCESSING", "HF_HUB_OFFLINE")},
           "versions": _versions(), "repo_sha": _git_sha(ROOT), "lightnav_sha": _git_sha(LIGHTNAV_ROOT)}
    download = Path(info["model_path"]) / "DOWNLOAD.json"
    if download.is_file():
        try:
            doc["checkpoint"] = json.loads(download.read_text())
        except ValueError:
            doc["checkpoint"] = download.read_text()
    if served is not None:
        doc["engine"] = served.facts
        if served.prune is not None:
            doc["prune"] = {**served.prune.describe(), "log_dir": str(served.prune.log_dir)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")


def main(argv=None) -> None:
    import uvicorn
    args = parse_args(argv)
    info = server_info(args)
    if args.config_out:
        write_config(args.config_out, info)
    if args.step_log:
        args.step_log.parent.mkdir(parents=True, exist_ok=True)

    def on_ready(served):
        info["engine"] = served.facts
        if args.config_out:
            write_config(args.config_out, info, served)
        print(f"[serve] ready: {json.dumps(served.facts, default=str)}", flush=True)

    app = create_app(lambda: build_served(args), info=info, step_log=args.step_log,
                     instruction_template=args.instruction_template, expect_video_size=args.expect_video_size,
                     on_ready=on_ready, turn_deg=args.turn_deg)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
