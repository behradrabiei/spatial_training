#!/usr/bin/env python3
"""Re-drive the showcase episode of paper/selector_kl to save its raw RGB observations.

The probe run's video.mp4 frames carry burned-in text and a lossy encode, so the figure's frame
comes from a fresh, deterministic re-drive: one habitat sim (Ray actor, vln env) and the
LongNav-R1 worker in-process on the probe preset (selector probe off -- decisions do not depend
on it), exactly as paper/layer_token_info/probe_episode.py's survey drives an episode. Frames
land in paper_results/selector_kl/runs/showcase/<label>/frames/step_<t>.png up to the showcase
step, and the replayed actions are checked against the run's action_history.

Runs in longnav_vlm:  python paper/selector_kl/showcase_frames.py [--showcase LABEL:STEP] [--force]
(default: results/showcase.json written by aggregate.py; skipped when the frame exists)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = ROOT / "paper_results/selector_kl"
ARM = "r1_selector_probe"


def load_driver():
    spec = importlib.util.spec_from_file_location("probe_episode", ROOT / "paper/layer_token_info/probe_episode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_actions(label):
    """action_history of the probe run's episode (list of action ids) and its goal."""
    from longnav.analysis.influence import _r1_sequences
    for got, d in _r1_sequences(RESULTS_ROOT / "runs" / ARM):
        if got == label:
            return [int(a) for a in d["action_history"]]
    raise SystemExit(f"{label}: no sequence.json under runs/{ARM}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--showcase", help="LABEL:STEP (0-based); default: results/showcase.json")
    ap.add_argument("--force", action="store_true", help="re-drive even when the frame exists")
    args = ap.parse_args()
    if args.showcase:
        label, step = args.showcase.rsplit(":", 1)
        step = int(step)
    else:
        sc = json.loads((RESULTS_ROOT / "results/showcase.json").read_text())
        label, step = sc["episode_label"], int(sc["step"])
    out = RESULTS_ROOT / "runs/showcase" / label
    frame = out / "frames" / f"step_{step:03d}.png"
    if frame.exists() and not args.force:
        print(f"Skipping re-drive: {frame} exists")
        return
    (out / "frames").mkdir(parents=True, exist_ok=True)
    recorded = run_actions(label)
    if step >= len(recorded):
        raise SystemExit(f"{label}: step {step} beyond the run's {len(recorded)} decisions")

    import ray
    from omegaconf import OmegaConf
    from PIL import Image
    from longnav.utils.factories import ExpBootstrapper
    drv = load_driver()
    cfg = drv.compose_config([f"task.run_name=showcase", f"task.output_dir={RESULTS_ROOT / 'runs'}",
                              "vlm.selector_probe=false", f"rollout.max_steps={step + 1}"],
                             experiment="paper_selector_probe")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    rollout = resolved["rollout"]
    boot = ExpBootstrapper(cfg)
    boot.setup_cluster()
    sim = boot.bootstrap_sims(None)[0]
    worker = drv.build_worker(cfg, resolved, "prune")
    t0 = time.time()
    try:
        ray.get(sim.assign_shard.remote([label]))
        worker.reset()
        rgb0, state0 = ray.get(sim.reset.remote())
        goal = state0["obs"]["instr_or_goal"]
        got = state0.get("info", {}).get("episode_label", label)
        if got != label:
            raise RuntimeError(f"asked for {label}, sim served {got}")

        def sim_step(action_id, _first=[rgb0, state0]):
            if action_id is None:
                return _first
            res = ray.get(sim.step.remote(int(action_id), supplementary_logs={}))
            return res[0], res[1]

        def on_step(rec):
            t = rec["step"]
            Image.fromarray(rec["rgb"]).save(out / "frames" / f"step_{t:03d}.png")
            if rec["action_id"] != recorded[t]:
                raise SystemExit(f"step {t}: replayed action {rec['action_id']} != recorded {recorded[t]}; "
                                 "the re-drive does not reproduce the run")

        records = drv.drive(worker, rollout, sim_step, goal, on_step=on_step)
    finally:
        drv.release(worker)
        ray.kill(sim)
        ray.shutdown()
        for d in (RESULTS_ROOT / "runs/showcase/rollout", RESULTS_ROOT / "runs/showcase/showcase"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    info = {"episode_label": label, "step": step, "goal": goal, "steps_driven": len(records),
            "actions": [r["action"] for r in records], "probs": [r["probs"] for r in records],
            "action_space": rollout["action_space"], "seconds": time.time() - t0}
    (out / "trajectory.json").write_text(json.dumps(info, indent=1))
    print(f"[showcase] {label}: {len(records)} steps re-driven and matched in {time.time() - t0:.0f}s -> {frame}")


if __name__ == "__main__":
    main()
