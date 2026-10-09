#!/usr/bin/env python3
"""Driver for the layer-wise visual-token information figure (paper/layer_token_info/README.md).

Two sub-commands, both run from the repo root in the longnav_vlm env:

  survey [--episodes L ...]
      Boot one habitat sim (Ray actor in the vln env) and the LongNav-R1 worker in-process
      (evict mode, full context, sparse filter disabled), drive every manifest episode for
      rollout.max_steps (10) steps with deterministic decoding, save the frames and actions, and
      score each episode's 10th observation by the leak-free joint KL of hiding the whole frame.
      Writes runs/<label>/{frames/step_XX.png, trajectory.json}, runs/survey.csv and
      runs/selected.json (the episode with the largest score among those that reach step 10).

  probe --arm {recompute,mask} [--episode L] [--layers 0 4 ...] [--steps 5 10 ...] [--measures ...]
      No habitat. Re-feed the saved frames to a fresh worker for the arm (recompute = evict
      mode, leak-free; mask = prune substrate, leaky), check the decisions reproduce the
      recorded ones, then run the per-token, per-layer probes after the last step and save
      runs/<label>/token_info_<arm>.npz. With --steps the probes run after each listed decision
      (1-based: --steps 10 probes the 10th decision, frame step_09.png) in one replay and land in
      runs/<label>/token_info_<arm>_t<step>.npz; existing files are skipped.

--study NAME (before the sub-command) moves everything to paper_results/layer_token_info/NAME/runs,
so a second protocol (more scenes, longer drives, other measures) does not touch the headline
runs; `survey --max-steps N` then drives longer episodes (an episode that stops earlier is kept,
its last decision is the stop) and logs the leak-free joint score at every step.

The Hydra config is composed programmatically with the same overrides longnav.scripts.eval takes
(+checkpoint=longnav +dataset=hm3d_v2_val +experiment=paper_token_info +resources=single).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "paper_results/layer_token_info/runs"
CTX_PER_EPISODE = ROOT / "paper_results/ctx_ablation/results/per_episode.csv"
BASE_OVERRIDES = ["+checkpoint=longnav", "+dataset=hm3d_v2_val", "+experiment=paper_token_info",
                  "+resources=single", "resources.num_sims=1", "resources.num_vlms=1"]
DEFAULT_EPISODE = "VBzV5z6i1WS_1"  # fallback when no survey has been run


def compose_config(extra_overrides=(), experiment="paper_token_info"):
    """The resolved RLConfig, exactly as longnav.scripts.eval would see it (``experiment`` swaps
    the +experiment preset; paper/selector_kl re-drives its showcase episode this way)."""
    from hydra import compose, initialize_config_dir
    from longnav.conf.register_configs import register_configs

    if Path.cwd() != ROOT:
        # ${read_text:...} resolves relative to the cwd outside a Hydra app
        os.chdir(ROOT)
    register_configs()
    overrides = [o if not o.startswith("+experiment=") else f"+experiment={experiment}" for o in BASE_OVERRIDES]
    with initialize_config_dir(config_dir=str(ROOT / "src/longnav/config"), version_base=None):
        return compose(config_name="rl_config", overrides=overrides + list(extra_overrides))


def build_worker(cfg, resolved, mode):
    """LongNav-R1 in-process: base model + LoRA adapter, as bootstrap_vlms_rl(training=False)."""
    from longnav.utils.factories import get_base_model, resolve_checkpoint_path
    from longnav.utils.rollout_core import RLActor

    vlm = dict(resolved["vlm"])
    vlm["context_window_mode"] = mode
    ckpt = resolve_checkpoint_path(cfg.training.checkpoint)
    base = get_base_model(ckpt)
    if base is not None:
        vlm["model_id"] = base
    worker = RLActor(rollout_config=dict(resolved["rollout"]), **vlm)
    worker._setup_peft(cfg.training)
    worker.load_checkpoint(ckpt, False, False)
    return worker


def release(worker):
    import torch
    del worker
    gc.collect()
    torch.cuda.empty_cache()


def set_study(name):
    """Redirect the run directory to paper_results/layer_token_info/<name>/runs."""
    global RUNS
    if name:
        RUNS = ROOT / "paper_results/layer_token_info" / name / "runs"


def episode_dir(label):
    return RUNS / label


def load_manifest(path):
    labels = json.loads(Path(path).read_text())
    assert isinstance(labels, list) and all(isinstance(x, str) for x in labels)
    return labels


def ctx_prior():
    """r1_full success / SPL per episode from the context-window ablation, if available."""
    prior = {}
    if CTX_PER_EPISODE.exists():
        for row in csv.DictReader(open(CTX_PER_EPISODE)):
            if row["arm"] == "r1_full":
                prior[row["episode_label"]] = (float(row["success"]), float(row["spl"]), int(float(row["n_steps"])))
    return prior


# --------------------------------------------------------------------------- the episode loop


def drive(worker, rollout, images_or_sim, goal, on_step=None):
    """Feed frames to the worker exactly as EpisodeRolloutMixin.run_episode does.

    ``images_or_sim`` is either a list of PIL images (replay) or a callable ``step(action_id)``
    returning the next ``(rgb, state_dict)`` (live sim). Returns the list of per-step records.
    """
    from PIL import Image
    from longnav.utils.rollout_core import select_action, substitute_convo_template

    messages = substitute_convo_template(rollout["convo_start_template"], {"instr_or_goal": goal} | rollout)
    records = []
    live = callable(images_or_sim)
    rgb = None if live else np.asarray(images_or_sim[0])
    t = 0
    while True:
        if live and t == 0:
            rgb = images_or_sim(None)[0]
        elif not live:
            if t >= len(images_or_sim):
                break
            rgb = np.asarray(images_or_sim[t])
        probs, _, _ = worker.infer_probs(images=[Image.fromarray(rgb)], messages=messages,
                                         temperature=rollout["temperature"], pos_id_kwargs={"mode": "standard"})
        action_id, _ = select_action(probs, deterministic=rollout.get("deterministic", False),
                                     stop_prob_threshold=rollout.get("stop_prob_threshold"))
        rec = {"step": t, "rgb": rgb, "probs": [float(p) for p in probs], "action_id": int(action_id),
               "action": rollout["action_space"][action_id]}
        records.append(rec)
        if on_step is not None:
            on_step(rec)
        messages = substitute_convo_template(rollout["convo_turn_template"], {"action": rec["action"]})
        t += 1
        if live:
            if t >= rollout["max_steps"]:
                break
            rgb, state = images_or_sim(action_id)
            if state["done"]:
                rec["done_after"] = True
                break
    return records


# --------------------------------------------------------------------------- survey


def cmd_survey(args):
    import ray
    from omegaconf import OmegaConf
    from PIL import Image
    from longnav.utils import token_info as ti
    from longnav.utils.factories import ExpBootstrapper, save_hydra_config

    extra = [f"rollout.max_steps={args.max_steps}"] if args.max_steps else []
    cfg = compose_config([f"task.run_name=survey", f"task.output_dir={RUNS}"] + extra)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    rollout = resolved["rollout"]
    labels = args.episodes or load_manifest(cfg.task.episode_json)
    RUNS.mkdir(parents=True, exist_ok=True)
    save_hydra_config(cfg, str(RUNS))

    boot = ExpBootstrapper(cfg)
    boot.setup_cluster()
    sim = boot.bootstrap_sims(None)[0]
    worker = build_worker(cfg, resolved, "evict")
    prior = ctx_prior()
    rows = []
    try:
        for label in labels:
            out = episode_dir(label)
            (out / "frames").mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            ray.get(sim.assign_shard.remote([label]))
            worker.reset()
            first = ray.get(sim.reset.remote())
            if len(first) != 2:
                raise RuntimeError("the sim attached patch coords; this driver expects (rgb, state)")
            rgb0, state0 = first
            goal = state0["obs"]["instr_or_goal"]
            got = state0.get("info", {}).get("episode_label", label)
            if got != label:
                raise RuntimeError(f"asked for {label}, sim served {got}")

            def sim_step(action_id, _first=[rgb0, state0]):
                if action_id is None:
                    return _first
                res = ray.get(sim.step.remote(int(action_id), supplementary_logs={}))
                return res[0], res[1]

            def save_frame(rec):
                Image.fromarray(rec["rgb"]).save(out / "frames" / f"step_{rec['step']:02d}.png")
                if args.step_scores:  # how much this observation moves its decision (2 prefills)
                    rec["score"] = ti.joint_current_frame_kl(worker, 0)[0]

            records = drive(worker, rollout, sim_step, goal, on_step=save_frame)
            steps = len(records)
            reached = steps >= rollout["max_steps"] and not records[-1].get("done_after")
            score = float("nan")
            if reached or args.step_scores:
                score, _, _ = ti.joint_current_frame_kl(worker, 0)
            grid = ti.current_frame_grid(worker)
            traj = {"episode_label": label, "goal": goal, "steps": steps, "reached_probe_step": reached,
                    "actions": [r["action"] for r in records], "action_ids": [r["action_id"] for r in records],
                    "probs": [r["probs"] for r in records], "action_space": rollout["action_space"],
                    "grid_thw": list(grid), "sparse_threshold": resolved["vlm"]["sparse_threshold"],
                    "survey_score_leakfree_kl": score, "seconds": time.time() - t0}
            if args.step_scores:
                traj["step_scores_leakfree_kl"] = [r["score"] for r in records]
            (out / "trajectory.json").write_text(json.dumps(traj, indent=1))
            p = prior.get(label, (float("nan"), float("nan"), -1))
            rows.append({"episode_label": label, "steps": steps, "reached_probe_step": int(reached),
                         "score_leakfree_kl": score, "actions": " ".join(traj["actions"]),
                         "r1_full_success": p[0], "r1_full_spl": p[1], "r1_full_steps": p[2]})
            print(f"[survey] {label}: {steps} steps, actions {' '.join(traj['actions'])}, "
                  f"score {score:.4f}, {time.time() - t0:.0f}s", flush=True)
    finally:
        release(worker)
        ray.kill(sim)
        ray.shutdown()
        # the sim actor needs a log dir (<output_dir>/<run_name>/rollout) but never flushes here
        for d in (RUNS / "survey" / "rollout", RUNS / "survey"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    with open(RUNS / "survey.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["reached_probe_step"] and np.isfinite(r["score_leakfree_kl"])]
    if not ok:
        if args.max_steps:  # a long drive: every episode may stop before the cap, nothing to select
            print("[survey] no episode ran to the step cap; no selected.json written")
            return
        raise SystemExit("no episode reached the probe step")
    best = max(ok, key=lambda r: r["score_leakfree_kl"])
    (RUNS / "selected.json").write_text(json.dumps({"episode_label": best["episode_label"],
                                                    "score_leakfree_kl": best["score_leakfree_kl"],
                                                    "rule": "max leak-free KL(full || current frame hidden) at step 10"}, indent=1))
    print(f"[survey] selected {best['episode_label']} (score {best['score_leakfree_kl']:.4f})")


# --------------------------------------------------------------------------- probe


def selected_episode():
    sel = RUNS / "selected.json"
    if sel.exists():
        return json.loads(sel.read_text())["episode_label"]
    return DEFAULT_EPISODE


def cmd_probe(args):
    from omegaconf import OmegaConf
    from PIL import Image
    from longnav.utils import token_info as ti

    label = args.episode or selected_episode()
    out = episode_dir(label)
    traj = json.loads((out / "trajectory.json").read_text())
    if args.steps:
        steps = sorted({t for t in args.steps if 1 <= t <= traj["steps"]})
        skipped = sorted(set(args.steps) - set(steps))
        if skipped:
            print(f"[probe:{args.arm}] {label}: steps {skipped} lie outside the {traj['steps']}-step trajectory")
        targets = {t: out / f"token_info_{args.arm}_t{t:03d}.npz" for t in steps}
        targets = {t: f for t, f in targets.items() if not f.exists()}
    else:
        if not traj["reached_probe_step"]:
            raise SystemExit(f"{label} ended before the probe step; pick another episode")
        targets = {traj["steps"]: out / f"token_info_{args.arm}.npz"}
    if not targets:
        print(f"[probe:{args.arm}] {label}: nothing to do")
        return
    mode = {"recompute": "evict", "mask": "prune"}[args.arm]
    cfg = compose_config([f"task.run_name={label}", f"vlm.context_window_mode={mode}"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    rollout = resolved["rollout"]
    last = max(targets)
    frames = [Image.open(out / "frames" / f"step_{t:02d}.png").convert("RGB") for t in range(last)]
    worker = build_worker(cfg, resolved, mode)
    layers = args.layers
    fn = ti.probe_recompute_arm if args.arm == "recompute" else ti.probe_mask_arm
    worst = [0.0]

    def on_step(rec):
        t = rec["step"]
        d = float(np.abs(np.asarray(rec["probs"]) - np.asarray(traj["probs"][t])).max())
        worst[0] = max(worst[0], d)
        if rec["action_id"] != traj["action_ids"][t]:
            raise SystemExit(f"step {t}: replayed argmax {rec['action']} != recorded "
                             f"{traj['actions'][t]} (max |dp| {d:.2e}); the arm does not "
                             "reproduce the trajectory")
        if t + 1 not in targets:
            return
        print(f"[probe:{args.arm}] {label}: decision {t + 1} ({rec['action']}) reproduced, max |dp| so far "
              f"{worst[0]:.2e}" + (" (WARNING: above 1e-2)" if worst[0] > 1e-2 else ""), flush=True)
        t0 = time.time()

        def progress(li, layer, seconds):
            n = len(layers) if layers else len(worker.language_model.layers)
            print(f"[probe:{args.arm}] t={t + 1} layer {layer} done ({li + 1}/{n}, {seconds:.0f}s)", flush=True)

        res = fn(worker, layers, progress=progress, measures=args.measures)
        res |= {"episode_label": label, "arm": args.arm, "mode": mode, "goal": traj["goal"],
                "action": rec["action"], "action_probs": np.asarray(rec["probs"]),
                "action_space": np.asarray(rollout["action_space"]), "steps": np.int64(t + 1),
                "replay_max_dp": np.float64(worst[0]), "wall_seconds": np.float64(time.time() - t0)}
        np.savez(targets[t + 1], **res)
        means = ", ".join(f"mean {m} {res[m].mean():.3e}" for m in res["measures"])
        print(f"[probe:{args.arm}] saved {targets[t + 1]}: {int(res['n_probes'])} probes, "
              f"{float(res['seconds']):.0f}s; {means}", flush=True)

    try:
        worker.reset()
        drive(worker, rollout, frames, traj["goal"], on_step=on_step)
    finally:
        release(worker)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", help="run under paper_results/layer_token_info/<study>/runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--episodes", nargs="*", help="labels to drive (default: the preset's manifest)")
    s.add_argument("--max-steps", type=int, help="drive this many steps (default: the preset's 10)")
    s.add_argument("--step-scores", action="store_true",
                   help="log the leak-free KL of hiding the whole current frame at every step")
    p = sub.add_parser("probe")
    p.add_argument("--arm", choices=("recompute", "mask"), required=True)
    p.add_argument("--episode", help="default: runs/selected.json, else " + DEFAULT_EPISODE)
    p.add_argument("--layers", nargs="*", type=int, help="decoder layers to probe (default: all)")
    p.add_argument("--steps", nargs="*", type=int, help="1-based decisions to probe (default: the last)")
    p.add_argument("--measures", nargs="*", help="subset of leave keep_cur keep_all leave_nohist "
                                                 "(default: the first three)")
    args = ap.parse_args()
    set_study(args.study)
    {"survey": cmd_survey, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
