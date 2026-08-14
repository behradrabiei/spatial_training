"""Is the attribution map causal, or just decorative?

A heat map is only worth reading if the patches it highlights are the ones the
decision actually rests on. This drops the top-k highlighted visual keys out of the
KV cache, replays the action-decision token, and measures how far the chosen
action's logprob falls -- against the same number of randomly chosen visual keys as
a control. Top-k should hurt substantially more than random-k. If it does not, the
map is not describing the decision.

Works for any attn_weighting, so it doubles as an A/B between the attention modes
and "grad" on the same episode.

Run inside the VLM conda env (longnav_vlm). Point it at real frames -- synthetic
noise gives the model nothing to look at, so every patch is equally irrelevant and
the audit cannot separate a good map from a bad one:

    python tools/attn_faithfulness.py --weighting grad \\
        --video dump/longnav_eval/<run>/rollout/<scene>/<episode>/video.mp4 --goal bed
    python tools/attn_faithfulness.py --weighting wo_norm --images 'frames/*.png'
"""
import argparse
import glob
import os
from string import Template

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker

DEFAULT_MODEL = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
PROMPT_PATH = "src/longnav/conf/prompts/objectnav_prompt.txt"
ACTION_SPACE = "[stop, forward, left, right, up, down]"
# Rollout videos are a horizontal strip of panels; the first is the raw RGB.
RGB_PANEL_W = 640


def build_messages(goal):
    with open(PROMPT_PATH) as f:
        prompt = Template(f.read()).substitute(instr_or_goal=goal, action_space_str=ACTION_SPACE)
    start = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    ]
    turn = [
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    ]
    return start, turn


def load_images(args, n):
    if args.video:
        import cv2  # imageio lives in the habitat env, not this one

        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(frame[:, :RGB_PANEL_W, ::-1]))
        cap.release()
        if not frames:
            raise SystemExit(f"no frames read from {args.video}")
        return frames
    if args.images:
        paths = sorted(glob.glob(args.images))[:n]
        if not paths:
            raise SystemExit(f"no images matched {args.images!r}")
        return [Image.open(p).convert("RGB") for p in paths]
    print("WARNING: using synthetic noise frames; the audit cannot say anything "
          "meaningful about a map of nothing. Pass --video or --images.")
    rng = np.random.RandomState(0)
    return [Image.fromarray(rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(n)]


class Auditor:
    """Replays the decision token against a cache with chosen keys removed.

    The decision token's own KV entry always comes out first, exactly as the
    attribution replay does: left in, the query would attend to a duplicate of
    itself. Variants are built by indexing the saved tensors, so the originals are
    never mutated and restoring is just putting the references back.
    """

    def __init__(self, worker):
        self.worker = worker

    def visual_positions(self):
        """Cache positions of every visual key still resident, in row order."""
        recs = [r["abs_kv_idx"] for r in self.worker._frame_records if r["abs_kv_idx"] is not None]
        return torch.cat(recs) if recs else torch.empty(0, dtype=torch.long)

    def logprobs_without(self, drop):
        """Action logprobs with `drop` (cache positions) removed. Cache is restored."""
        worker = self.worker
        layers = worker.past_key_values.layers
        saved = [(l.keys, l.values) for l in layers]
        kv_len = saved[0][0].shape[-2]

        keep = torch.ones(kv_len, dtype=torch.bool)
        keep[-1] = False  # the decision token replays as the query, not as a key
        if len(drop):
            keep[torch.as_tensor(drop, dtype=torch.long)] = False
        index = keep.nonzero(as_tuple=False).squeeze(-1).to(saved[0][0].device)

        try:
            for layer, (keys, values) in zip(layers, saved):
                layer.keys = keys[..., index, :]
                layer.values = values[..., index, :]
            with torch.no_grad():
                out = worker.model.forward(
                    **worker._decision_inputs,
                    attention_mask=None,
                    seq_keep_mask="all",
                    past_key_values=worker.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            logits = out.logits[0, -1, worker.vocab_ids].float()
            return torch.log_softmax(logits, dim=-1).cpu().numpy()
        finally:
            for layer, (keys, values) in zip(layers, saved):
                layer.keys, layer.values = keys, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--weighting", default="grad", help="raw | value_norm | wo_norm | grad")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--audit-from", type=int, default=3, help="first step to audit")
    ap.add_argument("--topk", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--trials", type=int, default=5, help="random-k control draws")
    ap.add_argument("--images", default=None, help="glob of RGB frames")
    ap.add_argument("--video", default=None, help="rollout video.mp4; its first panel is the RGB")
    ap.add_argument("--goal", default="bed", help="navigation target named in the prompt")
    args = ap.parse_args()

    images = load_images(args, args.steps)
    start_msgs, turn_msgs = build_messages(args.goal)
    worker = VLMWorker(model_id=args.model, attn_impl="eager", dtype="bfloat16", use_sparse=True,
                       visualize_attention_3d=True, attn3d_layers=[args.layer],
                       attn_weighting=args.weighting)
    auditor = Auditor(worker)
    rng = np.random.RandomState(0)
    results = {k: [] for k in args.topk}

    for step, image in enumerate(images):
        worker.infer_step(start_msgs if step == 0 else turn_msgs, [image])
        if step < args.audit_from:
            continue
        layer_id = worker.attn3d_layer_ids[0]
        row = worker.attn_probe.rows.get(layer_id)
        if row is None:
            print(f"step {step}: no attribution row, skipping")
            continue
        visual = auditor.visual_positions()
        if not len(visual):
            continue

        base = auditor.logprobs_without([])
        chosen = int(base.argmax())
        # Rank visual keys by the same numbers the heat map is painted with.
        order = visual[torch.argsort(row[visual], descending=True)]

        line = [f"step {step}: action={worker.vocab[chosen]} visual_keys={len(visual)}"]
        for k in args.topk:
            k = min(k, len(visual))
            top = auditor.logprobs_without(order[:k].tolist())
            ctrl = [auditor.logprobs_without(visual[rng.choice(len(visual), k, replace=False)].tolist())
                    for _ in range(args.trials)]
            d_top = float(base[chosen] - top[chosen])
            d_ctrl = float(np.mean([base[chosen] - c[chosen] for c in ctrl]))
            results[k].append((d_top, d_ctrl))
            line.append(f"  k={k:3d}: top-k drop={d_top:+.4f}  random-k drop={d_ctrl:+.4f}")
        print("\n".join(line))

    print("\n=== summary "
          f"(weighting={args.weighting}, layer={args.layer}) ===")
    for k, pairs in results.items():
        if not pairs:
            continue
        top = np.mean([p[0] for p in pairs])
        ctrl = np.mean([p[1] for p in pairs])
        verdict = "causal" if top > 2 * max(ctrl, 1e-6) else "NOT distinguishable from random"
        print(f"k={k:3d}: mean top-k drop={top:+.4f}  mean random-k drop={ctrl:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
