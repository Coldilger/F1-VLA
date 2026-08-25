#!/usr/bin/env python3
"""
Experiment 4 (F1-VLA), live-rollout variant: same gen_out extraction and
same (pos, rot6) delta-pose label as extract_features.py, but sourced from a
live, randomized SimplerEnv-Bridge closed-loop rollout instead of replaying
bridge_orig_lerobot frames.

## Why this exists alongside extract_features.py, unchanged

extract_features.py's images are real photographs from the actual
fine-tuning dataset (expert teleop recordings) -- there is no simulator
involved at all, and therefore no privileged object-pose state anywhere in
that pipeline. This script instead drives a live SimplerEnv rollout, the
same simulated-render setting Experiment 1/2's own closed-loop evals use.
It isolates one specific variable -- real photograph vs. simulated render --
that the offline probe's negative Exp4 result (see PROBING_RESULTS.md) can't
rule out on its own. This is a new, additional data source; it does not
replace, modify, or invalidate extract_features.py, whose own results stand
as they are.

Caveat that must travel with any result from this script: F1-VLA was never
trained on simulated Bridge frames (only on bridge_orig_lerobot's real
photographs), so a live-sim rollout is out-of-distribution for this model in
a way it is not for LDA-1B (trained natively in RoboCasa sim). A null result
here is therefore ambiguous between "content-blind representation" and
"real-vs-sim domain shift swamping the signal" -- it does not cleanly
confirm or overturn the offline finding either way. Treat as one more data
point, not a tiebreaker.

## Mechanics reused verbatim from extract_features.py (not reimplemented)

- capturing_forward / collect_real_tokens: wraps
  paligemma_with_expert.forward exactly the same way, capturing gen_out from
  the same 4 calls per replan and slicing them identically (call 0 -> last
  position only; calls 1-3 as-is) -> (30, 1024).
- make_projection / GEN_HIDDEN / PROJECT_DIM / PROJECTION_SEED: identical
  fixed random projection (seed 12345) down to 32 channels -> 960-dim
  feature vector, so features from this script and from extract_features.py
  live in the same projected space.
- pose_vector's output convention (position in meters + the rotation
  matrix's first two rows, flattened to 6): reproduced here as
  pose_vector_from_pq, built from a live pose's (p, q) instead of a bridge
  dataset row's (pos, euler rpy) -- same underlying rotation, just a
  different input format, so deltas are comparable across both scripts.

## What's structurally different (unavoidably, given a live rollout)

extract_features.py freely samples --samples-per-episode independent
timestamps per already-fully-recorded episode. Here, a "sample" is a real
replan as the episode is actually driven forward -- you get however many
replans naturally occur (execute_steps ticks apart), not a chosen count.
gen_out can only be captured inside _predict_new_chunk (once per replan),
but the future frame needed for this sample's label (tick + horizon) hasn't
happened yet at capture time -- ticks are buffered per-episode and the
(state_t, state_future) delta is resolved in a post-processing pass after
the whole rollout finishes. A replan within horizon ticks of its episode's
end has no future state to pair with and is dropped.

Usage (after main_inference_live_oracle.py's own SimplerEnv CLI conventions):
  python extract_features_live.py \\
    --f1-checkpoint-path .../outputs/bridge_finetune \\
    --f1-stats-path .../outputs/bridge_finetune/bridge_orig_stats.json \\
    --out features_live.npz --horizon 5 \\
    --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \\
    --obj-variation-mode episode --obj-episode-range 0 24 ...
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from f1_vla_policy import F1VLAInference  # noqa: E402

GEN_HIDDEN = 1024
PROJECT_DIM = 32
PROJECTION_SEED = 12345  # same value as extract_features.py -- shared projected space


def make_projection(device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    proj = torch.randn(GEN_HIDDEN, PROJECT_DIM, generator=g) / (GEN_HIDDEN**0.5)
    return proj.to(device)


def pose_vector_from_pq(p, q_wxyz) -> np.ndarray:
    """Same (pos, rot6) convention as extract_features.py's pose_vector, built
    from a live pose's (p, q) instead of a dataset row's (pos, euler rpy).
    q_wxyz is scalar-first, matching f1_vla_policy.py's own
    _build_state (Rotation.from_quat(ee_pose_proprio.q, scalar_first=True)) --
    the same object handed to step() is read the same way here. Absolute,
    un-rebased pose (no _build_state-style episode-start reference rotation):
    a fixed rotation-of-reference offset does not need to be removed to take
    a delta between two ticks in the same episode."""
    pos = np.asarray(p, dtype=np.float64)
    rot6 = Rotation.from_quat(np.asarray(q_wxyz, dtype=np.float64), scalar_first=True).as_matrix()[:2].reshape(6)
    return np.concatenate([pos, rot6])


def add_live_feature_extraction(model: F1VLAInference, projection: torch.Tensor, horizon: int):
    """Wraps step() (per-tick pose buffering) and _predict_new_chunk (per-replan
    gen_out capture), mirroring main_inference_live_oracle.py's wrapping style.
    Returns (tick_histories, replan_samples) -- both filled in place as the
    rollout runs; resolve_labels() below turns them into (features, targets)
    after the rollout finishes."""
    original_step = model.step
    original_predict_new_chunk = model._predict_new_chunk
    original_reset = model.reset

    pge = model.policy.model.paligemma_with_expert
    original_forward = pge.forward
    captured_calls = []

    def capturing_forward(*a, **kw):
        out = original_forward(*a, **kw)
        (_, gen_out, _), _past_kv = out
        if gen_out is not None:
            captured_calls.append(gen_out.detach())
        return out

    pge.forward = capturing_forward

    def collect_real_tokens() -> torch.Tensor:
        assert len(captured_calls) == 4, f"expected 4 calls (num_resolutions=4), got {len(captured_calls)}"
        first = captured_calls[0][:, -1:]
        rest = captured_calls[1:]
        tokens = torch.cat([first, *rest], dim=1).squeeze(0)
        return tokens

    episode_idx = {"n": -1}
    tick_histories = {}  # episode_idx -> list[pose_vector], in tick order
    replan_samples = []  # each: dict(episode_idx, tick_idx, feat)

    def wrapped_reset(task_description):
        episode_idx["n"] += 1
        tick_histories[episode_idx["n"]] = []
        return original_reset(task_description)

    def wrapped_step(image, task_description, ee_pose_proprio, gripper_proprio):
        ep = episode_idx["n"]
        pv = pose_vector_from_pq(ee_pose_proprio.p, ee_pose_proprio.q)
        tick_histories[ep].append(pv)
        return original_step(image, task_description, ee_pose_proprio, gripper_proprio)

    def wrapped_predict_new_chunk(image, task_description):
        ep = episode_idx["n"]
        tick_idx = len(tick_histories[ep]) - 1  # this tick's pose was just appended by wrapped_step
        captured_calls.clear()
        result = original_predict_new_chunk(image, task_description)
        tokens = collect_real_tokens()
        projected = (tokens.float() @ projection).cpu().numpy()
        feat = projected.reshape(-1)
        replan_samples.append(dict(episode_idx=ep, tick_idx=tick_idx, feat=feat))
        print(
            f"LIVE_FEATURE_SAMPLE n={len(replan_samples)} ep={ep} tick={tick_idx} feat_dim={feat.shape[0]}",
            flush=True,
        )
        return result

    model.reset = wrapped_reset
    model.step = wrapped_step
    model._predict_new_chunk = wrapped_predict_new_chunk
    return tick_histories, replan_samples


def resolve_labels(tick_histories, replan_samples, horizon: int):
    feats, targets, ep_of_sample = [], [], []
    dropped_no_future = 0
    for s in replan_samples:
        ep, t = s["episode_idx"], s["tick_idx"]
        history = tick_histories[ep]
        if t + horizon >= len(history):
            dropped_no_future += 1
            continue
        delta = history[t + horizon] - history[t]
        feats.append(s["feat"])
        targets.append(delta)
        ep_of_sample.append(ep)
    print(f"resolved {len(feats)} samples ({dropped_no_future} dropped: no future tick within horizon)", flush=True)
    return feats, targets, ep_of_sample


def parse_extraction_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f1-checkpoint-path", type=str, required=True)
    parser.add_argument("--f1-stats-path", type=str, required=True)
    parser.add_argument("--f1-device", type=str, default="cuda")
    parser.add_argument("--f1-execute-steps", type=int, default=None)
    parser.add_argument("--f1-seed", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--variant", type=str, default="finetuned_live")
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
    ext_args, remaining_argv = parse_extraction_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = F1VLAInference(
        checkpoint_path=ext_args.f1_checkpoint_path,
        stats_path=ext_args.f1_stats_path,
        device=ext_args.f1_device,
        seed=ext_args.f1_seed,
        execute_steps=ext_args.f1_execute_steps,
    )
    device = next(model.policy.model.paligemma_with_expert.parameters()).device
    projection = make_projection(device)

    tick_histories, replan_samples = add_live_feature_extraction(model, projection, ext_args.horizon)

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    feats, targets, ep_of_sample = resolve_labels(tick_histories, replan_samples, ext_args.horizon)
    if not feats:
        print("no samples resolved -- nothing to save", flush=True)
        sys.exit(1)

    feats = np.stack(feats).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    ep_of_sample = np.array(ep_of_sample)

    np.savez_compressed(
        ext_args.out,
        features=feats,
        targets=targets,
        episode_ids=ep_of_sample,
        horizon=ext_args.horizon,
        variant=ext_args.variant,
        project_dim=PROJECT_DIM,
        projection_seed=PROJECTION_SEED,
    )
    print(f"\nsaved {ext_args.out}: features {feats.shape}, targets {targets.shape}", flush=True)
    print(f"target delta magnitude (mean |.|): {np.abs(targets).mean():.5f}", flush=True)
