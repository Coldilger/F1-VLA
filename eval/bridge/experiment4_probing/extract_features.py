#!/usr/bin/env python3
"""
Experiment 4 (F1-VLA): extract frozen world-model features.

Captures the "gen" (world-model) expert's real, causally-used token
representations -- the exact channel Experiment 1's ablation empties by
passing gen_embs=None into the shared KV cache
(../experiment1_ablation/sample_without_world_model.py) -- and saves them
alongside the ground-truth future end-effector pose delta.

## Why gen_out, not the multi-layer KV cache

The action expert technically reads per-layer keys/values from the shared
cache (18 layers), not a single summary vector -- so no tensor is literally
"the thing the action expert attends to". `gen_out` (the world-model expert's
own final-layer hidden state) is the same kind of choice mimic-video's
`crossattn_emb` was: the single tensor crossing the boundary from
world-model expert to action expert, analogous across both models even
though each model's action-consuming module also reads it at every one of
its own layers. Probing the full per-layer KV structure would need an
attention-based probe head, inconsistent with the single-frozen-hidden-state
design used for every other model in this experiment.

## The real structure (confirmed empirically, not from config)

`gen_expert_config.pn = '1_2_3_4_5_6_8_10_13_16'` names a 10-scale VAR
pyramid, but `gen_expert_config.num_resolutions = 4` -- and
`modeling_f1.py`'s generation loop breaks at `si == num_resolutions - 1`.
Confirmed via probe_true_shape.py on real data: exactly 4
`paligemma_with_expert.forward` calls happen per step, not 10. So the real
signal is 1+4+9+16 = 30 tokens across scales (1x1, 2x2, 3x3, 4x4), not the
680 a naive reading of `pn` would suggest.

The first call's `gen_out` has shape (1, 680, 1024) -- it's an artifact of
how the full nominal prefix gets vectorized before any real generation has
happened at scales 1-9; the actual code only ever uses its LAST position
(`gen_out[:, -1:] if si == 0 else gen_out`, modeling_f1.py ~line 586). This
script replicates that exact slicing, so what's captured here is exactly
what feeds the KV cache in real inference -- no more, no less.

## Pooling

30 tokens x 1024 channels is small enough that no spatial/scale pooling is
needed at all (unlike mimic-video's 19200-token case) -- every token's
identity (which scale, which position within scale) is kept. Channels are
compressed 1024 -> 32 via a fixed (untrained, seed-based) random projection,
for the same reason as mimic-video's v2 extraction: fitting a
data-dependent projection (e.g. PCA) on the full sample set would leak
validation-episode structure into the projection basis. Final feature size:
30 * 32 = 960.

## Checkpoint comparison caveat

Unlike mimic-video's pretrained_cosmos_bridge/finetuned_cosmos_bridge pair
(which differ ONLY in the probed component, sharing an identical action
decoder), F1-VLA's pretrained and Bridge-finetuned checkpoints are one joint
model -- understanding, generation AND action experts are presumably all
finetuned together. So this comparison is weaker than mimic's: a probe
accuracy gap between variants shows Bridge finetuning changed what gen_out
carries, but doesn't rule out the action expert changing too (irrelevant to
what's probed here, but worth being honest that the control isn't as clean
as mimic's single-component isolation).

Usage:
  python extract_features.py --variant finetuned --out features_finetuned.npz
  python extract_features.py --variant pretrained --out features_pretrained.npz
"""

import argparse
import contextlib
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from f1_vla_policy import F1VLAInference  # noqa: E402

GEN_HIDDEN = 1024
PROJECT_DIM = 32
PROJECTION_SEED = 12345  # same convention/value as mimic-video's v2 extraction

VARIANTS = {
    "finetuned": dict(
        checkpoint_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune",
        stats_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json",
    ),
    "pretrained": dict(
        # config.json here is a patched COPY of the HF snapshot's, with
        # language_tokenizer_path/pretrained_path repointed at this cluster's
        # local checkpoints (the original names an author-machine path,
        # /fs-computility/efm/..., that doesn't resolve here). model.safetensors
        # is an unmodified symlink into the original snapshot -- see
        # checkpoints/pretrained_patched/ for how it was built.
        checkpoint_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/checkpoints/pretrained_patched",
        # Bridge normalization stats aren't part of the pretrained release (it
        # was never finetuned on Bridge). Reusing the finetuned run's stats
        # only rescales predicted actions for readability; it has no effect
        # on gen_out, which is what this experiment probes -- unnormalized
        # inputs (images, state) go into the model identically either way.
        stats_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json",
    ),
}


@contextlib.contextmanager
def _fast_path_is_file():
    original_is_file = pathlib.Path.is_file
    cache = {}

    def fast_is_file(self):
        parent = self.parent
        if parent not in cache:
            try:
                cache[parent] = set(os.listdir(parent))
            except (FileNotFoundError, NotADirectoryError):
                cache[parent] = set()
        return self.name in cache[parent]

    pathlib.Path.is_file = fast_is_file
    try:
        yield
    finally:
        pathlib.Path.is_file = original_is_file


class _PoseProxy:
    def __init__(self, p, q):
        self.p = p
        self.q = q


def pose_from_bridge_state(state: np.ndarray) -> _PoseProxy:
    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    return _PoseProxy(p=pos, q=quat_xyzw[[3, 0, 1, 2]])


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def pose_vector(state: np.ndarray) -> np.ndarray:
    pos = state[0:3].astype(np.float64)
    rot6 = Rotation.from_euler("xyz", state[3:6].astype(np.float64)).as_matrix()[:2].reshape(6)
    return np.concatenate([pos, rot6])


def make_projection(device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    proj = torch.randn(GEN_HIDDEN, PROJECT_DIM, generator=g) / (GEN_HIDDEN**0.5)
    return proj.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--samples-per-episode", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-root", default="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    ap.add_argument("--camera-key", default="observation.images.image_0")
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    print(f"variant={args.variant}", flush=True)
    print(f"  checkpoint_path={cfg['checkpoint_path']}", flush=True)

    model = F1VLAInference(checkpoint_path=cfg["checkpoint_path"], stats_path=cfg["stats_path"], seed=args.seed)

    device = next(model.policy.model.paligemma_with_expert.parameters()).device
    projection = make_projection(device)

    # Capture every paligemma_with_expert.forward call's gen_out during one
    # sample_actions_with_world_model invocation, then replicate the real
    # code's own slicing (call 0 -> last position only; calls 1+ -> as-is).
    captured_calls = []
    pge = model.policy.model.paligemma_with_expert
    original_forward = pge.forward

    def capturing_forward(*a, **kw):
        out = original_forward(*a, **kw)
        (_, gen_out, _), _past_kv = out
        if gen_out is not None:
            captured_calls.append(gen_out.detach())
        return out

    pge.forward = capturing_forward

    def collect_real_tokens() -> torch.Tensor:
        """[30, GEN_HIDDEN] -- exactly what the real loop uses, in scale order."""
        assert len(captured_calls) == 4, f"expected 4 calls (num_resolutions=4), got {len(captured_calls)}"
        first = captured_calls[0][:, -1:]  # (1,1,1024) -- si==0's real slice
        rest = captured_calls[1:]  # each (1, pn*pn, 1024) for si=1,2,3
        tokens = torch.cat([first, *rest], dim=1).squeeze(0)  # (30, 1024)
        return tokens

    rng = np.random.default_rng(args.seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=args.dataset_root)
        episode_ids = rng.choice(meta.total_episodes, size=args.n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot",
            root=args.dataset_root,
            episodes=episode_ids,
            video_backend="pyav",
        )
    print(f"dataset loaded: {len(episode_ids)} of {meta.total_episodes} episodes", flush=True)

    feats, targets, ep_of_sample = [], [], []

    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < args.horizon + 2:
            continue

        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - args.horizon)
        ts = rng.choice(
            candidate_ts, size=min(args.samples_per_episode, len(candidate_ts)), replace=False
        )

        for t in ts:
            t0 = time.time()
            t = int(t)
            frame_t = ds[t]
            state_t = frame_t["observation.state"].numpy()
            state_future = ds[t + args.horizon]["observation.state"].numpy()

            image_t = to_uint8_hwc(frame_t[args.camera_key])
            proprio_t = pose_from_bridge_state(state_t)
            gripper_t = float(state_t[7])

            captured_calls.clear()
            model.reset(task_description)
            model.step(image_t, task_description, proprio_t, gripper_t)

            tokens = collect_real_tokens()  # (30, 1024)
            projected = (tokens.float() @ projection).cpu().numpy()  # (30, 32)
            feat = projected.reshape(-1)  # (960,)

            delta = pose_vector(state_future) - pose_vector(state_t)

            feats.append(feat)
            targets.append(delta)
            ep_of_sample.append(ep_id)
            print(
                f"sample {len(feats)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s, feat dim {feat.shape[0]}",
                flush=True,
            )

    feats = np.stack(feats).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    ep_of_sample = np.array(ep_of_sample)

    np.savez_compressed(
        args.out,
        features=feats,
        targets=targets,
        episode_ids=ep_of_sample,
        horizon=args.horizon,
        variant=args.variant,
        project_dim=PROJECT_DIM,
        projection_seed=PROJECTION_SEED,
    )
    print(f"\nsaved {args.out}: features {feats.shape}, targets {targets.shape}", flush=True)
    print(f"target delta magnitude (mean |.|): {np.abs(targets).mean():.5f}", flush=True)


if __name__ == "__main__":
    main()
