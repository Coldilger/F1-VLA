#!/usr/bin/env python3
"""
Offline probe for Experiment 1, variant 1 (F1-VLA): "remove the module".

Same protocol/infrastructure as ../experiment2_oracle/oracle_offline_probe.py
(real BridgeDataV2 episodes, no SimplerEnv/ManiSkill2). For a sampled
(episode, t), predicts an action chunk twice on the SAME real logged moment:
  - baseline: model.select_action_with_world_model (unmodified, default path)
  - ablated:  sample_actions_without_world_model (this experiment's new
              function, see sample_without_world_model.py) -- world-model
              tokens never enter the sequence at all for this call
and compares both against the real logged action.

If ablated L1 is close to baseline L1, the foresight computation isn't
doing much for action selection even in the direction of "removing it
doesn't hurt" -- interesting on its own, distinct from Experiment 2's
"does a *perfect* future help" question. If ablated is clearly worse than
baseline, the module is load-bearing at inference (subject to the sequence-
shape confound noted in sample_without_world_model.py's docstring).
"""

import contextlib
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from f1_vla_policy import F1VLAInference  # noqa: E402
from sample_without_world_model import sample_actions_without_world_model  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
    """Same patch as experiment2_oracle/oracle_offline_probe.py -- avoids the
    per-file Path.is_file() beegfs round-trip that hung dataset construction
    for the full Bridge episode set."""
    original_is_file = pathlib.Path.is_file
    dir_listing_cache = {}

    def fast_is_file(self):
        parent = self.parent
        if parent not in dir_listing_cache:
            try:
                dir_listing_cache[parent] = set(os.listdir(parent))
            except (FileNotFoundError, NotADirectoryError):
                dir_listing_cache[parent] = set()
        return self.name in dir_listing_cache[parent]

    pathlib.Path.is_file = fast_is_file
    try:
        yield
    finally:
        pathlib.Path.is_file = original_is_file


class PoseProxy:
    def __init__(self, p: np.ndarray, q: np.ndarray):
        self.p = p
        self.q = q


def pose_from_bridge_state(state: np.ndarray) -> PoseProxy:
    from scipy.spatial.transform import Rotation

    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return PoseProxy(p=pos, q=quat_wxyz)


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


class F1VLAAblatedInference(F1VLAInference):
    """F1VLAInference + the ablated (no-world-model) sampling path. reset()/
    step() are untouched -- adds a method the offline probe calls instead,
    so normal closed-loop rollout eval (main_inference.py) is unaffected."""

    def predict_action_without_world_model(self, image: np.ndarray, task_description: str) -> np.ndarray:
        main_img = self._preprocess_main_image(image).unsqueeze(0).to(self.device)
        hist_frames = list(self.image_history)
        if len(hist_frames) < self.n_obs_img_steps:
            hist_frames = [hist_frames[0]] * (self.n_obs_img_steps - len(hist_frames)) + hist_frames
        hist_stack = torch.stack(hist_frames).unsqueeze(0).to(self.device)

        batch = {
            "observation.images.image0": main_img,
            "observation.images.image0_mask": torch.tensor([True], device=self.device),
            "observation.images.image0_history": hist_stack,
            "observation.state": self._current_state.unsqueeze(0).to(self.device),
            "task": [task_description],
        }
        images, image_masks = self.policy.prepare_mix_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens, lang_masks = self.policy.prepare_language(batch)

        with torch.no_grad():
            actions = sample_actions_without_world_model(
                self.policy.model, images, image_masks, lang_tokens, lang_masks, state
            )
        actions = actions[:, :, :7]  # unpad, same as select_action_with_world_model does internally
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean
        return actions.numpy()

    def predict_action_baseline(self, image: np.ndarray, task_description: str) -> np.ndarray:
        main_img = self._preprocess_main_image(image).unsqueeze(0).to(self.device)
        hist_frames = list(self.image_history)
        if len(hist_frames) < self.n_obs_img_steps:
            hist_frames = [hist_frames[0]] * (self.n_obs_img_steps - len(hist_frames)) + hist_frames
        hist_stack = torch.stack(hist_frames).unsqueeze(0).to(self.device)

        batch = {
            "observation.images.image0": main_img,
            "observation.images.image0_mask": torch.tensor([True], device=self.device),
            "observation.images.image0_history": hist_stack,
            "observation.state": self._current_state.unsqueeze(0).to(self.device),
            "task": [task_description],
        }
        with torch.no_grad():
            actions = self.policy.select_action_with_world_model(batch, rng=self._rng)
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean
        return actions.numpy()


def run_ablation_probe(
    checkpoint_path: str,
    stats_path: str,
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",
    n_episodes: int = 24,
    samples_per_episode: int = 5,
    seed: int = 0,
):
    model = F1VLAAblatedInference(checkpoint_path=checkpoint_path, stats_path=stats_path, seed=seed)

    rng = np.random.default_rng(seed)
    t_ds = time.time()
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )
    print(f"dataset loaded in {time.time() - t_ds:.1f}s, {len(episode_ids)} episodes selected "
          f"of {meta.total_episodes} total", flush=True)

    l1_ablated, l1_baseline, l1_zero = [], [], []

    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < 3:
            continue

        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - 1)
        ts = rng.choice(candidate_ts, size=min(samples_per_episode, len(candidate_ts)), replace=False)

        for t in ts:
            t0 = time.time()
            t = int(t)
            frame_t = ds[t]
            image_t = to_uint8_hwc(frame_t[camera_key])
            state_t = frame_t["observation.state"].numpy()
            true_action = frame_t["action"].numpy()

            model.reset(task_description)
            model.image_history.append(model._preprocess_history_image(image_t))
            model._current_state = model._build_state(pose_from_bridge_state(state_t), float(state_t[7]))

            pred_ablated = model.predict_action_without_world_model(image_t, task_description)[0]
            pred_baseline = model.predict_action_baseline(image_t, task_description)[0]

            l1_ablated.append(np.abs(pred_ablated - true_action).mean())
            l1_baseline.append(np.abs(pred_baseline - true_action).mean())
            l1_zero.append(np.abs(true_action).mean())
            print(f"sample {len(l1_ablated)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s", flush=True)

    l1_ablated, l1_baseline, l1_zero = map(np.array, (l1_ablated, l1_baseline, l1_zero))
    print(f"n samples: {len(l1_ablated)}")
    print(f"full action L1, ablated (no world model):  {l1_ablated.mean():.4f}  (sd {l1_ablated.std():.4f})")
    print(f"full action L1, baseline (default):         {l1_baseline.mean():.4f}  (sd {l1_baseline.std():.4f})")
    print(f"full action L1, zero-action baseline:        {l1_zero.mean():.4f}")


if __name__ == "__main__":
    run_ablation_probe(
        checkpoint_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune",
        stats_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
