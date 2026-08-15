#!/usr/bin/env python3
"""
Offline oracle probe for F1-VLA on real BridgeDataV2 episodes.

Bypasses SimplerEnv/ManiSkill2 entirely, same protocol as
mimic-video/eval/bridge/oracle_offline_probe.py and matching the metric
convention of LDA-1B/lda/eval/eval_bridge_openloop.py. For a sampled (episode, t):
encode the REAL next frame through F1's own VQ-VAE (vae.img_to_idxBl) and inject
those indices into the VAR foresight loop (modeling_f1.py's oracle_indices hook,
added for this experiment) instead of letting the world-model head sample/imagine
the future. Then compare the predicted action chunk's first step against the
actual logged action at t.

Unlike the mimic-video probe, this compares directly against the raw `action`
column, not against the next observation.state. F1VLAInference already
unnormalizes its predictions back into Bridge's native action space (x,y,z,roll,
pitch,yaw,gripper) as part of its normal step() -- see f1_vla_policy.py's own
docstring -- so predicted and logged actions live in the same representation
with no SimplerEnv/frame-convention translation involved. This is a cleaner
comparison than mimic's (no orientation-frame risk), because the SimplerEnv
controller conversion in F1VLAInference.step() only happens AFTER this point
and this script never calls step().

STATUS: first-draft harness, mirrors oracle_offline_probe.py's structure.
One VERIFY below (camera key) is analogous to the mimic script's.
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from f1_vla_policy import F1VLAInference  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
    """Same patch as f1_vla/src/processors/data_processors/data_loader.py.
    LeRobotDataset.__init__ calls Path.is_file() once per episode/video file;
    on beegfs each call is an uncached network round-trip, which for 53,192
    Bridge episodes takes HOURS unpatched (confirmed: the first run of this
    script hung silently for the full 1h SLURM wall-time with zero output and
    got killed before finishing dataset construction). This lists each parent
    dir once instead."""
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
    """Stand-in for ManiSkill2's sapien.Pose. F1VLAInference._build_state only
    reads .p and .q (f1_vla_policy.py:198-199), so a plain object works without
    a live simulator behind it -- same trick as the mimic-video probe."""

    def __init__(self, p: np.ndarray, q: np.ndarray):
        self.p = p
        self.q = q  # scalar-first [w,x,y,z], matches Rotation.from_quat(..., scalar_first=True)


def pose_from_bridge_state(state: np.ndarray) -> PoseProxy:
    # bridge_orig_lerobot observation.state: [x, y, z, roll, pitch, yaw, pad, gripper]
    from scipy.spatial.transform import Rotation

    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    # NOTE: F1VLAInference._build_state re-expresses rotation relative to the
    # pose captured at the *first* step of the episode (its own self._ref_rot
    # logic, f1_vla_policy.py:210-212), which is exactly how training data is
    # structured (RESULTS.md fix #4). Feeding it raw per-frame Bridge state
    # through .reset() + this proxy reproduces that correctly AS LONG AS the
    # first proprio fed for a given episode is this episode's own t=0 -- true
    # here since each sample calls reset() itself; no cross-sample state leaks.
    return PoseProxy(p=pos, q=quat_wxyz)


class F1VLAOracleInference(F1VLAInference):
    """F1VLAInference + ground-truth-future injection. reset()/step() are
    untouched; this only adds a method the offline probe calls instead of
    step(), so normal closed-loop rollout eval (main_inference.py) is
    unaffected."""

    def predict_action_given_true_next_frame(
        self, image: np.ndarray, task_description: str, true_next_frame: np.ndarray
    ) -> np.ndarray:
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

        # Same preprocessing as training-time / step()-time history frames
        # (_preprocess_history_image: square -> Resize -> CenterCrop -> [-1,1]),
        # so the VAE sees the true future frame in the distribution it expects.
        future_img = self._preprocess_history_image(true_next_frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            oracle_indices = self.policy.model.vae.img_to_idxBl(future_img)
            actions = self.policy.select_action_with_world_model(
                batch, rng=self._rng, oracle_indices=oracle_indices
            )
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean
        return actions.numpy()


def to_uint8_hwc(img) -> np.ndarray:
    """Confirmed against lerobot source (decode_video_frames_torchvision:
    "convert to the pytorch format which is float32 in [0,1] range (and
    channel first)") -- bridge_orig_lerobot frames are float32 CHW [0,1],
    so this conversion is not a guess."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def run_oracle_probe(
    checkpoint_path: str,
    stats_path: str,
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",  # VERIFY: matches F1's training camera
    n_episodes: int = 24,
    samples_per_episode: int = 5,
    seed: int = 0,
):
    model = F1VLAOracleInference(checkpoint_path=checkpoint_path, stats_path=stats_path, seed=seed)

    # Root cause of the two earlier 1h TIMEOUTs, confirmed via a standalone
    # diagnostic: LeRobotDataset(repo_id=..., root=...) with no `episodes=`
    # arg loads/generates the HF `datasets` split for ALL 53,192 episodes
    # ("Generating train split" in its own log), which never finished inside
    # a 1h SLURM job. The SAME dataset with an explicit small episodes= list
    # built in 7.3s for 50 episodes (measured). We only ever sample
    # n_episodes anyway, so get the episode count from the lightweight
    # metadata object first and hand LeRobotDataset only the ids we need.
    rng = np.random.default_rng(seed)
    t_ds = time.time()
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )  # default backend (torchcodec) needs FFmpeg .so's not installed here; pyav worked in the diagnostic
    print(f"dataset loaded in {time.time() - t_ds:.1f}s, {len(episode_ids)} episodes selected "
          f"of {meta.total_episodes} total", flush=True)

    action_l1s, action_l1s_baseline = [], []
    pos_l1s, grip_l1s = [], []

    # NOTE: with episodes=episode_ids passed to LeRobotDataset above,
    # episode_data_index is re-indexed POSITIONALLY (0..len(episode_ids)-1,
    # in episode_ids' own order), not by the original global episode id --
    # see get_episode_data_index in lerobot's source. Index by position here,
    # not by ep_id's numeric value (which would IndexError or silently hit
    # the wrong episode for any id >= n_episodes).
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
            frame_next = ds[t + 1]

            image_t = to_uint8_hwc(frame_t[camera_key])
            next_frame_img = to_uint8_hwc(frame_next[camera_key])
            state_t = frame_t["observation.state"].numpy()
            true_action = frame_t["action"].numpy()  # [x,y,z,roll,pitch,yaw,gripper], native Bridge space

            model.reset(task_description)
            model.image_history.append(model._preprocess_history_image(image_t))
            model._current_state = model._build_state(pose_from_bridge_state(state_t), float(state_t[7]))

            pred_chunk = model.predict_action_given_true_next_frame(image_t, task_description, next_frame_img)
            pred_action = pred_chunk[0]  # first step of the chunk, comparable to a single-step true_action

            action_l1s.append(np.abs(pred_action - true_action).mean())
            pos_l1s.append(np.abs(pred_action[0:3] - true_action[0:3]).mean())
            grip_l1s.append(abs(pred_action[6] - true_action[6]))
            action_l1s_baseline.append(np.abs(true_action).mean())  # "predict zero action" baseline
            print(f"sample {len(action_l1s)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s", flush=True)

    action_l1s, pos_l1s, grip_l1s, action_l1s_baseline = map(
        np.array, (action_l1s, pos_l1s, grip_l1s, action_l1s_baseline)
    )
    print(f"n samples: {len(action_l1s)}")
    print(f"full action L1 (oracle):        {action_l1s.mean():.4f}  (sd {action_l1s.std():.4f})")
    print(f"full action L1 (zero-action baseline): {action_l1s_baseline.mean():.4f}")
    print(f"position L1 (oracle):           {pos_l1s.mean():.4f}")
    print(f"gripper L1 (oracle):            {grip_l1s.mean():.4f}")


if __name__ == "__main__":
    run_oracle_probe(
        checkpoint_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune",
        stats_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
