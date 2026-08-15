"""Closed-loop SimplerEnv wrapper for Experiment 1 variant 2 (F1-VLA):
"shuffle the KV entries across episodes". Subclasses F1VLAInference
unchanged, reusing Experiment 2's already-validated oracle_indices hook
(modeling_f1.py's sample_actions_with_world_model) -- only the injected
frame's source differs.

Real closed-loop control has no access to "the true next frame" at decision
time (it hasn't happened yet -- it depends on the very action being chosen),
so Experiment 2's oracle condition is offline-only by construction. Variant
2 doesn't have this problem: the injected frame comes from a completely
independent source (a random moment from the real Bridge dataset,
unconnected to the live rollout), which exists and is available regardless
of what decision is about to be made. A small pool of real Bridge frames is
loaded once at init and one is sampled per control step -- this is
literally "wrong episode" content, since the injected frame has nothing to
do with the live SimplerEnv episode at all.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")

from f1_vla_policy import F1VLAInference  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
    """Same patch as experiment2_oracle/oracle_offline_probe.py."""
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


def _to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _load_frame_pool(
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",
    pool_size: int = 64,
    seed: int = 123,
) -> list[np.ndarray]:
    """A fixed pool of real Bridge frames, loaded once, sampled from
    per-step. Independent of any specific SimplerEnv rollout by
    construction -- there is no "right" answer to be wrong about."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    rng = np.random.default_rng(seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=pool_size, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )
    frames = []
    for pos in range(len(episode_ids)):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        t = int(np.random.default_rng(seed + pos).integers(ep_from, ep_to))
        frames.append(_to_uint8_hwc(ds[t][camera_key]))
    return frames


class F1VLAShuffledInference(F1VLAInference):
    def __init__(self, *args, bridge_dataset_root: str, frame_pool_size: int = 64, seed: int | None = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        print(f"Loading {frame_pool_size}-frame shuffle pool from real Bridge data...", flush=True)
        self._shuffle_pool = _load_frame_pool(
            pathlib.Path(bridge_dataset_root), pool_size=frame_pool_size
        )
        # Independent, deterministic RNG for picking which pool frame to
        # inject per step -- tied to the same --f1-seed used for the world
        # model's own sampling, so each seed's run is reproducible.
        self._shuffle_rng = np.random.default_rng(seed if seed is not None else 0)
        print(f"Loaded {len(self._shuffle_pool)} frames.", flush=True)

    def _predict_new_chunk(self, image, task_description: str):
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

        idx = self._shuffle_rng.integers(0, len(self._shuffle_pool))
        wrong_frame = self._shuffle_pool[idx]
        future_img = self._preprocess_history_image(wrong_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            oracle_indices = self.policy.model.vae.img_to_idxBl(future_img)
            actions = self.policy.select_action_with_world_model(
                batch, rng=self._rng, oracle_indices=oracle_indices
            )
        actions = actions[0].cpu()
        actions = actions * self.action_std + self.action_mean
        return actions.numpy()
