#!/usr/bin/env python3
"""
Offline probe for Experiment 1, variant 2 (F1-VLA): "shuffle the KV entries
across episodes" -- the clean version of variant 1, without its sequence-
shape confound.

Reuses Experiment 2's already-validated oracle mechanism
(../experiment2_oracle/oracle_offline_probe.py's F1VLAOracleInference,
modeling_f1.py's `oracle_indices` hook) unchanged -- no new model code. The
only difference from the oracle condition: instead of encoding THIS
episode's own real next frame and injecting it, this encodes a real next
frame from a DIFFERENT, randomly paired episode. Sequence shape, token
count, and position are identical to the oracle condition (real VQ tokens,
right place in the sequence) -- only the *content* is wrong-episode. This
directly isolates "is input-specific information used?" from variant 1's
confound ("is this an unfamiliar sequence shape?").

Uses the SAME sampling protocol/seed as variant 1 and Experiment 2 (24
episodes x 5 moments, seed=0) so all four conditions (baseline, ablated,
oracle, shuffled) are paired on the same underlying samples and directly
comparable -- see ../README.md for the combined table.
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
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment2_oracle")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from oracle_offline_probe import F1VLAOracleInference, pose_from_bridge_state  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
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


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def run_kv_shuffle_probe(
    checkpoint_path: str,
    stats_path: str,
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",
    n_episodes: int = 24,
    samples_per_episode: int = 5,
    seed: int = 0,
):
    model = F1VLAOracleInference(checkpoint_path=checkpoint_path, stats_path=stats_path, seed=seed)

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

    # Same (episode, t) sampling as variant 1 / Experiment 2, same seed and
    # order, so this list is identical to what those scripts iterated over --
    # keeps all conditions paired on the same underlying moments.
    samples = []  # (pos, ep_id, t, task_description)
    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < 3:
            continue
        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - 1)
        ts = rng.choice(candidate_ts, size=min(samples_per_episode, len(candidate_ts)), replace=False)
        for t in ts:
            samples.append((pos, ep_id, int(t), task_description))

    # Pair each sample with a "foresight source" from a DIFFERENT episode --
    # shift by samples_per_episode so consecutive samples (same episode's
    # block) never pair with themselves; wraps around the list.
    n = len(samples)
    shuffle_offset = samples_per_episode if samples_per_episode < n else 1
    pairs = [(i, (i + shuffle_offset) % n) for i in range(n)]
    for i, j in pairs:
        assert samples[i][1] != samples[j][1], f"sample {i} paired with its own episode {samples[i][1]}"

    l1_shuffled = []

    for i, j in pairs:
        t0 = time.time()
        _, ep_id, t, task_description = samples[i]
        _, src_ep_id, src_t, _ = samples[j]

        frame_t = ds[t]
        image_t = to_uint8_hwc(frame_t[camera_key])
        true_action = frame_t["action"].numpy()

        # Wrong-episode "foresight source": a real next frame, but from a
        # different episode/moment than the one we're predicting for.
        wrong_next_frame = to_uint8_hwc(ds[src_t + 1][camera_key])

        model.reset(task_description)
        model.image_history.append(model._preprocess_history_image(image_t))
        state_t = frame_t["observation.state"].numpy()
        model._current_state = model._build_state(pose_from_bridge_state(state_t), float(state_t[7]))

        pred_chunk = model.predict_action_given_true_next_frame(image_t, task_description, wrong_next_frame)
        pred_action = pred_chunk[0]

        l1_shuffled.append(np.abs(pred_action - true_action).mean())
        print(f"sample {len(l1_shuffled)} (ep {ep_id}, t {t}, foresight from ep {src_ep_id}): "
              f"{time.time() - t0:.1f}s", flush=True)

    l1_shuffled = np.array(l1_shuffled)
    print(f"n samples: {len(l1_shuffled)}")
    print(f"full action L1, shuffled (wrong-episode foresight): {l1_shuffled.mean():.4f}  (sd {l1_shuffled.std():.4f})")


if __name__ == "__main__":
    run_kv_shuffle_probe(
        checkpoint_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune",
        stats_path="/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
