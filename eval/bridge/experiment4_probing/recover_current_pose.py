#!/usr/bin/env python3
"""
Recovers the current end-effector pose for each already-extracted sample.

The sharper positive control (see mimic-video's copy of this script for the
full reasoning): episode identity is a global scene property; future/current
pose is spatially localized. If the pooled features can't recover current
pose -- unambiguously present in the input frame -- the extraction/pooling is
broken and no conclusion about future information is possible.

No GPU and no model: extract_features.py's sampling is fully determined by its
seed, so replaying the same RNG draws in the same order recovers the exact
(episode, t) pairs. The replayed episode ids are checked against the saved
ones and the script refuses to write anything if they disagree.

NOTE: uses Rotation.from_euler("xyz", ...) (lowercase, intrinsic), matching
F1's own extract_features.py/oracle_offline_probe.py convention -- NOT
mimic-video's "XYZ" (uppercase, extrinsic). Silently using the wrong
convention wouldn't error, just produce a subtly wrong pose vector -- copied
exactly from extract_features.py's pose_vector rather than mimic's script.
"""

import argparse
import contextlib
import os
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402


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


def pose_vector(state: np.ndarray) -> np.ndarray:
    pos = state[0:3].astype(np.float64)
    rot6 = Rotation.from_euler("xyz", state[3:6].astype(np.float64)).as_matrix()[:2].reshape(6)
    return np.concatenate([pos, rot6])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="npz written by extract_features.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--samples-per-episode", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-root", default="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    args = ap.parse_args()

    saved = np.load(args.features, allow_pickle=True)
    saved_eps = saved["episode_ids"]

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

    cur_poses, replayed_eps = [], []
    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < args.horizon + 2:
            continue
        candidate_ts = np.arange(ep_from, ep_to - args.horizon)
        ts = rng.choice(
            candidate_ts, size=min(args.samples_per_episode, len(candidate_ts)), replace=False
        )
        for t in ts:
            state_t = ds[int(t)]["observation.state"].numpy()
            cur_poses.append(pose_vector(state_t))
            replayed_eps.append(ep_id)

    replayed_eps = np.array(replayed_eps)
    if not np.array_equal(replayed_eps, saved_eps):
        print("REPLAY_MISMATCH: recovered sample order differs from the saved features")
        print(f"  saved    n={len(saved_eps)}  first 10: {saved_eps[:10]}")
        print(f"  replayed n={len(replayed_eps)}  first 10: {replayed_eps[:10]}")
        raise SystemExit(1)

    cur_poses = np.stack(cur_poses).astype(np.float32)
    np.savez_compressed(args.out, current_pose=cur_poses, episode_ids=replayed_eps)
    print(f"REPLAY_OK: {len(cur_poses)} samples matched the saved episode order")
    print(f"saved {args.out}: current_pose {cur_poses.shape}")


if __name__ == "__main__":
    main()
