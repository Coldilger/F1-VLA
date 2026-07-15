import os

os.environ.setdefault("HF_HOME", "/mnt/beegfsnew/scratch/3295540/hf_cache")

import sys
import time

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/F1-VLA")

from f1_vla.src.processors.data_processors.data_loader import _fast_path_is_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

DS = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"
delta_timestamps = {"observation.images.image_0": [0.0], "action": [0.0]}

print(f"HF_HOME={os.environ['HF_HOME']}", flush=True)

meta = LeRobotDatasetMetadata("bridge_orig", root=DS)
all_episodes = list(range(meta.total_episodes))
print(f"total_episodes={meta.total_episodes}", flush=True)

t0 = time.time()
with _fast_path_is_file():
    ds = LeRobotDataset(DS, episodes=all_episodes, delta_timestamps=delta_timestamps, video_backend="pyav")
print(f"FIRST_LOAD_DONE: {time.time()-t0:.1f}s, episodes={ds.num_episodes}, frames={ds.num_frames}", flush=True)

# Immediately reload to confirm the cache actually speeds things up before we
# trust it for the real training job.
t0 = time.time()
with _fast_path_is_file():
    ds2 = LeRobotDataset(DS, episodes=all_episodes, delta_timestamps=delta_timestamps, video_backend="pyav")
print(f"SECOND_LOAD_DONE (should be fast if cache works): {time.time()-t0:.1f}s, "
      f"episodes={ds2.num_episodes}, frames={ds2.num_frames}", flush=True)

print("ALL_DONE", flush=True)
