import contextlib
import os
import pathlib
import sys
import time

DATASET_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")


@contextlib.contextmanager
def fast_path_is_file():
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


def log(msg):
    print(f"[{time.time():.1f}] {msg}", flush=True)


log("start")
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

log("imports done")

t0 = time.time()
meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(DATASET_ROOT))
log(f"metadata loaded in {time.time() - t0:.2f}s, total_episodes={meta.total_episodes}, root={meta.root}")

t0 = time.time()
with fast_path_is_file():
    ds = LeRobotDataset(
        repo_id="bridge_orig_lerobot",
        root=str(DATASET_ROOT),
        episodes=list(range(50)),  # small slice first, not all 53k
        video_backend="pyav",
    )
log(f"LeRobotDataset (50 episodes, pyav) built in {time.time() - t0:.2f}s, "
    f"num_episodes={ds.num_episodes}, num_frames={ds.num_frames}")

t0 = time.time()
item = ds[0]
log(f"ds[0] fetched in {time.time() - t0:.2f}s, keys={list(item.keys())}")
log(f"task={item.get('task')!r}")
img = item["observation.images.image_0"]
log(f"image_0 shape={tuple(img.shape)} dtype={img.dtype} min={img.min():.3f} max={img.max():.3f}")
log("done")
