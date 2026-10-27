"""Resumable downloader for IPEC-COMMUNITY/bridge_orig_lerobot (~53k parquet +
~213k mp4 files). snapshot_download makes one Hub API call per file and hits
the 1000-req/5-min API limit almost immediately; this instead builds the
/resolve/ URLs from meta/info.json and stays under the separate resolver
limit (observed 5000 req / 300 s). Skips files already on disk, so rerunning
continues where a previous run stopped. Recovered verbatim (plus this
docstring) from the July 2026 script that produced the original copy.
Run via slurm/download_bridge_dataset.slurm.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

DS = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"
REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TOKEN = os.environ["HF_TOKEN"]
WORKERS = 20
# HF "resolvers" rate limit observed: 5000 req / 300s (fixed window) = 16.67 req/s.
# Stay a bit under that to leave headroom for retries.
RATE_PER_SEC = 14.0

info = json.load(open(f"{DS}/meta/info.json"))
total_episodes = info["total_episodes"]
chunks_size = info["chunks_size"]
camera_keys = sorted(k for k in info["features"] if k.startswith("observation.images"))

existing_parquet = set()
if os.path.isdir(f"{DS}/data"):
    out = subprocess.run(["find", f"{DS}/data", "-name", "*.parquet"], capture_output=True, text=True).stdout
    existing_parquet = {os.path.relpath(p, DS) for p in out.splitlines()}

existing_videos = set()
if os.path.isdir(f"{DS}/videos"):
    out = subprocess.run(["find", f"{DS}/videos", "-name", "*.mp4"], capture_output=True, text=True).stdout
    existing_videos = {os.path.relpath(p, DS) for p in out.splitlines()}

todo = []
for j in range(total_episodes):
    chunk = j // chunks_size
    p = f"data/chunk-{chunk:03d}/episode_{j:06d}.parquet"
    if p not in existing_parquet:
        todo.append(p)
    for cam in camera_keys:
        p = f"videos/chunk-{chunk:03d}/{cam}/episode_{j:06d}.mp4"
        if p not in existing_videos:
            todo.append(p)

print(f"files to download: {len(todo)}", flush=True)
print(f"estimated time at {RATE_PER_SEC}/s: {len(todo)/RATE_PER_SEC/3600:.1f} hours", flush=True)

dirs = {os.path.dirname(p) for p in todo}
for d in dirs:
    os.makedirs(os.path.join(DS, d), exist_ok=True)

session = requests.Session()
adapter = HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS, max_retries=0)
session.mount("https://", adapter)
session.headers.update({"Authorization": f"Bearer {TOKEN}"})

_lock = threading.Lock()
_next_time = [time.monotonic()]
_interval = 1.0 / RATE_PER_SEC


def throttle():
    with _lock:
        now = time.monotonic()
        target = max(now, _next_time[0])
        _next_time[0] = target + _interval
    delay = target - now
    if delay > 0:
        time.sleep(delay)


def _dl(rel_path):
    url = f"{BASE_URL}/{rel_path}"
    dest = os.path.join(DS, rel_path)
    tmp = dest + ".part"
    for attempt in range(6):
        throttle()
        try:
            with session.get(url, stream=True, timeout=60) as r:
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 20))
                    r.close()
                    time.sleep(min(retry_after, 60))
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            os.replace(tmp, dest)
            return rel_path, None
        except Exception as e:
            if attempt == 5:
                return rel_path, str(e)
            time.sleep(3 * (attempt + 1))
    return rel_path, "exhausted retries (429s)"


done = 0
failed = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {}
    it = iter(todo)
    # bound in-flight futures instead of submitting all ~250k at once (memory)
    for _ in range(min(WORKERS * 4, len(todo))):
        p = next(it, None)
        if p is None:
            break
        futures[ex.submit(_dl, p)] = p

    while futures:
        fut = next(as_completed(futures))
        del futures[fut]
        rel_path, err = fut.result()
        done += 1
        if err:
            failed.append((rel_path, err))
        nxt = next(it, None)
        if nxt is not None:
            futures[ex.submit(_dl, nxt)] = nxt
        if done % 250 == 0 or done == len(todo):
            rate = done / (time.time() - t0)
            eta_min = (len(todo) - done) / max(rate, 0.1) / 60
            print(f"[{done}/{len(todo)}] rate={rate:.1f}/s failed={len(failed)} eta={eta_min:.0f}min", flush=True)

print("DOWNLOAD_DONE", flush=True)
print(f"total_failed={len(failed)}", flush=True)
for rel_path, err in failed[:80]:
    print("FAIL", rel_path, err, flush=True)
sys.exit(0 if not failed else 1)
