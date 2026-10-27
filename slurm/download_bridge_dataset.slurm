#!/bin/bash
#SBATCH --job-name=bridge_dl
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:10:00
#SBATCH --output=/home/3295540/F1-VLA/slurm/logs/bridge-dl-%j.out
# Re-downloads IPEC-COMMUNITY/bridge_orig_lerobot (LeRobot v2.0, 53,192
# episodes, ~21 GB) to the shared data/ location every repo's training and
# offline scripts point at (lost in the 2026-10 scratch purge). LDA-1B's
# bridge_lda is converted from this copy by LDA-1B/slurm/convert_bridge.slurm.
#
# Self-chaining: the full download can outlast medium_cpu's 6h limit, so each
# run queues a follow-up that starts only if this one ends badly (timeout,
# failure). The downloader skips files already on disk, so the follow-up
# resumes. Capped at MAX_WAVES runs so a persistent error can't loop forever.
set -euo pipefail
WAVE=${WAVE:-1}
MAX_WAVES=4
if [ "$WAVE" -lt "$MAX_WAVES" ]; then
  sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes \
         --export=ALL,WAVE=$((WAVE + 1)) /home/3295540/F1-VLA/slurm/download_bridge_dataset.slurm >/dev/null
fi
echo "wave $WAVE/$MAX_WAVES"

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="$(cat /home/3295540/.cache/huggingface/token)"
DST=/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot
mkdir -p "$DST"

# meta/ first: the downloader builds every file URL from meta/info.json
/home/3295540/.conda/envs/f1_vla/bin/python - <<EOF
from huggingface_hub import hf_hub_download
for f in ["meta/info.json", "meta/stats.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
    hf_hub_download("IPEC-COMMUNITY/bridge_orig_lerobot", f, repo_type="dataset", local_dir="$DST")
EOF

/home/3295540/.conda/envs/f1_vla/bin/python /home/3295540/F1-VLA/slurm/download_bridge_dataset.py
du -sh --exclude=.cache "$DST"
