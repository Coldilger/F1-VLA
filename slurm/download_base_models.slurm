#!/bin/bash
#SBATCH --job-name=f1_base_dl
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3295540/F1-VLA/slurm/logs/base-dl-%j.out
# Re-downloads the base checkpoints that the Bridge finetune config and the
# eval policy point at under data/checkpoints/ (lost in the 2026-10 scratch
# purge). Layout matches f1_vla/config/bridge_finetune.yaml exactly.
# pi0_base is pinned to the revision current at the original July download
# (later commits only touched the model card).
set -euo pipefail
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_TOKEN="$(cat /home/3295540/.cache/huggingface/token)"   # paligemma is gated
CK=/mnt/beegfsnew/scratch/3295540/data/checkpoints
mkdir -p "$CK"

/home/3295540/.conda/envs/f1_vla/bin/python - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
ck = "$CK"
snapshot_download("InternRobotics/F1-VLA", local_dir=f"{ck}/f1_vla_stage2")
snapshot_download("lerobot/pi0_base", revision="25c379b52b", local_dir=f"{ck}/pi0_base")
snapshot_download("google/paligemma-3b-pt-224", local_dir=f"{ck}/paligemma_tokenizer",
                  allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json"])
hf_hub_download("FoundationVision/var", "vae_ch160v4096z32.pth", local_dir=f"{ck}/var_vae")
EOF
echo "DOWNLOAD_EXIT=$?"
du -sh "$CK"/*
