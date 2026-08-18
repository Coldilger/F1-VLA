#!/bin/bash
#SBATCH --job-name=f1_probe_extract
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/extract-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/extract_features.py \
  --variant "$VARIANT" \
  --out "$OUT" \
  --n-episodes "${N_EPISODES:-40}" \
  --samples-per-episode "${SAMPLES_PER_EPISODE:-10}"

echo "EXTRACT_EXIT=$? VARIANT=$VARIANT"
