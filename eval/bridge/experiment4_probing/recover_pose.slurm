#!/bin/bash
#SBATCH --job-name=f1_recover_pose
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/recover-pose-%j.out

set -u
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing
cd "$DIR"
/home/3295540/.conda/envs/f1_vla_eval/bin/python recover_current_pose.py \
  --features features_finetuned.npz \
  --out current_pose.npz
echo "RECOVER_EXIT=$?"
