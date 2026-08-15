#!/bin/bash
#SBATCH --job-name=f1_ablation_offline
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/ablation-offline-%j.out

# Experiment 1, variant 1 (F1-VLA): full offline probe, 24 episodes x 5
# samples, matching Experiment 2's protocol so the two are comparable.
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation/ablation_offline_probe.py
