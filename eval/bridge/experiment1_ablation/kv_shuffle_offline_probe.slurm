#!/bin/bash
#SBATCH --job-name=f1_kvshuffle_offline
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/kvshuffle-offline-%j.out

# Experiment 1, variant 2 (F1-VLA): full offline probe, 24 episodes x 5
# samples, same seed/protocol as variant 1 and Experiment 2 for direct
# comparability.
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation/kv_shuffle_offline_probe.py
