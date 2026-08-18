#!/bin/bash
#SBATCH --job-name=f1_probe_shape
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/probe-shape-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/probe_true_shape.py

echo "PROBESHAPE_EXIT=$?"
