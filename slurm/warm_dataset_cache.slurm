#!/bin/bash
#SBATCH --job-name=f1vla_cache_warm
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/cachewarm-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
/home/3295540/.conda/envs/f1_vla/bin/python /mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/warm_dataset_cache.py
