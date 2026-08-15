#!/bin/bash
#SBATCH --job-name=f1_oracle_offline
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/oracle-offline-%j.out

# Offline oracle-injection probe (Experiment 2). No SimplerEnv/ManiSkill2/vulkan
# needed -- reads bridge_orig_lerobot directly and injects ground-truth VQ
# tokens into the VAR foresight loop (see modeling_f1.py's oracle_indices hook).
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment2_oracle/oracle_offline_probe.py
