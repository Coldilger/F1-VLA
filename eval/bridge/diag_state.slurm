#!/bin/bash
#SBATCH --job-name=f1_diag_state
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/diag-state-%j.out
export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv
/home/3295540/.conda/envs/f1_vla_eval/bin/python /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/diag_state.py
