#!/bin/bash
#SBATCH --job-name=f1vla_chunk30
#SBATCH --partition=long_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/full-chunk30-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT=f1-vla-bridge
cd /mnt/beegfsnew/scratch/3295540/F1-VLA

/home/3295540/.conda/envs/f1_vla/bin/python train_hf.py \
  --config-file f1_vla/config/bridge_finetune_chunk30.yaml \
  exp.training_args.bf16=True \
  exp.training_args.report_to=[tensorboard,wandb]
