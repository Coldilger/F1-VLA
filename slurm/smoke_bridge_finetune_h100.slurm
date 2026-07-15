#!/bin/bash
#SBATCH --job-name=f1vla_smoke_h100
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/smoke-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
cd /mnt/beegfsnew/scratch/3295540/F1-VLA

/home/3295540/.conda/envs/f1_vla/bin/python train_hf.py \
  --config-file f1_vla/config/bridge_finetune.yaml \
  exp.training_args.output_dir=outputs/bridge_finetune_smoke_h100 \
  exp.training_args.run_name=bridge_finetune_smoke_h100 \
  exp.training_args.max_steps=20 \
  exp.training_args.logging_steps=1 \
  exp.training_args.torch_compile=False \
  exp.training_args.bf16=True
