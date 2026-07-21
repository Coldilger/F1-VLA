#!/bin/bash
#SBATCH --job-name=f1vla_chunk30
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/full-chunk30-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT=f1-vla-bridge
cd /mnt/beegfsnew/scratch/3295540/F1-VLA

# Auto-resume: if a checkpoint already exists, continue from it
# (overwrite_output_dir=False lets train_hf pick up the last checkpoint).
# On a clean first run the output dir has no checkpoint yet, so use
# overwrite_output_dir=True to avoid the "dir exists and is not empty" abort
# caused by the stats/config files train_hf writes early.
if ls outputs/bridge_finetune_chunk30/checkpoint-* >/dev/null 2>&1; then
  OVERWRITE=False
  echo ">>> checkpoint found -> resuming (overwrite_output_dir=False)"
else
  OVERWRITE=True
  echo ">>> no checkpoint -> clean start (overwrite_output_dir=True)"
fi

/home/3295540/.conda/envs/f1_vla/bin/python train_hf.py \
  --config-file f1_vla/config/bridge_finetune_chunk30.yaml \
  exp.training_args.bf16=True \
  exp.training_args.report_to=[tensorboard,wandb] \
  exp.training_args.overwrite_output_dir=$OVERWRITE
