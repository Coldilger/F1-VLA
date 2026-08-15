#!/bin/bash
#SBATCH --job-name=f1_kvshuffle_smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/kvshuffle-smoke-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
cd /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation

/home/3295540/.conda/envs/f1_vla_eval/bin/python -c "
from kv_shuffle_offline_probe import run_kv_shuffle_probe
import pathlib
run_kv_shuffle_probe(
    checkpoint_path='/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune',
    stats_path='/mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json',
    dataset_root=pathlib.Path('/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot'),
    n_episodes=4,
    samples_per_episode=2,
)
"
