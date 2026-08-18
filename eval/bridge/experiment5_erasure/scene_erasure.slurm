#!/bin/bash
#SBATCH --job-name=f1_scene_erasure
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/erasure-%j.out

set -u
DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment5_erasure
E4=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing
PY=/home/3295540/.conda/envs/f1_vla_eval/bin/python
cd "$DIR"
"$PY" scene_erasure_diagnostic.py \
  --features "$E4/features_${VARIANT}.npz" \
  --current-pose "$E4/current_pose.npz" \
  --json-out "result_${VARIANT}.json"
echo "ERASURE_EXIT=$? VARIANT=$VARIANT"
