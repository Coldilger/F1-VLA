#!/bin/bash
#SBATCH --job-name=f1_cmp_targets
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/cmptargets-%j.out

set -u
DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing
PY=/home/3295540/.conda/envs/f1_vla_eval/bin/python
cd "$DIR"
for VARIANT in finetuned pretrained; do
  echo ""
  echo "############ $VARIANT ############"
  "$PY" compare_targets.py --features "features_${VARIANT}.npz"
done
echo "CMP_EXIT=$?"
