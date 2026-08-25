#!/bin/bash
#SBATCH --job-name=f1_train_probe_live
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/probe-live-%j.out

# Exp4, live-rollout variant, step 2: train_probe.py unmodified, pointed at
# whichever live-extraction job (n24 H100 or its H200 twin) finished first.
# FEATURES is set by the submitting sbatch call, e.g.:
#   sbatch --export=ALL,FEATURES=features_live_n24.npz train_probe_live.slurm

set -u
DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing
PY=/home/3295540/.conda/envs/f1_vla_eval/bin/python
cd "$DIR"

echo "features file: ${FEATURES}"
"$PY" train_probe.py \
  --features "${FEATURES}" \
  --json-out "probe_live_$(basename "${FEATURES}" .npz).json"
