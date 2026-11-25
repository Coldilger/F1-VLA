#!/bin/bash
#SBATCH --job-name=f1_diag_replay
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/F1-VLA/eval/bridge/experiment6_7_swap/diag-%j.out
ulimit -t unlimited
export MS2_REAL2SIM_ASSET_DIR=/home/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data DISPLAY=""
cd /home/3295540/F1-VLA/eval/bridge/SimplerEnv
for T in PutEggplantInBasketScene-v0 PutSpoonOnTableClothInScene-v0; do
  echo "== $T"; /home/3295540/.conda/envs/f1_vla_eval/bin/python ../experiment6_7_swap/diag_replay.py "$T" 0 2>&1 | grep -E '^step|Error|error' | grep -v GLFW
done
