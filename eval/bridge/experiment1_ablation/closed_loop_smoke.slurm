#!/bin/bash
#SBATCH --job-name=f1_ablated_cl_smoke
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/ablated-cl-smoke-%j.out

export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

SIMPLER=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

# 3-episode smoke test: confirm the ablated closed-loop wrapper runs
# end-to-end in the real simulator (not just the offline probe) before
# committing to the full 4-task x 3-seed x 24-episode grid.
/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment1_ablation/main_inference_ablated.py \
  --f1-checkpoint-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune \
  --f1-stats-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json \
  --f1-seed 0 \
  --ckpt-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 3 \
  --additional-env-save-tags ablated_smoke \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0
