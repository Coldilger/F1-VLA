#!/bin/bash
#SBATCH --job-name=f1_live_extract_smoke_h200
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/F1-VLA/slurm/logs/live-extract-smoke-h200-%j.out

# Live-rollout Exp4 feature extraction, smoke stage: does the whole pipeline
# (real rollout + per-tick pose buffering + per-replan gen_out capture +
# post-hoc label resolution) run end to end and produce a sane .npz, before
# committing to a larger episode count. Single F1-VLA instance (no second
# oracle model needed here).

export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache

SIMPLER=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

/home/3295540/.conda/envs/f1_vla_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/extract_features_live.py \
  --f1-checkpoint-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune \
  --f1-stats-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune/bridge_orig_stats.json \
  --f1-seed 0 \
  --out /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/features_live_smoke_h200.npz \
  --horizon 5 \
  --ckpt-path /mnt/beegfsnew/scratch/3295540/F1-VLA/outputs/bridge_finetune \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 6 \
  --additional-env-save-tags live_extract_smoke_h200 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0
